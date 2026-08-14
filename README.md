# morphline

**A BIDS-aware longitudinal neuroimaging derivatives pipeline.** Ingests FreeSurfer `.stats` output, accounts for every observation it gains and loses, quality-controls it, harmonizes across scanners, fits longitudinal mixed-effects models, and emits a self-contained HTML report you could reconstruct the entire run from.

> **Status: past the `v0.1.0` walking skeleton.** Ingestion, data accounting, QC, and ComBat harmonization are implemented and validated; the model still fits a single region. See [Build.md](Build.md) for what lands when.

[![CI](https://github.com/mattglittenberg/morphline/actions/workflows/ci.yml/badge.svg)](https://github.com/mattglittenberg/morphline/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Quickstart

No data required — the pipeline generates its own fixtures.

```bash
uv sync
uv run morphline run --config config/test.yaml --outdir results/
open results/report.html
```

Or through Nextflow, which is how it runs in CI:

```bash
nextflow run . -profile test,docker --outdir results/
```

---

## Why synthetic fixtures are the primary substrate

Development runs against a synthetic stats-file generator, not real data. This is not a compromise — it is the stronger engineering position:

- It forces the ingestion boundary to be a real interface rather than an accident.
- It makes CI possible at all: a cold clone runs the full pipeline with zero external data and zero data-use agreements.
- **It permits statistical validation that real data cannot support.** Real data has no known true slope, so "did the model recover the effect?" is unanswerable. The generator injects known site effects, age effects, diagnosis effects, and a known diagnosis × time interaction, then the test suite asks whether each stage recovers them.

Real datasets are integration targets the ingestion layer points at — not the thing that gates progress.

## Architecture

Parsing and entity resolution are separate concerns, and the separation is load-bearing:

```
stats files
    ↓
FreeSurferStatsParser          knows aseg/aparc structure ONLY
    ↓                          knows nothing about datasets, sites, or BIDS
parsed FreeSurfer records
    ↓
DatasetAdapter                 knows subject/session conventions, site,
    ↓                          scanner, demographics, dataset layout
canonical observations (Parquet)
    ↓
accounting → QC → harmonization → modeling → report
```

**Architectural rule: everything downstream of ingestion reads only the canonical schema.** QC, harmonization, modeling, and reporting never import the parser, never touch a `.stats` file, and never contain a FreeSurfer-specific string. Adding a fifth dataset requires zero changes downstream of the adapter layer.

That rule is enforced by [`tests/test_architecture_boundary.py`](tests/test_architecture_boundary.py), which walks the AST of every stage module. A docstring asking nicely does not survive a deadline; a failing test does.

## Data accounting

Every boundary reports what it gained and lost, and every drop has a stated cause:

```
raw files → parsed files → canonical observations → QC-passing observations → modeled observations
```

Unexplained loss is treated as a bug, not a rounding error. Parse failures carry machine-readable reason codes rather than vanishing.

---

## Honest scope

Written before the work, kept accurate as it lands.

- **`pybids` is not a derivatives parser.** It indexes raw BIDS well; its BIDS-Derivatives support is partial, and FreeSurfer's native `SUBJECTS_DIR` output is not BIDS-organized at all. morphline uses a purpose-built resolver for FreeSurfer subject-directory conventions and reserves pybids for raw layout traversal inside adapters.
- **Surface holes are not Euler numbers.** Hole counts are extracted verbatim from the `aseg.stats` header; the Euler number is *derived* as `2 - 2 × holes`. On FreeSurfer 5.3, which reports no hole counts, the value is null — never zero, which would imply a flawless surface and rank the oldest data in a study as its best.
- **ComBat is implemented in-repo rather than via `neuroHarmonize`.** That package pins `numpy==1.26.4`, which would hold the entire project on the numpy 1.x line. Implementing the empirical-Bayes estimator directly removes the pin and turns the harmonization test suite into a test of *this* estimator against injected truth, rather than a test that someone else's library executed.

  The obvious objection is that a hand-rolled estimator can quietly diverge from the method it cites, so morphline checks: `tests/test_combat_crosscheck.py` compares adjusted values, γ\*, and δ\* against `neuroCombat` and requires agreement to 1e-6. It runs in CI behind an optional extra (`uv sync --extra combat-xcheck`), kept out of the default dependency set so the numpy floor stays put.

  **That cross-check earned its place immediately: it found a real defect.** The shrinkage was pooling across batches within a region, where Johnson et al. (2007) pool across regions within a batch — so each prior was built from 3 values instead of 28. Every internal test passed anyway, because a wrong-axis prior still shrinks and still recovers injected effects. Only an independent implementation of the same paper disagreed. Fixing it measurably improved recovery. Residual disagreement is now ~7 significant figures and runs the other way: `neuroCombat` solves the design by explicit normal equations, morphline by SVD least squares.
- **amd64 only.** arm64 is a timeboxed follow-on, not a promise.
- **Real-data integration is cross-sectional only.** The ABIDE PCP adapter reads real per-subject FreeSurfer 5.1 `.stats` files, and the full-scale run over all 1112 subjects is done: 3314 files parsed with zero failures, 30,910 canonical observations, and an accounting funnel that reconciles with zero unexplained loss — including the 9 known-incomplete subjects, attributed by reason code rather than quietly absent. That validates the parser, adapter, accounting, QC, and harmonization stages against real files. It validates **nothing longitudinal**. Longitudinal integration targets a public OpenNeuro accession and is contingent on one that actually ships `.stats` files alongside ≥2 structural sessions per subject.
- **ABIDE is cross-sectional, and the modeling stage says so by producing nothing.** One session per participant means the longitudinal formula has no time variation to fit, so every observation is reported as attributable loss at the modeling boundary rather than silently yielding a coefficient. That is the intended behaviour: a cross-sectional dataset should not be able to produce a longitudinal result by accident.
- **Aggregated tables are not a parser test.** The widely-linked ABIDE FreeSurfer 6 deposit ships `asegstats2table` output — one row per subject — which bypasses the parser entirely and validates only the adapter. morphline therefore targets ABIDE PCP's *per-subject* `.stats` files (FreeSurfer 5.1, 1112 subjects, 17 sites) for parser validation, and uses the aggregate tables as an independent cross-check of the ingestion path — see the next entry, which is not the cross-check that was planned. Reading real per-subject files immediately found two silent defects that the synthetic fixtures had not: `?h.aparc.stats` reports all three of its header measures under the short name `Cortex`, so keying on that name alone kept only one of them; and intracranial volume is `IntraCranialVol, ICV` on FreeSurfer 5.1 against `EstimatedTotalIntraCranialVol, eTIV` on 6+, so neither field alone resolves eTIV across versions. Both are fixed — header measures are now indexed by short name *and* alias — and both are regression-tested against the real header shapes. The fixture generator emits those shapes too, since omitting them is what let the first defect survive.
- **That FreeSurfer 6 deposit is not FreeSurfer 6, at least in its stats tables.** The cross-check was planned as a cross-version comparison — our FreeSurfer 5.1 numbers against someone else's FreeSurfer 6 run, on rank correlation, because the releases differ systematically. Building it found that all 28,976 shared observations (1035 subjects × 28 regions, mm³ volumes and mm thickness alike) are *bit-identical* to the ABIDE PCP FreeSurfer 5.1 values, maximum absolute difference `0.0`. Two `recon-all` runs at different releases cannot do that. The deposit's own independently-derived `brainvol` file corroborates it, carrying the FreeSurfer 6 column name `EstimatedTotalIntraCranialVol` against the FreeSurfer 5.1 value our `aseg.stats` reports as `IntraCranialVol, ICV`. Its README states FreeSurfer v6, so this is a discrepancy in the deposit rather than a misreading of it; the finding is scoped to what was checked (the Desikan aparc and aseg tables, plus `brainvol`) and says nothing about its Zenodo volume, mesh, or lGI archives.

  What survives is a **sharper** check than the one planned. `asegstats2table` and `aparcstats2table` are FreeSurfer's own tools, run by a third party over the same source files, so the expected answer is exact equality — and a mis-read column, a mis-mapped `StructName`, or a swapped hemisphere shows up as an exact mismatch instead of vanishing into "well, the versions differ." It passes. What it cannot show, and no longer claims to: version tolerance, or that the values are *correct* — both sides descend from one recon-all run, so a segmentation error is present in both and invisible here.
- **The longitudinal path is validated against synthetic ground truth, not real longitudinal data.** This is stated first because it is the limitation a reviewer should notice. Public longitudinal *and* multi-scanner MRI with redistributable FreeSurfer derivatives barely exists: OASIS-3 is the obvious candidate and requires institutional registration, which this project does not have. Rather than wait on an approval that cannot arrive, morphline validates within-subject slope recovery against injected truth — which real data cannot do, since no real dataset has a known true slope — and names the gap here.

  **This is a searched-for gap, not an assumed one.** [`scripts/audit_openneuro.py`](scripts/audit_openneuro.py) audits OpenNeuro for accessions shipping FreeSurfer `.stats` files alongside ≥2 structural sessions per subject, using the GraphQL index and the anonymously-listable S3 bucket. The best candidate is **Penn LEAD** (CC0, published as raw [`ds007116`](https://openneuro.org/datasets/ds007116) plus two fMRIPrep derivative accessions), and it is a near miss worth describing precisely: **73 of its 132 subjects have ≥2 sessions and 20 have three**, its session tables carry per-session `age` and interval-preserving `acq_time`, and `participants.tsv` carries sex and ten diagnosis columns — so the model specification is fully supported by the metadata. What fails is the derivatives. The 131-subject anatomical deposit ran one FreeSurfer pass per subject with `--fs-no-resume` against a shared subjects directory, so its stats belong to no single timepoint; per-session stats survive only as an incidental byproduct in the *functional* derivatives, for 8 subjects. Eight will not carry random slopes across 28 tests. Full evidence in [docs/openneuro_audit.md](docs/openneuro_audit.md).

  The generalizable lesson, which cost an afternoon: **an accession can ship complete FreeSurfer derivatives and still not be longitudinal**, because the pipeline that produced them collapsed sessions. No filename search detects that — it takes reading the deposit's own processing invocation and counting session entities in directory names.

  Penn LEAD is therefore the named post-ship target, at the cost of a FreeSurfer derivation run over its repeat sessions. It is single-scanner (one Siemens Prisma_fit 3T, one console software version, across every sidecar sampled), so when it lands expect it to validate the longitudinal path but *not* the scanner/time confound. Those are separate claims and this README will keep them separate.
- **IXI is a better harmonization dataset than ABIDE, and is not used.** Three scanners, healthy subjects only, so a batch effect is not entangled with case-mix the way ABIDE's site effects are. It distributes NIfTI images with no FreeSurfer derivatives, so feeding it to a `.stats`-ingesting pipeline costs ~600 `recon-all` runs. That is a post-v1.0 item, and until then ABIDE's site/diagnosis entanglement is a real caveat on the harmonization results.

Limitations that will matter more as the statistics land — the scanner/time non-identifiability problem, the sensitivity-analysis-versus-inference distinction, the missing-at-random assumption — are documented in [BUILD_PLAN.md](BUILD_PLAN.md) §2.3.1 and §2.5.4 and will be restated here as those stages are built.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/
uv run pytest -v
```

## License

MIT — see [LICENSE](LICENSE).
