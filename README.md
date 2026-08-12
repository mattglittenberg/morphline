# morphline

**A BIDS-aware longitudinal neuroimaging derivatives pipeline.** Ingests FreeSurfer `.stats` output, accounts for every observation it gains and loses, quality-controls it, harmonizes across scanners, fits longitudinal mixed-effects models, and emits a self-contained HTML report you could reconstruct the entire run from.

> **Status: past the `v0.1.0` walking skeleton.** Ingestion, data accounting, QC, and ComBat harmonization are implemented and validated; the model still fits a single region. See [Build.md](Build.md) for what lands when.

<!-- Badges: replace OWNER with the GitHub account before the repo goes public. -->
[![CI](https://github.com/OWNER/morphline/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/morphline/actions/workflows/ci.yml)
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
- **amd64 only.** arm64 is a timeboxed follow-on, not a promise.
- **Real-data integration is demonstrated, not completed.** The ABIDE PCP adapter reads real per-subject FreeSurfer 5.1 `.stats` files and has been verified end to end on a small real sample — parsed with no failures, full 28-region coverage, sites resolved, accounting funnel reconciling. The full-scale run over all 1112 subjects has not been done, so no claim rests on it. Longitudinal integration targets a public OpenNeuro accession and is contingent on an accession that actually ships `.stats` files alongside ≥2 structural sessions per subject.
- **ABIDE is cross-sectional, and the modeling stage says so by producing nothing.** One session per participant means the longitudinal formula has no time variation to fit, so every observation is reported as attributable loss at the modeling boundary rather than silently yielding a coefficient. That is the intended behaviour: a cross-sectional dataset should not be able to produce a longitudinal result by accident.
- **Aggregated tables are not a parser test.** The widely-linked ABIDE FreeSurfer 6 deposit ships `asegstats2table` output — one row per subject — which bypasses the parser entirely and validates only the adapter. morphline therefore targets ABIDE PCP's *per-subject* `.stats` files (FreeSurfer 5.1, 1112 subjects, 17 sites) for parser validation and keeps the FS6 tables as an independent cross-check. Reading real per-subject files immediately found two silent defects that the synthetic fixtures had not: `?h.aparc.stats` reports all three of its header measures under the short name `Cortex`, so keying on that name alone kept only one of them; and intracranial volume is `IntraCranialVol, ICV` on FreeSurfer 5.1 against `EstimatedTotalIntraCranialVol, eTIV` on 6+, so neither field alone resolves eTIV across versions. Both are fixed — header measures are now indexed by short name *and* alias — and both are regression-tested against the real header shapes. The fixture generator emits those shapes too, since omitting them is what let the first defect survive.
- **The longitudinal path is validated against synthetic ground truth, not real longitudinal data.** This is stated first because it is the limitation a reviewer should notice. Public longitudinal *and* multi-scanner MRI with redistributable FreeSurfer derivatives barely exists: OASIS-3 is the obvious candidate and requires institutional registration, which this project does not have. Rather than wait on an approval that cannot arrive, morphline validates within-subject slope recovery against injected truth — which real data cannot do, since no real dataset has a known true slope — and names the gap here. When an OpenNeuro accession lands, expect it to validate the longitudinal path but *not* the scanner/time confound; those are separate claims and this README will keep them separate.
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
