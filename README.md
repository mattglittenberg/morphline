# morphline

**A BIDS-aware longitudinal neuroimaging derivatives pipeline.** Ingests FreeSurfer `.stats` output, accounts for every observation it gains and loses, quality-controls it, harmonizes across scanners, fits longitudinal mixed-effects models, and emits a self-contained HTML report you could reconstruct the entire run from.

> **Status: `v0.1.0` — walking skeleton.** Every stage exists and runs end to end on synthetic fixtures, but only ingestion and data accounting are deep. QC passes everything, harmonization is an identity transform, and the model fits a single region. See [Build.md](Build.md) for what lands when.

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
- **No real-data integration yet.** Cross-sectional (ABIDE) and longitudinal (OASIS-3) integration are weeks 4 and beyond.

Limitations that will matter more as the statistics land — the scanner/time non-identifiability problem, the sensitivity-analysis-versus-inference distinction, the missing-at-random assumption — are documented in [BUILD_PLAN.md](BUILD_PLAN.md) §2.3.1 and §2.5.4 and will be restated here as those stages are built.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/
uv run pytest -v
```

## License

MIT — see [LICENSE](LICENSE).
