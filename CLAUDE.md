# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`morphline` — a BIDS-aware longitudinal neuroimaging derivatives pipeline. Python package lives at [src/morphline/](src/morphline/); the working directory name (`neuroimaging-pipeline`) is not the project name.

Status is `v0.1.0` walking skeleton: every stage exists and runs end to end on synthetic fixtures, but only ingestion and data accounting are deep. QC marks everything `PASS`, harmonization is an identity transform, and the model fits one region. Stage docstrings state precisely what is stubbed and what week 3–6 replaces.

## Code Comments & Documentation Guardrails

* **No Inline Comments**: Do not write inline comments, explanatory comments, or restate what the adjacent code does. Code must be self-documenting.
* **No Commentary on Edits**: Do not add comments explaining what changed, why it changed, or leaving "TODO" markers unless explicitly requested.
* **Permitted Comments Only**: The only acceptable comments are legally required headers, critical architectural warnings, complex algorithmic edge cases, or public API docstrings (JSDoc/Docstrings).
* **Refactor Over Explaining**: If you feel code requires an explanation, refactor the code for clarity instead of adding a comment.
* **Delete, Don't Comment Out**: Never leave commented-out code blocks; delete them entirely.

## Commands

```bash
uv sync                                                    # install (uv.lock is committed)
uv run morphline run --config config/test.yaml --outdir results/
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/
uv run pytest -v
uv run pytest tests/test_accounting.py::test_name          # single test
uv run pytest -m "not slow"                                # skip statistical recovery tests
nextflow run . -profile test,docker --outdir results/      # CI-only in practice; see below
```

Three run configs, all exercised: `config/test.yaml` (tiny, CI), `config/recovery.yaml` (larger, statistical recovery), `config/confounded.yaml` (regime B, site confounded with time). No external data is ever required — the pipeline generates its own fixtures.

Markers declared in `pyproject.toml`: `slow` (statistical recovery over larger fixtures), `optional` (cross-checks needing optional deps). Coverage is printed but not gated — there is deliberately no `--cov-fail-under`.

## Architecture

### The ingestion split

```
.stats files → FreeSurferStatsParser → parsed records → DatasetAdapter → canonical Parquet
                (file structure only)                    (dataset conventions)
                                                                ↓
                              accounting → QC → harmonization → modeling → report
```

**Everything downstream of ingestion reads only the canonical schema.** Those stages must never import `morphline.parsers`/`adapters`/`fixtures`, and must not contain FreeSurfer vocabulary (`aseg`, `aparc`, `.stats`, `StructName`, …) in executable code. Prose in docstrings and comments is exempt.

This is enforced by AST-walking tests in [tests/test_architecture_boundary.py](tests/test_architecture_boundary.py) — including a meta-test that the check itself can still fail. `stages/ingest.py` is the only exemption, and a test asserts the exemption stays narrow. Adding a new dataset means writing an adapter and nothing else.

### Canonical schema is the contract

[src/morphline/schema.py](src/morphline/schema.py) is the single source of truth. Long format, one row per `subject × session × region × measure`, persisted as Parquet — Parquet is the boundary between *every* stage. Every row carries `source_file` + `source_file_checksum` so any coefficient walks back to the files that produced it. `write_canonical` conforms and validates (non-null columns, no duplicate observation keys) on the way out. Changing this schema is expensive; treat it as an interface, not a convenience.

Canonical region names live in [src/morphline/regions.py](src/morphline/regions.py) (no FreeSurfer strings, so downstream stages may import it); FreeSurfer `StructName` → canonical mapping lives in `adapters/freesurfer_regions.py`.

### Two orchestration paths, one set of stages

- **In-process**: [src/morphline/pipeline.py](src/morphline/pipeline.py), driven by `morphline run`. Local dev and the fast CI e2e job.
- **Staged**: one CLI subcommand per stage, each reading paths and writing paths. This is what [main.nf](main.nf) and `modules/local/*.nf` invoke.

Nextflow channels carry **file paths, never dataframes**. Gathering steps `collect()` paths and the consuming process reads them with pyarrow inside its own container. Do not put a serialized table or large value on a channel.

[tests/test_nextflow_parity.py](tests/test_nextflow_parity.py) drives the staged CLI as subprocesses in the DAG's order and asserts the artifacts match the in-process run. Nextflow itself is not runnable on the arm64 dev machine, so if parity passes and the Nextflow run still fails, the bug is in workflow wiring.

### Data accounting

`raw files → parsed files → canonical observations → QC-passing → modeled`. Every boundary reports what it lost and attributes each drop to a cause; unexplained loss is a bug. `morphline account` and `morphline run` **exit non-zero** when the funnel does not reconcile. Loss counters are tracked exactly at ingestion (never derived as a remainder — a catch-all would make the funnel reconcile by construction) and travel to the accounting stage via `ingest_counters.json`.

Parse failures never escape as exceptions; they become reason-coded records (`ParseFailureCode` in `parsers/errors.py`).

### Config and provenance

[src/morphline/config.py](src/morphline/config.py) loads YAML into frozen pydantic models with `extra="forbid"` — a typo in a threshold name must fail, not silently keep the default. The **fully resolved** config (defaults included) goes into the provenance block, on the rule that a reader holding only `report.html` can reconstruct the run.

## Conventions that carry rationale

- **`# Measure` lines are indexed by short name *and* alias.** Neither field alone is stable: `?h.aparc.stats` reports `NumVert`, `WhiteSurfArea`, and `MeanThickness` all under the short name `Cortex`, so short-name-only keying silently keeps one of three; and intracranial volume is `IntraCranialVol, ICV` on FreeSurfer 5.1 but `EstimatedTotalIntraCranialVol, eTIV` on 6+, so alias-only keying loses eTIV on 5.1. An alias is distinguished from a long description by containing no whitespace. Both failures were silent — no exception, no `ParseFailureCode`, no row-count change — and both were invisible to the fixtures until `_aparc_header` was taught to emit the real three-measure shape. When adding fixture realism, prefer reproducing a header shape that *collides* over one that merely parses.
- **Surface holes are not Euler numbers.** `lhSurfaceHoles`/`rhSurfaceHoles` are extracted verbatim; Euler is derived as `2 - 2 × holes`. FreeSurfer 5.3 omits hole counts → the value is **null, never 0**, because 0 implies a flawless surface and would rank the oldest data in a study as its best. The fixture generator deliberately mixes in 5.3 so this stays testable.
- **QC classifies; the analysis layer decides.** `qc_status` is three-level (`PASS`/`WARNING`/`FAIL`) and separate from `analysis_included`, which comes from `AnalysisConfig`. Do not collapse them.
- **Harmonized vs unharmonized is a sensitivity analysis, not a solution.** Where scanner is confounded with time the effects are not identifiable from the data alone; the report must say so in those words. The confound diagnostics in `stages/harmonize.py` are real even though the estimator is a stub.
- **ComBat is implemented in-repo**, not via `neuroHarmonize`, which pins `numpy==1.26.4` and would hold the project on numpy 1.x.
- **28 tests, not 14 regions.** Hemispheres are independent tests and count in the multiplicity family. Regions and tests are reported separately.
- **Regime B failing loudly is the correct result** — demonstrating the failure mode is the artifact.

## Docs and placeholders

`BUILD_PLAN.md` is the spec; `Build.md` is the build order with every open question resolved. Code comments reference BUILD_PLAN sections as `§2.7` — when changing behavior those comments describe, check the section rather than guessing.

`OWNER` is a deliberate placeholder in [README.md](README.md) badges and [nextflow.config](nextflow.config)'s GHCR image; it is find-and-replaced before the repo goes public. There is no git remote yet.

## Environment notes

- Python 3.12 only (`>=3.12,<3.13`), matching the `python:3.12-slim-bookworm` container base. Container is **amd64 only**.
- The dev machine is arm64, so the Docker and Nextflow paths cannot be exercised locally at all — the `nf` CI job is their only verification, and it is `continue-on-error: true` until proven stable.
- `mypy --strict` over `src/` from the start; ruff runs with `D` (google docstrings) and `ANN` enabled, so new public functions need docstrings and full annotations. Line length 100.
- `work/` and `results/` are gitignored pipeline outputs; `config/` is morphline YAML and `conf/` is Nextflow profile config — different tools, kept distinct.
