# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`morphline` — a BIDS-aware longitudinal neuroimaging derivatives pipeline. Python package lives at [src/morphline/](src/morphline/); the working directory name (`neuroimaging-pipeline`) is not the project name.

Status is past the `v0.1.0` walking skeleton: ingestion, data accounting, QC, and harmonization are deep; the model still fits one region (`model.py`'s `[:1]` truncation, week 5). Stage docstrings state precisely what is stubbed and what remains.

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
uv run pytest -m "not slow"                                # what CI's `test` job runs
uv run pytest -m slow                                      # what CI's `recovery` job runs
uv run --extra combat-xcheck pytest -m optional            # the neuroCombat cross-check
nextflow run . -profile test,docker --outdir results/      # CI-only in practice; see below
```

Four run configs, all exercised: `config/test.yaml` (tiny, CI), `config/recovery.yaml` (regime A, §2.3.2 recovery suite), `config/confounded.yaml` (regime B, site confounded with time), `config/abide.yaml` (real data, needs `data/abide_pcp`). Only the last needs external data; the rest generate their own fixtures.

Markers declared in `pyproject.toml`: `slow` (statistical recovery over larger fixtures — the model slope suite and the four §2.3.2 harmonization criteria), `optional` (cross-checks needing optional deps — currently `neuroCombat`, which is a `[project.optional-dependencies]` extra deliberately kept out of the dev group so it cannot enter the default resolution). Coverage is printed but not gated — there is deliberately no `--cov-fail-under`; note it now comes from two jobs, so a badge will need `coverage combine`.

CI jobs: `lint`, `types`, `test` (`-m "not slow"`), `recovery` (`-m slow`), `xcheck` (`-m optional`, with the extra), `e2e`, `nf`, `publish`. **ruff is pinned exactly** in both `pyproject.toml` and `.pre-commit-config.yaml`, and the two must stay in lockstep — when they drifted, pre-commit and CI formatted the same file differently and CI failed on a commit that had not touched it.

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

Two adapters exist: `synthetic.py` and `abide_pcp.py`. The table → measurement-row conversion they share lives in `adapters/freesurfer_rows.py`; only *metadata* resolution is dataset-specific, and two copies of the row extraction would drift into datasets disagreeing about what a thickness is.

**Adapters select stats files by exact filename, never by `*.stats` glob.** The parser identifies tables by filename, so `lh.aparc.a2009s.stats` is also an lh aparc table and `lh.entorhinal_exvivo.stats` re-reports a structure the Desikan-Killiany table already reports. A glob emits two rows for one `subject × session × region × measure` key, which `write_canonical` rejects — correctly, since neither row is wrong; they answer different questions. `AbidePcpAdapter.CORE_TABLES` is the allowlist, and a test asserts the extra tables are ignored when present.

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

The **modeling boundary** is decomposed, not labelled. `ModelExclusion` separates `outside_modeled_region_set` (a scope decision) from `incomplete_model_covariates` (a data limitation that biases the remaining sample), and falls back to `not_modeled_cause_unavailable` when the model's per-region inputs were not supplied. These have opposite implications, so the single catch-all that used to sit here let the funnel reconcile while reporting a cause that was false — which is worse than not reconciling, because it looks like an answer. Synthetic runs show pure scope exclusion; ABIDE shows covariate incompleteness. Passing `model_fits` into `build_accounting` is what makes the split possible.

### Config and provenance

[src/morphline/config.py](src/morphline/config.py) loads YAML into frozen pydantic models with `extra="forbid"` — a typo in a threshold name must fail, not silently keep the default. The **fully resolved** config (defaults included) goes into the provenance block, on the rule that a reader holding only `report.html` can reconstruct the run.

`fixtures:` is required only when fixtures are generated; `dataset.path` makes it optional, and a config with neither is rejected at load time. `random_seeds` is then empty for real-data runs rather than carrying a seed that governed nothing. `ingest_versions.json` is a sidecar beside `ingest_counters.json`, carrying observed versions and their verbatim declarations to the staged provenance step — `version_declaration` is deliberately *not* a canonical column, since the schema is an interface.

## Conventions that carry rationale

- **`# Measure` lines are indexed by short name *and* alias.** Neither field alone is stable: `?h.aparc.stats` reports `NumVert`, `WhiteSurfArea`, and `MeanThickness` all under the short name `Cortex`, so short-name-only keying silently keeps one of three; and intracranial volume is `IntraCranialVol, ICV` on FreeSurfer 5.1 but `EstimatedTotalIntraCranialVol, eTIV` on 6+, so alias-only keying loses eTIV on 5.1. An alias is distinguished from a long description by containing no whitespace. Both failures were silent — no exception, no `ParseFailureCode`, no row-count change — and both were invisible to the fixtures until `_aparc_header` was taught to emit the real three-measure shape. When adding fixture realism, prefer reproducing a header shape that *collides* over one that merely parses.
- **`cvs_version` is not always a version.** FreeSurfer 6/7 declare `# cvs_version 7.2.0`; 5.1 declares the *source file's* CVS revision (`$Id: mri_segstats.c,v 1.75.2.2 ... $`), which names neither FreeSurfer nor its version and differs between `mri_segstats` and `mris_anatomical_stats` — so taking it at face value made one release look like two. A declaration that does not start with a digit yields `freesurfer_version = None`, on the same rule as surface holes; the raw string is kept verbatim on `version_declaration` and reaches provenance as `freesurfer_version_declarations`. The release a derivative distribution belongs to is `dataset_version`, which is knowledge about the download, not about the file.
- **Surface holes are not Euler numbers.** `lhSurfaceHoles`/`rhSurfaceHoles` are extracted verbatim; Euler is derived as `2 - 2 × holes`. FreeSurfer 5.3 omits hole counts → the value is **null, never 0**, because 0 implies a flawless surface and would rank the oldest data in a study as its best. The fixture generator deliberately mixes in 5.3 so this stays testable.
- **QC classifies; the analysis layer decides.** `qc_status` is three-level (`PASS`/`WARNING`/`FAIL`) and separate from `analysis_included`, which comes from `AnalysisConfig`. Do not collapse them.
- **Harmonized vs unharmonized is a sensitivity analysis, not a solution.** Where scanner is confounded with time the effects are not identifiable from the data alone; the report must say so in those words. The confound diagnostics in `stages/harmonize.py` are independent of the estimator and must survive it being cut (§7).
- **ComBat is implemented in-repo** (`combat.py`), not via `neuroHarmonize`, which pins `numpy==1.26.4` and would hold the project on numpy 1.x. `stages/harmonize.py` owns the *policy* — which rows estimate the batch terms, what happens to a small batch, what the report is told; `combat.py` stays a pure function over a frame.
- **ComBat's γ absorbs a multiplicative mean shift.** The fixture generator injects `observed = biological·mult + add + ε`, so γ estimates `add + (mult−1)·mean(biological)`, not `add`. A recovery test comparing γ against the configured `additive_effect` fails a *correct* estimator. The exact quantity is computable from `ground_truth.parquet`, which carries both `value` and `true_biological_value`.
- **γ is identified only up to a size-weighted centering.** The batch terms satisfy `Σ_b (n_b/n)·γ_b = 0`, not `mean(γ) = 0`. With equal batch sizes the two coincide and the distinction is invisible; with unequal ones they diverge badly. Truth quantities must be centered the same way before comparison.
- **`report_and_exclude` excludes from harmonization, never from the dataset.** Dropping rows at the harmonization boundary would resurface at the modeling boundary under a cause that is false. Harmonization is not a funnel boundary and must not become one.
- **28 tests, not 14 regions.** Hemispheres are independent tests and count in the multiplicity family. Regions and tests are reported separately.
- **Regime B does not attenuate, and that is the measured result.** BUILD_PLAN §2.3.2 predicted harmonization would visibly flatten the longitudinal effect under a scanner/time confound. It does the opposite: *unharmonized*, the `time` coefficient comes out sign-flipped (+58 against an injected −20), because the scanner step reads as biology; harmonization recovers it to −19.7. The cause is a tension with §2.3.4, which requires `time_from_baseline_years` be preserved in the design matrix — and a preserved covariate is exactly the one the batch term cannot absorb. Drop it and the predicted attenuation appears immediately (−7.5), which is what `TestCovariatePreservationIsWhatSavesRegimeB` demonstrates. The failure mode is real but belongs to the *configuration*, not the regime. Recorded as BUILD_PLAN revision 5; do not "fix" the tests back toward the original prediction.
- **Regime B's estimates stay non-interpretable regardless.** Recovery there is knowable only because the truth was injected; from the data alone a scanner step and a biological change of the same size are the same observation. `interpretable: false` must not soften because a number lands close.
- **The `time:dx_baseline` interaction is robust to the scanner/time confound** in a way the `time` main effect is not — a scanner change shifts both diagnosis groups alike and largely cancels. Independent support for §2.5.3 making the interaction the primary hypothesis.

## Docs and placeholders

`BUILD_PLAN.md` is the spec; `Build.md` is the build order with every open question resolved. Code comments reference BUILD_PLAN sections as `§2.7` — when changing behavior those comments describe, check the section rather than guessing.

`OWNER` is a deliberate placeholder in [README.md](README.md) badges and [nextflow.config](nextflow.config)'s GHCR image; it is find-and-replaced before the repo goes public. There is no git remote yet.

## Environment notes

- Python 3.12 only (`>=3.12,<3.13`), matching the `python:3.12-slim-bookworm` container base. Container is **amd64 only**.
- The dev machine is arm64, so the Docker and Nextflow paths cannot be exercised locally at all — the `nf` CI job is their only verification, and it is `continue-on-error: true` until proven stable.
- `mypy --strict` over `src/` from the start; ruff runs with `D` (google docstrings) and `ANN` enabled, so new public functions need docstrings and full annotations. Line length 100.
- `work/` and `results/` are gitignored pipeline outputs; `config/` is morphline YAML and `conf/` is Nextflow profile config — different tools, kept distinct.
