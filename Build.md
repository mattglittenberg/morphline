# Build.md — morphline

Execution plan derived from [BUILD_PLAN.md](BUILD_PLAN.md). BUILD_PLAN is the *spec*; this is the *build order*, with every open question resolved into a decision.

Section references (§) point back into BUILD_PLAN.md.

---

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Name | `morphline` — package `morphline`, CLI `morphline`, GitHub repo `morphline` | Working directory stays `neuroimaging-pipeline` |
| Scope of this build | **Weeks 1–2 only** → tag `v0.1.0`, review before week 3 | §0.1 walking skeleton |
| Local run path | Python CLI (`morphline run`) | No Docker, Java, or Nextflow on the dev machine; it is arm64 |
| Docker + Nextflow | Authored now, **verified in CI**, not run locally | §2.6 amd64-first |
| ComBat | **Implemented in-repo** (week 4), not `neuroHarmonize` | `neuroHarmonize` pins `numpy==1.26.4`, holding the whole project on numpy 1.x |
| Git | `git init` + commits, no remote | Badges/GHCR use an `OWNER` placeholder to find-and-replace |
| CI gates | ruff lint+format, `mypy --strict src/`, pytest, Python e2e, Docker+Nextflow job | |
| Coverage | Measured and printed, **no** `--cov-fail-under` gate | Badge deferred to week 6 |
| Real data | None on hand | ABIDE (cross-sectional) / OpenNeuro (longitudinal) are user TODOs below |
| Longitudinal dataset | **OpenNeuro accession**, not OASIS-3 | OASIS-3 needs institutional registration an individual cannot satisfy (§1.3); OpenNeuro is `datalad clone`-able today |
| ABIDE source | **ABIDE PCP** (per-subject FS 5.1 `.stats`) for the parser; dfsp-spirit `.tsv` tables as a cross-check | Only per-subject `.stats` validate the parser; the aggregate deposit bypasses it entirely (§1.3). Its FreeSurfer 6 label is wrong — the tables carry FS 5.1 values, so the cross-check is exact equality rather than cross-version (BUILD_PLAN revision 7) |
| Python | 3.12 | Matches `python:3.12-slim-bookworm` container base |

### Deviations from BUILD_PLAN.md, stated explicitly

1. **In-repo ComBat instead of `neuroHarmonize`/`neuroCombat` (§2.3.4).** The dependency resolution is real: `neuroHarmonize` caps numpy below 2.0 and drags `pandas`, `scipy`, and `statsmodels` back with it. Writing the empirical-Bayes ComBat (Johnson et al. 2007) in ~150 lines of numpy removes the pin and converts the §2.3.2 validation suite from "does the library run" into "does *our* estimator recover the injected batch parameter" — which is the stronger portfolio artifact.
   *A reviewer will ask why not the standard library. The README must answer it.* Week-4 mitigation: an optional `combat-xcheck` dependency extra and a `@pytest.mark.optional` test comparing our output to `neuroCombat` in an isolated environment.
2. **arm64 dev machine, amd64 deliverable (§2.6).** The container and Nextflow paths cannot be exercised locally at all during weeks 1–2. This is why the `nf` CI job is not optional — it is the *only* verification those paths get.
3. **OpenNeuro replaces OASIS-3 as Track B; IXI is a stretch, not a swap (§1.3).** OASIS-3 requires institutional registration, so BUILD_PLAN revision 2's "apply on day 1 and forget about it" is not executable — it would spend the whole 6 weeks waiting on an approval that cannot arrive. OpenNeuro converts an *access* risk into a *fit* risk, which the §1.2 audit settles in an afternoon. The cost is honest: expect a smaller, shorter, probably single-site dataset, which validates the longitudinal path but not the scanner/time confound. IXI is the better harmonization design (three scanners, healthy controls only, no diagnosis confound) but distributes NIfTI images with **no FreeSurfer derivatives**, so it cannot feed a `.stats`-ingesting pipeline without ~600 × `recon-all`. ABIDE therefore stays Track A.
   *A reviewer will ask why a portfolio project has no longitudinal real data.* The README must answer it: the longitudinal path is validated against injected ground truth, which real data cannot provide, and the gap is named rather than papered over.

---

## User TODOs — not blocked by code, not doable by the agent

These are §1.2 / §1.3 day-1 items. None gate weeks 1–2, all gate week 4.

- [x] **Download ABIDE PCP per-subject FreeSurfer 5.1 stats** — the Track A *parser* target, ~36 MB for the core three files, anonymous HTTPS, no credentials:

  ```bash
  printf '\n# source datasets (not redistributable)\ndata/\n' >> .gitignore
  scripts/fetch_abide_pcp.sh
  ```

  [scripts/fetch_abide_pcp.sh](scripts/fetch_abide_pcp.sh) builds `data/subjects.txt` from the bucket listing (which is paginated — two requests, not one) and writes `data/abide_pcp/<SUBJECT>/stats/`. Serial, ~10–20 minutes, resumable: non-empty files are skipped, so an interrupted run is simply re-run. `--all-tables` fetches all 10 stats files (~67 MB) instead of the three the 28-test region set needs.

  **Destination must be gitignored.** ABIDE PCP is CC BY-NC-SA and this repo is headed for public; `data/` is neither `work/` (Nextflow's default `workDir`, which `nextflow clean` may delete) nor `results/` (pipeline output). The script warns via `git check-ignore` if the destination is not ignored.

  Expect 1112 subject directories, 1103 complete, 9 known-incomplete (`UCLA_51233`, `UCLA_51243`, `UCLA_51270`, `UM_2_0050423` empty; `UCLA_51244`, `UCLA_51310`, `UM_1_0050309`, `UM_1_0050323`, `UM_1_0050328` lh-only). That is **22 missing files, not 9** — the lh-only subjects lack `aseg.stats` as well as `rh.aparc.stats`. Final counts: `aseg.stats` 1103, `lh.aparc.stats` 1108, `rh.aparc.stats` 1103. The script classifies each gap as known or unknown and exits non-zero only on the latter, because 22 expected failures would otherwise hide a real one.
- [x] **Clone the dfsp-spirit tables** — `scripts/fetch_abide_fs6.sh`, ~5 MB, gitignored destination (the script refuses to write to a tracked path; the tables are CC BY-**NC**-SA). These are *aggregated* `.tsv` tables, so they validate neither the parser nor a second FreeSurfer version — **they turned out to carry FreeSurfer 5.1 values despite the deposit's FS6 label**, see BUILD_PLAN revision 7. What they do validate is morphline's parse → map → canonicalize path, by exact equality against FreeSurfer's own aggregation tools. Skip all six Zenodo deposits — 42 GB, no stats files among them.
- [ ] **Run the §1.2 dataset audit** (timebox 4 hours) — `uv run python scripts/audit_openneuro.py`. The script is written and verified; running it and judging the survivors is the remaining work. Accept a dataset only if actual `aseg.stats` / `aparc.stats` text files are locatable — not merely a `derivatives/` directory that turns out to hold MRIQC or fMRIPrep output — *and* it has ≥2 structural sessions for a usable number of subjects. **This is now the Track B access task.** If nothing clears the timebox, Track B is synthetic-only and that is an acceptable v1.0 (§0.3).
- [ ] **Optional, week 6 or later:** check whether a third-party IXI FreeSurfer derivative deposit exists. IXI ships NIfTI only; if no `.stats` exist, the adapter needs ~600 × `recon-all` and stays post-ship (§1.3). Do not start that compute during weeks 1–6.

**No longer a TODO:** the OASIS-3 DUA. OASIS-3 requires institutional registration, so it is dropped from v1.0 entirely rather than waited on — see BUILD_PLAN §1.3 and deviation 3 above.

---

## Week 1 — Parser, adapters, fixtures

### Step 1. Scaffold

`uv`-managed, Python 3.12, `src/` layout, **`uv.lock` committed** — §2.6 is explicit that "reproducible" without a lockfile is a claim, not a fact.

Runtime: `pandas`, `pyarrow`, `numpy` (2.x, unpinned — what dropping neuroHarmonize buys), `scipy`, `statsmodels`, `pydantic`, `typer`, `jinja2`, `plotly`, `pyyaml`.
Dev: `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `pre-commit`.

Two config directories, kept distinct because they serve different tools:

- `config/` — morphline YAML (`default.yaml`, `test.yaml`, `recovery.yaml`)

  A `fixtures:` block is required only when fixtures are generated. Setting `dataset.path` makes it optional, so a real-data config carries no fiction and provenance records no fixture seed that governed nothing:

  ```yaml
  dataset:
    name: abide-i-pcp
    version: freesurfer-5.1
    adapter: abide-pcp
    path: data/abide_pcp
    phenotypic_csv: data/Phenotypic_V1_0b_preprocessed1.csv   # optional
  ```

  A config with neither `dataset.path` nor `fixtures:` is rejected at load time rather than defaulted — `FixtureConfig.sites` has no sensible default, and inventing sites would fabricate the site effects the harmonization tests exist to recover.
- `conf/` — Nextflow profile configs (nf-core convention: `base.config`, `test.config`)

Then `git init -b main` and an initial commit.

### Step 2. Canonical schema — `src/morphline/schema.py`

**This is the contract; §4 warns that changing it later is expensive.** Implements §1.5 verbatim (identity, acquisition, measurement, covariates, global measures, provenance) as a single source of truth exposing a pydantic row model, a `pyarrow.Schema`, and `read_canonical` / `write_canonical` used by *every* stage.

Long format: one row per subject × session × region × measure. Parquet is the persistence boundary between stages (§2.7).

Traceability (§1.5): every row carries `source_file` + `source_file_checksum`, and a test asserts a model coefficient walks back to the files that produced it.

### Step 3. `FreeSurferStatsParser` — `src/morphline/parsers/freesurfer.py`

Structure only, **zero dataset knowledge** (§1.4). Returns `ParsedStatsFile`: `header_measures`, `col_headers`, `rows`, `freesurfer_version`, `source_file`, `checksum`.

Handles `# Measure` lines, `# ColHeaders`, whitespace-delimited numeric rows, FS 5.3/6/7 differences, extra/missing/reordered columns, malformed input.

Parse failures **never** escape as exceptions — they become reason-coded records, because §1.6 requires rejected files be reported *with reason codes*. Codes: `EMPTY_FILE`, `NO_COLHEADERS`, `MALFORMED_MEASURE`, `TRUNCATED_ROW`, `COLUMN_COUNT_MISMATCH`, `UNPARSEABLE_NUMERIC`, `ENCODING_ERROR`, `UNKNOWN_TABLE_TYPE`.

**Surface holes and Euler, landed with the parser rather than in week 3** (§2.2) — it costs nothing here and this is the detail the spec flags as commonly got subtly wrong:

- `lhSurfaceHoles` / `rhSurfaceHoles` are **extracted verbatim**, no arithmetic.
- Euler is **derived**: `euler_lh = 2 - 2 * lhSurfaceHoles`.
- Absent on FS 5.3 → **null, never 0**. Zero implies a flawless surface, which would make the oldest data look like the highest-quality data in the study.

### Step 4. Adapters — `src/morphline/adapters/`

`base.py` defines the `DatasetAdapter` protocol (discover files; resolve subject/session IDs, site, scanner, field strength, dates, demographics). `synthetic.py` and `abide_pcp.py` implement it; OpenNeuro (and IXI if §1.3's stretch lands) slot in later with **zero downstream change** (§1.4).

`freesurfer_rows.py` holds the table → measurement-row conversion both adapters share. Only *metadata* is dataset-specific; two copies of the row extraction would drift, and the drift would surface as datasets disagreeing about what a thickness is.

`abide_pcp.py` reads its three tables by **exact filename**, never by `*.stats` glob. The parser identifies tables by filename, so `lh.aparc.a2009s.stats` is also an lh aparc table and `lh.entorhinal_exvivo.stats` re-reports a structure the Desikan-Killiany table already reports — globbing emits two rows for one `subject × session × region × measure` key and the schema rejects that (§5.2). Correctly: neither row is wrong, they answer different questions. `--all-tables` downloads make this reachable, so a test asserts it.

**The architectural rule is enforced by a test, not a comment.** `tests/test_architecture_boundary.py` walks the AST of `stages/*` asserting no import of `morphline.parsers`, and greps those modules for `freesurfer`, `aseg`, `aparc`, `.stats`. A failing test is the only enforcement that survives contact with a deadline.

### Step 5. Fixture generator — `src/morphline/fixtures/`

The highest-value component in the repo (§3). Two layers, so the truth model is testable independently of the file format.

**`truth.py` — injected-effect model.** Documented in the module docstring as an explicit equation so tests can invert it:

```
value = base_r
      * (1 + b_age·(age_base_i − 70)/10 + b_dx·dx_i + (b_time + b_dxtime·dx_i)·t_ij + u0_i + u1_i·t_ij)
      * mult_site[s,r] + add_site[s,r] + ε
```

Independently controllable per §3.2: additive **and** multiplicative site effects, age, diagnosis, **diagnosis × time** (the primary modeled hypothesis), subject random intercepts and slopes, measurement noise with known variance, planted QC failures, known-clean observations, and missingness split into `missing_acquisition` vs `missing_derivative`.

Two regimes, both exercised in CI: **A** site ⊥ time, **B** site confounded with time. Everything seeded; the seed enters provenance.

**`generator.py` — file writer.** Emits real `.stats` text so the parser is exercised end to end rather than bypassed:

```
<out>/derivatives/freesurfer/sub-XXX/ses-YY/stats/{aseg.stats,lh.aparc.stats,rh.aparc.stats}
<out>/truth/ground_truth.parquet
<out>/truth/manifest.json     # seed, regime, effect sizes, planted cases, missingness causes
```

Structural realism (§3.1): FS 5.3/6/7 flavors including versions omitting `SurfaceHoles`; extra, missing, and reordered columns; malformed and truncated headers; files truncated mid-row; Unicode in comments; empty files; locale-flavoured numerics. Paired with Hypothesis property tests over generated header/row structures.

Region set is the §2.5.2 AD/aging default: 7 subcortical + 7 cortical, bilateral = **28 regional tests**. Regions and tests are counted separately in the report.

**Exit week 1:** `pytest` green; fixtures → canonical Parquet; parser contains zero dataset-specific code.

---

## Week 2 — Walking skeleton

### Step 6. Stages — `src/morphline/stages/`

Every stage exists from day 14. Each reads canonical Parquet, writes canonical Parquet, emits `versions.yml`.

| Stage | Week 2 behaviour | Deepened in | Status |
|---|---|---|---|
| `ingest.py` | Real — adapter + parser → canonical Parquet | done | **done** |
| `accounting.py` | **Real, not a stub** — full §1.6 funnel with reason codes | week 3 | **done** |
| `qc.py` | Marks everything `PASS`, emits the real field structure | week 3 | **done** — four checks; longitudinal change flag declared as a known miss |
| `harmonize.py` | Identity passthrough, toggleable | week 4 | **done** — in-repo ComBat, three small-batch policies, cross-checked against `neuroCombat` |
| `model.py` | MixedLM on one region | week 5 | **done** — full 28-test region set, primary and secondary FDR families corrected separately, harmonized-vs-unharmonized sensitivity arm, tolerance-bounded slope recovery against injected truth |
| `report.py` | Jinja2 + Plotly; provenance block, funnel, one table | week 6 | funnel, QC, harmonization, and batch parameters render; polish outstanding |

Accounting is built for real immediately: it is the cheapest defence against silent data loss and §7 lists it as never-cut. The funnel — `raw files → parsed files → canonical observations → QC-passing observations → modeled observations` — must reconcile exactly with every drop attributed. `test_accounting_funnel_reconciles_exactly` is a week-2 test.

QC ships its full field shape now (`qc_status` / `qc_flags` / `qc_score` / `qc_notes` / `analysis_included`, §2.4.1) even though every value is `PASS`, so week 3 changes logic rather than schema.

`provenance.py` collects the §2.8 block: git SHA + dirty flag, container digest, Python/Nextflow versions, observed FreeSurfer versions, dataset + version, fully resolved config, seeds, timestamp, duration.

### Step 7. CLI — `src/morphline/cli.py`

Typer. One subcommand per stage plus `run` for the whole chain; this is the local development path.

```
morphline fixtures generate --config config/test.yaml --outdir work/fixtures
morphline ingest | account | qc | harmonize | model | report
morphline run --config config/test.yaml --outdir results/
```

`config.py` loads YAML into pydantic models and dumps the **fully resolved** config into provenance — §2.8: if a parameter changed the output, it appears in the block.

### Step 8. Container — `containers/Dockerfile`

`python:3.12-slim-bookworm`, multi-stage, non-root user, installed from `uv.lock`. **amd64 only** (§2.6). Published to GHCR tagged by both git SHA and semver.

### Step 9. Nextflow — `main.nf`, `nextflow.config`, `modules/local/`, `conf/`

DSL2, profiles `test` / `docker` / `apptainer` / `local`. nf-core *conventions* without the template: `--outdir`, `nextflow_schema.json`, per-process `versions.yml`, and a `-profile test` that runs on generated fixtures with no external data.

`GENERATE_FIXTURES` (test only) → `PARSE_SUBJECT` (per subject) → `COLLECT_CANONICAL` → `ACCOUNTING` → `QC` → `HARMONIZE` → `MODEL` → `REPORT`.

**Channels carry file paths, never dataframes** (§2.7). Each process writes a Parquet file and emits its path; gathers `collect()` paths and the consumer reads them with pyarrow inside its own container. No serialized dataframe, large value channel, or in-memory table ever enters a channel. Free at v1 scale, and it means the architecture survives two orders of magnitude of growth.

`-with-report` / `-with-trace` / `-with-dag` outputs get committed to `docs/` once CI produces a real run — §2.7 calls the DAG image the cheapest high-impact thing in the project.

### Step 10. CI — `.github/workflows/ci.yml`

| Job | Command | Blocking |
|---|---|---|
| `lint` | `ruff check .` + `ruff format --check .` | yes |
| `types` | `mypy --strict src/` | yes |
| `test` | `pytest` + hypothesis, coverage printed, no threshold | yes |
| `e2e` | `morphline run -c config/test.yaml`, seconds | yes |
| `nf` | build amd64 image → `nextflow run . -profile test,docker` | non-blocking at first, promoted once stable |

`mypy --strict` from the start means annotating while writing rather than retrofitting, and it catches schema drift between stages — the specific failure mode a seven-stage Parquet-boundary pipeline invites.

**Exit week 2:** CI green, badge in README, `nextflow run . -profile test,docker` completes end to end on a clean machine with zero data present. **Shippable.** Tag `v0.1.0`.

---

## Verification

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/
uv run pytest -v
uv run morphline fixtures generate --config config/recovery.yaml --outdir work/fx
uv run morphline run --config config/test.yaml --outdir results/
open results/report.html
```

Week-2 exit gate:

- `pytest` green, including `test_surface_holes_absent_in_fs53_fixture_yields_null_not_zero`, `test_euler_number_derived_correctly`, `test_accounting_funnel_reconciles_exactly`, and the architecture-boundary test.
- Funnel reconciles with **zero unexplained loss**.
- `results/report.html` opens standalone with a complete provenance block.
- Cold-clone e2e works with no data present.

Verifiable only in CI: amd64 image build, GHCR push, `nextflow run . -profile test,docker`.

---

## Deferred — weeks 3–6

Carried forward from BUILD_PLAN §4. Not part of the weeks 1–2 build this document plans.

- **Week 3 — QC for real. ✅ Done.** Within-site robust outliers (median/MAD), eTIV bounds, asymmetry outliers, three-level status model, sensitivity **and** specificity against planted cases with the confusion matrix in the report. Targets met and asserted against `QCConfig` fields rather than literals. The **longitudinal suspicious-change flag (§2.4.3) was deliberately not implemented** — it needs interval handling and population change distributions to mean anything — and the omission is declared: `stages/qc.py` says so and the validation suite reports planted extreme changes as a known miss rather than counting them clean.
- **Week 4 — Real cross-sectional data + harmonization. ✅ Done, two items carried.** ABIDE PCP at full scale (1112 sessions, 3314 files, zero parse failures, funnel reconciling), labelled cross-sectional integration. The parser did break on real files — six defects total, all fixed. In-repo ComBat with covariate preservation, three behaviourally distinct small-batch policies, and all four §2.3.2 criteria passing on `recovery.yaml`.

  Two corrections the week produced, both recorded in BUILD_PLAN revisions 5–6: **Regime B does not attenuate** under the covariate-preserving configuration §2.3.4 mandates (the *unharmonized* estimate is the wrong one), and the `neuroCombat` cross-check promised by deviation 1 **found a real defect** — shrinkage pooling across batches within a region where Johnson et al. pool across regions within a batch. No internal recovery test could have caught it.

  **Carried into week 5:** the dfsp-spirit FS6 aggregate cross-check (the only outstanding *external* validation of the ingested numbers) and the §1.2 OpenNeuro audit (which gates whether week 5 has a longitudinal real-data target).
- **Week 5 — Longitudinal model. ✅ Done.** MixedLM per §2.5.1 across all 28 tests, with escalating optimizers and a random-intercept-only fallback behind per-region convergence reporting — 28/28 converge on the recovery fixture with no slope dropped. BH-FDR within the declared primary family; each secondary effect corrected in its own family (§2.5.3), with raw *p* and *q* reported for every test. Completer/non-completer baseline comparison by standardized mean difference. Harmonized-vs-unharmonized sensitivity arm, labelled as sensitivity analysis and reported as *not applicable* where harmonization changed nothing.

  Slope recovery across the region set: every 95% interval contains the injected slope (28/28), worst-case relative error 10.3% against a declared 20% bound, no systematic bias. On regime B the two arms agree on direction everywhere but disagree on significance in 5 of 28 regions — the interaction is robust to the scanner/time confound in a way the `time` main effect is not, which is independent support for §2.5.3's choice of primary hypothesis.
- **Week 6 — Ship.** README with architecture diagram, Nextflow DAG, cold-clone quickstart, and an honest limitations section covering §2.1 (pybids is not a derivatives parser), §2.3.1 (non-identifiability of scanner vs time), §2.5.4 (MAR assumption), and the real-data scope actually achieved. Full provenance block. **Synthetic** example report on GitHub Pages (§5.3). Coverage badge, CITATION.cff. Tag `v1.0.0`. Open 5–8 roadmap issues.

**Never cut** (§7): CI, the fixture suite with recovery tests, the data accounting funnel, the provenance block, the README limitations section, the published synthetic report.
