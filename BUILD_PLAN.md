# Build Plan — BIDS-aware Longitudinal Neuroimaging Derivatives Pipeline

**Ship date: 6 weeks from start. v1.0 tagged, README written, GitHub bio updated, applications going out.**
If a week slips, cut scope, not the date. Cut order is defined in §7.

*Revision 2 — methodological and architectural corrections applied. Scope unchanged.*
*Revision 3 (2026-08-09) — dataset strategy changed: OASIS-3 dropped (institutional registration), Track B is now OpenNeuro, IXI added as a stretch harmonization target. §0.3, §1.2, §1.3, §1.4, §1.5, §3.4, §4, §5.1, §5.3, §6, §7, §8 touched. Scope and ship date unchanged.*
*Revision 4 (2026-08-10) — Track A split across two ABIDE sources after verifying that the FS6 deposit ships only aggregated tables: ABIDE PCP per-subject FreeSurfer 5.1 `.stats` is the parser target, the FS6 tables are a cross-check. §1.3, §4, §5.2 touched. Two parser defects this exposed are fixed in code. Scope and ship date unchanged.*

*Revision 5 (2026-08-11) — three corrections from implementing §2.3 and measuring it. §2.3.2's criterion 1 and its regime B prediction are both restated; the corrections are recorded inline in those subsections rather than changing the surrounding plan. Scope and ship date unchanged.*
*Revision 6 (2026-08-11) — §4 checkboxes reconciled against the repo for the first time since week 1: weeks 1–3 marked complete, week 4 marked complete except the dfsp-spirit cross-check and the OpenNeuro adapter. §2.3.4 corrected to record the in-repo estimator and its `neuroCombat` cross-check; §3.2, §8 corrected where they repeated the regime B prediction revision 5 restated. Two carried-forward items are named in §4 rather than left implicit. Scope and ship date unchanged.*

*Revision 7 (2026-08-12) — the dfsp-spirit deposit's aggregated stats tables are **not** FreeSurfer 6. They carry values bit-identical to ABIDE PCP's FreeSurfer 5.1 output across all 28,976 shared observations. §1.3 and §4 corrected; the cross-check was rebuilt as an exact-equality test and is now done. Scope and ship date unchanged.*

**Status at revision 7:** weeks 1–5 complete; week 6 (ship) not started. **One** item still carries — the **§1.2 OpenNeuro audit**, which decides whether v1.0 has a longitudinal *real-data* target. It no longer gates week 5: the longitudinal model is validated against injected truth, which is the validation real data cannot provide (§3.2), so a null audit result costs v1.0 the real-data longitudinal claim and nothing else (§0.3, §5.1).

---

## 0. Three structural commitments

### 0.1 Walking skeleton first

The original sequence (ingest → QC → containers → harmonize → model → report/CI) means there is no runnable artifact until week 6. Invert it. Build a thin end-to-end path through *every* stage by end of week 2, using synthetic fixtures only. Every stage is a stub that does the dumbest correct thing. Then deepen stages in place.

Consequences:
- You are taggable from day 14. Every subsequent week is an improvement on something that already works.
- Nextflow wiring gets debugged when the processes are trivial, not when they're the most complex thing in the repo.
- CI is green from week 2, so the badge is real for four extra weeks of commit history.

### 0.2 Fixtures are the primary substrate; real data is a swap-in

Do not develop against real data. Build the synthetic stats-file generator in week 1 and make it the thing you iterate against. Real datasets become integration targets you point the ingestion layer at, not the thing that gates your progress.

This is not a compromise — it's the stronger engineering story. It forces the ingestion boundary to be a real interface, it makes CI possible, and it lets you write *statistical* validation tests (§3) that you cannot write against real data because you don't know ground truth.

### 0.3 The MVP is explicit, and it does not depend on any gated dataset

**v1.0 = a complete synthetic longitudinal pipeline with statistical recovery tests, plus one real cross-sectional dataset (ABIDE) validating the parser, adapter, QC, and harmonization stages, plus documented longitudinal real-data integration as a named subsequent target.**

None of the following are on the critical path:
- longitudinal real-data access
- whole-brain region coverage
- a longitudinal ComBat implementation
- sophisticated or exhaustive QC

*Revision 3 supersedes OASIS-3 here.* OASIS-3 requires institutional registration, which no individual can satisfy on a 6-week timeline, so it is out of scope entirely rather than a slow-arriving bonus. Track B is now a public OpenNeuro accession (§1.3), gated on the §1.2 audit rather than on anyone's approval.

If a qualifying OpenNeuro accession is found, longitudinal real-data integration becomes a v1.0 bonus. If it is not, v1.0 ships anyway with the longitudinal path validated against synthetic ground truth and the real-data gap stated plainly in the README. **A clearly documented limitation is a stronger portfolio signal than a missed ship date.**

---

## 1. Dataset strategy and ingestion architecture

### 1.1 The requirements conflict

| Stage | Needs |
|---|---|
| ComBat harmonization | Multi-site / multi-scanner, adequate batch sizes (see §2.3.3) |
| Longitudinal mixed model | ≥2 sessions per subject, ideally ≥3 |

Few public datasets deliver both. Plan for two datasets, not one — and label what each one actually validates.

### 1.2 Day 1 task: dataset audit (timebox to 4 hours)

Before writing any code, confirm a specific dataset ID satisfies your needs. Script it:

```bash
# for a candidate OpenNeuro accession, check derivatives + session structure
datalad clone https://github.com/OpenNeuroDatasets/dsXXXXXX.git
find dsXXXXXX/derivatives -name "aseg.stats" | head
find dsXXXXXX -maxdepth 2 -name "ses-*" -type d | wc -l
```

Accept a dataset only if you can locate actual `aseg.stats` / `aparc.stats` text files, not just a `derivatives/` directory that turns out to hold MRIQC or fMRIPrep output.

**This audit is now on the critical path for Track B**, not a nicety — it is the only gate between the pipeline and longitudinal real data. A candidate accession must clear all three: ≥2 structural sessions per subject for a usable number of subjects, locatable FreeSurfer `.stats` files, and enough scanner or site variation to be worth harmonizing. Datasets failing only the third are still worth taking; that just makes them a longitudinal-only target. If nothing clears the first two within the timebox, stop looking and ship synthetic-only — that is the §0.3 outcome, not a failure.

### 1.3 Two validation tracks, clearly labelled

**Track A — cross-sectional harmonization and parser/integration validation: ABIDE I, from two sources.**

The two ABIDE sources are not interchangeable, and the distinction decides whether the parser gets validated at all:

| Source | Shape | Size | Validates |
|---|---|---|---|
| **ABIDE PCP**, `s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/<SUBJECT>/stats/` | **Per-subject `.stats` files**, FreeSurfer **5.1**, 10 per subject | 67 MB all files; ~36 MB for `aseg` + `?h.aparc` | The parser against real files, per-file provenance, real recon failures |
| **dfsp-spirit**, `github.com/dfsp-spirit/abide_preproc_smri_freesurfer6` | **Aggregated `.tsv` tables**, ~~FreeSurfer **6**~~ **FreeSurfer 5.1 values — see revision 7**, one row per subject | ~5 MB, whole repo | morphline's parse → map → canonicalize path, by exact equality — *not* the parser, and **not** a second FreeSurfer version |

Verified counts on the PCP bucket: **1112 subject directories across 17 sites** (NYU 184 largest, CMU 27 smallest), **1103** with all of `aseg.stats` + `lh/rh.aparc.stats`, and **9 incomplete** — 4 empty, 5 lh-only. The bucket is anonymously readable, no credentials and no requester-pays. Those 9 are an asset: real recon failures give the accounting funnel genuine attributable loss instead of synthetic loss.

**Only per-subject `.stats` files validate the parser.** The dfsp-spirit tables are `asegstats2table` / `aparcstats2table` output — they bypass `FreeSurferStatsParser` entirely and exercise only the adapter. A pipeline whose ingestion story is "we parse FreeSurfer's native output" cannot demonstrate that against a table someone else already aggregated. Take PCP as the parser target; keep the FS6 tables as an independent cross-check that morphline's per-subject numbers aggregate to a table derived by different code. That cross-check is a stronger artifact than either source alone.

Two constraints this creates. PCP is **FreeSurfer 5.1 only**, so "FreeSurfer 6 stats" is the wrong claim for the parser target — 5.1 emits no `SurfaceHoles`, which usefully exercises the null-never-zero Euler convention (§2.2) against real data in its null branch. And because the two sources are different FreeSurfer versions over overlapping subjects, **version is confounded with source: never pool them in one ComBat run.**

> **Revision 7 correction — the deposit is not a second FreeSurfer version.** Its aggregated stats tables carry values *bit-identical* to ABIDE PCP's FreeSurfer 5.1 output: all 28,976 shared observations across 1035 subjects and 28 regions, spanning mm³ volumes and mm thickness, with a maximum absolute difference of **0.0**. Two `recon-all` runs at different releases cannot produce that; FreeSurfer 6 changed both the aseg segmentation and the surface pipeline. Corroborated by the deposit's own `stats/brainvol/` file, which it says it derived independently with `mris_anatomical_stats`: it carries the FreeSurfer 6 *column name* `EstimatedTotalIntraCranialVol` against the FreeSurfer 5.1 *value* (`1697598.984182`) that our `aseg.stats` reports as `IntraCranialVol, ICV`. The deposit's README states FreeSurfer v6, so this is a discrepancy in the deposit rather than a misreading of it — and it is scoped to what was checked: the Desikan aparc and aseg tables plus `brainvol`. Nothing here speaks to the Zenodo volume, mesh, or lGI archives.
>
> Consequences. The **cross-version comparison §4 planned is not possible** with this source, and "expect systematic offsets" was the wrong expectation — the right one is exact equality. The `never pool them in one ComBat run` warning above is moot for these tables, since there is only one version present; it stays as standing guidance for any genuinely multi-version source. And the check that *is* possible turns out to be sharper than the one planned: `asegstats2table` / `aparcstats2table` are FreeSurfer's own tools run by a third party over the same files, so a mis-read column, a mis-mapped `StructName`, or a swapped hemisphere surfaces as an exact mismatch rather than being absorbed into version noise. That is now `tests/test_abide_fs6_crosscheck.py`.
>
> What it consequently cannot validate: FreeSurfer version tolerance, and the correctness of the values themselves. Both sides descend from one recon-all run, so a segmentation error is present in both and invisible to the comparison.

ABIDE is unaffected by the access problem that removed OASIS-3: neither source carries an institutional gate, so Track A is downloadable by one person today. That is why it stays Track A even though IXI is the better harmonization design (below). Note the dfsp-spirit deposit is CC BY-**NC**-SA; its six Zenodo deposits are 42 GB of volumes, meshes, labels, and lGI, and **none of them contains stats files** — ignore them.

**What ABIDE cannot validate: anything longitudinal.** It cannot exercise longitudinal QC rules, within-subject slope estimation, longitudinal ComBat, or the scanner/time confound. Do not describe a successful ABIDE run as validating the pipeline end to end. Say "cross-sectional real-data integration" and mean it.

One caveat to carry into §2.3: ABIDE's diagnosis distribution varies by site, so site effect and case-mix are entangled. Covariate-preserving ComBat is the mitigation, but a residual site effect in ABIDE is not cleanly attributable to the scanner. This is the specific weakness IXI would fix.

**Track B — longitudinal analysis validation: OpenNeuro (preferred), synthetic (guaranteed).**
A public OpenNeuro accession with ≥2 structural sessions per subject, selected by the §1.2 audit. OpenNeuro is `datalad clone`-able immediately, mostly CC0, and requires no agreement with anyone — the access risk is replaced by a *fit* risk, which the audit resolves in an afternoon instead of over weeks.

The trade is deliberate and it is not free. No single OpenNeuro accession matches OASIS-3's 1378 participants / 2842 sessions across 30 years, and most accessions ship raw images with no FreeSurfer derivatives at all. Expect a smaller dataset, expect fewer sessions per subject, and expect the scanner variation to be thin or absent — a single-site longitudinal accession still validates the longitudinal path, it just cannot exercise the scanner/time confound on real data. Say which of the two it validated; do not let "real longitudinal data" imply both.

**Why not OASIS-3.** It requires institutional registration, which an individual cannot satisfy. It is not a slow bonus to be waited on — it is out of scope for v1.0 and stays a named post-ship target (§6).

Until and unless a qualifying accession is found, the longitudinal path is validated against synthetic fixtures with injected ground truth. That is a legitimate validation dimension, not a placeholder — it tests things real data cannot, because real data has no known true slope. It also remains the *only* substrate that exercises the regime B scanner/time confound against known truth.

**Stretch — dedicated harmonization dataset: IXI.**
IXI is ~600 healthy subjects across three scanners (Philips 3T at Hammersmith, Philips 1.5T at Guy's, GE 1.5T at the Institute of Psychiatry), CC BY-SA 3.0, no registration of any kind. On design it is a better harmonization substrate than ABIDE: scanner differences without a diagnosis distribution varying alongside them, so a batch effect is unambiguously a batch effect.

**It ships NIfTI images only — no FreeSurfer derivatives.** morphline ingests `.stats` files, so IXI is not a swap-in: it requires either locating a third-party FreeSurfer deposit or running `recon-all` on ~600 subjects at roughly 8 hours each. Neither belongs on a 6-week critical path. IXI is therefore a **stretch target contingent on the §1.2 audit locating usable derivatives**, and ABIDE remains Track A regardless of the outcome. Do not start recon-all compute during weeks 1–6.

Its value if it lands is a clean scanner-effect story and proof that the adapter abstraction wasn't accidental — the same value a third OpenNeuro accession would provide, at higher cost and higher quality.

### 1.4 Architecture: parsing and entity resolution are separate concerns

The original design collapsed "read FreeSurfer files" and "figure out what dataset this is" into one adapter. Split them:

```
stats files
    ↓
FreeSurferStatsParser          # knows aseg.stats / aparc.stats structure ONLY
    ↓                           # knows nothing about datasets, sites, or BIDS
parsed FreeSurfer records       # (file, header measures, column headers, rows)
    ↓
DatasetAdapter                  # knows subject/session ID conventions, site,
    ↓                           # scanner, demographics, dataset layout
canonical observations (Parquet)
```

- **`FreeSurferStatsParser`** handles header `# Measure` lines, `# ColHeaders`, whitespace-delimited numeric rows, FreeSurfer 5.3 / 6 / 7 differences, extra and missing columns, and malformed input. One parser, all datasets.
- **`DatasetAdapter`** (synthetic / ABIDE / OpenNeuro, plus IXI if §1.3's stretch lands) resolves subject IDs, session IDs, site, scanner, field strength, dates, and demographics. Three to four adapters, one parser.

**Architectural rule, stated explicitly in the README and enforced by module boundaries: everything downstream of ingestion reads only the canonical schema.** QC, harmonization, modeling, and reporting must never import the parser, never touch a `.stats` file, and never contain a FreeSurfer-specific string. Adding a fifth dataset should require zero changes downstream of the adapter layer.

### 1.5 Canonical intermediate schema

Long format, one row per subject × session × region × measure. Parquet on disk, and Parquet is the persistence boundary between pipeline stages (§2.7).

```
# identity
dataset                 # e.g. "abide-i", "openneuro-dsXXXXXX", "synthetic-v1"
dataset_version         # accession version / release tag / fixture seed
subject_id
session_id
time_from_baseline_years

# acquisition
site
scanner_manufacturer
scanner_model
field_strength_tesla

# measurement
region
hemisphere
measure_type            # volume | thickness | area | curvature
value
unit

# subject/session covariates
age_at_session
age_baseline
sex
dx_baseline
dx_at_session

# global measures
etiv
etiv_baseline
surface_holes_lh
surface_holes_rh

# provenance
source_file             # absolute path or dataset-relative path as ingested
source_file_checksum
freesurfer_version
parser_version
ingested_at
```

Traceability requirement: **any row in the final results table must be traceable back through canonical observations to the specific source file it came from.** If you cannot answer "which files produced this coefficient," the provenance design has failed.

### 1.6 New stage: data accounting (between ingestion and QC)

A dedicated stage that reports counts and losses at every boundary. This is the cheapest defence against silent data loss and it makes the real-data run auditable.

Emits a structured accounting table (and a report section):

| Boundary | Reported |
|---|---|
| Discovery | raw stats files found, by dataset and type |
| Parsing | files parsed successfully; files rejected, **with reason codes** |
| Entity resolution | subjects, sessions, sessions per subject distribution |
| Canonicalization | observations written; regions per session; missing observations |
| Metadata coverage | site / scanner / field strength / FreeSurfer version distributions; batch sizes |
| QC | PASS / WARNING / FAIL counts and rates, by flag type and by site |
| Analysis | observations entering the model; subjects and sessions retained |

The headline line in the report is a single funnel:

```
raw files → parsed files → canonical observations → QC-passing observations → modeled observations
```

Every drop in that funnel must have a stated cause. Unexplained loss is a bug, not a rounding error.

---

## 2. Technical decisions

### 2.1 pybids is not a derivatives parser — be honest about this

`pybids` indexes raw BIDS well; its BIDS-Derivatives support is partial, and FreeSurfer's native output (`SUBJECTS_DIR/sub-01_ses-01/stats/aseg.stats`) is not BIDS-organized at all. Use pybids for raw layout traversal and entity parsing inside the adapters, and a purpose-built resolver for FreeSurfer subject-directory conventions. Say exactly this in the README. Overselling pybids is the kind of thing a neuroimaging engineer spots in thirty seconds; describing the boundary accurately reads as competence.

### 2.2 Euler number: a well-established QC signal, not a silver bullet

The Euler number is a well-validated and widely used automated FreeSurfer QC signal (Rosen et al. 2018), and it has a practical advantage here: it is derivable from the stats tables alone, with no surface files required.

**Distinguish the two quantities carefully, because they are not the same thing:**

- **Surface hole counts** are *reported directly* in the `aseg.stats` header on FS 6+ as `lhSurfaceHoles`, `rhSurfaceHoles`, `SurfaceHoles`. These are extracted, not computed.
- **The Euler number** is *derived* from them, per hemisphere:
  ```
  euler_lh = 2 - 2 * lhSurfaceHoles
  euler_rh = 2 - 2 * rhSurfaceHoles
  ```
  It is a topological measure of surface defects; more negative means more defects. `SurfaceHoles` in the header is the summed/total hole count and should not be used as if it were a Euler number.

Document this relationship in the module docstring and the README. Getting it subtly wrong is common.

**Test it explicitly.** Fixtures carry known hole counts in their headers, so:

```python
def test_surface_holes_extracted_verbatim_from_header():
    # header says lhSurfaceHoles 42 -> field equals 42, no arithmetic applied

def test_euler_number_derived_correctly():
    # lhSurfaceHoles 42 -> euler_lh == -82

def test_surface_holes_absent_in_fs53_fixture_yields_null_not_zero():
    # missing metric must be null; 0 would silently mean "perfect surface"
```

That last one matters: on FreeSurfer versions that don't emit hole counts, a default of zero produces a Euler number of 2, i.e. a flawless surface, which would make old data look like the highest-quality data in the study.

Euler is **one input among several** to QC status (§2.4.1), never a sole determinant.

### 2.3 Harmonization

#### 2.3.1 The identifiability problem — state it precisely

Standard ComBat assumes independent observations, which longitudinal data violates. The deeper issue in aging cohorts is that **scanner is frequently confounded with time**: subjects are scanned on an older scanner early in the study and a newer one later.

When that confounding is strong, **the biological longitudinal effect and the scanner effect are not identifiable from the observed data alone.** No harmonization method recovers a unique answer from confounded data; the separation depends on assumptions that the data cannot check. This is a property of the study design, not a deficiency in the software.

What follows from that:

- **Running harmonized and unharmonized models is a sensitivity analysis, not a solution.** It shows how conclusions depend on the harmonization assumption. It does not establish which is correct.
- The report must **explicitly label these results as sensitivity analysis and distinguish them from validated biological inference.** Use those words. A results section that quietly presents harmonized estimates as biology is exactly the failure this section exists to prevent.
- Report the scanner × time crosstab and a quantitative confounding measure (e.g. association between scanner and time-from-baseline). Where confounding is strong, say so and mark the affected estimates as not interpretable as biology.
- Where scanner changes are *not* confounded with time — e.g. sites contributing across the full time range — harmonization is on much firmer ground. Distinguish these cases rather than treating the whole dataset uniformly.

#### 2.3.2 Validation strategy (replaces the "site means converge" test)

Checking that post-harmonization site means are equal is **circular**: equalizing site means is what ComBat does by construction. It tests that the library ran, not that harmonization worked.

Instead, the fixture generator injects **known batch effects and known biological effects simultaneously** (§3), and validation asks four separate questions:

1. **Batch-effect recovery.** Is the *estimated* batch parameter (ComBat's additive/multiplicative site terms) approximately equal to the *injected* site effect? This tests estimation, not enforcement.

   > **Revision 5 correction.** "Equal to the injected site effect" is wrong as written, and a test that takes it literally fails a *correct* estimator. The generator produces `observed = biological × mult[s,r] + add[s,r] + ε`, while standard ComBat carries one additive batch term against a shared covariate block — so γ estimates `add[s,r] + (mult[s,r] − 1) × mean(biological | s,r)`, absorbing the multiplicative *mean* shift. Measured on `recovery.yaml`, comparing against `add` alone is 8× worse than comparing against the realized shift. The realized shift is computable exactly, since `ground_truth.parquet` carries both `value` and `true_biological_value`.
   >
   > Second correction: γ is identified only up to a **size-weighted** centering — the batch terms satisfy `Σ_b (n_b/n)·γ_b = 0`, not `mean(γ) = 0`. Equal batch sizes hide the distinction; unequal ones do not. Truth must be centered the same way before comparison.
   >
   > Third: **δ has no injected truth to recover.** `EffectSpec` carries no per-site noise term, so no site-specific residual scale was ever injected. δ can be bounded, not recovered. A `SiteSpec.noise_sd_multiplier` would fix this and is a fixture-generator change, not a week-4 one.
2. **Biological preservation.** Are the injected age, diagnosis, and diagnosis-by-time effects still recoverable after harmonization, within tolerance of their pre-harmonization estimates and their true injected values?
3. **Site-association reduction.** Does the residual association between site and the outcome fall substantially, after adjusting for the biological covariates? (Adjustment matters: if sites differ in age composition, raw site association *should* remain.)
4. **No substantial attenuation.** Does the estimated longitudinal slope survive harmonization? A method that flattens the true within-subject trajectory has failed even if criteria 1 and 3 pass.

```python
def test_combat_recovers_injected_batch_effect():
    # estimated site term ≈ injected site offset, per region, within tolerance

def test_combat_preserves_injected_biological_effects():
    # age, dx, and dx×time coefficients recovered post-harmonization

def test_combat_reduces_covariate_adjusted_site_association():
    # residual site association drops substantially vs unharmonized

def test_combat_does_not_attenuate_injected_longitudinal_slope():
    # slope estimate not systematically shrunk toward zero
```

Criteria 2 and 4 are the ones that catch real failures. A harmonizer that eats your signal passes criterion 3 with flying colours.

Run this suite under two fixture regimes: **site independent of time** (harmonization should work cleanly) and **site confounded with time** (harmonization should visibly attenuate the longitudinal effect). The second regime failing loudly is the *correct* result, and demonstrating it is a stronger portfolio artifact than a suite where everything passes.

> **Revision 5 correction.** Regime B does not attenuate, and the reason is a conflict with §2.3.4 rather than a fixture defect. §2.3.4 requires `time_from_baseline_years` be preserved in the design matrix, and a preserved covariate is exactly the one the batch term cannot absorb. Measured on `confounded.yaml`: the *unharmonized* `time` coefficient comes out **sign-flipped** (+58 against an injected −20 on hippocampus) because the scanner step reads as biology; harmonization recovers it to −19.7. Attenuation appears immediately once `time` is dropped from the covariate set (−7.5), so the predicted failure mode is real but is a property of the **configuration**, not of the regime.
>
> Two findings worth carrying forward. The `time:dx_baseline` interaction is far more robust to the confound than the `time` main effect, because a scanner change shifts both diagnosis groups alike and largely cancels — independent support for §2.5.3 making the interaction primary. And none of this makes regime B interpretable: recovery is knowable only because the truth was injected, so the `interpretable: false` verdict must not soften because an estimate happens to land close.
>
> The suite therefore asserts the measured behavior, and carries a separate test class demonstrating the attenuation §2.3.2 predicted, under the covariate configuration that actually produces it.

#### 2.3.3 Minimum batch size is an engineering heuristic, not a methodological law

There is no universal minimum n per batch. Adequacy depends on the number of batches, covariate balance within batch, outcome variance, the magnitude of the batch effect relative to that variance, and the complexity of the harmonization model. Small batches with strong, clean effects can be fine; larger batches with imbalanced covariates can be worse.

Implement it as a **configurable conservative guard**, default ~20–25:

```yaml
harmonization:
  min_batch_size: 20        # conservative engineering default, not a methodological threshold
  small_batch_policy: report_and_exclude   # report_and_exclude | pool | passthrough
```

Requirements regardless of policy:
- **All batches below threshold are reported in the accounting output**, with their sizes and covariate composition, never dropped silently.
- Pooling is permitted only where scientifically defensible (e.g. same scanner model and protocol at one institution) and the pooling decision is recorded in provenance.
- `passthrough` is allowed but must emit a warning into the report.
- The README states that the default is an engineering choice for stability, and lists the factors that actually govern adequacy.

#### 2.3.4 Implementation for v1.0

- ~~`neuroHarmonize` / `neuroCombat`~~ **implemented in-repo** (`morphline/combat.py`), with biological covariates (`age_baseline`, `sex`, `dx_baseline`, `time_from_baseline_years`) preserved in the design matrix so they are not absorbed as batch effects.

  > **Revision 6.** `neuroHarmonize` pins `numpy==1.26.4`, which would hold the whole project on numpy 1.x — see Build.md deviation 1. The estimator is therefore written from Johnson et al. (2007) directly, and cross-checked against `neuroCombat` behind an optional extra: adjusted values, γ*, and δ* agree to 1e-6. That cross-check is not ceremony — it found the shrinkage pooling on the wrong axis (across batches within a region, where the paper pools across regions within a batch), a defect that passed all four §2.3.2 criteria because a wrong-axis prior still shrinks and still recovers.
  >
  > Covariate preservation turns out to be more load-bearing than this bullet implies: it is what determines whether regime B attenuates (§2.3.2, revision 5).
- Harmonization is a **toggleable stage**; the pipeline runs both ways and the report presents both. *The toggle and the diagnostics are in place; the two-arm sensitivity run is week-5 work (§4).*
- Longitudinal ComBat (Beer et al. 2020) is a **v1.1 issue**, not v1.0 work. `longCombat` is R-only, so the options are an R sidecar container or a Python implementation. A Python port is a genuinely valuable open-source contribution and an excellent second act — and it is a week of work by itself.

**If the schedule forces a cut, keep the written analysis in §2.3.1 even if the implementation goes.** The paragraph explaining non-identifiability is worth more to a reviewer than working ComBat code.

### 2.4 QC design

#### 2.4.1 Three-level status, not a binary exclusion list

QC **identifies and classifies** problems. The analysis layer **decides** what to include. Keeping those separate means review-worthy observations stay visible instead of vanishing into an exclusion list.

QC emits structured fields on every observation:

```
qc_status          # PASS | WARNING | FAIL
qc_flags           # list of triggered flag codes, e.g. ["euler_low", "long_change_extreme"]
qc_score           # optional continuous severity, where a meaningful scale exists
qc_notes           # human-readable detail per flag
analysis_included  # set by the ANALYSIS layer, from qc_status + an explicit policy
```

Inclusion policy is configuration, not hard-coded:

```yaml
analysis:
  include_qc_status: [PASS]          # WARNING observations excluded by default
  sensitivity_include: [PASS, WARNING]  # secondary run for sensitivity reporting
```

Report WARNING rates by site and by flag type. A site with a 40% WARNING rate is telling you something about that site.

#### 2.4.2 Checks

- **Euler / surface holes** (§2.2), thresholds evaluated within site — site distributions genuinely differ.
- **Robust cross-sectional outliers:** median/MAD z-scores per region, computed within site.
- **eTIV plausibility bounds.**
- **Left–right asymmetry index outliers.**
- **Longitudinal change flags** — see below.

#### 2.4.3 Longitudinal change: a review criterion, not an automatic failure

The earlier rule ("annualized hippocampal change >5% = segmentation failure") is wrong as stated. A fixed percentage cannot determine that segmentation failed.

Large apparent longitudinal change can reflect any of: genuine biological change (rapid progression, illness, intervention), scanner or protocol change between sessions, registration or resampling differences, segmentation error, or ordinary measurement noise. **The QC layer cannot distinguish these, so it must not claim to.**

Implement it as a **suspicious-change flag** feeding WARNING status, computed with:

- **Annualized percentage change**, not raw between-session difference.
- **A minimum inter-session interval.** Short intervals make percentage-per-year explode from noise alone; below the minimum, compute the flag on absolute change or suppress it.
- **Expected biological variability** for the region and population — hippocampal atrophy rates in aging cohorts differ by an order of magnitude from cortical thickness change in young adults.
- **Within-subject measurement variability**, from test–retest literature or from the cohort's own short-interval pairs where available.
- **Robust population-level distributions** of annualized change (site-stratified where n permits), so the flag is relative to the observed cohort rather than a hard-coded constant.

Two tiers: a wide WARNING band (review) and a narrow FAIL band for physiologically impossible values (e.g. sign-flipped or order-of-magnitude implausible). Both configurable. Both reported with the actual value, so a human can judge.

#### 2.4.4 Validation: both error directions

Recall alone is not a validation criterion — flagging every observation achieves recall 1.0. Fixtures carry known-clean and known-bad observations, so validate against both:

```python
def test_qc_sensitivity_to_planted_failures():
    # recall on planted FAIL-class observations >= configured floor

def test_qc_specificity_on_clean_observations():
    # false-positive rate on known-clean observations <= configured ceiling

def test_qc_precision_reported():
    # precision computed and asserted above floor; confusion matrix emitted
```

Acceptance criterion, thresholds configurable in the QC test config:

| Metric | Default target |
|---|---|
| Recall on planted FAIL | ≥ 0.95 |
| False-positive rate on clean | ≤ 0.05 |
| Precision | ≥ 0.80 |

Emit the full confusion matrix per flag type into the report, so it is visible which check is over-firing. Report metrics separately for FAIL-only and FAIL+WARNING treatments, since the three-level model has two operating points.

### 2.5 Longitudinal model

#### 2.5.1 Specification, fully disambiguated

```python
# statsmodels MixedLM
value ~ age_baseline + time + dx_baseline + time:dx_baseline + sex + etiv_baseline
# random: intercept + slope on time, grouped by subject_id
```

Every term pinned down for v1:

| Term | Definition | Why |
|---|---|---|
| `age_baseline` | Age at the subject's first included session | Captures **between-subject** age differences |
| `time` | `time_from_baseline_years`, continuous, 0 at first session | Captures **within-subject** longitudinal change |
| `dx_baseline` | **Baseline diagnosis**, fixed per subject | Primary v1 specification |
| `time:dx_baseline` | Diagnosis-by-time interaction | **The primary hypothesis** — differential rate of change by group |
| `etiv_baseline` | **Baseline eTIV**, fixed per subject | Primary v1 specification |
| `sex` | Fixed per subject | Covariate |

The `age_baseline` / `time` split is the point of the parameterization: it separates the cross-sectional age gradient (confounded with cohort effects) from the within-subject rate of change (which is what a longitudinal design is for). Do not put age-at-session in the same model as time; they are collinear by construction.

**`etiv_baseline`, not time-varying eTIV.** Within-subject eTIV should be approximately constant; observed fluctuation is mostly measurement variability, and putting a noisy time-varying regressor in the model lets measurement error in the covariate leak into the slope estimate. Fix it at baseline.

**`dx_baseline`, not time-varying diagnosis.** Time-varying diagnosis is a post-baseline variable that can be affected by the process being modeled — conversion is partly a *consequence* of atrophy — so conditioning on it invites collider bias and complicates interpretation of the very slope you are estimating.

**Sensitivity analyses / future work, explicitly not v1 primary:** time-varying eTIV; time-varying or conversion-status diagnosis; non-linear age or time terms; alternative head-size normalization (proportional rather than covariate adjustment). Each gets a roadmap issue.

Also non-negotiable:
- **Head-size adjustment is mandatory.** Regional volumes without it is the most common error in this literature and it will be noticed immediately. Covariate adjustment is the v1 choice; state it and justify it.
- **Report convergence status per region.** Some fits will fail. Failing loudly beats silently reporting a non-converged model.
- `statsmodels` MixedLM handles random intercept + slope fine. Crossed random effects are painful and unnecessary here.

#### 2.5.2 Region set: 10–20 hypothesis-driven regions is the v1 default

This is a default, not an emergency cut. The portfolio value is the reproducible ingestion → accounting → QC → harmonization → modeling architecture, not the region count. A focused set reduces convergence failures, multiple-testing burden, visualization complexity, and interpretation load, while demonstrating exactly the same engineering.

Default set (AD/aging-oriented, adjust to your dataset):

*Subcortical (aseg):* hippocampus, amygdala, thalamus, caudate, putamen, lateral ventricle, inferior lateral ventricle — bilateral.
*Cortical (aparc):* entorhinal, parahippocampal, inferior parietal, precuneus, posterior cingulate, middle temporal, superior frontal — bilateral.

That is **14 structures × 2 hemispheres = 28 regional tests**, which is the primary-family size used in §2.5.3. Count regions and count tests separately in the report; hemispheres are independent tests and must be counted in the multiplicity family, so a "14-region analysis" is a 28-test family. Stating the wrong one is an easy way to understate your own correction burden.

Whole-region analysis (~113 structures) remains available as a **secondary mode** behind a flag (`--region_set all`), with its own FDR family. It is not the default and is not part of the definition of done.

#### 2.5.3 Multiplicity: define the family explicitly

Ambiguous multiplicity handling is a standard reviewer complaint. Pin it:

- **Primary family:** the `time:dx_baseline` coefficient across the regions in the v1 region set. Benjamini–Hochberg FDR is applied **within this family and only this family**.
- Family size is stated numerically in the report (e.g. "BH-FDR across 28 regional tests of the time × diagnosis interaction").
- **Secondary effects** (main effect of `time`, `dx_baseline`, `age_baseline`) each form their **own separate family**, corrected separately, and are labelled secondary/exploratory in the report.
- Raw *p* and *q* are both reported for every test. Hemispheres are separate tests, counted in the family size.
- Whole-region mode uses the same rule with its own, much larger family — and the report says so, because the same nominal *p* means something different there.

#### 2.5.4 Missing-data policy — document the assumptions

The pipeline distinguishes missingness **by cause**, and records the cause in the accounting output:

| Cause | Recorded as |
|---|---|
| Session absent from dataset (subject did not return, dropout, acquisition failure) | `missing_acquisition` |
| Session present but FreeSurfer output absent or unparseable | `missing_derivative` |
| Session parsed but excluded by QC | `excluded_qc`, with flags retained |

Stated assumptions for v1, to appear in the README and the report:

- Mixed-effects models handle **unbalanced** longitudinal designs naturally and use all available observations; complete cases per subject are not required.
- Validity rests on a **missing-at-random (MAR)** assumption — conditional on the model covariates, missingness is unrelated to the unobserved outcome.
- **This assumption is questionable in aging cohorts**, where dropout can be caused by the disease progression under study, and where QC failure is plausibly associated with atrophy severity and motion. That makes some missingness potentially **not** at random.
- **v1 does not attempt MNAR modeling** (no pattern-mixture models, selection models, or joint dropout modeling). This is an explicit, documented limitation.
- Mitigation within v1 scope: report missingness rates by group, by site, and by timepoint, and compare baseline characteristics of completers versus non-completers. Cheap, honest, and it shows you know the assumption is load-bearing.

### 2.6 Containers

- Base: `python:3.12-slim-bookworm`, multi-stage build, non-root user.
- Pin everything with a lockfile (`uv` or `pip-tools`). "Reproducible" without a lockfile is a claim, not a fact.
- Publish to GHCR, tagged by both git SHA and semver.

**Multi-arch is a bounded engineering task, not a promise.** The order is:

1. **amd64 first.** CI and any cloud execution run amd64. Get that green and locked before anything else.
2. **arm64 via buildx as a bounded follow-on**, timeboxed. Do not let it block the scientific core.

Most of the scientific Python stack ships ARM64 wheels now, but "no compilation surprises" is not a guarantee to plan around — transitive dependencies, pinned older versions, and build-time toolchain differences can all bite, and the failure surfaces during a build you were counting on being fast. If arm64 resists, ship amd64-only for v1.0, note it in the README, and open an issue. Local development on ARM can run the amd64 image under emulation if needed; it is slower, and for tabular data at this scale, slower is still seconds.

### 2.7 Nextflow

- DSL2. Profiles: `test`, `docker`, `apptainer`, `local`.
- Adopt nf-core *conventions* without the full template: `--outdir`, `nextflow_schema.json`, `versions.yml` per process, a `-profile test` that runs on fixtures with no external data.

**Channels carry file paths, not datasets.** The v1 flow (per-subject parse → gather → accounting → QC → harmonize → model → report) is fine, but implement gathering as **Parquet files as persistence boundaries**, not large in-memory channel objects:

- Each process **writes a Parquet file** and emits its **path**.
- Gathering steps `collect()` **paths**, and the consuming process reads them with `pandas`/`pyarrow` inside its own container.
- No process should emit a serialized dataframe, a large value channel, or an in-memory table into a channel.

This costs nothing at v1 scale and means the architecture doesn't need rewriting if the dataset grows by two orders of magnitude — which is a better answer in an interview than "it fit in memory."

- Commit `-with-report`, `-with-trace`, `-with-dag` outputs from a real run into `docs/`. The DAG image in the README is the cheapest high-impact thing in this entire project.

### 2.8 The report is a reproducibility artifact

The HTML report is not just visualization. Every report embeds a provenance block:

```
pipeline_version        git_sha (+ dirty flag)
container_image:tag     container_digest
python_version          nextflow_version
freesurfer_version(s)   observed in the input data
dataset                 dataset_version / accession + version
input_path_or_id        run_parameters (full resolved config)
harmonization_enabled   region_set
qc_config               inclusion_policy
run_timestamp           duration
random_seed(s)
```

Rule: **a reader holding only the HTML file should be able to reconstruct the run.** If a parameter changed the output, it appears in the block.

Report sections, in order: provenance → **data accounting funnel** → QC summary (status distribution, flag breakdown, confusion matrix on fixtures) → harmonization diagnostics (including the scanner × time confound assessment and the sensitivity-analysis labelling from §2.3.1) → model results (primary family first, secondary families clearly separated) → limitations.

Self-contained single HTML: Jinja2 + Plotly with everything inlined, no external asset fetches.

---

## 3. The fixture generator (build this in week 1)

Generate **structurally valid, representative FreeSurfer stats fixtures** — not byte-for-byte reproductions of FreeSurfer output. The goal is to exercise the parser and the statistics realistically, not to clone `mri_segstats`.

### 3.1 Structural realism

Fixtures must exercise: realistic `# Measure` header lines; `# ColHeaders` declarations; whitespace-delimited numeric rows with realistic value ranges; FreeSurfer 5.3 / 6 / 7 format differences (including versions that omit `SurfaceHoles`); extra columns; missing columns; reordered columns; malformed and truncated headers; truncated files mid-row; Unicode in comment fields; empty files; and locale-flavoured numeric formatting.

Pair this with Hypothesis property-based tests over generated header/row structures.

### 3.2 Statistical ground truth — the strongest part of the project

The generator injects, simultaneously and independently controllable:

| Injected | Purpose |
|---|---|
| **Site/batch effects** — additive and multiplicative, per region, per site | Harmonization recovery testing (§2.3.2) |
| **Age effect** (between-subject) | Biological preservation testing |
| **Diagnosis effect** (baseline group difference) | Biological preservation testing |
| **Diagnosis × time interaction** (differential atrophy rate) | The primary modeled hypothesis |
| **Subject random intercepts and slopes** | Realistic within-subject correlation |
| **Measurement noise** with a known variance | Separating signal recovery from noise |
| **Known QC failures** — inflated hole counts, implausible eTIV, extreme longitudinal jumps | QC sensitivity testing |
| **Known-clean observations** | QC specificity testing |
| **Missingness** — separately as `missing_acquisition` and `missing_derivative` | Accounting and missing-data policy testing |

Two configurable regimes, both exercised in CI:
- **Regime A — site independent of time.** Harmonization should recover batch effects and preserve biology.
- **Regime B — site confounded with time.** Harmonization should visibly attenuate the longitudinal effect, and the test asserts that it does. Demonstrating the failure mode is the point.

> **Revision 5 correction, restated here because this bullet says it too.** Regime B does not attenuate under the covariate-preserving configuration §2.3.4 mandates — the *unharmonized* estimate is the badly wrong one. See §2.3.2 for the measured numbers and the mechanism. The failure mode is still demonstrated, by a test class that drops `time_from_baseline_years` from the covariate set, which is the configuration that actually produces it.

One gap in this table, found while writing the §2.3.2 suite: `EffectSpec` injects a single global `noise_sd`, so **no site-specific residual scale is ever injected** and ComBat's δ has no truth to recover. Adding `SiteSpec.noise_sd_multiplier` would close it and make δ a genuine recovery target rather than a bounded quantity. Post-ship (§6).

Everything is seeded and the seed is recorded in provenance.

### 3.3 What this unlocks

Statistical recovery tests that no portfolio project of this type usually has, because real data has no known truth:

```python
def test_mixedlm_recovers_injected_dx_by_time_interaction()
def test_combat_recovers_injected_batch_effect()
def test_combat_preserves_injected_biological_effects()
def test_combat_does_not_attenuate_injected_longitudinal_slope()
def test_qc_sensitivity_and_specificity_on_planted_cases()
def test_accounting_funnel_reconciles_exactly()
```

Put one of these in the README with its actual output. It is the most persuasive artifact in the repo, and the public, verifiable version of the testing discipline your NDA'd work can only assert.

### 3.4 Two validation dimensions, kept separate

Do not conflate these in the README or in CI:

| Dimension | Question | Substrate | Verdict |
|---|---|---|---|
| **Statistical recovery** | Do the methods recover known truth? | Synthetic fixtures | Numeric tolerances, pass/fail in CI |
| **Real-data integration** | Does the pipeline handle real files and real metadata? | ABIDE (cross-sectional), OpenNeuro accession (longitudinal, if the §1.2 audit finds one) | Accounting checks and sanity criteria (§5.2) |

Synthetic recovery cannot prove the parser handles real headers. Real-data integration cannot prove the model recovers a true slope. You need both, and they answer different questions.

---

## 4. Week-by-week

### Week 1 — Access, parser, adapters, fixtures
- [ ] **Day 1:** §1.2 dataset audit run against candidate OpenNeuro accessions — this is the access task now, and it either resolves in an afternoon or Track B is synthetic-only. **Still outstanding as of revision 6, and it is now the gate on week 5**: it decides whether the longitudinal model has a real-data target. A null result is the §0.3 outcome, not a failure, but the answer is needed either way.
- [x] Canonical schema defined (§1.5). This is the contract; changing it later is expensive.
- [x] `FreeSurferStatsParser` — structure only, no dataset knowledge (§1.4).
- [x] `DatasetAdapter` interface + synthetic adapter.
- [x] Fixture generator: structural realism (§3.1) + injected ground truth (§3.2), both regimes.
- [x] pytest + Hypothesis running locally. ruff + pre-commit configured.

**Exit:** `pytest` green; fixtures → canonical Parquet; parser has zero dataset-specific code. **Met** (repo initialized, licensed, README written).

### Week 2 — Walking skeleton
- [x] Stub every remaining stage: accounting counts rows, QC marks everything PASS, harmonize is identity, model fits one region, report is a Jinja2 template with a provenance block and one table.
- [x] Data accounting stage scaffolded with the funnel (§1.6) — built for real rather than stubbed, since §7 never cuts it.
- [x] Dockerfile, **amd64** build, pushed to GHCR.
- [x] Nextflow DSL2 wiring all stages, paths-not-dataframes in channels (§2.7), `-profile test,docker` runs on fixtures.
- [x] GitHub Actions: lint → pytest → container build → **full `nextflow run` on fixtures**.

**Exit:** CI green, badge in README, `nextflow run . -profile test,docker` completes end to end on a clean machine with zero data present. **Met — `v0.1.0` tagged.**

### Week 3 — QC and accounting for real
- [x] Surface-hole extraction and Euler derivation, with the three explicit tests (§2.2). *Landed with the parser in week 1.*
- [x] Robust within-site outlier detection; eTIV bounds; asymmetry outliers.
- [ ] Longitudinal suspicious-change flag with interval and variability handling (§2.4.3). **Deliberately not implemented.** It needs interval handling, expected biological variability, and population distributions of annualized change to mean anything, and none of those are cheap. The omission is *visible* rather than hidden: `stages/qc.py`'s docstring states it, and the validation suite reports planted extreme changes as a declared known miss rather than quietly counting them clean. §7 item 5 permits this cut.
- [x] Three-level `qc_status` / `qc_flags` / `qc_score` / `analysis_included` model (§2.4.1).
- [x] Sensitivity **and** specificity validation against planted cases; confusion matrix in report (§2.4.4).
- [x] Data accounting funnel populated and reconciling exactly.

**Exit:** QC hits recall ≥ 0.95 and FPR ≤ 0.05 on fixtures; funnel reconciles with zero unexplained loss. **Met** — thresholds are asserted against `QCConfig` fields, not literals.

### Week 4 — Real cross-sectional data + harmonization
- [x] ABIDE PCP adapter over per-subject FreeSurfer 5.1 `.stats` (§1.3).
- [x] First full-scale real run — **explicitly labelled cross-sectional integration**, against the §5.2 criteria. 1112 sessions discovered, 3314 files parsed with zero failures, 30,910 canonical observations, funnel reconciling with zero unexplained loss.
- [x] Expect the parser to break on real files. **It did — four further defects, on top of the two found ahead of schedule.** That budget is now spent rather than unspent.
- [x] Cross-check: morphline's per-subject PCP numbers, aggregated, against the dfsp-spirit tables. **Done, and it disproved its own premise** — the tables are FreeSurfer 5.1, not 6 (revision 7), so the planned rank-correlation-with-offsets check became an exact-equality check instead. All 28,976 shared observations match bit-for-bit. That is the only *external* validation of the ingested numbers in the repo; everything else compares morphline against morphline or against truth morphline generated.
- [x] Reconcile the 9 known-incomplete PCP subjects through the accounting funnel with reason codes (§1.3). Attributed as `missing_derivative=4`, `absent_from_source=9`, `source_unavailable=105` — not a silent 1103-vs-1112 discrepancy.
- [x] ComBat with covariate preservation; configurable `min_batch_size` and small-batch policy (§2.3.3). All three policies are behaviourally distinct and separately tested; small-batch covariate composition is reported alongside sizes.
- [x] Batch-effect recovery, biological preservation, and non-attenuation tests on fixtures (§2.3.2). On `recovery.yaml`: γ recovery *r* = 0.998 with worst-case error 7.6% of the true effect spread, site R² 0.62 → 0.0001, slopes unchanged to within 1%.
- [x] Regime B (confounded) test — **asserting the measured behavior, which is not the predicted attenuation.** See the revision 5 correction in §2.3.2.
- [x] Real-data sanity checks per §5.2.
- [ ] If the §1.2 audit found a qualifying OpenNeuro accession: OpenNeuro adapter. **Blocked on the audit, which has not been run.**
- [x] *Added in revision 6, not in the original plan:* cross-check the in-repo ComBat against `neuroCombat` behind an optional extra (Build.md deviation 1's promise). It matches to 1e-6 on adjusted values, γ*, and δ* — and it found a real defect in the estimator's shrinkage axis that no internal test could have caught.

**Exit:** all four harmonization criteria pass on fixtures; ABIDE run passes §5.2 checks. **Met.** Two items carry forward: the dfsp-spirit cross-check and the OpenNeuro audit.

### Week 5 — Longitudinal model
- [x] MixedLM per §2.5.1, on the 10–20 region default set. **All 28 tests fit**; on `config/test.yaml` 28/28 converge with no random slope dropped, in ~2.5s for the whole pipeline, so the focused region set costs nothing in runtime.
- [x] Baseline eTIV, baseline dx, `age_baseline`/`time` split, convergence reporting. *Landed with the walking skeleton; the escalating-optimizer and random-intercept-only fallbacks are what make 28/28 convergence real rather than lucky.*
- [x] FDR within the declared primary family; secondary families separate (§2.5.3). Each of `time`, `dx_baseline`, `age_baseline` is corrected within its own family, never pooled with the primary or with each other, and the report states each family's size. Raw *p* and *q* are reported for every test in every family.
- [x] Missing-data accounting by cause + completer/non-completer comparison (§2.5.4). Baseline age, eTIV, sex, diagnosis, and site are compared between completers and non-completers, summarised by **standardized mean difference rather than a p-value** — the question is whether the groups differ enough for MAR to be doing real work, not whether the sample is large enough to detect that they do. Subjects with no observations at all are counted separately and never compared: that group is the most likely to be informatively missing, so dropping it silently would bias the check meant to detect bias. A cross-sectional dataset reports the comparison as *not applicable* with the reason stated, rather than emitting an empty table that reads as if it had been checked.
- [x] Slope-recovery tests pass against injected truth. `TestSlopeRecovery` fits all 28 regions on a clean regime-A fixture (90 subjects × 4 sessions) and bounds the error three ways: **every 95% interval contains the injected slope (28/28)**, worst-case relative error **10.3%** against a declared 20% bound, median **3.1%**, and mean signed error **−0.6%** against a 5% bound on systematic bias. All 28 survive BH-FDR. The target is measured off the recorded `true_biological_value` column by OLS rather than recomputed from the generator's coefficients — re-deriving `base × direction × b_dxtime` would make the test agree with the generator by construction and keep passing if the fixtures encoded ventricular expansion backwards. Two guards keep the suite non-vacuous: the injected effect must exceed 3 standard errors (recovering zero is trivial), and a deliberate 10% scaling of every estimate fails all three criteria.
- [x] Harmonized vs unharmonized sensitivity run, **labelled as sensitivity analysis** in the report. Both arms fit the same specification; the harmonized arm is primary. The report gives per-region estimates under each arm, direction changes, and significance changes, under a standing statement that agreement between arms is not evidence of correctness. **A run where harmonization changed no values reports the comparison as not applicable** rather than presenting one fit twice — `config/test.yaml` is exactly that case, since both its sites sit below `min_batch_size` by design.

**Exit:** model recovers the injected dx × time interaction within tolerance; sensitivity comparison rendered with correct labelling. **Met.**

**Measured on `config/confounded.yaml` (regime B):** 28 of 28 regions estimated under both arms, **0 direction changes**, **5 significance changes**. The absence of sign flips is independent support for §2.5.3's choice of the interaction as the primary hypothesis — a scanner step shifts both diagnosis groups alike and largely cancels in the interaction, which is exactly what the `time` main effect does *not* do (revision 5). The five significance changes are the honest cost: for those regions, whether the result clears q≤0.05 is a consequence of the harmonization choice rather than of the data.

### Week 6 — Ship
- [ ] README: what/why/how, architecture diagram, Nextflow DAG, quickstart that works from a cold clone, and an honest limitations section covering §2.1, §2.3.1, §2.5.4, and the real-data scope actually achieved.
- [ ] Full provenance block in the report (§2.8).
- [ ] **Publish the example report to GitHub Pages — generated entirely from synthetic fixtures.** See §5.3.
- [ ] Coverage badge, CITATION.cff, LICENSE.
- [ ] Tag `v1.0.0`. Update GitHub bio. **Start applying.**
- [ ] Open 5–8 roadmap issues.

---

## 5. Definition of done for v1.0

### 5.1 Core

1. Cold clone → `nextflow run . -profile test,docker` → green, no data required.
2. CI badge green, including the end-to-end pipeline run on fixtures.
3. **Statistical recovery tests pass:** model slope recovery, ComBat batch recovery + biological preservation + non-attenuation, QC sensitivity *and* specificity.
4. **One real dataset (ABIDE) passes cross-sectional integration**, with §5.2 checks satisfied and the cross-sectional scope stated plainly.
5. Data accounting funnel present in the report and reconciling exactly.
6. Report carries a complete provenance block (§2.8).
7. One published, clickable HTML report — synthetic (§5.3).
8. README documents: the non-identifiability problem, the sensitivity-vs-inference distinction, the MAR assumption, the region-set choice, and the FDR family.

Longitudinal real-data integration is **desirable, not required**. If the §1.2 audit found a qualifying OpenNeuro accession, include it and celebrate — and state which of longitudinal-path and scanner/time confound it actually exercised (§1.3). If not, the README says so.

### 5.2 Real-data integration criteria (replaces "it ran through")

A real-data run passes only if all of these are checked and reported:

- **Counts match expectation:** subjects and sessions found reconcile against the dataset's published participant/session counts, with any discrepancy explained. For ABIDE PCP the expected numbers are known in advance (§1.3): 1112 directories, 1103 complete, 9 incomplete by name — so this check has a right answer, and "1103 subjects ingested" without the 9 attributed is a failure.
- **Header measures survive parsing:** measure counts per file are reported, not assumed. Both real breaks found in §1.3 were silent losses that no row-count or failure-rate check would have caught.
- **Parsing:** success and failure counts reported; every failure has a reason code; failure rate below a stated threshold.
- **Missingness:** rates reported by cause (§2.5.4), by site, and by timepoint.
- **Metadata coverage:** site, scanner, field strength, and FreeSurfer version distributions reported; unexpected or unmapped values surfaced, not silently coerced.
- **Batch sizes:** per-site n reported; batches below `min_batch_size` listed with the policy applied.
- **QC rates:** PASS/WARNING/FAIL rates overall and by site; implausible rates (0% or >50% flagged) treated as a bug in QC, not a finding about the data.
- **Model convergence:** per-region convergence status reported; non-convergence rate below a stated threshold.
- **Output sanity:** volumes in plausible physiological ranges; eTIV distribution sane; no duplicated subject × session × region rows; no all-zero or all-null regions; hemispheric measures roughly symmetric in aggregate.

### 5.3 The public example report must be synthetic

The report published to GitHub Pages is **generated entirely from synthetic fixtures**, unless redistribution of real-data-derived results is explicitly permitted by that dataset's terms. Do not build the public artifact around any dataset whose terms you have not read — and note that CC0/CC BY-SA terms on OpenNeuro and IXI make this *permitted* rather than *advisable*: the synthetic report is still the better demo, for the reasons below.

This is not a limitation — the synthetic report is arguably the better demo. It can show the injected ground truth alongside the recovered estimates, the QC confusion matrix, the harmonization recovery diagnostics under both regimes, and the full accounting funnel. A real-data report can show none of that, because real data has no known truth. And it carries zero licensing or DUA exposure.

---

## 6. Post-ship roadmap (the public commit history)

- Longitudinal ComBat implementation (Python port of Beer et al. 2020) — the headline
- Longitudinal real-data integration, if no qualifying OpenNeuro accession made v1.0
- IXI adapter with a FreeSurfer derivation run — the clean three-scanner harmonization story, once ~600 × `recon-all` is affordable off the critical path
- OASIS-3, if institutional affiliation ever makes the registration possible — still the best single longitudinal multi-scanner source, just not obtainable by an individual
- Whole-region mode promoted from flag to fully supported, with its own multiplicity treatment
- Additional adapters (further OpenNeuro accessions, CAT12, FastSurfer)
- Sensitivity specifications from §2.5.1: time-varying eTIV, time-varying diagnosis, non-linear time
- Missingness sensitivity analyses (pattern-mixture / selection models)
- `nf-core` template compliance + lint passing
- arm64 image, if deferred from v1.0
- Interactive report

---

## 7. Cut order (invoke when a week slips)

1. **Harmonization implementation** — keep the §2.3.1 written analysis and the confound diagnostics.
2. **OpenNeuro / longitudinal real-data integration** — synthetic validates the longitudinal path; document the gap. IXI is already a stretch (§1.3) and is cut before this one.
3. **Second and third adapters** — ABIDE alone satisfies real-data integration.
4. **Model complexity** — random intercepts only, drop random slopes.
5. **QC check breadth** — Euler + robust outliers only; drop asymmetry and longitudinal-change flags. Keep the three-level status model and the sensitivity/specificity validation regardless.

**Never cut:** CI, the fixture suite with recovery tests, the data accounting funnel, the provenance block, the README limitations section, the published synthetic report. Those are the parts a reviewer actually sees, and they are what distinguishes this from a notebook.

Note that region coverage is no longer a cut item — 10–20 regions is the v1 default (§2.5.2), not a fallback.

---

## 8. Standing risks

| Risk | Mitigation |
|---|---|
| No OpenNeuro accession clears the §1.2 audit | Not on critical path (§0.3); synthetic validates the longitudinal path; ABIDE covers real-data integration. Timebox the search to 4 hours and accept the synthetic-only outcome |
| Chosen OpenNeuro accession is single-site or small | Expected, not a surprise (§1.3). It validates the longitudinal path but not the scanner/time confound; report which one and never imply both |
| IXI derivatives never materialize | IXI is a stretch by construction; ABIDE is Track A regardless. Never let ~600 × `recon-all` onto the 6-week path |
| Real stats files break the parser | Week 4 buffer; property-based tests; version-tolerant parsing from day 1; reason-coded parse failures. **Realized: six defects total, all fixed. The buffer was needed** |
| Scanner/time confound is total in real data | Report it as a finding; label affected estimates as sensitivity analysis, not inference (§2.3.1) |
| Harmonization eats the longitudinal signal | Demonstrated by a fixture test that drops the time covariate — which is the configuration that causes it. Preserving the covariate (§2.3.4) prevents it, so the real risk is a *misconfigured* covariate set, not the regime (§2.3.2, revision 5) |
| An in-repo estimator silently diverges from the method it cites | The `neuroCombat` cross-check, run in CI behind an optional extra. It has already caught one such divergence; internal recovery tests did not |
| QC over-flags | Specificity and precision are acceptance criteria, not afterthoughts (§2.4.4) |
| MixedLM non-convergence | Focused region set reduces exposure; per-region status reported; fall back to random-intercept-only |
| arm64 build resists | amd64 is the deliverable; arm64 is timeboxed and droppable (§2.6) |
| Scope creep | §7 cut order; the ship date does not move |
