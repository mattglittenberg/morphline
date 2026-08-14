# OpenNeuro audit — BUILD_PLAN §1.2, Track B

**Verdict: no qualifying accession. Track B stays synthetic-only, which is the §0.3 outcome.**

The best candidate found is **Penn LEAD**, and it fails on a distinction no automated
check could have drawn: the study is genuinely longitudinal, but its *public FreeSurfer
derivatives* are not.

`scripts/audit_openneuro.py` writes its machine-readable output to
`data/openneuro_audit.json`, which is gitignored along with the rest of `data/`. This
file is the committed record. Regenerate the machine output with:

```bash
uv run python scripts/audit_openneuro.py --check ds007088 ds007089
```

---

## The candidate: Penn LEAD

Penn Longitudinal Executive functioning in Adolescent Development — a transdiagnostic
adolescent executive-function study, published across three OpenNeuro accessions. All
three are **CC0**.

| Accession | What it is | FreeSurfer layout |
|---|---|---|
| [ds007116](https://openneuro.org/datasets/ds007116) | Raw BIDS | — |
| [ds007089](https://openneuro.org/datasets/ds007089) | fMRIPrep **anatomical** derivatives | `sub-X/` — 131 subjects, **no session dimension** |
| [ds007088](https://openneuro.org/datasets/ds007088) | fMRIPrep **functional** derivatives | `sub-X_ses-Y/` — 14 subjects with stats, **8 with ≥2 sessions** |

The automated audit surfaces **ds007088** as the candidate, because that is where
per-session `aseg.stats` / `?h.aparc.stats` files actually live. That verdict is correct
on its own terms and misleading in practice, for the reasons below.

## §1.2's three criteria

| Criterion | Verdict |
|---|---|
| Locatable FreeSurfer `.stats` files | **Pass** — 22 complete `aseg` + `lh/rh.aparc` triples in ds007088 |
| ≥2 structural sessions per subject, for a usable number of subjects | **Fail on count** — 8 subjects, against a specification (§2.5.1) carrying random intercept *and* slope across 28 tests |
| Scanner or site variation | **Fail cleanly** — a single scanner, zero variation |

§1.2 permits taking a dataset that fails only the third criterion. This one fails the
first on its qualifying clause, which is the disqualifying failure.

## What the manual review established

### The session dimension is in the raw data, not the derivatives

Sessions per subject across all 132 raw subjects in ds007116:

| Sessions | Subjects |
|---|---|
| 1 | 59 |
| 2 | 53 |
| 3 | 20 |

**73 of 132 subjects have ≥2 sessions.** That is a real longitudinal cohort. The
derivatives do not carry it:

- **ds007089** holds one FreeSurfer run per subject, with no session entity in any of its
  131 directory names (complete listing, not truncated). Its README's own fMRIPrep
  invocation passes `--fs-subjects-dir .../fmriprep_anat/sourcedata/freesurfer` with
  `--fs-no-resume` — a single shared anatomical run reused across every session. Those
  stats are therefore **not attributable to any one timepoint**, so they cannot serve as a
  longitudinal observation *or* as a clean cross-sectional baseline.
- **ds007088** holds genuinely per-session output — the `aseg.stats` header records
  `SUBJECTS_DIR .../job-7567866-177-sub-21554-ses-1/...`, a session-specific BABS job —
  but only for 8 multi-session subjects, as an incidental byproduct of session-independent
  processing. The key scan was complete, not truncated: 27,753 keys, 22 files per stats
  type, 24 session directories over 15 subject IDs.

Producing longitudinal derivatives for the 73 eligible subjects means 73+ `recon-all`
runs. §1.3 bars that from the six-week path.

### The time axis exists and is usable

`ds007116/sub-*/sub-*_sessions.tsv` carries `session_id`, `age`, and `acq_time`.
`acq_time` is date-shifted to the year 1800 (BIDS anonymization) but **intervals are
preserved** — checked on two subjects, where the `age` deltas and `acq_time` deltas agree
to 0.01 yr. Either column yields `time_from_baseline_years`.

Note for any future adapter: date arithmetic here must use intervals only. A plausibility
check on absolute dates would reject every row.

### The model's covariates are present

`ds007116/participants.tsv` carries `sex`, `age`, `age_months`, `study_group`, and binary
diagnosis columns (`dx_none`, `dx_prodromal`, `dx_prodromal_remit`, `dx_psychosis`,
`dx_moodnos`, `dx_mdd`, `dx_bp`, `dx_adhd`, `dx_ptsd`, `dx_ptsd_remit`). So `dx_baseline`
is constructible rather than absent, and `age_baseline` follows from the session tables.
`etiv_baseline` comes from `aseg.stats`. The §2.5.1 specification is fully supported by
the metadata — it is only the sample size that fails.

### Scanner metadata is uniform

Every `_T1w.json` sampled — 31 sidecars spanning both the multi-session subjects and the
wider cohort — reports identically:

```
Siemens Prisma_fit  3T  serial=167024  station=HUP FNDBA MR2  sw=syngo MR E11  institution=HUP
```

One scanner, one console, one software version, one site, with no mid-study upgrade. This
is not thin variation; it is none. ComBat would have a single batch and nothing to
estimate, and the two-arm sensitivity comparison would correctly report itself
not-applicable — the degenerate path `config/test.yaml` already exercises.

### FreeSurfer 7.3.2

ds007088's stats are FreeSurfer **7.3.2** (`# cvs_version 7.3.2`), which would have been
the first real 7.x data in the repo — 7.x currently exists only in fixtures. Three header
properties are each one the ingestion layer already reasons about:

- `cvs_version` starts with a digit, so it yields `freesurfer_version = "7.3.2"`, unlike
  ABIDE 5.1's `$Id: mri_segstats.c,v ...$`.
- `lhSurfaceHoles` / `rhSurfaceHoles` are present (10 / 7), so Euler derivation lands in
  its non-null branch rather than 5.1's null branch.
- Intracranial volume appears as `EstimatedTotalIntraCranialVol, eTIV` — the 6+ naming
  that short-name-*and*-alias indexing exists to handle.

## Consequence for v1.0

v1.0 ships with the longitudinal path validated against injected synthetic truth, and
names this gap concretely rather than generically. The post-ship target is **ds007116 with
a FreeSurfer derivation run over its 73 multi-session subjects** (§6) — a specific,
CC0-licensed, single-site longitudinal cohort, which would validate the longitudinal path
but *not* the scanner/time confound. Those are separate claims and must stay separate.
