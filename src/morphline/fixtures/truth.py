r"""Injected ground truth for synthetic fixtures (BUILD_PLAN §3.2).

This is the strongest part of the project: real data has no known true slope,
so "did the model recover the effect?" is unanswerable against it. Here every
effect is injected deliberately and recorded, so each stage can be asked
whether it recovered what was put in.

The generative equation
-----------------------
For subject *i*, session *j*, region *r*, at site *s*::

    biological_ijr = base_r * (1
        + b_age      * (age_baseline_i - 70) / 10
        + b_dx       * dx_i
        + b_sex      * female_i
        + (b_time + b_dxtime * dx_i) * t_ij
        + u0_ir + u1_ir * t_ij)

    observed_ijr = biological_ijr * mult_site[s, r] + add_site[s, r] + eps_ijr

where ``dx_i`` is 1 for patients and 0 for controls, ``t_ij`` is years from the
subject's baseline session, ``u0``/``u1`` are subject random intercept and
slope, and ``eps`` is measurement noise. All of ``b_*``, ``u*`` and ``eps`` are
expressed as *fractions of the region's baseline value*, so they are
comparable across regions spanning four orders of magnitude in absolute size.

The equation is written out here so tests can invert it. ``b_dxtime`` is the
primary modeled hypothesis (§2.5.1) — the differential rate of change by
diagnosis group — and ``test_mixedlm_recovers_injected_dx_by_time_interaction``
exists to recover exactly this coefficient.

Two regimes (§3.2)
------------------
* **Regime A — site independent of time.** Subjects at every site are enrolled
  across the full time range. Harmonization should recover batch effects and
  preserve biology.
* **Regime B — site confounded with time.** Sessions migrate to a later
  scanner partway through follow-up, exactly as happens in a real aging cohort
  when a site replaces its scanner. Harmonization should visibly attenuate the
  longitudinal effect, and the Regime B test asserts that it does.

The second regime failing loudly is the *correct* result. Where scanner is
confounded with time, the biological and scanner effects are not identifiable
from the observed data alone — that is a property of the study design, not a
deficiency in the software (§2.3.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from morphline.coerce import as_float, as_str
from morphline.config import FixtureConfig, Regime
from morphline.regions import MEASURE_FOR_STRUCTURE, V1_STRUCTURES, region_key
from morphline.schema import Hemisphere, MeasureType

#: Plausible baseline values per structure, in native units — mm³ for
#: subcortical volumes, mm for cortical thickness. Approximate adult means;
#: the point is a realistic dynamic range, not clinical accuracy.
BASE_VALUES: Final[dict[str, float]] = {
    "hippocampus": 4000.0,
    "amygdala": 1600.0,
    "thalamus": 7500.0,
    "caudate": 3500.0,
    "putamen": 4800.0,
    "lateral-ventricle": 12000.0,
    "inferior-lateral-ventricle": 550.0,
    "entorhinal": 3.30,
    "parahippocampal": 2.70,
    "inferiorparietal": 2.45,
    "precuneus": 2.35,
    "posteriorcingulate": 2.50,
    "middletemporal": 2.85,
    "superiorfrontal": 2.75,
}

#: Structures that *grow* with atrophy rather than shrink. Ventricles expand as
#: surrounding tissue is lost, so an injected "atrophy" effect must flip sign
#: for them or the fixtures encode biology backwards.
EXPANDING_STRUCTURES: Final = frozenset({"lateral-ventricle", "inferior-lateral-ventricle"})


@dataclass(frozen=True, slots=True)
class SiteEffect:
    """The batch effect injected at one site, per region.

    Attributes:
        additive: Additive offset per region, in the region's native units.
        multiplicative: Multiplicative scaling per region. 1.0 is no effect.
    """

    additive: dict[str, float]
    multiplicative: dict[str, float]


@dataclass(slots=True)
class GroundTruth:
    """Everything injected into a synthetic dataset, kept for validation.

    Attributes:
        observations: One row per subject × session × region, carrying both the
            noise-free biological value and the observed value after site
            effects and noise. Recovery tests compare estimates against these.
        subjects: One row per subject, with design variables and random effects.
        sessions: One row per subject × session, including planted problems and
            missingness causes.
        site_effects: The injected batch parameters, keyed by site name.
        effects: The injected biological coefficients.
        manifest: A JSON-serialisable summary written alongside the fixtures.
    """

    observations: pd.DataFrame
    subjects: pd.DataFrame
    sessions: pd.DataFrame
    site_effects: dict[str, SiteEffect]
    effects: dict[str, float]
    manifest: dict[str, Any] = field(default_factory=dict)


def _region_specs() -> list[tuple[str, str, str, MeasureType]]:
    """Return ``(region, structure, hemisphere, measure_type)`` for the v1 set."""
    return [
        (region_key(structure, hemi), structure, hemi.value, MEASURE_FOR_STRUCTURE[structure])
        for structure in V1_STRUCTURES
        for hemi in (Hemisphere.LEFT, Hemisphere.RIGHT)
    ]


def _draw_site_effects(
    cfg: FixtureConfig, regions: list[str], rng: np.random.Generator
) -> dict[str, SiteEffect]:
    """Draw per-site, per-region additive and multiplicative batch effects.

    Each site's configured effect sets the *scale*; the per-region values vary
    around it, because a real scanner does not bias every structure identically.
    """
    effects: dict[str, SiteEffect] = {}
    for site in cfg.sites:
        additive = {}
        multiplicative = {}
        for region in regions:
            structure = region.split("-", 1)[1]
            base = BASE_VALUES[structure]
            # Additive effect is configured as a fraction of the base value so
            # one number is meaningful across mm and mm³ regions alike.
            additive[region] = float(
                base * site.additive_effect * (1.0 + 0.3 * rng.standard_normal())
            )
            spread = site.multiplicative_effect - 1.0
            multiplicative[region] = float(1.0 + spread * (1.0 + 0.3 * rng.standard_normal()))
        effects[site.name] = SiteEffect(additive=additive, multiplicative=multiplicative)
    return effects


def _build_subjects(cfg: FixtureConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Assign subjects to sites and draw their baseline characteristics."""
    rows: list[dict[str, Any]] = []
    subject_index = 0
    for site in cfg.sites:
        for _ in range(site.n_subjects):
            subject_index += 1
            is_patient = bool(rng.random() < cfg.patient_fraction)
            rows.append(
                {
                    "subject_id": f"sub-{subject_index:04d}",
                    "site": site.name,
                    "scanner_manufacturer": site.scanner_manufacturer,
                    "scanner_model": site.scanner_model,
                    "field_strength_tesla": site.field_strength_tesla,
                    "age_baseline": float(rng.normal(cfg.baseline_age_mean, cfg.baseline_age_sd)),
                    "sex": "F" if rng.random() < 0.5 else "M",
                    "dx_baseline": "patient" if is_patient else "control",
                    "dx_code": 1.0 if is_patient else 0.0,
                    "etiv_baseline": float(rng.normal(cfg.etiv_mean, cfg.etiv_sd)),
                    "freesurfer_version": _sample_version(cfg, rng),
                }
            )
    return pd.DataFrame(rows)


def _sample_version(cfg: FixtureConfig, rng: np.random.Generator) -> str:
    """Sample a FreeSurfer version for one subject, per the configured mix."""
    versions = list(cfg.freesurfer_version_mix)
    weights = np.array([cfg.freesurfer_version_mix[v] for v in versions], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(versions, p=weights))


def _build_sessions(
    cfg: FixtureConfig, subjects: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Expand subjects into sessions, planting missingness and QC problems."""
    planted = cfg.planted
    rows: list[dict[str, Any]] = []

    for subject in subjects.itertuples():
        for visit in range(cfg.n_sessions):
            # Real follow-up never lands exactly on the nominal interval.
            jitter = float(rng.normal(0.0, 0.08)) if visit > 0 else 0.0
            t = max(0.0, visit * cfg.session_interval_years + jitter)

            missing_cause: str | None = None
            if visit > 0:
                if rng.random() < planted.missing_acquisition_fraction:
                    missing_cause = "missing_acquisition"
                elif rng.random() < planted.missing_derivative_fraction:
                    missing_cause = "missing_derivative"

            rows.append(
                {
                    "subject_id": subject.subject_id,
                    "session_id": f"ses-{visit + 1:02d}",
                    "visit_index": visit,
                    "time_from_baseline_years": t,
                    "age_at_session": as_float(subject.age_baseline) + t,
                    "site": subject.site,
                    "missing_cause": missing_cause,
                    "planted_high_holes": bool(
                        missing_cause is None and rng.random() < planted.qc_high_holes_fraction
                    ),
                    "planted_bad_etiv": bool(
                        missing_cause is None and rng.random() < planted.qc_bad_etiv_fraction
                    ),
                    "planted_extreme_change": bool(
                        missing_cause is None
                        and visit > 0
                        and rng.random() < planted.qc_extreme_change_fraction
                    ),
                    "planted_malformed": bool(
                        missing_cause is None and rng.random() < planted.malformed_file_fraction
                    ),
                }
            )

    sessions = pd.DataFrame(rows)
    sessions["is_clean"] = ~(
        sessions["planted_high_holes"]
        | sessions["planted_bad_etiv"]
        | sessions["planted_extreme_change"]
        | sessions["planted_malformed"]
    )
    return _apply_regime(cfg, sessions, rng)


def _apply_regime(
    cfg: FixtureConfig, sessions: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Set each session's *acquisition* site according to the regime.

    Regime A leaves the subject's site alone — site is a stable subject
    property, independent of time. Regime B migrates later sessions onto a
    different scanner, reproducing the aging-cohort pattern where subjects are
    scanned on an older scanner early and a newer one later. That is precisely
    the confound that makes the biological and scanner effects non-identifiable
    (§2.3.1), and Regime B exists so the pipeline can be shown failing on it.
    """
    sessions = sessions.copy()
    if cfg.regime is Regime.A_INDEPENDENT or len(cfg.sites) < 2:
        sessions["acquisition_site"] = sessions["site"]
        return sessions

    late_site = cfg.sites[-1].name
    switch_after = max(1, cfg.n_sessions // 2)
    sessions["acquisition_site"] = np.where(
        sessions["visit_index"] >= switch_after, late_site, sessions["site"]
    )
    return sessions


def generate_ground_truth(cfg: FixtureConfig) -> GroundTruth:
    """Generate a synthetic dataset's ground truth.

    Args:
        cfg: Fixture configuration, including the seed and regime.

    Returns:
        The injected truth: observations, subject and session tables, site
        batch parameters, and a manifest. Everything is seeded and the seed is
        recorded in provenance (§3.2).
    """
    rng = np.random.default_rng(cfg.seed)
    specs = _region_specs()
    regions = [region for region, _, _, _ in specs]

    subjects = _build_subjects(cfg, rng)
    sessions = _build_sessions(cfg, subjects, rng)
    site_effects = _draw_site_effects(cfg, regions, rng)
    eff = cfg.effects

    # Subject random effects are drawn per subject × region: a subject whose
    # hippocampus sits high need not have a high thalamus, and pretending
    # otherwise would build a correlation structure the model does not assume.
    subject_ids = subjects["subject_id"].tolist()
    u0 = {
        (sid, region): float(rng.normal(0.0, eff.random_intercept_sd))
        for sid in subject_ids
        for region in regions
    }
    u1 = {
        (sid, region): float(rng.normal(0.0, eff.random_slope_sd))
        for sid in subject_ids
        for region in regions
    }

    subject_lookup = subjects.set_index("subject_id")
    present = sessions[sessions["missing_cause"].isna()]

    records: list[dict[str, Any]] = []
    for session in present.itertuples():
        subj = subject_lookup.loc[session.subject_id]
        dx = as_float(subj["dx_code"])
        female = 1.0 if subj["sex"] == "F" else 0.0
        age_centered = (as_float(subj["age_baseline"]) - 70.0) / 10.0
        t = as_float(session.time_from_baseline_years)
        site_effect = site_effects[as_str(session.acquisition_site)]

        for region, structure, hemisphere, measure_type in specs:
            base = BASE_VALUES[structure]
            # Ventricles expand as tissue is lost, so atrophy coefficients flip
            # sign for them. Without this the fixtures would encode biology
            # backwards and any test that "recovers" it would be recovering a
            # mistake.
            direction = -1.0 if structure in EXPANDING_STRUCTURES else 1.0

            fractional = (
                direction * eff.age_per_decade * age_centered
                + direction * eff.dx_baseline * dx
                + direction * eff.sex_effect * female
                + direction * (eff.time_per_year + eff.dx_by_time_per_year * dx) * t
                + u0[(session.subject_id, region)]
                + u1[(session.subject_id, region)] * t
            )
            biological = base * (1.0 + fractional)

            noise = float(rng.normal(0.0, eff.noise_sd)) * base
            observed = (
                biological * site_effect.multiplicative[region]
                + site_effect.additive[region]
                + noise
            )

            if session.planted_extreme_change:
                # A segmentation-scale failure, not a biological signal: a
                # large step in one session only, which is exactly what the
                # longitudinal change flag exists to surface for review.
                observed *= 1.0 + direction * -0.35

            records.append(
                {
                    "subject_id": session.subject_id,
                    "session_id": session.session_id,
                    "region": region,
                    "structure": structure,
                    "hemisphere": hemisphere,
                    "measure_type": str(measure_type),
                    "true_biological_value": biological,
                    "value": observed,
                    "time_from_baseline_years": t,
                    "site": session.acquisition_site,
                    "is_clean": session.is_clean,
                }
            )

    observations = pd.DataFrame(records)

    manifest: dict[str, Any] = {
        "seed": cfg.seed,
        "regime": str(cfg.regime),
        "n_subjects": len(subjects),
        "n_sessions_planned": len(sessions),
        "n_sessions_present": len(present),
        "n_regions": len(regions),
        "n_observations": len(observations),
        "injected_effects": eff.model_dump(mode="json"),
        "site_additive_effects": {name: effect.additive for name, effect in site_effects.items()},
        "site_multiplicative_effects": {
            name: effect.multiplicative for name, effect in site_effects.items()
        },
        "planted": cfg.planted.model_dump(mode="json"),
        "missingness": (sessions["missing_cause"].value_counts(dropna=True).to_dict()),
        "freesurfer_version_mix": cfg.freesurfer_version_mix,
    }

    return GroundTruth(
        observations=observations,
        subjects=subjects,
        sessions=sessions,
        site_effects=site_effects,
        effects=eff.model_dump(mode="json"),
        manifest=manifest,
    )
