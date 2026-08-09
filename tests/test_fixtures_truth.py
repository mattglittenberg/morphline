"""Ground-truth generator tests (BUILD_PLAN §3.2).

The recovery tests in later weeks are only as trustworthy as the truth they
compare against, so the generator itself gets validated: are the effects
actually injected, is the seed actually deterministic, and do the two regimes
actually differ in the way they claim to?
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_fixture_config
from morphline.config import EffectSpec, PlantedSpec, Regime, SiteSpec
from morphline.fixtures import generate_ground_truth
from morphline.fixtures.truth import BASE_VALUES, EXPANDING_STRUCTURES


def test_seed_is_deterministic() -> None:
    a = generate_ground_truth(make_fixture_config(seed=99))
    b = generate_ground_truth(make_fixture_config(seed=99))
    np.testing.assert_allclose(
        a.observations["value"].to_numpy(), b.observations["value"].to_numpy()
    )


def test_different_seeds_give_different_data() -> None:
    a = generate_ground_truth(make_fixture_config(seed=1))
    b = generate_ground_truth(make_fixture_config(seed=2))
    assert not np.allclose(
        a.observations["value"].to_numpy()[:50], b.observations["value"].to_numpy()[:50]
    )


def test_every_v1_region_is_generated() -> None:
    truth = generate_ground_truth(make_fixture_config())
    assert truth.observations["region"].nunique() == 28


def test_values_are_in_plausible_ranges() -> None:
    """Volumes in mm^3, thicknesses in mm — §5.2 output sanity."""
    truth = generate_ground_truth(make_fixture_config())
    for structure, base in BASE_VALUES.items():
        rows = truth.observations[truth.observations["structure"] == structure]
        assert not rows.empty
        # Injected effects and noise are all fractional, so nothing should
        # stray beyond a factor of two from its base value.
        assert rows["value"].min() > base * 0.4
        assert rows["value"].max() < base * 2.0


def test_dx_by_time_interaction_is_actually_injected() -> None:
    """The primary hypothesis must be present in the data before any stage
    can be tested for recovering it."""
    cfg = make_fixture_config(
        seed=5,
        n_sessions=4,
        sites=(SiteSpec(name="s", n_subjects=60),),
        effects=EffectSpec(
            dx_by_time_per_year=-0.05,
            noise_sd=0.005,
            random_intercept_sd=0.01,
            random_slope_sd=0.001,
        ),
        planted=PlantedSpec(
            missing_acquisition_fraction=0.0,
            missing_derivative_fraction=0.0,
            malformed_file_fraction=0.0,
            qc_extreme_change_fraction=0.0,
            qc_high_holes_fraction=0.0,
            qc_bad_etiv_fraction=0.0,
        ),
    )
    truth = generate_ground_truth(cfg)
    obs = truth.observations.merge(truth.subjects[["subject_id", "dx_baseline"]], on="subject_id")
    hippo = obs[obs["region"] == "lh-hippocampus"]

    def slope(group_name: str) -> float:
        g = hippo[hippo["dx_baseline"] == group_name]
        return float(np.polyfit(g["time_from_baseline_years"], g["true_biological_value"], 1)[0])

    patient_slope = slope("patient")
    control_slope = slope("control")
    # Patients were given an extra -5%/yr of hippocampal atrophy.
    assert patient_slope < control_slope
    expected_gap = BASE_VALUES["hippocampus"] * -0.05
    assert patient_slope - control_slope == pytest.approx(expected_gap, rel=0.25)


def test_ventricles_expand_while_grey_matter_shrinks() -> None:
    """Atrophy must not be encoded backwards for expanding structures.

    A fixture set where ventricles shrink with disease would let a downstream
    stage 'recover' a sign error and call it success.
    """
    cfg = make_fixture_config(
        seed=11,
        n_sessions=4,
        sites=(SiteSpec(name="s", n_subjects=50),),
        effects=EffectSpec(
            time_per_year=-0.02,
            dx_by_time_per_year=-0.03,
            noise_sd=0.005,
            random_intercept_sd=0.01,
            random_slope_sd=0.001,
        ),
    )
    truth = generate_ground_truth(cfg)

    for structure in ("hippocampus", "lateral-ventricle"):
        rows = truth.observations[truth.observations["structure"] == structure]
        slope = float(
            np.polyfit(rows["time_from_baseline_years"], rows["true_biological_value"], 1)[0]
        )
        if structure in EXPANDING_STRUCTURES:
            assert slope > 0, f"{structure} should expand over time"
        else:
            assert slope < 0, f"{structure} should shrink over time"


def test_site_effects_are_injected_per_region() -> None:
    truth = generate_ground_truth(make_fixture_config())
    assert set(truth.site_effects) == {"site-a", "site-b"}
    for effect in truth.site_effects.values():
        assert len(effect.additive) == 28
        assert len(effect.multiplicative) == 28
    # Sites were configured with opposite-sign additive effects.
    a = np.mean(list(truth.site_effects["site-a"].additive.values()))
    b = np.mean(list(truth.site_effects["site-b"].additive.values()))
    assert a > 0 > b


def test_regime_a_keeps_site_independent_of_time() -> None:
    truth = generate_ground_truth(make_fixture_config(regime=Regime.A_INDEPENDENT, n_sessions=4))
    sessions = truth.sessions
    assert (sessions["acquisition_site"] == sessions["site"]).all()


def test_regime_b_confounds_site_with_time() -> None:
    """Later sessions migrate to a different scanner — the aging-cohort pattern."""
    truth = generate_ground_truth(make_fixture_config(regime=Regime.B_CONFOUNDED, n_sessions=4))
    sessions = truth.sessions
    early = sessions[sessions["visit_index"] < 2]["acquisition_site"].unique()
    late = sessions[sessions["visit_index"] >= 2]["acquisition_site"].unique()
    assert len(late) == 1, "late sessions should all be on the newer scanner"
    assert set(early) != set(late)


def test_missingness_is_recorded_by_cause() -> None:
    """§2.5.4 distinguishes missingness by cause; the fixtures must too."""
    cfg = make_fixture_config(
        seed=20260800,
        sites=(SiteSpec(name="a", n_subjects=8), SiteSpec(name="b", n_subjects=8)),
        planted=PlantedSpec(missing_acquisition_fraction=0.10, missing_derivative_fraction=0.08),
    )
    truth = generate_ground_truth(cfg)
    causes = set(truth.sessions["missing_cause"].dropna())
    assert causes <= {"missing_acquisition", "missing_derivative"}
    assert causes, "expected some planted missingness"


def test_clean_observations_are_labelled() -> None:
    """QC specificity testing needs known-clean cases (§2.4.4)."""
    truth = generate_ground_truth(make_fixture_config())
    assert "is_clean" in truth.observations.columns
    assert truth.observations["is_clean"].any()


def test_manifest_records_seed_and_regime() -> None:
    """Everything is seeded and the seed is recorded in provenance (§3.2)."""
    truth = generate_ground_truth(make_fixture_config(seed=777, regime=Regime.B_CONFOUNDED))
    assert truth.manifest["seed"] == 777
    assert truth.manifest["regime"] == "B"
    assert "injected_effects" in truth.manifest
    assert "site_additive_effects" in truth.manifest


def test_missing_sessions_produce_no_observations() -> None:
    cfg = make_fixture_config(
        seed=20260800,
        planted=PlantedSpec(missing_acquisition_fraction=0.5, missing_derivative_fraction=0.0),
    )
    truth = generate_ground_truth(cfg)
    missing = truth.sessions[truth.sessions["missing_cause"].notna()]
    for row in missing.itertuples():
        match = truth.observations[
            (truth.observations["subject_id"] == row.subject_id)
            & (truth.observations["session_id"] == row.session_id)
        ]
        assert match.empty
