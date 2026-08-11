"""Model stage tests (BUILD_PLAN §2.5).

v0.1.0 fits one region, so the statistical recovery test here is a *smoke*
check that the wiring produces an estimate of the right sign and rough
magnitude — not the tolerance-bounded recovery test week 5 will add across all
28 regions. It is marked accordingly. What is tested strictly is the machinery
that must be right before recovery testing means anything: the BH-FDR
implementation, the family definition, and convergence reporting.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_fixture_config
from morphline.adapters import SyntheticAdapter
from morphline.config import AnalysisConfig, EffectSpec, PlantedSpec, QCConfig, SiteSpec
from morphline.fixtures import write_fixtures
from morphline.regions import V1_TEST_COUNT
from morphline.stages.ingest import ingest
from morphline.stages.model import PRIMARY_TERM, RegionFit, apply_fdr, fit_model, fit_region
from morphline.stages.qc import apply_qc


def make_fit(region: str, p: float | None, *, converged: bool = True) -> RegionFit:
    return RegionFit(
        region=region,
        measure_type="volume",
        n_observations=100,
        n_subjects=25,
        converged=converged,
        p_values={PRIMARY_TERM: p} if p is not None else {},
    )


class TestBenjaminiHochberg:
    """FDR correctness, checked against hand-computable cases."""

    def test_known_values(self) -> None:
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
        fits = [make_fit(f"r{i}", p) for i, p in enumerate(p_values)]
        apply_fdr(fits, 0.05)

        n = len(p_values)
        expected = np.array(p_values) * n / np.arange(1, n + 1)
        expected = np.minimum.accumulate(expected[::-1])[::-1]

        actual = np.array([f.q_value for f in fits])
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_q_values_are_monotone_in_p(self) -> None:
        fits = [make_fit(f"r{i}", p) for i, p in enumerate([0.5, 0.01, 0.2, 0.001])]
        apply_fdr(fits, 0.05)
        ordered = sorted(fits, key=lambda f: f.p_values[PRIMARY_TERM])
        q = [f.q_value for f in ordered]
        assert all(a <= b + 1e-12 for a, b in itertools.pairwise(q))

    def test_q_never_below_p(self) -> None:
        fits = [make_fit(f"r{i}", p) for i, p in enumerate([0.01, 0.02, 0.03])]
        apply_fdr(fits, 0.05)
        for fit in fits:
            assert fit.q_value >= fit.p_values[PRIMARY_TERM] - 1e-12

    def test_q_is_clipped_to_one(self) -> None:
        fits = [make_fit(f"r{i}", p) for i, p in enumerate([0.9, 0.95, 0.99])]
        apply_fdr(fits, 0.05)
        assert all(f.q_value <= 1.0 for f in fits)

    def test_single_test_family_leaves_p_unchanged(self) -> None:
        fits = [make_fit("r0", 0.03)]
        apply_fdr(fits, 0.05)
        assert fits[0].q_value == pytest.approx(0.03)

    def test_non_converged_fits_are_excluded_from_the_family(self) -> None:
        """A failed fit must not inflate the family size and dilute the
        correction for the fits that did converge."""
        fits = [
            make_fit("r0", 0.01),
            make_fit("r1", None, converged=False),
            make_fit("r2", 0.02),
        ]
        apply_fdr(fits, 0.05)
        assert fits[1].q_value is None
        # Family size is 2, not 3.
        assert fits[0].q_value == pytest.approx(0.01 * 2 / 1)

    def test_empty_family_does_not_crash(self) -> None:
        fits: list[RegionFit] = []
        apply_fdr(fits, 0.05)


class TestConvergenceReporting:
    """Failing loudly beats silently reporting a non-converged model (§2.5.1)."""

    def test_insufficient_data_is_reported_not_raised(self, fixture_tree: Path) -> None:
        obs = ingest(SyntheticAdapter(fixture_tree)).observations
        tiny = obs[obs["subject_id"].isin(obs["subject_id"].unique()[:2])]
        fit = fit_region(tiny, "lh-hippocampus", "volume")
        assert not fit.converged
        assert "insufficient data" in fit.message

    def test_missing_region_is_reported_not_raised(self, fixture_tree: Path) -> None:
        obs = ingest(SyntheticAdapter(fixture_tree)).observations
        fit = fit_region(obs, "lh-nonexistent-region", "volume")
        assert not fit.converged
        assert fit.n_observations == 0


class TestCrossSectionalDesign:
    """A design that cannot answer the question is not a fit that failed."""

    @staticmethod
    def _cross_sectional(fixture_tree: Path) -> pd.DataFrame:
        """Keep one session per subject, which is what ABIDE actually is."""
        obs = ingest(SyntheticAdapter(fixture_tree)).observations
        baseline = obs[obs["time_from_baseline_years"] == 0.0].copy()
        assert not baseline.empty
        return baseline

    def test_zero_time_variance_is_not_estimable(self, fixture_tree: Path) -> None:
        fit = fit_region(self._cross_sectional(fixture_tree), "lh-hippocampus", "volume")
        assert not fit.estimable
        assert not fit.converged
        assert "not estimable" in fit.message
        assert "cross-sectional" in fit.message

    def test_not_estimable_is_distinct_from_non_convergence(self, fixture_tree: Path) -> None:
        """The optimizer never ran, so it cannot be blamed for the outcome.

        Reporting this as non-convergence would put a study-design limitation
        into the convergence rate §5.2 asks to be held below a threshold, where
        no amount of model work could ever move it.
        """
        included = apply_qc(self._cross_sectional(fixture_tree), QCConfig(), AnalysisConfig())
        results = fit_model(included, AnalysisConfig())
        assert results.fits[0].n_subjects > 0
        assert results.n_estimable == 0
        assert all(not f.estimable for f in results.fits)
        assert any("not identifiable" in note for note in results.notes)
        assert not any("Non-converged regions" in note for note in results.notes)

    def test_no_coefficients_are_invented(self, fixture_tree: Path) -> None:
        fit = fit_region(self._cross_sectional(fixture_tree), "lh-hippocampus", "volume")
        assert fit.coefficients == {}
        assert fit.primary_estimate is None
        assert fit.n_observations == 0

    def test_longitudinal_fixtures_remain_estimable(self, fixture_tree: Path) -> None:
        """The guard must not fire on the data it is supposed to let through."""
        obs = ingest(SyntheticAdapter(fixture_tree)).observations
        assert fit_region(obs, "lh-hippocampus", "volume").estimable


def test_family_size_is_reported_and_matches_the_region_set(fixture_tree: Path) -> None:
    """Family size must be stated numerically (§2.5.3), and the full-scale
    family is 28 tests, not 14 regions."""
    obs = apply_qc(
        ingest(SyntheticAdapter(fixture_tree)).observations, QCConfig(), AnalysisConfig()
    )
    results = fit_model(obs, AnalysisConfig())
    assert results.family_size == len(results.fits)
    assert V1_TEST_COUNT == 28
    assert any("28" in note for note in results.notes)


def test_only_analysis_included_observations_reach_the_model(fixture_tree: Path) -> None:
    """QC classifies; the analysis layer decides (§2.4.1)."""
    obs = apply_qc(
        ingest(SyntheticAdapter(fixture_tree)).observations, QCConfig(), AnalysisConfig()
    )
    obs.loc[obs.index[:200], "analysis_included"] = False
    results = fit_model(obs, AnalysisConfig())
    fitted = results.fits[0]
    excluded_region = obs.loc[obs.index[:200]]
    excluded_in_region = excluded_region[excluded_region["region"] == fitted.region]
    if not excluded_in_region.empty:
        assert fitted.n_observations < len(obs[obs["region"] == fitted.region])


@pytest.mark.slow
def test_mixedlm_recovers_injected_dx_by_time_interaction(tmp_path: Path) -> None:
    """Smoke-level recovery of the primary hypothesis.

    v0.1.0 asserts sign and order of magnitude only. Week 5 replaces this with
    a tolerance-bounded recovery test across the full region set, which is
    what the fixture generator's injected truth exists to support.
    """
    cfg = make_fixture_config(
        seed=31337,
        n_sessions=4,
        sites=(SiteSpec(name="single-site", n_subjects=70),),
        effects=EffectSpec(
            age_per_decade=-0.015,
            dx_baseline=-0.03,
            time_per_year=-0.005,
            dx_by_time_per_year=-0.020,
            random_intercept_sd=0.03,
            random_slope_sd=0.002,
            noise_sd=0.010,
        ),
        planted=PlantedSpec(
            qc_high_holes_fraction=0.0,
            qc_bad_etiv_fraction=0.0,
            qc_extreme_change_fraction=0.0,
            missing_acquisition_fraction=0.0,
            missing_derivative_fraction=0.0,
            malformed_file_fraction=0.0,
        ),
    )
    root = tmp_path / "fx"
    write_fixtures(cfg, root)

    obs = apply_qc(ingest(SyntheticAdapter(root)).observations, QCConfig(), AnalysisConfig())
    fit = fit_region(obs, "lh-hippocampus", "volume")

    assert fit.converged, fit.message
    estimate = fit.primary_estimate
    assert estimate is not None

    # Injected: an extra -2.0%/yr of hippocampal atrophy for patients, on a
    # base volume of 4000 mm^3 -> about -80 mm^3/yr.
    from morphline.fixtures.truth import BASE_VALUES

    expected = BASE_VALUES["hippocampus"] * cfg.effects.dx_by_time_per_year

    assert estimate < 0, "patients should lose volume faster than controls"
    assert estimate == pytest.approx(expected, rel=0.5), (
        f"estimated {estimate:.1f} mm^3/yr vs injected {expected:.1f} mm^3/yr"
    )
    assert fit.primary_p is not None and fit.primary_p < 0.05


def test_head_size_adjustment_is_in_the_specification() -> None:
    """Head-size adjustment is mandatory; omitting it is the most common error
    in this literature (§2.5.1)."""
    from morphline.stages.model import FIXED_EFFECTS_FORMULA

    assert "etiv_baseline" in FIXED_EFFECTS_FORMULA


def test_age_at_session_is_not_in_the_model() -> None:
    """Age-at-session and time are collinear by construction (§2.5.1)."""
    from morphline.stages.model import FIXED_EFFECTS_FORMULA

    assert "age_at_session" not in FIXED_EFFECTS_FORMULA
    assert "age_baseline" in FIXED_EFFECTS_FORMULA


def test_time_varying_diagnosis_is_not_in_the_model() -> None:
    """Conditioning on post-baseline diagnosis invites collider bias (§2.5.1)."""
    from morphline.stages.model import FIXED_EFFECTS_FORMULA

    assert "dx_at_session" not in FIXED_EFFECTS_FORMULA
    assert "dx_baseline" in FIXED_EFFECTS_FORMULA


def test_random_slope_on_time_is_specified() -> None:
    from morphline.stages.model import RANDOM_EFFECTS_FORMULA

    assert "time" in RANDOM_EFFECTS_FORMULA
