"""Model stage tests (BUILD_PLAN §2.5).

Two layers, kept separate on purpose. The fast tests cover the machinery that
must be right before recovery testing means anything: the BH-FDR
implementation, the family *definitions* — primary and secondary, corrected
separately (§2.5.3) — convergence reporting, and the sensitivity arm.

:class:`TestSlopeRecovery` is the statistical layer, marked ``slow``. It fits
all 28 regions against injected ground truth and bounds the error, which is
the thing the fixture generator exists to make possible (§3.2): real data has
no known true slope, so the same question cannot be asked of it at all.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_fixture_config
from morphline.adapters import SyntheticAdapter
from morphline.config import (
    AnalysisConfig,
    EffectSpec,
    FixtureConfig,
    PlantedSpec,
    QCConfig,
    SiteSpec,
)
from morphline.fixtures import write_fixtures
from morphline.fixtures.truth import GroundTruth
from morphline.regions import V1_TEST_COUNT, v1_region_set
from morphline.stages.ingest import ingest
from morphline.stages.model import (
    PRIMARY_TERM,
    SECONDARY_TERMS,
    ModelResults,
    RegionFit,
    apply_fdr,
    apply_multiplicity,
    compare_arms,
    family_sizes,
    fit_model,
    fit_region,
    fit_with_sensitivity,
    term_slug,
)
from morphline.stages.qc import apply_qc


def make_fit(
    region: str,
    p: float | None,
    *,
    converged: bool = True,
    secondary_p: dict[str, float] | None = None,
) -> RegionFit:
    p_values = {PRIMARY_TERM: p} if p is not None else {}
    p_values.update(secondary_p or {})
    return RegionFit(
        region=region,
        measure_type="volume",
        n_observations=100,
        n_subjects=25,
        converged=converged,
        p_values=p_values,
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


class TestSecondaryFamilies:
    """§2.5.3: each secondary effect forms its own family, corrected
    separately from the primary family and from the other secondaries."""

    def test_each_family_is_corrected_against_its_own_size(self) -> None:
        """The pooling error this guards against inflates every family to the
        union's size, which would make the primary q four times too large."""
        fits = [
            make_fit(f"r{i}", p, secondary_p={term: p for term in SECONDARY_TERMS})
            for i, p in enumerate([0.01, 0.02, 0.03, 0.04])
        ]
        apply_multiplicity(fits, 0.05)

        # Four tests per family, not sixteen across the union.
        assert fits[0].q_value == pytest.approx(0.01 * 4 / 1)
        for term in SECONDARY_TERMS:
            assert fits[0].q_values[term] == pytest.approx(0.01 * 4 / 1)

    def test_a_secondary_family_does_not_change_the_primary_q(self) -> None:
        primary_only = [make_fit(f"r{i}", p) for i, p in enumerate([0.01, 0.2, 0.5])]
        apply_multiplicity(primary_only, 0.05)
        expected = [f.q_value for f in primary_only]

        with_secondary = [
            make_fit(f"r{i}", p, secondary_p={term: 0.001 for term in SECONDARY_TERMS})
            for i, p in enumerate([0.01, 0.2, 0.5])
        ]
        apply_multiplicity(with_secondary, 0.05)

        assert [f.q_value for f in with_secondary] == pytest.approx(expected)

    def test_families_with_different_p_distributions_get_different_q(self) -> None:
        """A shared q across families would mean the terms were pooled."""
        fits = [
            make_fit(f"r{i}", p, secondary_p={"time": 1.0 - p})
            for i, p in enumerate([0.001, 0.002, 0.003])
        ]
        apply_multiplicity(fits, 0.05)
        assert fits[0].q_value != pytest.approx(fits[0].q_values["time"])

    def test_family_sizes_exclude_non_converged_fits(self) -> None:
        fits = [
            make_fit("r0", 0.01, secondary_p={"time": 0.01}),
            make_fit("r1", None, converged=False),
            make_fit("r2", 0.02, secondary_p={"time": 0.02}),
        ]
        sizes = family_sizes(fits)
        assert sizes[PRIMARY_TERM] == 2
        assert sizes["time"] == 2

    def test_q_values_are_keyed_by_term(self) -> None:
        fits = [make_fit("r0", 0.01, secondary_p={term: 0.02 for term in SECONDARY_TERMS})]
        apply_multiplicity(fits, 0.05)
        assert set(fits[0].q_values) == {PRIMARY_TERM, *SECONDARY_TERMS}
        assert fits[0].q_value == pytest.approx(0.01)


class TestFullRegionSet:
    """The model fits the declared region set, not a truncation of it."""

    def test_every_region_in_the_set_is_attempted(self, fixture_tree: Path) -> None:
        obs = apply_qc(
            ingest(SyntheticAdapter(fixture_tree)).observations, QCConfig(), AnalysisConfig()
        )
        results = fit_model(obs, AnalysisConfig())

        assert len(results.fits) == V1_TEST_COUNT == 28
        assert {f.region for f in results.fits} == {r for r, _hemi, _m in v1_region_set()}

    def test_both_hemispheres_are_fitted_as_separate_tests(self, fixture_tree: Path) -> None:
        """28 tests, not 14 regions — hemispheres count in the family."""
        obs = apply_qc(
            ingest(SyntheticAdapter(fixture_tree)).observations, QCConfig(), AnalysisConfig()
        )
        results = fit_model(obs, AnalysisConfig())
        regions = {f.region for f in results.fits}

        assert sum(1 for r in regions if r.startswith("lh-")) == 14
        assert sum(1 for r in regions if r.startswith("rh-")) == 14

    def test_the_results_frame_carries_p_and_q_for_every_family(self, fixture_tree: Path) -> None:
        """§2.5.3: raw p and q are both reported for every test."""
        obs = apply_qc(
            ingest(SyntheticAdapter(fixture_tree)).observations, QCConfig(), AnalysisConfig()
        )
        frame = fit_model(obs, AnalysisConfig()).to_frame()

        assert {"estimate", "p_value", "q_value"} <= set(frame.columns)
        for term in SECONDARY_TERMS:
            slug = term_slug(term)
            assert {f"estimate_{slug}", f"p_value_{slug}", f"q_value_{slug}"} <= set(frame.columns)


class TestSensitivityArm:
    """Harmonized vs unharmonized is a sensitivity analysis (§2.3.1)."""

    def make_results(self, estimates: dict[str, float], q: float = 0.01) -> ModelResults:
        fits = []
        for region, estimate in estimates.items():
            fit = make_fit(region, 0.01)
            fit.coefficients[PRIMARY_TERM] = estimate
            fit.q_values[PRIMARY_TERM] = q
            fits.append(fit)
        return ModelResults(fits=fits, family_size=len(fits))

    def test_sign_flips_are_counted(self) -> None:
        harmonized = self.make_results({"lh-hippocampus": -20.0, "rh-hippocampus": -15.0})
        unharmonized = self.make_results({"lh-hippocampus": 30.0, "rh-hippocampus": -14.0})

        comparison = compare_arms(harmonized, unharmonized, 0.05)

        assert comparison.n_sign_flips == 1
        flipped = next(r for r in comparison.rows if r.sign_flipped)
        assert flipped.region == "lh-hippocampus"
        assert flipped.difference == pytest.approx(-50.0)

    def test_a_zero_estimate_is_not_a_sign_flip(self) -> None:
        """Zero has no direction to disagree about."""
        harmonized = self.make_results({"lh-hippocampus": 0.0})
        unharmonized = self.make_results({"lh-hippocampus": -20.0})
        assert compare_arms(harmonized, unharmonized, 0.05).n_sign_flips == 0

    def test_significance_changes_are_counted(self) -> None:
        harmonized = self.make_results({"lh-hippocampus": -20.0}, q=0.01)
        unharmonized = self.make_results({"lh-hippocampus": -19.0}, q=0.40)

        comparison = compare_arms(harmonized, unharmonized, 0.05)

        assert comparison.n_significance_changes == 1
        assert "consequence of" in " ".join(comparison.notes)

    def test_a_non_converged_arm_is_not_comparable(self) -> None:
        harmonized = self.make_results({"lh-hippocampus": -20.0})
        unharmonized = ModelResults(fits=[make_fit("lh-hippocampus", None, converged=False)])

        comparison = compare_arms(harmonized, unharmonized, 0.05)

        assert comparison.n_comparable == 0
        assert not comparison.rows[0].comparable
        assert "is not agreement" in " ".join(comparison.notes)

    def test_identical_arms_are_reported_as_not_applicable(self, fixture_tree: Path) -> None:
        """Two identical arms are one run reported twice, and saying so is the
        whole point — a table of zero differences reads as reassurance."""
        obs = apply_qc(
            ingest(SyntheticAdapter(fixture_tree)).observations, QCConfig(), AnalysisConfig()
        )
        results = fit_with_sensitivity(obs, obs.copy(), AnalysisConfig())

        assert results.sensitivity is not None
        assert not results.sensitivity.applicable
        assert "same run" in " ".join(results.sensitivity.notes)

    def test_differing_arms_are_applicable(self, fixture_tree: Path) -> None:
        obs = apply_qc(
            ingest(SyntheticAdapter(fixture_tree)).observations, QCConfig(), AnalysisConfig()
        )
        shifted = obs.copy()
        shifted["value"] = shifted["value"] * 1.05

        results = fit_with_sensitivity(obs, shifted, AnalysisConfig())

        assert results.sensitivity is not None
        assert results.sensitivity.applicable

    def test_the_frame_carries_the_second_arm_even_when_it_was_not_run(
        self, fixture_tree: Path
    ) -> None:
        """The parquet schema must not depend on how the stage was invoked."""
        obs = apply_qc(
            ingest(SyntheticAdapter(fixture_tree)).observations, QCConfig(), AnalysisConfig()
        )
        frame = fit_model(obs, AnalysisConfig()).to_frame()

        assert "estimate_unharmonized" in frame.columns
        assert frame["estimate_unharmonized"].isna().all()


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


#: Every estimate must sit within this many standard errors of the injected
#: truth. This is the criterion that actually means "recovered": it is scale
#: free, and it is the only one that stays meaningful when the effect size or
#: the sample size changes. 1.96 is the nominal 95% interval.
RECOVERY_Z = 1.96

#: Regions whose 95% interval must contain the truth. Two of 28 are allowed to
#: miss, because a 95% interval that never missed would mean the standard
#: errors were overstated.
MIN_COVERED_REGIONS = 26

#: Secondary, scale-free bounds on the error itself. A z-score check alone
#: would pass an estimator that was wildly wrong but honest about its own
#: uncertainty, so the size of the error is bounded too.
MAX_RELATIVE_ERROR = 0.20
MAX_MEDIAN_RELATIVE_ERROR = 0.075

#: Bound on the *mean signed* relative error across regions. Per-region bounds
#: cannot see a small bias that pushes every region the same way — a units slip
#: or a scaling error — because it hides inside each region's own tolerance.
MAX_SYSTEMATIC_BIAS = 0.05


def recovery_fixture_config() -> FixtureConfig:
    """Config for the slope-recovery suite.

    ``multiplicative_effect`` is pinned at exactly 1.0 at both sites, and that
    is load bearing rather than incidental. The generative equation is
    ``observed = biological * mult_site + add_site + eps``, so a multiplicative
    site effect *scales the slope itself* and the injected interaction would
    differ per site — making the pooled estimate a weighted average across
    sites with no single right answer to compare against. An additive site
    effect shifts intercepts only and leaves every slope untouched, so the
    sites still differ, and the target stays exact.
    """
    return make_fixture_config(
        seed=90210,
        n_sessions=4,
        sites=(
            SiteSpec(name="site-a", n_subjects=45, additive_effect=0.02, multiplicative_effect=1.0),
            SiteSpec(
                name="site-b", n_subjects=45, additive_effect=-0.03, multiplicative_effect=1.0
            ),
        ),
        effects=EffectSpec(
            age_per_decade=-0.015,
            dx_baseline=-0.030,
            time_per_year=-0.005,
            dx_by_time_per_year=-0.020,
            random_intercept_sd=0.030,
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


def injected_interaction(truth: GroundTruth, region: str) -> float:
    """Return the differential slope actually injected into one region.

    Measured off the recorded ``true_biological_value`` column by ordinary
    least squares on the same fixed-effect design, rather than recomputed from
    the generator's coefficients. Re-deriving ``base * direction * b_dxtime``
    here would make the test agree with the generator by construction — it
    would keep passing if the generator encoded ventricular expansion
    backwards, since the expectation would be wrong in exactly the same way.
    Reading it off the injected values instead means the fixtures and the model
    are checked against each other.

    Args:
        truth: The ground truth returned by the fixture writer.
        region: Canonical region name.

    Returns:
        The injected ``time × diagnosis`` slope in the region's native units.
    """
    frame = truth.observations[truth.observations["region"] == region].merge(
        truth.subjects[["subject_id", "dx_baseline"]], on="subject_id"
    )
    dx = (frame["dx_baseline"] == "patient").to_numpy(dtype=float)
    t = frame["time_from_baseline_years"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(t), t, dx, t * dx])
    beta, *_ = np.linalg.lstsq(
        design, frame["true_biological_value"].to_numpy(dtype=float), rcond=None
    )
    return float(beta[3])


@pytest.fixture(scope="module")
def recovery(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModelResults, dict[str, float]]:
    """Fit all 28 regions once and pair each with its injected truth."""
    root = tmp_path_factory.mktemp("recovery") / "fx"
    truth = write_fixtures(recovery_fixture_config(), root)
    observations = apply_qc(
        ingest(SyntheticAdapter(root)).observations, QCConfig(), AnalysisConfig()
    )
    results = fit_model(observations, AnalysisConfig())
    targets = {fit.region: injected_interaction(truth, fit.region) for fit in results.fits}
    return results, targets


def relative_errors(results: ModelResults, targets: dict[str, float]) -> dict[str, float]:
    """Return signed relative error against injected truth, per region."""
    return {
        fit.region: (fit.primary_estimate - targets[fit.region]) / abs(targets[fit.region])
        for fit in results.fits
        if fit.primary_estimate is not None
    }


@pytest.mark.slow
class TestSlopeRecovery:
    """Tolerance-bounded recovery of the primary hypothesis across all 28 tests.

    This is what the fixture generator exists for (§3.2): real data has no
    known true slope, so "did the model recover the effect?" is unanswerable
    against it. Here every region carries an injected differential slope and
    the model is asked to find all 28.
    """

    def test_every_region_converges(self, recovery: tuple[ModelResults, dict[str, float]]) -> None:
        results, _ = recovery
        assert results.n_estimable == V1_TEST_COUNT
        failed = [(f.region, f.message) for f in results.fits if not f.converged]
        assert not failed, f"regions failed to converge: {failed}"
        assert not [f.region for f in results.fits if f.random_slope_dropped]

    def test_the_fixtures_inject_an_effect_worth_recovering(
        self, recovery: tuple[ModelResults, dict[str, float]]
    ) -> None:
        """Otherwise the suite proves nothing.

        A fixture config that injected no differential slope would be recovered
        perfectly by an estimator that always returned zero, and every
        tolerance below would pass. The injected effect must be large enough
        relative to the estimator's own uncertainty for its recovery to be
        evidence of anything.
        """
        results, targets = recovery
        assert len(targets) == V1_TEST_COUNT
        for fit in results.fits:
            error = fit.std_errors.get(PRIMARY_TERM)
            assert error is not None and error > 0
            assert abs(targets[fit.region]) > 3 * error, (
                f"injected effect in {fit.region} is within noise of zero"
            )

    def test_the_direction_of_every_effect_is_recovered(
        self, recovery: tuple[ModelResults, dict[str, float]]
    ) -> None:
        """Including the ventricles, whose injected effect points the other way.

        Ventricles expand as surrounding tissue is lost. A pipeline that
        recovered magnitude everywhere but flipped these would be recovering a
        sign convention rather than biology, and every aggregate check would
        still pass.
        """
        results, targets = recovery
        wrong = [
            fit.region
            for fit in results.fits
            if fit.primary_estimate is not None and fit.primary_estimate * targets[fit.region] <= 0
        ]
        assert not wrong, f"estimated effect points the wrong way in: {wrong}"

        expanding = [r for r in targets if "ventricle" in r]
        assert len(expanding) == 4
        assert all(targets[r] > 0 for r in expanding), "ventricles must expand, not shrink"

    def test_every_estimate_is_within_tolerance_of_the_injected_truth(
        self, recovery: tuple[ModelResults, dict[str, float]]
    ) -> None:
        results, targets = recovery
        errors = relative_errors(results, targets)
        assert len(errors) == V1_TEST_COUNT

        worst = max(errors.items(), key=lambda kv: abs(kv[1]))
        assert abs(worst[1]) <= MAX_RELATIVE_ERROR, (
            f"{worst[0]} missed the injected slope by {worst[1]:.1%}"
        )

        median = float(np.median([abs(e) for e in errors.values()]))
        assert median <= MAX_MEDIAN_RELATIVE_ERROR, f"median relative error {median:.1%}"

    def test_confidence_intervals_cover_the_injected_truth(
        self, recovery: tuple[ModelResults, dict[str, float]]
    ) -> None:
        """The criterion that means "recovered" independently of effect size."""
        results, targets = recovery
        covered = 0
        for fit in results.fits:
            estimate = fit.primary_estimate
            error = fit.std_errors.get(PRIMARY_TERM)
            if estimate is None or error is None:
                continue
            target = targets[fit.region]
            if abs(estimate - target) <= RECOVERY_Z * error:
                covered += 1
        assert covered >= MIN_COVERED_REGIONS, (
            f"only {covered} of {V1_TEST_COUNT} intervals contained the injected slope"
        )

    def test_there_is_no_systematic_bias_across_regions(
        self, recovery: tuple[ModelResults, dict[str, float]]
    ) -> None:
        """A shared scaling error hides inside every per-region tolerance."""
        results, targets = recovery
        bias = float(np.mean(list(relative_errors(results, targets).values())))
        assert abs(bias) <= MAX_SYSTEMATIC_BIAS, f"mean signed relative error {bias:+.1%}"

    def test_the_injected_effect_survives_multiplicity_correction(
        self, recovery: tuple[ModelResults, dict[str, float]]
    ) -> None:
        """Every region carries a real injected effect, so BH-FDR across the
        declared 28-test family must not correct it away."""
        results, _ = recovery
        assert results.family_sizes[PRIMARY_TERM] == V1_TEST_COUNT
        missed = [
            fit.region
            for fit in results.fits
            if fit.q_value is None or fit.q_value > results.fdr_alpha
        ]
        assert not missed, f"injected effect not detected after FDR in: {missed}"


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
