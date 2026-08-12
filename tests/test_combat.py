"""Unit tests for the in-repo ComBat estimator.

These test the *estimator*, over hand-built arrays with injected batch effects,
and they run in milliseconds. They are deliberately not the §2.3.2 validation
suite: that one asks whether harmonization recovers truth injected by the
fixture generator and whether biology survives it, over a real fixture tree,
and it lives in ``test_harmonization_validation.py``.

The distinction matters because the cheap check here — "do site means converge
after adjustment" — is *circular* as a validation criterion (§2.3.2): equalizing
site means is what ComBat does by construction. It is a fine unit test of
whether the transform does what it claims, and worthless as evidence that
harmonization worked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphline.combat import (
    MIN_BATCHES,
    ComBatSkipCode,
    CovariateDropCode,
    run_combat,
)

BASE = 4000.0
NOISE_SD = 40.0


def _frame(
    *,
    shifts: dict[str, float],
    scales: dict[str, float] | None = None,
    n_per_batch: int = 150,
    base: float = BASE,
    noise_sd: float = NOISE_SD,
    age_slope: float = 0.0,
    seed: int = 7,
    region: str = "lh-hippocampus",
    measure_type: str = "volume",
) -> pd.DataFrame:
    """Build one region's observations with known additive batch shifts.

    Args:
        shifts: Additive offset injected per batch, in native units.
        scales: Multiplier on the noise SD per batch. Defaults to 1.0.
        n_per_batch: Observations per batch.
        base: Region base value before any effect.
        noise_sd: Measurement noise standard deviation.
        age_slope: Native-unit change per year of age, a biological covariate.
        seed: RNG seed.
        region: Canonical region name.
        measure_type: Measure the rows carry.

    Returns:
        A canonical-shaped frame carrying ``value``, ``site``, ``age_baseline``.
    """
    rng = np.random.default_rng(seed)
    scales = scales or {}
    rows: list[dict[str, object]] = []
    subject = 0
    for batch, shift in shifts.items():
        scale = scales.get(batch, 1.0)
        for _ in range(n_per_batch):
            age = float(rng.normal(70.0, 8.0))
            noise = float(rng.normal(0.0, noise_sd * scale))
            rows.append(
                {
                    "subject_id": f"sub-{subject:04d}",
                    "site": batch,
                    "region": region,
                    "measure_type": measure_type,
                    "age_baseline": age,
                    "value": base + age_slope * (age - 70.0) + shift + noise,
                }
            )
            subject += 1
    return pd.DataFrame(rows)


def _batch_means(frame: pd.DataFrame, values: pd.Series) -> dict[str, float]:
    """Mean adjusted value per batch."""
    return {str(k): float(v) for k, v in values.groupby(frame["site"]).mean().items()}


class TestBatchRecovery:
    """Does the estimator recover an additive shift it was not told about?"""

    def test_gamma_recovers_injected_shift_without_shrinkage(self) -> None:
        """Unshrunk, gamma is a batch mean and must land on the injected offset.

        ``empirical_bayes=False`` is the honest setting for an exact-recovery
        test: shrinkage deliberately biases each batch toward the prior, so a
        shrunk estimate *should* miss the injected value slightly.
        """
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        result = run_combat(frame, empirical_bayes=False)

        params = result.fit.parameters[("lh-hippocampus", "volume")]
        native = params.native_gamma()

        assert native["site-a"] == pytest.approx(-120.0, abs=12.0)
        assert native["site-b"] == pytest.approx(120.0, abs=12.0)

    def test_grand_mean_and_pooled_sd_are_sane(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        params = run_combat(frame).fit.parameters[("lh-hippocampus", "volume")]

        assert params.grand_mean == pytest.approx(BASE, abs=15.0)
        assert params.pooled_sd == pytest.approx(NOISE_SD, rel=0.15)

    def test_adjustment_removes_the_batch_difference(self) -> None:
        """A unit test of the transform, *not* a validation criterion (§2.3.2).

        Equalizing site means is what ComBat does by construction, so this
        confirms the code runs the method it claims to. It is not evidence that
        harmonization worked.
        """
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        result = run_combat(frame)

        before = _batch_means(frame, frame["value"])
        after = _batch_means(frame, result.values)

        assert abs(before["site-a"] - before["site-b"]) > 200.0
        assert abs(after["site-a"] - after["site-b"]) < 15.0

    def test_three_batches_are_all_recovered(self) -> None:
        shifts = {"site-a": -150.0, "site-b": 0.0, "site-c": 200.0}
        frame = _frame(shifts=shifts)
        params = run_combat(frame, empirical_bayes=False).fit.parameters[
            ("lh-hippocampus", "volume")
        ]

        centre = float(np.mean(list(shifts.values())))
        for batch, injected in shifts.items():
            assert params.native_gamma()[batch] == pytest.approx(injected - centre, abs=15.0)

    def test_gamma_is_centered_by_the_size_weighted_mean(self) -> None:
        """gamma is identified only up to a centering constraint, and the
        constraint is size-weighted.

        The grand mean is the n-weighted mean of the batch intercepts, so the
        batch terms satisfy ``sum_b (n_b/n) * gamma_b == 0`` — not
        ``mean(gamma) == 0``. With equal batch sizes the two agree and the
        distinction is invisible; with unequal ones they differ by a lot. The
        §2.3.2 recovery suite compares gamma against a truth quantity it must
        centre the same way, so pin the convention here rather than discovering
        it as an unexplained bias there.
        """
        frame = pd.concat(
            [
                _frame(shifts={"site-a": -150.0}, n_per_batch=300, seed=1),
                _frame(shifts={"site-b": 50.0}, n_per_batch=100, seed=2),
                _frame(shifts={"site-c": 200.0}, n_per_batch=50, seed=3),
            ],
            ignore_index=True,
        )
        injected = {"site-a": -150.0, "site-b": 50.0, "site-c": 200.0}

        params = run_combat(frame, empirical_bayes=False).fit.parameters[
            ("lh-hippocampus", "volume")
        ]
        gamma = params.native_gamma()
        counts = params.n_per_batch
        total = sum(counts.values())

        weighted = sum(gamma[b] * counts[b] / total for b in gamma)
        assert weighted == pytest.approx(0.0, abs=1e-6)

        unweighted = float(np.mean(list(gamma.values())))
        assert abs(unweighted) > 50.0

        offset = sum(injected[b] * counts[b] / total for b in counts)
        for batch, shift in injected.items():
            assert gamma[batch] == pytest.approx(shift - offset, abs=15.0)


class TestScaleParameter:
    """delta is a residual-scale term, and must behave like one."""

    def test_delta_is_near_one_when_noise_is_homoscedastic(self) -> None:
        """Equal noise across batches means no scale correction is warranted.

        This holds tightly here because these rows carry nothing but a batch
        shift and homoscedastic noise. It does **not** transfer unchanged to
        the fixture generator: there the residual after covariate adjustment is
        ``mult[s,r] * base * (u0 + u1*t) + eps``, so the subject random effects
        are scaled by the site's multiplicative term while ``eps`` is not. With
        ``random_intercept_sd`` well above ``noise_sd`` the random effects
        dominate the residual variance, and delta lands *near* 1.0 rather than
        at it. The §2.3.2 suite must therefore assert a band derived from those
        two SDs, not equality with 1.0.

        Note also that the generator has no per-site noise knob, so delta has
        no injected truth to recover — only a bound to stay inside.
        """
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        params = run_combat(frame).fit.parameters[("lh-hippocampus", "volume")]

        for value in params.delta_star.values():
            assert value == pytest.approx(1.0, abs=0.15)

    def test_delta_tracks_an_injected_variance_difference(self) -> None:
        frame = _frame(
            shifts={"site-a": 0.0, "site-b": 0.0},
            scales={"site-a": 1.0, "site-b": 2.5},
            n_per_batch=400,
        )
        params = run_combat(frame).fit.parameters[("lh-hippocampus", "volume")]

        ratio = params.delta_star["site-b"] / params.delta_star["site-a"]
        assert ratio == pytest.approx(2.5, rel=0.2)

    def test_adjustment_equalizes_residual_spread(self) -> None:
        frame = _frame(
            shifts={"site-a": 0.0, "site-b": 0.0},
            scales={"site-a": 1.0, "site-b": 2.5},
            n_per_batch=400,
        )
        result = run_combat(frame)
        spread = result.values.groupby(frame["site"]).std()

        assert float(spread["site-b"]) / float(spread["site-a"]) == pytest.approx(1.0, abs=0.2)


class TestCovariatePreservation:
    """§2.3.4 — biology in the design matrix, not in the batch term."""

    def test_injected_age_effect_survives_harmonization(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0}, age_slope=-25.0)
        result = run_combat(frame, covariates=("age_baseline",))

        slope_after = float(
            np.polyfit(frame["age_baseline"].to_numpy(float), result.values.to_numpy(float), 1)[0]
        )
        assert slope_after == pytest.approx(-25.0, rel=0.15)

    def test_covariate_is_recorded_as_used(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0}, age_slope=-25.0)
        fit = run_combat(frame, covariates=("age_baseline",)).fit

        assert fit.covariates_used == ("age_baseline",)
        assert fit.covariates_dropped == {}

    def test_categorical_covariate_is_dummy_coded(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        rng = np.random.default_rng(3)
        frame["dx_baseline"] = rng.choice(["control", "patient"], size=len(frame))
        frame.loc[frame["dx_baseline"] == "patient", "value"] -= 300.0

        result = run_combat(frame, covariates=("dx_baseline",))
        params = result.fit.parameters[("lh-hippocampus", "volume")]

        assert params.covariate_terms == ("dx_baseline[T.patient]",)
        assert params.covariate_coefficients["dx_baseline[T.patient]"] == pytest.approx(
            -300.0, abs=20.0
        )

    def test_group_difference_is_not_absorbed_as_a_batch_effect(self) -> None:
        """The failure this test exists for is silent: a harmonizer that omits
        the covariate eats the diagnosis effect and still passes a site-mean
        check."""
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        rng = np.random.default_rng(11)
        frame["dx_baseline"] = rng.choice(["control", "patient"], size=len(frame))
        frame.loc[frame["dx_baseline"] == "patient", "value"] -= 300.0

        result = run_combat(frame, covariates=("dx_baseline",))
        adjusted = result.values.groupby(frame["dx_baseline"]).mean()

        difference = float(adjusted["patient"]) - float(adjusted["control"])
        assert difference == pytest.approx(-300.0, abs=30.0)


class TestCovariateDrops:
    """A covariate that cannot enter the design is reported, never ignored."""

    def test_absent_covariate_is_reported(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        fit = run_combat(frame, covariates=("not_a_column",)).fit

        assert fit.covariates_dropped["not_a_column"] == str(CovariateDropCode.ABSENT)

    def test_all_null_covariate_is_reported(self) -> None:
        """``dx_baseline`` is all-null on healthy-control-only datasets."""
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        frame["dx_baseline"] = None
        fit = run_combat(frame, covariates=("dx_baseline",)).fit

        assert fit.covariates_dropped["dx_baseline"] == str(CovariateDropCode.ALL_NULL)

    def test_constant_covariate_is_reported(self) -> None:
        """``time_from_baseline_years`` is constant on every cross-sectional
        dataset, ABIDE included."""
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        frame["time_from_baseline_years"] = 0.0
        fit = run_combat(frame, covariates=("time_from_baseline_years",)).fit

        assert fit.covariates_dropped["time_from_baseline_years"] == str(CovariateDropCode.CONSTANT)

    def test_covariate_collinear_with_batch_is_dropped_not_fatal(self) -> None:
        """A site that recruited one diagnosis only makes the design singular.

        The batch term is kept and the covariate goes, because a batch effect
        the method cannot estimate is worse than a covariate it cannot preserve.
        """
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        frame["dx_baseline"] = np.where(frame["site"] == "site-a", "control", "patient")

        fit = run_combat(frame, covariates=("dx_baseline",)).fit

        assert fit.covariates_dropped["dx_baseline"] == str(CovariateDropCode.COLLINEAR_WITH_BATCH)
        assert fit.n_groups_harmonized == 1

    def test_a_dropped_covariate_does_not_stop_the_others(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0}, age_slope=-25.0)
        frame["dx_baseline"] = None

        fit = run_combat(frame, covariates=("age_baseline", "dx_baseline")).fit

        assert fit.covariates_used == ("age_baseline",)
        assert "dx_baseline" in fit.covariates_dropped


class TestEstimationMask:
    """Decision: estimate on trusted rows, apply to every row."""

    def test_masked_outliers_do_not_move_the_batch_term(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        corrupted = frame.copy()
        bad = corrupted.index[corrupted["site"] == "site-a"][:20]
        corrupted.loc[bad, "value"] = 40.0

        mask = pd.Series(True, index=corrupted.index)
        mask.loc[bad] = False

        masked = run_combat(corrupted, estimation_mask=mask, empirical_bayes=False)
        unmasked = run_combat(corrupted, empirical_bayes=False)

        masked_gamma = masked.fit.parameters[("lh-hippocampus", "volume")].native_gamma()
        unmasked_gamma = unmasked.fit.parameters[("lh-hippocampus", "volume")].native_gamma()

        assert masked_gamma["site-a"] == pytest.approx(-120.0, abs=15.0)
        assert abs(unmasked_gamma["site-a"] + 120.0) > 100.0

    def test_masked_rows_are_still_adjusted(self) -> None:
        """Excluding a row from *estimation* is not excluding it from output."""
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        mask = pd.Series(True, index=frame.index)
        bad = frame.index[frame["site"] == "site-a"][:20]
        mask.loc[bad] = False

        result = run_combat(frame, estimation_mask=mask)
        params = result.fit.parameters[("lh-hippocampus", "volume")]

        assert params.n_adjusted == len(frame)
        assert not np.allclose(result.values.loc[bad], frame.loc[bad, "value"])

    def test_counts_reflect_the_estimation_set_not_the_frame(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0}, n_per_batch=100)
        mask = pd.Series(True, index=frame.index)
        mask.loc[frame.index[frame["site"] == "site-a"][:30]] = False

        params = run_combat(frame, estimation_mask=mask).fit.parameters[
            ("lh-hippocampus", "volume")
        ]

        assert params.n_per_batch == {"site-a": 70, "site-b": 100}


PROBE = ("region-00", "volume")


def _multi_region_frame(
    *,
    shifts: dict[str, float],
    tiny: dict[str, float] | None = None,
    n_per_batch: int = 120,
    n_tiny: int = 3,
    n_regions: int = 10,
    seed: int = 11,
) -> pd.DataFrame:
    """Build several regions sharing a batch structure.

    Shrinkage pools across regions within a batch, so a prior only exists once
    a batch appears in more than one region. Every test that exercises the
    empirical-Bayes step therefore needs a multi-region frame; a single region
    is the degenerate case where there is nothing to borrow from.

    Args:
        shifts: Additive offset per batch, applied in every region.
        tiny: An additional small batch, if one is wanted.
        n_per_batch: Observations per ordinary batch per region.
        n_tiny: Observations for the small batch per region.
        n_regions: How many regions to emit.
        seed: RNG seed; each region draws from a different stream so the
            per-region variances differ and the prior is non-degenerate.

    Returns:
        A canonical-shaped frame spanning ``n_regions`` regions.
    """
    parts: list[pd.DataFrame] = []
    for index in range(n_regions):
        region = f"region-{index:02d}"
        parts.append(
            _frame(shifts=shifts, n_per_batch=n_per_batch, region=region, seed=seed + 2 * index)
        )
        if tiny:
            parts.append(
                _frame(shifts=tiny, n_per_batch=n_tiny, region=region, seed=seed + 2 * index + 1)
            )
    return pd.concat(parts, ignore_index=True)


class TestShrinkage:
    """Empirical Bayes is what makes a small batch survivable.

    The prior for a batch is estimated across the regions that batch appears
    in, so these all use multi-region frames. That is the pooling direction
    Johnson et al. specify, and it is why a one-region frame gets no shrinkage
    at all.
    """

    def test_small_batch_is_pulled_toward_the_prior(self) -> None:
        frame = _multi_region_frame(
            shifts={"site-a": -100.0, "site-b": 100.0}, tiny={"site-tiny": 900.0}
        )

        shrunk = run_combat(frame).fit.parameters[PROBE]
        raw = run_combat(frame, empirical_bayes=False).fit.parameters[PROBE]

        assert abs(shrunk.native_gamma()["site-tiny"]) < abs(raw.native_gamma()["site-tiny"])
        assert shrunk.shrinkage_applied
        assert not raw.shrinkage_applied

    def test_large_batches_are_barely_moved(self) -> None:
        frame = _multi_region_frame(
            shifts={"site-a": -100.0, "site-b": 100.0}, tiny={"site-tiny": 900.0}
        )

        shrunk = run_combat(frame).fit.parameters[PROBE]
        raw = run_combat(frame, empirical_bayes=False).fit.parameters[PROBE]

        for batch in ("site-a", "site-b"):
            moved = abs(shrunk.native_gamma()[batch] - raw.native_gamma()[batch])
            assert moved < 15.0

    def test_a_single_region_gets_no_shrinkage(self) -> None:
        """One value has no variance, so there is no prior to shrink toward."""
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        params = run_combat(frame).fit.parameters[("lh-hippocampus", "volume")]

        assert not params.shrinkage_applied
        assert params.n_iterations == 0

    def test_convergence_is_reported(self) -> None:
        frame = _multi_region_frame(shifts={"site-a": -120.0, "site-b": 120.0})
        params = run_combat(frame).fit.parameters[PROBE]

        assert params.converged
        assert params.n_iterations >= 1

    def test_iteration_cap_reports_non_convergence_rather_than_hanging(self) -> None:
        frame = _multi_region_frame(shifts={"site-a": -120.0, "site-b": 120.0})
        params = run_combat(frame, max_iterations=1, tolerance=1e-12).fit.parameters[PROBE]

        assert not params.converged
        assert params.n_iterations == 1

    def test_prior_pools_across_regions_not_across_batches(self) -> None:
        """The axis the implementation used to have backwards.

        Adding regions must change a batch's shrunk estimate, because they are
        what the prior is estimated from. If the prior were formed across
        batches within a region instead, the estimate would be identical no
        matter how many regions were present.
        """
        shifts = {"site-a": -100.0, "site-b": 100.0}
        tiny = {"site-tiny": 900.0}

        few = run_combat(_multi_region_frame(shifts=shifts, tiny=tiny, n_regions=2))
        many = run_combat(_multi_region_frame(shifts=shifts, tiny=tiny, n_regions=12))

        assert few.fit.parameters[PROBE].gamma_star["site-tiny"] != pytest.approx(
            many.fit.parameters[PROBE].gamma_star["site-tiny"], rel=1e-6
        )


class TestPerRegionEstimation:
    """The batch effect is per site per region, so the fit must be too."""

    def test_regions_are_estimated_independently(self) -> None:
        hippocampus = _frame(shifts={"site-a": -120.0, "site-b": 120.0}, seed=1)
        thickness = _frame(
            shifts={"site-a": 0.10, "site-b": -0.10},
            base=2.5,
            noise_sd=0.05,
            region="lh-entorhinal",
            measure_type="thickness",
            seed=2,
        )
        frame = pd.concat([hippocampus, thickness], ignore_index=True)

        fit = run_combat(frame, empirical_bayes=False).fit

        volume = fit.parameters[("lh-hippocampus", "volume")].native_gamma()
        thick = fit.parameters[("lh-entorhinal", "thickness")].native_gamma()

        assert volume["site-a"] == pytest.approx(-120.0, abs=15.0)
        assert thick["site-a"] == pytest.approx(0.10, abs=0.02)

    def test_units_never_mix_across_regions(self) -> None:
        """mm and mm³ in one pooled fit would be dimensionally incoherent."""
        hippocampus = _frame(shifts={"site-a": -120.0, "site-b": 120.0}, seed=1)
        thickness = _frame(
            shifts={"site-a": 0.10, "site-b": -0.10},
            base=2.5,
            noise_sd=0.05,
            region="lh-entorhinal",
            measure_type="thickness",
            seed=2,
        )
        frame = pd.concat([hippocampus, thickness], ignore_index=True)

        fit = run_combat(frame).fit

        assert fit.parameters[("lh-hippocampus", "volume")].pooled_sd > 1.0
        assert fit.parameters[("lh-entorhinal", "thickness")].pooled_sd < 1.0

    def test_one_unestimable_region_does_not_block_the_others(self) -> None:
        good = _frame(shifts={"site-a": -120.0, "site-b": 120.0}, seed=1)
        lonely = _frame(shifts={"site-a": 0.0}, region="lh-amygdala", measure_type="volume", seed=2)
        frame = pd.concat([good, lonely], ignore_index=True)

        fit = run_combat(frame).fit

        assert fit.n_groups_harmonized == 1
        assert fit.skipped[("lh-amygdala", "volume")] == str(ComBatSkipCode.SINGLE_BATCH)


class TestDegenerateInput:
    """Nothing here raises. Failures are reason codes, as in the parser."""

    def test_single_batch_is_skipped_and_left_alone(self) -> None:
        frame = _frame(shifts={"site-a": 0.0})
        result = run_combat(frame)

        assert result.fit.skipped[("lh-hippocampus", "volume")] == str(ComBatSkipCode.SINGLE_BATCH)
        pd.testing.assert_series_equal(
            result.values, frame["value"].astype("float64"), check_names=False
        )

    def test_zero_variance_is_skipped(self) -> None:
        frame = _frame(shifts={"site-a": 0.0, "site-b": 0.0}, noise_sd=0.0)
        result = run_combat(frame)

        assert result.fit.skipped[("lh-hippocampus", "volume")] == str(ComBatSkipCode.ZERO_VARIANCE)

    def test_batch_below_minimum_rows_is_not_estimated(self) -> None:
        frame = pd.concat(
            [
                _frame(shifts={"site-a": -100.0, "site-b": 100.0}, n_per_batch=100, seed=5),
                _frame(shifts={"site-solo": 500.0}, n_per_batch=1, seed=6),
            ],
            ignore_index=True,
        )

        result = run_combat(frame)
        params = result.fit.parameters[("lh-hippocampus", "volume")]

        assert "site-solo" not in params.n_per_batch
        assert params.n_unadjusted == 1

        solo = frame.index[frame["site"] == "site-solo"][0]
        assert float(result.values.loc[solo]) == pytest.approx(float(frame.loc[solo, "value"]))

    def test_rows_with_null_values_pass_through(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0})
        frame.loc[frame.index[:5], "value"] = np.nan

        result = run_combat(frame)

        assert result.values.iloc[:5].isna().all()
        assert result.fit.n_groups_harmonized == 1

    def test_rows_with_incomplete_covariates_pass_through_unadjusted(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0}, age_slope=-25.0)
        incomplete = frame.index[:5]
        frame.loc[incomplete, "age_baseline"] = np.nan

        result = run_combat(frame, covariates=("age_baseline",))

        assert np.allclose(result.values.loc[incomplete], frame.loc[incomplete, "value"])
        assert result.fit.parameters[("lh-hippocampus", "volume")].n_unadjusted == 5

    def test_empty_frame_returns_empty(self) -> None:
        result = run_combat(pd.DataFrame())

        assert result.values.empty
        assert result.fit.n_groups_harmonized == 0

    def test_missing_required_column_is_not_fatal(self) -> None:
        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0}).drop(columns=["region"])
        result = run_combat(frame)

        assert result.fit.n_groups_harmonized == 0
        pd.testing.assert_series_equal(
            result.values, frame["value"].astype("float64"), check_names=False
        )


class TestSerialization:
    """The fit has to reach the report and provenance as JSON."""

    def test_fit_round_trips_through_json(self) -> None:
        import json

        frame = _frame(shifts={"site-a": -120.0, "site-b": 120.0}, age_slope=-25.0)
        fit = run_combat(frame, covariates=("age_baseline",)).fit

        payload = json.loads(json.dumps(fit.as_dict()))

        assert payload["n_groups_harmonized"] == 1
        assert payload["covariates_used"] == ["age_baseline"]
        assert payload["parameters"][0]["region"] == "lh-hippocampus"
        assert set(payload["parameters"][0]["gamma_native"]) == {"site-a", "site-b"}

    def test_skip_keys_survive_json(self) -> None:
        import json

        frame = _frame(shifts={"site-a": 0.0})
        fit = run_combat(frame).fit

        payload = json.loads(json.dumps(fit.as_dict()))
        assert payload["skipped"] == {"lh-hippocampus:volume": str(ComBatSkipCode.SINGLE_BATCH)}

    def test_minimum_batches_is_two(self) -> None:
        assert MIN_BATCHES == 2
