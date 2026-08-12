"""§2.3.2 harmonization validation against injected ground truth.

Checking that post-harmonization site means are equal is **circular**:
equalizing site means is what ComBat does by construction, so it tests that the
estimator ran. These tests ask four separate questions instead, against truth
the fixture generator injected and the estimator was never told:

1. **Batch-effect recovery** — is the estimated batch term the injected one?
2. **Biological preservation** — do the injected effects survive?
3. **Site-association reduction** — does covariate-adjusted site association fall?
4. **No attenuation** — does the longitudinal slope survive?

Criteria 2 and 4 are the ones that catch real failures. A harmonizer that eats
your signal passes criterion 3 with flying colours.

Substrate is ``config/recovery.yaml`` (Regime A, site independent of time) and
``config/confounded.yaml`` (Regime B, site confounded with time) — the two
committed configs that existed for this suite and were, until now, exercised by
no test at all. Both trees are built once per module: 480 and 320 sessions is
not something to pay for per test.

Every tolerance is a ``HarmonizationConfig.target_*`` field rather than a
literal, on the precedent ``QCConfig.target_recall`` sets, so tightening a
criterion is a config edit rather than a hunt through assertions.

**Seed dependence.** These tolerances were calibrated against the seeds the two
configs commit to (424242 and 991155). They are stated relative to the spread of
the true effects rather than as absolute values, which degrades gracefully
across draws, but a new seed should be re-measured rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import FixtureBundle, build_fixture_bundle

# Deliberate use of the estimator's own design builder: criterion 3 must adjust
# for exactly the covariates ComBat adjusted for, and a reimplementation here
# would be asserting against the test's copy of the rule rather than the rule.
from morphline.combat import _encode_covariates
from morphline.config import HarmonizationConfig, RunConfig, load_config
from morphline.fixtures.truth import BASE_VALUES, EXPANDING_STRUCTURES
from morphline.stages.harmonize import HarmonizationResult, harmonize
from morphline.stages.model import RegionFit, fit_region

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Three regions spanning the scale and sign diversity of the v1 set: a
#: shrinking 4000 mm³ volume, a 3.3 mm thickness four orders of magnitude
#: smaller, and an *expanding* structure whose injected coefficients flip sign.
#: A sign bug survives the first two and dies on the third.
PROBE_REGIONS = (
    ("lh-hippocampus", "volume", "hippocampus"),
    ("lh-entorhinal", "thickness", "entorhinal"),
    ("lh-lateral-ventricle", "volume", "lateral-ventricle"),
)

TIME = "time"
INTERACTION = "time:dx_baseline[T.patient]"


def _injected(structure: str, per_year: float) -> float:
    """Return an injected per-year coefficient in the region's native units."""
    direction = -1.0 if structure in EXPANDING_STRUCTURES else 1.0
    return direction * BASE_VALUES[structure] * per_year


def _realized_site_shift(bundle: FixtureBundle, mask: pd.Series) -> pd.DataFrame:
    """Return the site shift ComBat's gamma actually estimates, per region.

    **Not** ``manifest["site_additive_effects"]``. The generator produces
    ``observed = biological * mult[s,r] + add[s,r] + eps``, and standard ComBat
    carries one additive batch term with a shared covariate block, so gamma
    absorbs the multiplicative *mean* shift as well::

        gamma[s,r]  ~  add[s,r] + (mult[s,r] - 1) * mean(biological | s,r)

    That quantity is computable exactly, because ``ground_truth.parquet``
    carries both ``value`` and ``true_biological_value``. Comparing gamma
    against the configured additive effect alone would fail a *correct*
    estimator, which is why this function exists rather than a dict lookup.

    The result is re-centered by the size-weighted mean, because gamma is
    identified only up to that constraint: the batch terms satisfy
    ``sum_b (n_b/n) * gamma_b == 0``, not ``mean(gamma) == 0``.

    Args:
        bundle: The fixture tree and its injected truth.
        mask: The rows that estimated the batch terms.

    Returns:
        One row per ``(region, site)`` with the expected shift in native units.
    """
    observed = bundle.observations.loc[mask, ["subject_id", "session_id", "region", "site"]]
    truth = bundle.ground_truth[
        ["subject_id", "session_id", "region", "value", "true_biological_value"]
    ]
    joined = observed.merge(truth, on=["subject_id", "session_id", "region"], how="inner")
    joined["shift"] = joined["value"] - joined["true_biological_value"]

    rows: list[dict[str, object]] = []
    for region, group in joined.groupby("region"):
        per_site = group.groupby("site")["shift"].agg(["mean", "size"])
        centre = float((per_site["mean"] * per_site["size"]).sum() / per_site["size"].sum())
        for site, entry in per_site.iterrows():
            rows.append(
                {
                    "region": str(region),
                    "site": str(site),
                    "expected": float(entry["mean"]) - centre,
                }
            )
    return pd.DataFrame(rows)


def _recovery_frame(bundle: FixtureBundle, result: HarmonizationResult) -> pd.DataFrame:
    """Pair every estimated batch term with the shift it should have recovered."""
    assert result.fit is not None
    mask = bundle.observations["analysis_included"].fillna(False).astype(bool)
    expected = _realized_site_shift(bundle, mask).set_index(["region", "site"])

    rows: list[dict[str, object]] = []
    for (region, _measure), params in result.fit.parameters.items():
        native = params.native_gamma()
        for site, gamma in native.items():
            if (region, site) not in expected.index:
                continue
            rows.append(
                {
                    "region": region,
                    "site": site,
                    "observed": float(gamma),
                    "expected": float(expected.loc[(region, site), "expected"]),
                    "delta": float(params.delta_star[site]),
                }
            )
    frame = pd.DataFrame(rows)
    frame["abs_error"] = (frame["observed"] - frame["expected"]).abs()
    spread = frame.groupby("region")["expected"].agg(lambda s: float(s.max() - s.min()))
    frame["relative_error"] = frame["abs_error"] / frame["region"].map(spread)
    return frame


def _site_r2(frame: pd.DataFrame, config: HarmonizationConfig) -> dict[str, float]:
    """Return the covariate-adjusted one-way site R² per region.

    Adjustment is the whole point of criterion 3: where sites differ in age or
    diagnosis composition, a raw site association *should* remain, and reporting
    its reduction would credit harmonization for removing real biology.
    """
    scores: dict[str, float] = {}
    for (region, _measure), group in frame.groupby(["region", "measure_type"]):
        values = group["value"].astype("float64")
        design, _ = _encode_covariates(group, config.covariates)

        usable = values.notna()
        if not design.empty:
            usable &= design.notna().all(axis=1)
        values = values[usable]
        if len(values) < 10:
            continue

        y = values.to_numpy(dtype=np.float64)
        if design.empty:
            residuals = y - y.mean()
        else:
            x = np.column_stack([np.ones(len(y)), design.loc[usable].to_numpy(dtype=np.float64)])
            coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
            residuals = y - x @ coefficients

        total = float(((residuals - residuals.mean()) ** 2).sum())
        if total <= 0.0:
            continue
        sites = group.loc[usable, "site"].to_numpy()
        between = sum(
            int((sites == s).sum()) * (residuals[sites == s].mean() - residuals.mean()) ** 2
            for s in np.unique(sites)
        )
        scores[str(region)] = float(between) / total
    return scores


def _fits(frame: pd.DataFrame) -> dict[str, RegionFit]:
    """Fit every probe region, keyed by region name."""
    return {region: fit_region(frame, region, measure) for region, measure, _s in PROBE_REGIONS}


def _bundle_for(config: RunConfig, root: Path) -> FixtureBundle:
    assert config.fixtures is not None
    return build_fixture_bundle(config.fixtures, root, config.qc, config.analysis)


@pytest.fixture(scope="module")
def regime_a_config() -> RunConfig:
    return load_config(REPO_ROOT / "config" / "recovery.yaml")


@pytest.fixture(scope="module")
def regime_b_config() -> RunConfig:
    return load_config(REPO_ROOT / "config" / "confounded.yaml")


@pytest.fixture(scope="module")
def regime_a(
    tmp_path_factory: pytest.TempPathFactory, regime_a_config: RunConfig
) -> tuple[FixtureBundle, HarmonizationResult]:
    """Regime A — site independent of time. Harmonization should work cleanly.

    3 sites x 40 subjects x 4 sessions, additive effects of opposite sign,
    multiplicative effects straddling 1.0, and no planted extreme changes.
    """
    bundle = _bundle_for(regime_a_config, tmp_path_factory.mktemp("regime_a"))
    return bundle, harmonize(bundle.observations, regime_a_config.harmonization)


@pytest.fixture(scope="module")
def regime_b(
    tmp_path_factory: pytest.TempPathFactory, regime_b_config: RunConfig
) -> tuple[FixtureBundle, HarmonizationResult]:
    """Regime B — site confounded with time."""
    bundle = _bundle_for(regime_b_config, tmp_path_factory.mktemp("regime_b"))
    return bundle, harmonize(bundle.observations, regime_b_config.harmonization)


class TestCriterion1BatchRecovery:
    """Is the *estimated* batch parameter the *injected* site effect?

    This tests estimation, not enforcement — the distinction §2.3.2 draws
    against the site-means check it replaces.
    """

    def test_every_region_was_estimated(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult]
    ) -> None:
        _bundle, result = regime_a
        assert result.applied
        assert result.fit is not None
        assert result.fit.n_groups_skipped == 0
        assert result.fit.n_groups_harmonized == 28

    def test_gamma_recovers_the_realized_site_shift(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult], regime_a_config: RunConfig
    ) -> None:
        bundle, result = regime_a
        frame = _recovery_frame(bundle, result)
        tolerance = regime_a_config.harmonization.target_batch_recovery_tolerance

        assert len(frame) == 84
        assert frame["relative_error"].max() <= tolerance

    def test_gamma_tracks_the_injected_structure_across_regions(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult]
    ) -> None:
        bundle, result = regime_a
        frame = _recovery_frame(bundle, result)

        correlation = float(np.corrcoef(frame["observed"], frame["expected"])[0, 1])
        assert correlation >= 0.95

    def test_gamma_absorbs_the_multiplicative_mean_shift(
        self,
        regime_a: tuple[FixtureBundle, HarmonizationResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The executable form of the subtlety, not just a docstring.

        A reader checking criterion 1 reaches for ``site_additive_effects``,
        which is what the config injected. Standard ComBat does not estimate
        that: it estimates the additive effect *plus* the mean shift the
        multiplicative effect induces. This asserts the correct target fits
        substantially better, so the naive comparison cannot be quietly
        reintroduced later.
        """
        bundle, result = regime_a
        frame = _recovery_frame(bundle, result)
        additive = bundle.truth.site_effects

        naive: list[float] = []
        for _index, row in frame.iterrows():
            region, site = str(row["region"]), str(row["site"])
            per_site = {s: additive[s].additive[region] for s in additive}
            centre = float(np.mean(list(per_site.values())))
            naive.append(per_site[site] - centre)

        rmse_correct = float(np.sqrt(((frame["observed"] - frame["expected"]) ** 2).mean()))
        rmse_naive = float(np.sqrt(((frame["observed"] - np.asarray(naive)) ** 2).mean()))

        with capsys.disabled():
            print(
                f"\n  gamma target RMSE: realized shift {rmse_correct:.2f} "
                f"vs additive-only {rmse_naive:.2f} ({rmse_naive / rmse_correct:.1f}x worse)"
            )

        assert rmse_correct < 0.5 * rmse_naive

    def test_delta_stays_near_one_but_has_no_injected_truth(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult]
    ) -> None:
        """A declared gap, asserted as a bound rather than a recovery.

        ``EffectSpec`` has no per-site noise knob, so no site-specific residual
        scale was ever injected and delta has nothing to recover. It should
        still stay near 1.0 — but not *at* it: the generator multiplies the
        subject random effects by ``mult[s,r]`` while leaving ``eps`` alone, and
        with ``random_intercept_sd`` above ``noise_sd`` those random effects
        dominate the residual, so each site's spread inherits its own
        multiplicative term.

        Adding ``noise_sd_multiplier`` to ``SiteSpec`` would give delta a real
        recovery test. This assertion will start failing the day that lands,
        which is the intent.
        """
        _bundle, result = regime_a
        frame = _recovery_frame(_bundle, result)

        assert frame["delta"].between(0.75, 1.25).all()


class TestCriterion2BiologicalPreservation:
    """Are the injected effects still recoverable after harmonization?

    Only the two *within-subject* time terms are asserted. The ``dx_baseline``
    main effect is deliberately excluded: it is a between-subject contrast, and
    where diagnosis composition varies by site its unharmonized estimate is
    site-contaminated, so neither the pre- nor the post-harmonization value is a
    clean target. §2.5.3 already puts it in a separate, secondary family for the
    same reason.
    """

    def test_all_probe_regions_converge(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult]
    ) -> None:
        _bundle, result = regime_a
        for region, fit in _fits(result.observations).items():
            assert fit.converged, f"{region}: {fit.message}"

    @pytest.mark.parametrize(("region", "measure", "structure"), PROBE_REGIONS)
    def test_injected_interaction_survives(
        self,
        regime_a: tuple[FixtureBundle, HarmonizationResult],
        regime_a_config: RunConfig,
        region: str,
        measure: str,
        structure: str,
    ) -> None:
        """The primary hypothesis (§2.5.1) must survive the transform."""
        _bundle, result = regime_a
        effects = regime_a_config.fixtures.effects if regime_a_config.fixtures else None
        assert effects is not None

        expected = _injected(structure, effects.dx_by_time_per_year)
        estimate = fit_region(result.observations, region, measure).coefficients[INTERACTION]
        tolerance = regime_a_config.harmonization.target_biological_preservation_tolerance

        assert np.sign(estimate) == np.sign(expected)
        assert estimate == pytest.approx(expected, rel=tolerance)

    @pytest.mark.parametrize(("region", "measure", "structure"), PROBE_REGIONS)
    def test_injected_time_effect_survives(
        self,
        regime_a: tuple[FixtureBundle, HarmonizationResult],
        regime_a_config: RunConfig,
        region: str,
        measure: str,
        structure: str,
    ) -> None:
        _bundle, result = regime_a
        effects = regime_a_config.fixtures.effects if regime_a_config.fixtures else None
        assert effects is not None

        expected = _injected(structure, effects.time_per_year)
        estimate = fit_region(result.observations, region, measure).coefficients[TIME]
        tolerance = regime_a_config.harmonization.target_biological_preservation_tolerance

        assert np.sign(estimate) == np.sign(expected)
        assert estimate == pytest.approx(expected, rel=tolerance)

    def test_expanding_structures_keep_their_inverted_sign(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult]
    ) -> None:
        """Ventricles expand as tissue is lost, so their coefficients flip.

        A sign error survives every shrinking region and dies here.
        """
        _bundle, result = regime_a
        ventricle = fit_region(result.observations, "lh-lateral-ventricle", "volume")
        hippocampus = fit_region(result.observations, "lh-hippocampus", "volume")

        assert ventricle.coefficients[INTERACTION] > 0
        assert hippocampus.coefficients[INTERACTION] < 0


class TestCriterion3SiteAssociation:
    """Does covariate-adjusted site association fall substantially?"""

    def test_unharmonized_site_association_exists_to_be_reduced(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult]
    ) -> None:
        """Without this precondition the criterion proves nothing."""
        bundle, _result = regime_a
        before = _site_r2(bundle.observations, HarmonizationConfig())

        assert len(before) == 28
        assert float(np.median(list(before.values()))) > 0.05

    def test_site_association_drops_substantially(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult], regime_a_config: RunConfig
    ) -> None:
        bundle, result = regime_a
        config = regime_a_config.harmonization
        before = _site_r2(bundle.observations, config)
        after = _site_r2(result.observations, config)

        regions = sorted(set(before) & set(after))
        median_before = float(np.median([before[r] for r in regions]))
        median_after = float(np.median([after[r] for r in regions]))
        reduction = 1.0 - median_after / median_before

        assert reduction >= config.target_site_association_reduction

    def test_no_region_retains_meaningful_site_association(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult], regime_a_config: RunConfig
    ) -> None:
        _bundle, result = regime_a
        after = _site_r2(result.observations, regime_a_config.harmonization)

        assert max(after.values()) <= 0.02


class TestCriterion4NoAttenuation:
    """Does the longitudinal slope survive harmonization?

    Two-sided on purpose. Inflating a slope is as much a failure as flattening
    one, and a one-sided test would pass an estimator that manufactures signal.
    """

    @pytest.mark.parametrize(("region", "measure", "structure"), PROBE_REGIONS)
    def test_slope_is_neither_shrunk_nor_inflated(
        self,
        regime_a: tuple[FixtureBundle, HarmonizationResult],
        regime_a_config: RunConfig,
        region: str,
        measure: str,
        structure: str,
    ) -> None:
        bundle, result = regime_a
        before = fit_region(bundle.observations, region, measure).coefficients[INTERACTION]
        after = fit_region(result.observations, region, measure).coefficients[INTERACTION]
        allowed = regime_a_config.harmonization.target_slope_attenuation_max

        assert np.sign(after) == np.sign(before)
        assert abs(1.0 - after / before) <= allowed

    def test_time_main_effect_is_not_flattened(
        self, regime_a: tuple[FixtureBundle, HarmonizationResult], regime_a_config: RunConfig
    ) -> None:
        bundle, result = regime_a
        allowed = regime_a_config.harmonization.target_slope_attenuation_max

        for region, measure, _structure in PROBE_REGIONS:
            before = fit_region(bundle.observations, region, measure).coefficients[TIME]
            after = fit_region(result.observations, region, measure).coefficients[TIME]
            assert abs(1.0 - after / before) <= allowed, region


class TestRegimeBConfounded:
    """Site confounded with time — where the effects are not identifiable.

    BUILD_PLAN §2.3.2 predicts harmonization "should visibly attenuate the
    longitudinal effect" here, and asks the test to assert that it does. **It
    does not, and these tests assert what actually happens instead.** The reason
    is a genuine tension inside the spec: §2.3.4 requires
    ``time_from_baseline_years`` be preserved in the design matrix, and a
    preserved covariate is precisely one the batch term cannot absorb.
    ``TestCovariatePreservationIsWhatSavesRegimeB`` below demonstrates the
    predicted attenuation appearing as soon as that covariate is dropped.

    None of which makes the estimates interpretable. Recovery here is knowable
    only because the truth was injected; from the data alone a scanner step and
    a biological change of the same size are the same observation. That is the
    §2.3.1 point, and it is why ``interpretable`` stays ``False`` no matter how
    close the estimate lands.
    """

    def test_the_confound_is_detected(
        self, regime_b: tuple[FixtureBundle, HarmonizationResult]
    ) -> None:
        """Generating the regime is not the same as detecting it."""
        _bundle, result = regime_b
        assert result.diagnostics is not None
        assert result.diagnostics.severity in {"moderate", "severe"}
        assert not result.diagnostics.interpretable

    def test_unharmonized_time_effect_is_badly_wrong(
        self, regime_b: tuple[FixtureBundle, HarmonizationResult], regime_b_config: RunConfig
    ) -> None:
        """This is the actual failure mode: the scanner step reads as biology.

        Unharmonized, the ``time`` coefficient comes out with the *wrong sign*
        on shrinking structures — a scanner that changed mid-study manufactures
        apparent growth in an atrophying brain.
        """
        bundle, _result = regime_b
        effects = regime_b_config.fixtures.effects if regime_b_config.fixtures else None
        assert effects is not None

        for region, measure, structure in PROBE_REGIONS[:2]:
            expected = _injected(structure, effects.time_per_year)
            estimate = fit_region(bundle.observations, region, measure).coefficients[TIME]
            assert np.sign(estimate) != np.sign(expected), region

    def test_harmonization_improves_rather_than_attenuates(
        self,
        regime_b: tuple[FixtureBundle, HarmonizationResult],
        regime_b_config: RunConfig,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Contradicts §2.3.2's stated expectation; asserts the measured truth."""
        bundle, result = regime_b
        effects = regime_b_config.fixtures.effects if regime_b_config.fixtures else None
        assert effects is not None

        with capsys.disabled():
            print("\n  Regime B, time coefficient (injected | unharmonized | harmonized)")
        for region, measure, structure in PROBE_REGIONS:
            expected = _injected(structure, effects.time_per_year)
            before = fit_region(bundle.observations, region, measure).coefficients[TIME]
            after = fit_region(result.observations, region, measure).coefficients[TIME]

            with capsys.disabled():
                print(f"    {region:<24} {expected:>10.3f} {before:>12.3f} {after:>12.3f}")

            assert abs(after - expected) < abs(before - expected), region

    def test_the_primary_hypothesis_is_robust_to_the_confound(
        self, regime_b: tuple[FixtureBundle, HarmonizationResult], regime_b_config: RunConfig
    ) -> None:
        """Why §2.5.3 makes the interaction primary rather than the main effect.

        A scanner change shifts patients and controls alike, so it largely
        cancels in the diagnosis-by-time interaction while destroying the time
        main effect. Measured on this fixture, unharmonized: the interaction
        keeps its sign and stays within a factor of two, while ``time`` inverts
        entirely. Robust is not the same as unbiased — the unharmonized
        interaction is still attenuated to roughly 0.6x on the thickness
        region, which is why it is asserted as a bound and not a tolerance.
        """
        bundle, result = regime_b
        effects = regime_b_config.fixtures.effects if regime_b_config.fixtures else None
        assert effects is not None
        tolerance = regime_b_config.harmonization.target_biological_preservation_tolerance

        for region, measure, structure in PROBE_REGIONS:
            expected = _injected(structure, effects.dx_by_time_per_year)

            unharmonized = fit_region(bundle.observations, region, measure).coefficients[
                INTERACTION
            ]
            assert np.sign(unharmonized) == np.sign(expected), region
            assert 0.5 <= abs(unharmonized / expected) <= 2.0, region

            harmonized = fit_region(result.observations, region, measure).coefficients[INTERACTION]
            assert harmonized == pytest.approx(expected, rel=tolerance), region

    def test_the_time_main_effect_is_the_casualty_not_the_interaction(
        self, regime_b: tuple[FixtureBundle, HarmonizationResult], regime_b_config: RunConfig
    ) -> None:
        """The contrast is the finding: same data, same confound, one term
        destroyed and the other survivable."""
        bundle, _result = regime_b
        effects = regime_b_config.fixtures.effects if regime_b_config.fixtures else None
        assert effects is not None

        for region, measure, structure in PROBE_REGIONS[:2]:
            time_expected = _injected(structure, effects.time_per_year)
            interaction_expected = _injected(structure, effects.dx_by_time_per_year)
            fit = fit_region(bundle.observations, region, measure)

            time_error = abs(fit.coefficients[TIME] / time_expected - 1.0)
            interaction_error = abs(fit.coefficients[INTERACTION] / interaction_expected - 1.0)

            assert time_error > interaction_error, region

    def test_estimates_stay_labelled_not_interpretable(
        self, regime_b: tuple[FixtureBundle, HarmonizationResult]
    ) -> None:
        """Landing near the truth is not evidence the data could have told you.

        The run has no access to the injected truth, so a close estimate and a
        badly confounded one are indistinguishable from inside. The label must
        not soften because the number happens to be right.
        """
        _bundle, result = regime_b
        assert result.diagnostics is not None
        assert not result.diagnostics.interpretable
        assert any(
            "confounded" in note.lower() or "identifiable" in note.lower() for note in result.notes
        )


class TestCovariatePreservationIsWhatSavesRegimeB:
    """§2.3.4's covariate preservation, shown to be load-bearing.

    Dropping ``time_from_baseline_years`` from the design matrix reproduces
    exactly the attenuation §2.3.2 predicts — the batch term absorbs the
    longitudinal signal it is no longer required to leave alone. This is the
    failure mode the spec wanted demonstrated; it is a property of the
    *configuration*, not of the regime.
    """

    def test_dropping_the_time_covariate_attenuates_the_slope(
        self,
        regime_b: tuple[FixtureBundle, HarmonizationResult],
        regime_b_config: RunConfig,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bundle, preserved = regime_b
        effects = regime_b_config.fixtures.effects if regime_b_config.fixtures else None
        assert effects is not None

        stripped = harmonize(
            bundle.observations,
            regime_b_config.harmonization.model_copy(
                update={"covariates": ("age_baseline", "sex", "dx_baseline")}
            ),
        )

        with capsys.disabled():
            print("\n  Regime B, time coefficient by covariate set (injected | kept | dropped)")
        for region, measure, structure in PROBE_REGIONS[:2]:
            expected = _injected(structure, effects.time_per_year)
            kept = fit_region(preserved.observations, region, measure).coefficients[TIME]
            lost = fit_region(stripped.observations, region, measure).coefficients[TIME]

            with capsys.disabled():
                print(f"    {region:<24} {expected:>10.3f} {kept:>10.3f} {lost:>10.3f}")

            assert abs(lost) < abs(kept), region

        # Deliberately not asserted: that the attenuated estimate sits *further*
        # from the injected value. Shrinking toward zero is the failure §2.3.2
        # names, but where the preserved estimate overshoots — entorhinal comes
        # out near 1.6x injected on this draw — shrinking it happens to land
        # closer to truth. That is a coincidence of the draw, not a defence of
        # dropping the covariate, and asserting it would make the test pass or
        # fail on which side of the truth the noise fell.

    def test_the_time_covariate_is_actually_in_the_design(
        self, regime_b: tuple[FixtureBundle, HarmonizationResult]
    ) -> None:
        """Declaring a covariate is not the same as using it."""
        _bundle, result = regime_b
        assert result.fit is not None
        assert "time_from_baseline_years" in result.fit.covariates_used


def test_both_committed_configs_are_exercised(
    regime_a: tuple[FixtureBundle, HarmonizationResult],
    regime_b: tuple[FixtureBundle, HarmonizationResult],
) -> None:
    """``recovery.yaml`` and ``confounded.yaml`` existed for this suite and were
    referenced by no test until it was written."""
    for _bundle, result in (regime_a, regime_b):
        assert result.applied
        assert result.fit is not None
        assert result.fit.n_groups_harmonized == 28
