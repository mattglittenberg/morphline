"""QC and harmonization stage tests.

These pin the *contracts* — the field structure, the three-level vocabulary,
the QC/analysis separation, and the small-batch policies — independently of
whether the numbers are any good. Whether QC's checks find planted failures is
``test_qc_validation.py``; whether ComBat recovers an injected batch effect is
``test_combat.py`` and the §2.3.2 recovery suite. The confound diagnostics are
tested here and must stay tested here, because §7 requires them to survive even
if the ComBat implementation is cut.

Note the shared ``fixture_tree`` has two sites of six subjects, both below the
default ``min_batch_size`` of 20. That is deliberate: it means the small-batch
path is exercised by every test in this module rather than only by a special
case.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import make_fixture_config
from morphline.adapters import SyntheticAdapter
from morphline.config import AnalysisConfig, HarmonizationConfig, QCConfig, Regime, SiteSpec
from morphline.fixtures import write_fixtures
from morphline.schema import QCStatus
from morphline.stages.harmonize import assess_confounding, find_small_batches, harmonize
from morphline.stages.ingest import ingest
from morphline.stages.qc import ALL_FLAGS, apply_qc


@pytest.fixture
def observations(fixture_tree: Path) -> pd.DataFrame:
    return ingest(SyntheticAdapter(fixture_tree)).observations


def _ingest_sites(root: Path, sites: tuple[SiteSpec, ...]) -> pd.DataFrame:
    """Write and ingest a fixture tree with the given site layout."""
    write_fixtures(make_fixture_config(seed=606060, n_sessions=2, sites=sites), root)
    return ingest(SyntheticAdapter(root)).observations


@pytest.fixture
def mixed_batch_observations(tmp_path: Path) -> pd.DataFrame:
    """One adequate batch plus two small ones sharing an acquisition setup.

    This is the layout pooling exists for. The shared ``fixture_tree`` has only
    small batches, so pooling there collapses to a single batch and cannot show
    the policy doing anything.
    """
    return _ingest_sites(
        tmp_path / "fx",
        (
            SiteSpec(name="site-big", n_subjects=25, additive_effect=0.02),
            SiteSpec(name="site-small-x", n_subjects=6, additive_effect=-0.03),
            SiteSpec(name="site-small-y", n_subjects=6, additive_effect=-0.03),
        ),
    )


@pytest.fixture
def unlike_batch_observations(tmp_path: Path) -> pd.DataFrame:
    """Two small batches on genuinely different scanners."""
    return _ingest_sites(
        tmp_path / "fx",
        (
            SiteSpec(name="site-big", n_subjects=25),
            SiteSpec(
                name="site-small-ge",
                n_subjects=6,
                scanner_manufacturer="GE",
                scanner_model="Discovery-MR750",
            ),
            SiteSpec(
                name="site-small-philips",
                n_subjects=6,
                scanner_manufacturer="Philips",
                scanner_model="Achieva",
                field_strength_tesla=1.5,
            ),
        ),
    )


class TestQCContract:
    """The field structure is final even though the checks are not."""

    def test_emits_all_four_qc_fields(self, observations: pd.DataFrame) -> None:
        out = apply_qc(observations, QCConfig(), AnalysisConfig())
        for column in ("qc_status", "qc_flags", "qc_score", "qc_notes", "analysis_included"):
            assert column in out.columns

    def test_status_uses_the_three_level_vocabulary(self, observations: pd.DataFrame) -> None:
        out = apply_qc(observations, QCConfig(), AnalysisConfig())
        assert set(out["qc_status"]) <= {str(s) for s in QCStatus}

    def test_disabling_qc_says_so_rather_than_claiming_checks_ran(
        self, observations: pd.DataFrame
    ) -> None:
        """Passing everything is fine; claiming a check passed it is not."""
        out = apply_qc(observations, QCConfig(enabled=False), AnalysisConfig())
        assert set(out["qc_status"]) == {str(QCStatus.PASS)}
        assert "disabled" in out["qc_notes"].iloc[0].lower()
        assert not any(out["qc_flags"].apply(bool))

    def test_flags_name_the_check_that_fired(self, observations: pd.DataFrame) -> None:
        out = apply_qc(observations, QCConfig(), AnalysisConfig())
        fired = {code for codes in out["qc_flags"] for code in codes}
        assert fired <= set(ALL_FLAGS)
        for codes, note in zip(out["qc_flags"], out["qc_notes"], strict=True):
            for code in codes:
                assert code in note

    def test_inclusion_policy_is_configuration_not_hardcoded(
        self, observations: pd.DataFrame
    ) -> None:
        """QC classifies; the analysis layer decides (§2.4.1)."""
        out = apply_qc(observations, QCConfig(), AnalysisConfig())
        passing = out["qc_status"] == str(QCStatus.PASS)
        assert (out["analysis_included"] == passing).all()

        exclude_all = AnalysisConfig(include_qc_status=())
        out2 = apply_qc(observations, QCConfig(), exclude_all)
        assert not out2["analysis_included"].any()

    def test_warning_excluded_by_default_but_available_for_sensitivity(self) -> None:
        cfg = AnalysisConfig()
        assert QCStatus.WARNING not in cfg.include_qc_status
        assert QCStatus.WARNING in cfg.sensitivity_include

    def test_empty_input_does_not_crash(self) -> None:
        from morphline.schema import empty_canonical

        out = apply_qc(empty_canonical(), QCConfig(), AnalysisConfig())
        assert out.empty


class TestConfoundDiagnostics:
    """§2.3.1 — these must survive even if the ComBat implementation is cut."""

    def test_regime_a_reports_no_confounding(self, observations: pd.DataFrame) -> None:
        diag = assess_confounding(observations)
        assert diag.severity == "none"
        assert diag.interpretable
        assert diag.max_site_time_r2 < 0.05

    def test_regime_b_detects_confounding(self, tmp_path: Path) -> None:
        """The confounded regime must be *detected*, not merely generated."""
        root = tmp_path / "fx"
        write_fixtures(make_fixture_config(regime=Regime.B_CONFOUNDED, n_sessions=4), root)
        obs = ingest(SyntheticAdapter(root)).observations
        diag = assess_confounding(obs)
        assert diag.severity in {"moderate", "severe"}
        assert not diag.interpretable
        assert diag.max_site_time_r2 > 0.2

    def test_confounded_run_says_not_interpretable_as_biology(self, tmp_path: Path) -> None:
        """The wording matters: this is the distinction §2.3.1 exists to force."""
        root = tmp_path / "fx"
        write_fixtures(make_fixture_config(regime=Regime.B_CONFOUNDED, n_sessions=4), root)
        obs = ingest(SyntheticAdapter(root)).observations
        message = assess_confounding(obs).message.lower()
        assert "confounded" in message or "identifiable" in message

    def test_single_site_cannot_be_confounded(self, observations: pd.DataFrame) -> None:
        single = observations[observations["site"] == "site-a"]
        diag = assess_confounding(single)
        assert diag.severity == "none"

    def test_crosstab_is_produced(self, observations: pd.DataFrame) -> None:
        """§2.3.1 requires reporting the scanner x time crosstab."""
        diag = assess_confounding(observations)
        assert not diag.crosstab.empty
        assert diag.crosstab.shape[0] == 2

    def test_empty_input_does_not_crash(self) -> None:
        from morphline.schema import empty_canonical

        diag = assess_confounding(empty_canonical())
        assert diag.severity == "unknown"

    def test_constant_time_is_not_assessable_rather_than_clean(
        self, observations: pd.DataFrame
    ) -> None:
        """Zero time variance makes the R2 zero by construction, not by design.

        A cross-sectional dataset reaches this on every run. Reporting it as
        "no confounding" would be indistinguishable in the report from the same
        verdict earned by a well-spread longitudinal cohort, so the stage must
        say that nothing was assessed.
        """
        cross_sectional = observations[observations["time_from_baseline_years"] == 0.0]
        diag = assess_confounding(cross_sectional)

        assert diag.severity == "not_assessable"
        assert "cannot be assessed" in diag.message
        assert "firm ground" not in diag.message


class TestSmallBatches:
    """§2.3.3 — below-threshold batches are reported, never silently dropped."""

    def test_small_batches_are_detected(self, observations: pd.DataFrame) -> None:
        small = find_small_batches(observations, HarmonizationConfig(min_batch_size=20))
        assert set(small) == {"site-a", "site-b"}
        assert all(n == 6 for n in small.values())

    def test_large_enough_batches_are_not_flagged(self, observations: pd.DataFrame) -> None:
        small = find_small_batches(observations, HarmonizationConfig(min_batch_size=2))
        assert small == {}

    def test_small_batches_appear_in_the_notes(self, observations: pd.DataFrame) -> None:
        result = harmonize(observations, HarmonizationConfig(min_batch_size=20))
        assert any("min_batch_size" in note for note in result.notes)

    def test_threshold_is_configurable(self) -> None:
        assert HarmonizationConfig().min_batch_size == 20
        assert HarmonizationConfig(min_batch_size=5).min_batch_size == 5


class TestHarmonizationPolicies:
    """§2.3.3 — the three small-batch policies must be behaviourally distinct.

    Both fixture sites sit below the default threshold, so each policy's
    handling of a small batch is what these exercise.
    """

    def test_disabled_config_is_a_real_identity_transform(self, observations: pd.DataFrame) -> None:
        """``enabled`` used to gate nothing but a note."""
        result = harmonize(observations, HarmonizationConfig(enabled=False))

        assert not result.applied
        assert result.n_values_changed == 0
        assert result.fit is None
        pd.testing.assert_series_equal(
            result.observations["value"].reset_index(drop=True),
            observations["value"].reset_index(drop=True),
        )

    def test_disabled_arm_is_labelled(self, observations: pd.DataFrame) -> None:
        result = harmonize(observations, HarmonizationConfig(enabled=False))
        assert any("unharmonized" in note.lower() for note in result.notes)

    def test_report_and_exclude_leaves_small_batch_rows_in_place(
        self, observations: pd.DataFrame
    ) -> None:
        """Excluded from harmonization, never from the dataset.

        Dropping the rows would reappear at the modeling boundary under a cause
        that is false, which is worse than not reconciling.
        """
        result = harmonize(observations, HarmonizationConfig())

        assert result.n_values_changed == 0
        assert len(result.observations) == len(observations)
        assert any("not from the dataset" in note for note in result.notes)

    def test_passthrough_harmonizes_small_batches_with_a_warning(
        self, observations: pd.DataFrame
    ) -> None:
        result = harmonize(observations, HarmonizationConfig(small_batch_policy="passthrough"))

        assert result.applied
        assert result.n_values_changed > 0
        assert any(note.startswith("WARNING") for note in result.notes)

    def test_pooling_every_batch_into_one_leaves_nothing_to_estimate(
        self, observations: pd.DataFrame
    ) -> None:
        """Both fixture sites are small, so pooling them leaves a single batch.

        Correct, and worth saying out loud: the report must explain this as the
        policy working as configured rather than as 28 anonymous skipped
        regions.
        """
        result = harmonize(observations, HarmonizationConfig(small_batch_policy="pool"))

        assert result.pooling is not None
        assert not result.applied
        assert any("fewer than two" in note for note in result.notes)

    def test_pool_merges_matching_batches_and_records_the_decision(
        self, mixed_batch_observations: pd.DataFrame
    ) -> None:
        """Pooling changed the estimate, so the decision has to be recoverable."""
        result = harmonize(mixed_batch_observations, HarmonizationConfig(small_batch_policy="pool"))

        assert result.applied
        assert result.pooling is not None
        merged = result.pooling["merged"]
        assert len(merged) == 1
        assert sorted(merged[0]["members"]) == ["site-small-x", "site-small-y"]
        assert merged[0]["n_subjects"] == 12
        assert any("not checkable from the data" in note for note in result.notes)

    def test_pooling_does_not_rewrite_the_canonical_site_column(
        self, mixed_batch_observations: pd.DataFrame
    ) -> None:
        """Pooling is an estimation device, not a claim about where a scan
        happened."""
        result = harmonize(mixed_batch_observations, HarmonizationConfig(small_batch_policy="pool"))

        assert set(result.observations["site"].unique()) == set(
            mixed_batch_observations["site"].unique()
        )

    def test_unlike_small_batches_are_excluded_rather_than_pooled(
        self, unlike_batch_observations: pd.DataFrame
    ) -> None:
        """Both being small is not a shared property of the scanners (§2.3.3)."""
        result = harmonize(
            unlike_batch_observations, HarmonizationConfig(small_batch_policy="pool")
        )

        assert result.pooling is None
        assert any("unlike scanner" in note for note in result.notes)

    def test_policies_differ_from_one_another(self, mixed_batch_observations: pd.DataFrame) -> None:
        """A policy that cannot be distinguished from another is not a policy.

        The discriminator is the *batch structure*, not the row count. Pool and
        passthrough both leave every row adjusted — they disagree about how many
        batches were estimated and therefore about what each row was adjusted
        by. Exclude is the one that changes nothing, because dropping the two
        small batches from estimation leaves a single batch and no contrast.
        """
        excluded = harmonize(mixed_batch_observations, HarmonizationConfig())
        pooled = harmonize(mixed_batch_observations, HarmonizationConfig(small_batch_policy="pool"))
        through = harmonize(
            mixed_batch_observations, HarmonizationConfig(small_batch_policy="passthrough")
        )

        assert excluded.n_values_changed == 0
        assert pooled.n_values_changed > 0
        assert through.n_values_changed > 0

        key = ("lh-hippocampus", "volume")
        assert pooled.fit is not None and through.fit is not None
        assert set(pooled.fit.parameters[key].n_per_batch) == {
            "site-big",
            "pooled:site-small-x+site-small-y",
        }
        assert set(through.fit.parameters[key].n_per_batch) == {
            "site-big",
            "site-small-x",
            "site-small-y",
        }
        assert not pooled.observations["value"].equals(through.observations["value"])


class TestHarmonizationContract:
    """What the stage promises regardless of policy."""

    def test_row_keys_and_counts_survive_the_transform(self, observations: pd.DataFrame) -> None:
        """The funnel and the parity test both rest on this."""
        result = harmonize(observations, HarmonizationConfig(small_batch_policy="passthrough"))
        keys = ["dataset", "subject_id", "session_id", "region", "measure_type"]

        assert len(result.observations) == len(observations)
        assert result.observations["value"].notna().sum() == observations["value"].notna().sum()
        pd.testing.assert_frame_equal(
            result.observations[keys].reset_index(drop=True),
            observations[keys].reset_index(drop=True),
        )

    def test_estimation_set_is_recorded(self, observations: pd.DataFrame) -> None:
        result = harmonize(observations, HarmonizationConfig(small_batch_policy="passthrough"))

        assert result.estimation["estimation_set"] == "analysis_included"
        assert "launder" in result.estimation["estimation_set_rationale"]

    def test_notes_name_the_estimator_rather_than_a_stub(self, observations: pd.DataFrame) -> None:
        result = harmonize(observations, HarmonizationConfig(small_batch_policy="passthrough"))
        assert any("ComBat" in note for note in result.notes)

    def test_constant_covariate_is_dropped_with_a_reason(self, observations: pd.DataFrame) -> None:
        """The ABIDE shape: cross-sectional data has no time variance."""
        cross_sectional = observations[observations["time_from_baseline_years"] == 0.0]
        result = harmonize(cross_sectional, HarmonizationConfig(small_batch_policy="passthrough"))

        assert result.fit is not None
        assert result.fit.covariates_dropped["time_from_baseline_years"] == "constant"
        assert result.applied

    def test_biological_covariates_reach_the_design_matrix(
        self, observations: pd.DataFrame
    ) -> None:
        """§2.3.4 — declaring them is not the same as using them."""
        result = harmonize(observations, HarmonizationConfig(small_batch_policy="passthrough"))

        assert result.fit is not None
        assert "age_baseline" in result.fit.covariates_used

    def test_small_batch_composition_is_reported(self, observations: pd.DataFrame) -> None:
        """§2.3.3 asks for sizes *and* covariate composition."""
        result = harmonize(observations, HarmonizationConfig())

        assert set(result.small_batch_composition) == {"site-a", "site-b"}
        entry = result.small_batch_composition["site-a"]
        assert entry["n_subjects"] == 6
        assert "sex" in entry
        assert "age_baseline_mean" in entry

    def test_result_round_trips_through_json(self, observations: pd.DataFrame) -> None:
        import json

        result = harmonize(observations, HarmonizationConfig(small_batch_policy="passthrough"))
        payload = json.loads(json.dumps(result.as_dict(), default=str))

        assert payload["applied"] is True
        assert payload["combat"]["n_groups_harmonized"] > 0
        assert payload["estimation"]["batch_column"] == "site"

    def test_biological_covariates_are_declared_for_preservation(self) -> None:
        """Covariates must be in the design matrix so they are not absorbed
        as batch effects (§2.3.4)."""
        covariates = HarmonizationConfig().covariates
        for term in ("age_baseline", "sex", "dx_baseline", "time_from_baseline_years"):
            assert term in covariates
