"""QC and harmonization stage tests.

Both are stubs in v0.1.0, so these tests pin the *contracts* that week 3 and
week 4 must not break — plus the confound diagnostics, which are real now
because §7 requires them to survive even if the ComBat implementation is cut.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import make_fixture_config
from morphline.adapters import SyntheticAdapter
from morphline.config import AnalysisConfig, HarmonizationConfig, QCConfig, Regime
from morphline.fixtures import write_fixtures
from morphline.schema import QCStatus
from morphline.stages.harmonize import assess_confounding, find_small_batches, harmonize
from morphline.stages.ingest import ingest
from morphline.stages.qc import apply_qc


@pytest.fixture
def observations(fixture_tree: Path) -> pd.DataFrame:
    return ingest(SyntheticAdapter(fixture_tree)).observations


class TestQCContract:
    """The field structure is final even though the checks are not."""

    def test_emits_all_four_qc_fields(self, observations: pd.DataFrame) -> None:
        out = apply_qc(observations, QCConfig(), AnalysisConfig())
        for column in ("qc_status", "qc_flags", "qc_score", "qc_notes", "analysis_included"):
            assert column in out.columns

    def test_stub_marks_everything_pass(self, observations: pd.DataFrame) -> None:
        out = apply_qc(observations, QCConfig(), AnalysisConfig())
        assert set(out["qc_status"]) == {str(QCStatus.PASS)}

    def test_stub_is_honest_about_being_a_stub(self, observations: pd.DataFrame) -> None:
        """Marking everything PASS is fine; claiming checks ran is not."""
        out = apply_qc(observations, QCConfig(), AnalysisConfig())
        assert "stub" in out["qc_notes"].iloc[0].lower()

    def test_inclusion_policy_is_configuration_not_hardcoded(
        self, observations: pd.DataFrame
    ) -> None:
        """QC classifies; the analysis layer decides (§2.4.1)."""
        out = apply_qc(observations, QCConfig(), AnalysisConfig())
        assert out["analysis_included"].all()

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


class TestHarmonizationStub:
    def test_identity_transform_preserves_values(self, observations: pd.DataFrame) -> None:
        result = harmonize(observations, HarmonizationConfig())
        assert not result.applied
        pd.testing.assert_series_equal(
            result.observations["value"].reset_index(drop=True),
            observations["value"].reset_index(drop=True),
        )

    def test_stub_is_honest_in_the_notes(self, observations: pd.DataFrame) -> None:
        result = harmonize(observations, HarmonizationConfig())
        assert any("identity" in note.lower() for note in result.notes)

    def test_disabled_arm_is_labelled(self, observations: pd.DataFrame) -> None:
        result = harmonize(observations, HarmonizationConfig(enabled=False))
        assert any("unharmonized" in note.lower() for note in result.notes)

    def test_biological_covariates_are_declared_for_preservation(self) -> None:
        """Covariates must be in the design matrix so they are not absorbed
        as batch effects (§2.3.4)."""
        covariates = HarmonizationConfig().covariates
        for term in ("age_baseline", "sex", "dx_baseline", "time_from_baseline_years"):
            assert term in covariates
