"""Data accounting funnel tests (BUILD_PLAN §1.6).

The governing rule is that every drop in the funnel must have a stated cause,
and unexplained loss is a bug rather than a rounding error. These tests are
what make that rule enforceable instead of aspirational.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from morphline.adapters import SyntheticAdapter
from morphline.config import FixtureConfig
from morphline.fixtures import write_fixtures
from morphline.stages.accounting import (
    FunnelStage,
    _model_exclusion_causes,
    build_accounting,
    completer_comparison,
)
from morphline.stages.ingest import ingest
from morphline.stages.qc import apply_qc


def build_from_tree(root: Path, *, with_qc: bool = True) -> object:
    adapter = SyntheticAdapter(root)
    result = ingest(adapter)
    qc_obs = None
    if with_qc:
        from morphline.config import AnalysisConfig, QCConfig

        qc_obs = apply_qc(result.observations, QCConfig(), AnalysisConfig())
    return build_accounting(
        observations=result.observations,
        parse_failures=result.failures_frame(),
        expected_sessions=adapter.expected_sessions(),
        files_discovered=result.files_discovered,
        sessions_discovered=result.sessions_discovered,
        sessions_without_files=result.sessions_without_files,
        qc_observations=qc_obs,
        modeled_observations=None,
    )


def test_accounting_funnel_reconciles_exactly(
    tmp_path: Path, lossy_fixture_config: FixtureConfig
) -> None:
    """The week-2 exit criterion: zero unexplained loss on fixtures."""
    root = tmp_path / "fx"
    write_fixtures(lossy_fixture_config, root)
    report = build_from_tree(root)
    assert report.reconcile() == []  # type: ignore[attr-defined]


def test_funnel_reconciles_when_nothing_is_planted(
    tmp_path: Path, clean_fixture_config: FixtureConfig
) -> None:
    """Nothing planted means nothing lost *up to* QC, which still classifies.

    QC is a statistical check, not an oracle: on data with nothing wrong in it
    a robust outlier rule still fires at its false-positive rate, and that rate
    is bounded and asserted in ``test_qc_validation.py``. Demanding zero loss
    here would make the accounting suite fail whenever QC does its job, so it
    asserts what it actually means — no loss anywhere ingestion is responsible
    for, and no unexplained loss anywhere at all.
    """
    root = tmp_path / "fx"
    write_fixtures(clean_fixture_config, root)
    report = build_from_tree(root)
    assert report.reconcile() == []  # type: ignore[attr-defined]

    funnel = report.funnel_frame()  # type: ignore[attr-defined]
    ingestion = funnel[funnel["boundary"] != "QC-passing observations"]
    assert ingestion["lost"].sum() == 0


def test_planted_losses_are_actually_attributed(
    tmp_path: Path, lossy_fixture_config: FixtureConfig
) -> None:
    """A funnel that reconciles only because nothing was lost proves nothing."""
    root = tmp_path / "fx"
    truth = write_fixtures(lossy_fixture_config, root)
    report = build_from_tree(root)
    funnel = report.funnel_frame()  # type: ignore[attr-defined]

    assert funnel["lost"].sum() > 0, "fixture config planted no losses to attribute"
    assert truth.manifest["missingness"], "expected planted missingness"

    causes = " ".join(funnel["causes"].tolist())
    assert "missing_acquisition" in causes
    assert "missing_derivative" in causes


def test_parse_failures_are_reported_by_reason_code(
    tmp_path: Path, lossy_fixture_config: FixtureConfig
) -> None:
    root = tmp_path / "fx"
    write_fixtures(lossy_fixture_config, root)
    report = build_from_tree(root)
    codes = report.parse_failures_by_code  # type: ignore[attr-defined]
    assert codes, "expected planted corruptions to produce reason-coded failures"
    assert all(isinstance(code, str) and code.isupper() for code in codes)


def test_unexplained_loss_is_detected() -> None:
    """The check must be able to fail, or it is not a check."""
    stage = FunnelStage(boundary="parsing", unit="files", count=90, lost=10, causes={"BAD": 4})
    assert stage.unexplained() == 6

    from morphline.stages.accounting import AccountingReport

    report = AccountingReport(funnel=[stage])
    errors = report.reconcile()
    assert len(errors) == 1
    assert "6 of 10" in errors[0]


def test_metadata_coverage_reported(tmp_path: Path, clean_fixture_config: FixtureConfig) -> None:
    """§5.2 requires site/scanner/field strength/version distributions."""
    root = tmp_path / "fx"
    write_fixtures(clean_fixture_config, root)
    report = build_from_tree(root)
    coverage = report.metadata_coverage  # type: ignore[attr-defined]
    for field in ("site", "scanner_manufacturer", "field_strength_tesla", "freesurfer_version"):
        assert field in coverage
        assert coverage[field], f"no values recorded for {field}"


def test_batch_sizes_reported(tmp_path: Path, clean_fixture_config: FixtureConfig) -> None:
    root = tmp_path / "fx"
    write_fixtures(clean_fixture_config, root)
    report = build_from_tree(root)
    batches = report.batch_sizes  # type: ignore[attr-defined]
    assert set(batches["subjects_per_site"]) == {"site-a", "site-b"}
    assert sum(batches["subjects_per_site"].values()) == 12


def test_sessions_per_subject_distribution(
    tmp_path: Path, clean_fixture_config: FixtureConfig
) -> None:
    root = tmp_path / "fx"
    write_fixtures(clean_fixture_config, root)
    report = build_from_tree(root)
    # No missingness planted, so every subject should have all three sessions.
    assert report.sessions_per_subject == {"3": 12}  # type: ignore[attr-defined]


def test_empty_dataset_does_not_crash_accounting() -> None:
    from morphline.schema import empty_canonical

    report = build_accounting(
        observations=empty_canonical(),
        parse_failures=pd.DataFrame(columns=["source_file", "failure_code"]),
        expected_sessions=pd.DataFrame(columns=["subject_id", "session_id"]),
        files_discovered=0,
        sessions_discovered=0,
        sessions_without_files=0,
    )
    assert report.reconcile() == []


def test_undocumented_absence_is_surfaced_as_a_note() -> None:
    """Loss the dataset did not warn us about must be called out, not absorbed."""
    from morphline.schema import empty_canonical

    report = build_accounting(
        observations=empty_canonical(),
        parse_failures=pd.DataFrame(columns=["source_file", "failure_code"]),
        expected_sessions=pd.DataFrame(
            {"subject_id": ["s1", "s2"], "session_id": ["ses-01", "ses-01"]}
        ),
        files_discovered=0,
        sessions_discovered=0,
        sessions_without_files=0,
    )
    causes = " ".join(report.funnel_frame()["causes"].tolist())
    assert "undocumented_absence" in causes
    assert any("without a recorded cause" in note for note in report.notes)
    # Still reconciles: the loss is attributed, even though the attribution is
    # itself a warning.
    assert report.reconcile() == []


@pytest.mark.parametrize("regime", ["A", "B"])
def test_funnel_reconciles_under_both_regimes(tmp_path: Path, regime: str) -> None:
    """Both fixture regimes are exercised in CI (§3.2)."""
    from conftest import make_fixture_config

    root = tmp_path / f"fx-{regime}"
    write_fixtures(make_fixture_config(regime=regime, n_sessions=4), root)
    report = build_from_tree(root)
    assert report.reconcile() == []  # type: ignore[attr-defined]


def modeling_frame(n_included: int, regions: list[str]) -> pd.DataFrame:
    """A minimal QC-annotated frame with the columns the boundary reads."""
    return pd.DataFrame(
        {
            "region": regions[:n_included],
            "analysis_included": [True] * n_included,
        }
    )


def modeling_causes(
    observations: pd.DataFrame,
    modeled: int,
    model_fits: pd.DataFrame | None,
) -> dict[str, int]:
    """Return the causes reported for the modeling boundary's shortfall."""
    return _model_exclusion_causes(len(observations) - modeled, observations, model_fits)


def test_modeling_loss_separates_scope_from_incomplete_covariates() -> None:
    """A region never attempted and a row with a null term are different losses.

    Reporting both as ``outside_modeled_region_set`` lets the funnel reconcile
    on a cause that is false: the first is a scope decision, the second is a
    data limitation that biases the remaining sample.
    """
    observations = modeling_frame(5, ["lh-hippocampus"] * 3 + ["lh-precuneus"] * 2)
    fits = pd.DataFrame({"region": ["lh-hippocampus"], "n_observations": [1]})
    causes = modeling_causes(observations, modeled=1, model_fits=fits)
    assert causes == {
        "outside_modeled_region_set": 2,
        "incomplete_model_covariates": 2,
    }


def test_modeling_loss_is_pure_scope_when_covariates_are_complete() -> None:
    """Every attempted observation fitted means the only loss is scope."""
    observations = modeling_frame(4, ["lh-hippocampus"] * 2 + ["lh-precuneus"] * 2)
    fits = pd.DataFrame({"region": ["lh-hippocampus"], "n_observations": [2]})
    causes = modeling_causes(observations, modeled=2, model_fits=fits)
    assert causes == {"outside_modeled_region_set": 2}


def test_modeling_loss_is_unattributed_without_the_fits() -> None:
    """Absent the model's inputs, the count is known but no cause is."""
    observations = modeling_frame(4, ["lh-hippocampus"] * 4)
    causes = modeling_causes(observations, modeled=1, model_fits=None)
    assert causes == {"not_modeled_cause_unavailable": 3}


def test_modeling_causes_always_sum_to_the_reported_loss() -> None:
    """Whatever the decomposition, it must close — that is the funnel's rule."""
    observations = modeling_frame(6, ["lh-hippocampus"] * 6)
    fits = pd.DataFrame({"region": ["lh-amygdala"], "n_observations": [0]})
    causes = modeling_causes(observations, modeled=0, model_fits=fits)
    assert sum(causes.values()) == 6


def test_no_modeling_loss_reports_no_causes() -> None:
    """Nothing lost means nothing to explain."""
    observations = modeling_frame(2, ["lh-hippocampus"] * 2)
    fits = pd.DataFrame({"region": ["lh-hippocampus"], "n_observations": [2]})
    assert modeling_causes(observations, modeled=2, model_fits=fits) == {}


class TestCompleterComparison:
    """§2.5.4: the MAR assumption is load-bearing, so dropout gets checked."""

    def roster(self, subjects: int, sessions: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"subject_id": f"sub-{s:03d}", "session_id": f"ses-{t:02d}"}
                for s in range(subjects)
                for t in range(sessions)
            ]
        )

    def observations(self, rows: list[tuple[str, int, float, str]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "subject_id": subject,
                    "session_id": f"ses-{t:02d}",
                    "age_baseline": age,
                    "etiv_baseline": 1_500_000.0,
                    "sex": sex,
                    "dx_baseline": "patient",
                    "site": "site-a",
                }
                for subject, sessions, age, sex in rows
                for t in range(sessions)
            ]
        )

    def test_a_cross_sectional_dataset_is_not_applicable(self) -> None:
        """Completion is not a meaningful category with one session each, and
        emitting an empty comparison would imply it had been checked."""
        result = completer_comparison(self.roster(10, 1), self.observations([]))

        assert not result["applicable"]
        assert "cross-sectional" in result["reason"]

    def test_completers_and_non_completers_are_separated(self) -> None:
        result = completer_comparison(
            self.roster(4, 3),
            self.observations(
                [
                    ("sub-000", 3, 70.0, "F"),
                    ("sub-001", 3, 72.0, "F"),
                    ("sub-002", 2, 60.0, "M"),
                    ("sub-003", 1, 62.0, "M"),
                ]
            ),
        )

        assert result["applicable"]
        assert result["n_completers"] == 2
        assert result["n_non_completers"] == 2

    def test_subjects_with_no_data_are_counted_but_not_compared(self) -> None:
        """The group most likely to be informatively missing must not vanish."""
        result = completer_comparison(
            self.roster(3, 2),
            self.observations([("sub-000", 2, 70.0, "F"), ("sub-001", 1, 71.0, "F")]),
        )

        assert result["n_without_baseline"] == 1
        assert result["n_completers"] == 1
        assert result["n_non_completers"] == 1
        assert any("no observations at all" in note for note in result["notes"])

    def test_baseline_imbalance_is_surfaced(self) -> None:
        """A large baseline difference must reach the report as a note, not sit
        in a table for a reader to notice."""
        result = completer_comparison(
            self.roster(6, 2),
            self.observations(
                [
                    ("sub-000", 2, 60.0, "F"),
                    ("sub-001", 2, 61.0, "F"),
                    ("sub-002", 2, 62.0, "F"),
                    ("sub-003", 1, 85.0, "M"),
                    ("sub-004", 1, 86.0, "M"),
                    ("sub-005", 1, 87.0, "M"),
                ]
            ),
        )

        smd = result["continuous"]["age_baseline"]["standardized_difference"]
        assert smd is not None and abs(smd) > 1.0
        assert any("differ at baseline" in note for note in result["notes"])

    def test_categorical_covariates_are_reported_as_proportions(self) -> None:
        result = completer_comparison(
            self.roster(4, 2),
            self.observations(
                [
                    ("sub-000", 2, 70.0, "F"),
                    ("sub-001", 2, 71.0, "M"),
                    ("sub-002", 1, 72.0, "F"),
                    ("sub-003", 1, 73.0, "F"),
                ]
            ),
        )

        assert result["categorical"]["sex"]["completers"] == pytest.approx({"F": 0.5, "M": 0.5})
        assert result["categorical"]["sex"]["non_completers"] == pytest.approx({"F": 1.0})

    def test_no_dropout_is_stated_rather_than_left_blank(self) -> None:
        result = completer_comparison(
            self.roster(2, 2),
            self.observations([("sub-000", 2, 70.0, "F"), ("sub-001", 2, 71.0, "M")]),
        )

        assert result["n_non_completers"] == 0
        assert any("no dropout to compare" in note for note in result["notes"])

    def test_an_absent_roster_is_reported_as_unanswerable(self) -> None:
        result = completer_comparison(pd.DataFrame(), self.observations([]))
        assert not result["applicable"]
        assert "no session roster" in result["reason"]


def test_the_funnel_carries_the_completer_comparison(fixture_tree: Path) -> None:
    report = build_from_tree(fixture_tree)
    assert "completers" in report.as_dict()
