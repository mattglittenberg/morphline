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
from morphline.stages.accounting import FunnelStage, build_accounting
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


def test_funnel_reconciles_when_nothing_is_lost(
    tmp_path: Path, clean_fixture_config: FixtureConfig
) -> None:
    root = tmp_path / "fx"
    write_fixtures(clean_fixture_config, root)
    report = build_from_tree(root)
    assert report.reconcile() == []  # type: ignore[attr-defined]
    funnel = report.funnel_frame()  # type: ignore[attr-defined]
    assert funnel["lost"].sum() == 0


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
