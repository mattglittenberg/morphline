"""QC sensitivity *and* specificity against planted ground truth (§2.4.4).

Recall alone is not a validation criterion: flagging every observation achieves
recall 1.0 and is useless. Fixtures carry known-bad *and* known-clean
observations precisely so both error directions can be measured, and the
acceptance thresholds live in :class:`~morphline.config.QCConfig` rather than
in assertions here.

**The classes are kept separate on purpose.** Two of the planted failure kinds
are detectable by the checks that exist; the third is not, and averaging them
into one recall number would let an unimplemented check hide behind an
implemented one. The known miss is asserted *as* a miss below, so it fails
loudly the day someone believes it is covered.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import make_fixture_config
from morphline.adapters import SyntheticAdapter
from morphline.config import AnalysisConfig, PlantedSpec, QCConfig, SiteSpec
from morphline.fixtures import write_fixtures
from morphline.schema import QCStatus
from morphline.stages.ingest import ingest
from morphline.stages.qc import FLAG_ETIV, FLAG_EULER, apply_qc


@pytest.fixture(scope="module")
def labelled(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """QC-annotated observations joined to the truth that generated them.

    Sized so each planted class has enough members for a rate to mean
    something: at the CI fixture's 16 subjects a single observation moves
    recall by six points, which measures the draw rather than the checks.
    """
    root = tmp_path_factory.mktemp("qc_validation")
    config = make_fixture_config(
        seed=909090,
        n_sessions=3,
        sites=(
            SiteSpec(name="site-a", n_subjects=30),
            SiteSpec(
                name="site-b",
                n_subjects=30,
                scanner_manufacturer="GE",
                scanner_model="Discovery-MR750",
            ),
        ),
        planted=PlantedSpec(
            qc_high_holes_fraction=0.10,
            qc_bad_etiv_fraction=0.10,
            qc_extreme_change_fraction=0.05,
            missing_acquisition_fraction=0.0,
            missing_derivative_fraction=0.0,
            malformed_file_fraction=0.0,
        ),
        # Hole counts are what the Euler check reads, and 5.3 omits them. A
        # version mix here would measure the draw's version split rather than
        # the check; the null branch has its own test below.
        freesurfer_version_mix={"6.0.0": 0.5, "7.2.0": 0.5},
    )
    write_fixtures(config, root)

    observations = ingest(SyntheticAdapter(root)).observations
    annotated = apply_qc(observations, QCConfig(), AnalysisConfig())

    sessions = pd.read_parquet(root / "truth" / "sessions.parquet")
    sessions["session_id"] = sessions["session_id"].astype(str)
    return annotated.merge(sessions, on=["subject_id", "session_id"], how="inner")


def _flagged(frame: pd.DataFrame) -> pd.Series:
    """Whether each observation drew any QC flag at all."""
    return frame["qc_status"] != str(QCStatus.PASS)


def test_qc_sensitivity_to_planted_etiv_failures(labelled: pd.DataFrame) -> None:
    """An implausible eTIV is the one planted class that must FAIL, not warn."""
    planted = labelled[labelled["planted_bad_etiv"]]
    assert not planted.empty

    recall = (planted["qc_status"] == str(QCStatus.FAIL)).mean()
    assert recall >= QCConfig().target_recall
    assert all(FLAG_ETIV in codes for codes in planted["qc_flags"])


def test_qc_sensitivity_to_planted_surface_defects(labelled: pd.DataFrame) -> None:
    """Inflated hole counts must be caught, at the WARNING operating point.

    Euler is one input among several and never a sole determinant (§2.2), so a
    surface defect warns rather than fails. Measured on sessions that are not
    also carrying a planted eTIV failure, or this would credit the eTIV check.
    """
    planted = labelled[labelled["planted_high_holes"] & ~labelled["planted_bad_etiv"]]
    assert not planted.empty

    assert _flagged(planted).mean() >= QCConfig().target_recall
    assert all(FLAG_EULER in codes for codes in planted["qc_flags"])


def test_qc_specificity_on_clean_observations(labelled: pd.DataFrame) -> None:
    """The false-positive rate on known-clean observations stays under the ceiling."""
    clean = labelled[labelled["is_clean"]]
    assert not clean.empty
    assert _flagged(clean).mean() <= QCConfig().target_false_positive_rate


def test_qc_precision_and_confusion_matrix(
    labelled: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    """Precision holds above the floor, and the full matrix is emitted."""
    flagged = _flagged(labelled)
    dirty = ~labelled["is_clean"]

    true_positive = int((flagged & dirty).sum())
    false_positive = int((flagged & ~dirty).sum())
    false_negative = int((~flagged & dirty).sum())
    true_negative = int((~flagged & ~dirty).sum())

    precision = true_positive / (true_positive + false_positive)
    assert precision >= QCConfig().target_precision

    with capsys.disabled():
        print(
            f"\nQC confusion matrix (observations, FAIL+WARNING treated as flagged):\n"
            f"  true positive  {true_positive:5d}   false negative {false_negative:5d}\n"
            f"  false positive {false_positive:5d}   true negative  {true_negative:5d}\n"
            f"  precision {precision:.4f}"
        )


def test_unimplemented_longitudinal_flag_is_a_declared_miss(labelled: pd.DataFrame) -> None:
    """The suspicious-change flag is not implemented, and no other check covers it.

    This asserts a *gap*. If it starts failing because planted extreme changes
    are now being caught, that is the §2.4.3 flag arriving and this test should
    be replaced by a recall assertion — not deleted quietly.
    """
    planted = labelled[
        labelled["planted_extreme_change"]
        & ~labelled["planted_bad_etiv"]
        & ~labelled["planted_high_holes"]
    ]
    if planted.empty:
        pytest.skip("this fixture draw planted no isolated extreme changes")

    from morphline.stages.qc import FLAG_ASYMMETRY, FLAG_REGION_OUTLIER

    change_specific = {FLAG_ETIV, FLAG_EULER}
    for codes in planted["qc_flags"]:
        assert not (set(codes) & change_specific)
        assert set(codes) <= {FLAG_REGION_OUTLIER, FLAG_ASYMMETRY}


def test_absent_hole_counts_are_not_scored_as_passing(tmp_path: Path) -> None:
    """A version reporting no hole counts must not be ranked as flawless (§2.2).

    Zero holes would mean a topologically perfect surface, which would make the
    oldest data in a study look like its best. Absent the counts the Euler
    check does not run, and must not silently contribute a pass.
    """
    config = make_fixture_config(
        seed=4242,
        planted=PlantedSpec(
            qc_high_holes_fraction=1.0,
            qc_bad_etiv_fraction=0.0,
            qc_extreme_change_fraction=0.0,
            missing_acquisition_fraction=0.0,
            missing_derivative_fraction=0.0,
            malformed_file_fraction=0.0,
        ),
        freesurfer_version_mix={"5.3.0": 1.0},
    )
    write_fixtures(config, tmp_path)

    observations = ingest(SyntheticAdapter(tmp_path)).observations
    assert observations["surface_holes_lh"].isna().all()

    annotated = apply_qc(observations, QCConfig(), AnalysisConfig())
    assert not any(FLAG_EULER in codes for codes in annotated["qc_flags"])
