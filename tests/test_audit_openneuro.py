"""Unit tests for the OpenNeuro audit script's decision logic.

The script itself is a one-off tool for BUILD_PLAN §1.2 and is not pipeline
code, so most of it needs no test. Two parts do, because both decide the
audit's answer and both fail *silently* in the direction that matters — a false
negative rejects the accession the audit exists to find, and nothing downstream
would ever notice.

No network. These load the module by path and exercise the parsing and verdict
rules against hand-written key listings.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load() -> ModuleType:
    """Import the audit script by path; ``scripts/`` is not a package."""
    path = REPO_ROOT / "scripts" / "audit_openneuro.py"
    spec = importlib.util.spec_from_file_location("audit_openneuro", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_openneuro"] = module
    spec.loader.exec_module(module)
    return module


audit = _load()


def _sessions_for(keys: list[str]) -> dict[str, set[str]]:
    """Reproduce the script's subject/session extraction over a key list."""
    found: dict[str, set[str]] = {}
    for key in keys:
        subject = audit._SUBJECT.search(key)
        if not subject:
            continue
        sessions = found.setdefault(subject.group(1), set())
        session = audit._SESSION.search(key)
        if session:
            sessions.add(session.group(1))
    return found


class TestEntityParsing:
    """Both FreeSurfer layouts must resolve, not just the nested one."""

    def test_nested_bids_layout(self) -> None:
        keys = [
            "ds1/derivatives/freesurfer/sub-01/ses-01/stats/aseg.stats",
            "ds1/derivatives/freesurfer/sub-01/ses-02/stats/aseg.stats",
        ]
        assert _sessions_for(keys) == {"01": {"01", "02"}}

    def test_underscore_subjects_dir_layout(self) -> None:
        """FreeSurfer's own convention, and the form BUILD_PLAN §2.1 cites.

        A pattern requiring ``ses-`` to be its own path segment reads this as
        single-session and downgrades a genuine longitudinal accession to
        ``stats-only``. That is the audit's worst possible failure: it rejects
        the dataset it was written to find, and reports a clean negative.
        """
        keys = [
            "ds1/derivatives/freesurfer/sub-01_ses-01/stats/aseg.stats",
            "ds1/derivatives/freesurfer/sub-01_ses-02/stats/aseg.stats",
        ]
        assert _sessions_for(keys) == {"01": {"01", "02"}}

    def test_single_session_is_not_inflated(self) -> None:
        keys = ["ds1/derivatives/freesurfer/sub-01/stats/aseg.stats"]
        assert _sessions_for(keys) == {"01": set()}

    def test_alphanumeric_labels_survive(self) -> None:
        keys = ["ds1/derivatives/freesurfer/sub-A01_ses-baseline2/stats/aseg.stats"]
        assert _sessions_for(keys) == {"A01": {"baseline2"}}


class TestVerdict:
    """ "Found nothing" and "stopped looking" must not collapse together."""

    def test_stats_with_repeat_sessions_is_a_candidate(self) -> None:
        finding = audit.Finding(
            dataset_id="ds1", counts={"aseg.stats": 40}, subjects_multi_session=20
        )
        assert finding.verdict == "CANDIDATE"

    def test_stats_without_repeat_sessions_is_cross_sectional(self) -> None:
        finding = audit.Finding(
            dataset_id="ds1", counts={"aseg.stats": 40}, subjects_multi_session=0
        )
        assert finding.verdict == "stats-only"

    def test_a_complete_empty_scan_is_a_rejection(self) -> None:
        assert audit.Finding(dataset_id="ds1", keys_scanned=900).verdict == "no-stats"

    def test_a_capped_empty_scan_is_not_a_rejection(self) -> None:
        """ds002785 reproduced this: the whole-prefix walk hit its page cap and
        the dataset was reported ``no-stats``, while its
        ``derivatives/freesurfer/`` holds ``aseg.stats`` for 216 subjects."""
        finding = audit.Finding(dataset_id="ds1", keys_scanned=40_000, truncated=True)
        assert finding.verdict == "truncated"

    def test_transport_failure_is_not_a_rejection(self) -> None:
        assert audit.Finding(dataset_id="ds1", error="URLError: timeout").verdict == "ERROR"

    @pytest.mark.parametrize(
        "verdict", ["CANDIDATE", "stats-only", "truncated", "ERROR", "no-stats"]
    )
    def test_every_verdict_is_rankable_in_the_report(self, verdict: str) -> None:
        """A verdict missing from the sort table raises KeyError mid-report,
        after the scan has already been paid for."""
        rank = {"CANDIDATE": 0, "stats-only": 1, "truncated": 2, "ERROR": 3, "no-stats": 4}
        assert verdict in rank


def test_stats_filenames_are_what_the_adapters_read() -> None:
    """The audit must look for the files morphline actually ingests.

    A dataset shipping only ``aparc.a2009s.stats`` is not a target: the v1
    region set is Desikan-Killiany, and the adapters select by exact filename.
    """
    from morphline.adapters.abide_pcp import CORE_TABLES

    assert set(audit.STATS_FILENAMES) == set(CORE_TABLES)
