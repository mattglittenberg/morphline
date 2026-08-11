"""Adapter decorator restricting discovery to a single subject.

This is what makes per-subject parallelism possible without every adapter
needing to know about it: Nextflow fans out one task per subject, each task
runs ingestion over exactly one subject's files, and the results are gathered
by path (§2.7).

A decorator rather than a parameter on :meth:`DatasetAdapter.discover`, so
adding a dataset never requires reimplementing the filter.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from morphline.adapters.base import DatasetAdapter, SubjectSession
from morphline.parsers import ParsedStatsFile


class SubjectFilter(DatasetAdapter):
    """Wraps an adapter, yielding only one subject's sessions.

    Args:
        inner: The adapter to restrict.
        subject_id: Subject to keep.
    """

    def __init__(self, inner: DatasetAdapter, subject_id: str) -> None:
        """Wrap ``inner``, restricting discovery to ``subject_id``."""
        self._inner = inner
        self._subject_id = subject_id

    @property
    def dataset(self) -> str:
        """Short dataset identifier, delegated to the wrapped adapter."""
        return self._inner.dataset

    @property
    def dataset_version(self) -> str:
        """Dataset version, delegated to the wrapped adapter."""
        return self._inner.dataset_version

    def discover(self) -> Iterator[SubjectSession]:
        """Yield only the wrapped adapter's sessions for the chosen subject."""
        for subject_session in self._inner.discover():
            if subject_session.subject_id == self._subject_id:
                yield subject_session

    def to_canonical(
        self, subject_session: SubjectSession, parsed: list[ParsedStatsFile]
    ) -> pd.DataFrame:
        """Delegate canonicalization to the wrapped adapter."""
        return self._inner.to_canonical(subject_session, parsed)

    def regions_in_scope(self, parsed: list[ParsedStatsFile]) -> set[str] | None:
        """Delegate region coverage to the wrapped adapter.

        Not inherited: the base default is ``None``, so a missing delegation
        here would silently downgrade every staged run's region-level losses to
        unattributed while the in-process run attributed them.
        """
        return self._inner.regions_in_scope(parsed)

    def expected_sessions(self) -> pd.DataFrame:
        """Return the wrapped adapter's expected sessions for this subject."""
        expected = self._inner.expected_sessions()
        if expected.empty or "subject_id" not in expected.columns:
            return expected
        return expected[expected["subject_id"] == self._subject_id].copy()
