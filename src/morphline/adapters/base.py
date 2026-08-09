"""Dataset adapters — the second half of the ingestion split (BUILD_PLAN §1.4).

The parser knows file *structure*; an adapter knows dataset *conventions*. It
resolves subject and session IDs, site, scanner, field strength, dates, and
demographics, and turns parsed records into canonical observations.

Four adapters, one parser. Adding a fifth dataset must require **zero** changes
downstream of this layer — that rule is enforced by
``tests/test_architecture_boundary.py``, not merely asserted in a docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from morphline.parsers import ParsedStatsFile


@dataclass(frozen=True, slots=True)
class SubjectSession:
    """One subject-session's worth of stats files, located but not yet parsed.

    Attributes:
        subject_id: Dataset-native subject identifier.
        session_id: Dataset-native session identifier. Cross-sectional datasets
            use a single synthetic session so the downstream schema stays
            uniform.
        stats_files: Paths to the stats files belonging to this session.
    """

    subject_id: str
    session_id: str
    stats_files: tuple[Path, ...]


class DatasetAdapter(ABC):
    """Resolves one dataset's layout and metadata into canonical observations.

    Subclasses supply three things: how to find subject-sessions on disk, what
    metadata attaches to each one, and the dataset's identity for provenance.
    """

    @property
    @abstractmethod
    def dataset(self) -> str:
        """Short dataset identifier, e.g. ``"abide-i"`` or ``"synthetic-v1"``."""

    @property
    @abstractmethod
    def dataset_version(self) -> str:
        """Accession version, release tag, or fixture seed (§1.5)."""

    @abstractmethod
    def discover(self) -> Iterator[SubjectSession]:
        """Locate every subject-session in the dataset.

        Yields:
            One :class:`SubjectSession` per subject × session found on disk.
        """

    @abstractmethod
    def to_canonical(
        self,
        subject_session: SubjectSession,
        parsed: list[ParsedStatsFile],
    ) -> pd.DataFrame:
        """Convert parsed stats files into canonical long-format rows.

        Args:
            subject_session: The session these files belong to.
            parsed: Successfully parsed files for that session. May be shorter
                than ``subject_session.stats_files`` when some were rejected —
                the rejections are accounted for separately, by reason code.

        Returns:
            A frame conforming to :data:`morphline.schema.CANONICAL_SCHEMA`,
            one row per region × measure.
        """

    @abstractmethod
    def expected_sessions(self) -> pd.DataFrame:
        """Return the sessions the dataset *claims* to contain.

        Reconciling this against what was actually discovered is what lets the
        accounting stage distinguish ``missing_acquisition`` from
        ``missing_derivative`` (§2.5.4) instead of reporting undifferentiated
        loss.

        Returns:
            A frame with at least ``subject_id`` and ``session_id`` columns.
        """
