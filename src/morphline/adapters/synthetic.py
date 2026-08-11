"""Adapter for synthetic fixture datasets.

Resolves the fixture layout written by
:func:`morphline.fixtures.generator.write_fixtures` and converts parsed
FreeSurfer records into canonical observations. This is the reference
implementation of :class:`~morphline.adapters.base.DatasetAdapter`; the ABIDE
and OpenNeuro adapters follow the same contract and require no downstream
changes (§1.4).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from morphline.adapters.base import DatasetAdapter, SubjectSession
from morphline.adapters.freesurfer_regions import APARC_STRUCT_MAP, ASEG_STRUCT_MAP
from morphline.coerce import as_float, as_str
from morphline.config import DatasetConfig
from morphline.parsers import PARSER_VERSION, ParsedStatsFile, StatsTableType
from morphline.regions import region_key
from morphline.schema import Hemisphere, MeasureType


class SyntheticAdapter(DatasetAdapter):
    """Reads a generated fixture tree into canonical observations.

    Args:
        root: Fixture root, the directory containing ``derivatives/`` and
            ``truth/``.
        dataset_config: Dataset identity for provenance.
    """

    def __init__(self, root: Path | str, dataset_config: DatasetConfig | None = None) -> None:
        """Initialise the adapter against a fixture root."""
        self.root = Path(root)
        self._config = dataset_config or DatasetConfig()
        self._derivatives = self.root / "derivatives" / "freesurfer"
        self._truth_dir = self.root / "truth"
        self._subjects: pd.DataFrame | None = None
        self._sessions: pd.DataFrame | None = None

    @property
    def dataset(self) -> str:
        """Short dataset identifier."""
        return self._config.name

    @property
    def dataset_version(self) -> str:
        """Fixture version; the seed is carried separately in provenance."""
        return self._config.version

    # -- metadata ------------------------------------------------------------

    @property
    def subjects(self) -> pd.DataFrame:
        """Subject-level metadata table, loaded lazily from the fixture truth."""
        if self._subjects is None:
            path = self._truth_dir / "subjects.parquet"
            self._subjects = (
                pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=["subject_id"])
            )
        return self._subjects

    @property
    def sessions(self) -> pd.DataFrame:
        """Session-level metadata table, loaded lazily from the fixture truth."""
        if self._sessions is None:
            path = self._truth_dir / "sessions.parquet"
            self._sessions = (
                pd.read_parquet(path)
                if path.is_file()
                else pd.DataFrame(columns=["subject_id", "session_id"])
            )
        return self._sessions

    def expected_sessions(self) -> pd.DataFrame:
        """Sessions the dataset claims to contain, with planned missingness.

        For fixtures this is exact, which is what makes the accounting funnel
        checkable to the row: any discrepancy between expected and discovered
        must be explainable by a recorded cause (§1.6).
        """
        if self.sessions.empty:
            return pd.DataFrame(columns=["subject_id", "session_id", "missing_cause"])
        cols = [c for c in ("subject_id", "session_id", "missing_cause") if c in self.sessions]
        return self.sessions[cols].copy()

    # -- discovery -----------------------------------------------------------

    def discover(self) -> Iterator[SubjectSession]:
        """Walk the fixture tree for subject-session stats directories.

        Yields:
            One :class:`SubjectSession` per session directory found, including
            those whose ``stats/`` directory is empty — an acquired session
            with no usable derivative is a *different* kind of loss from a
            session that never happened, and the funnel must distinguish them.
        """
        if not self._derivatives.is_dir():
            return
        for subject_dir in sorted(self._derivatives.iterdir()):
            if not subject_dir.is_dir():
                continue
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                stats_dir = session_dir / "stats"
                files = tuple(sorted(stats_dir.glob("*.stats"))) if stats_dir.is_dir() else ()
                yield SubjectSession(
                    subject_id=subject_dir.name,
                    session_id=session_dir.name,
                    stats_files=files,
                )

    # -- canonicalization ----------------------------------------------------

    def to_canonical(
        self, subject_session: SubjectSession, parsed: list[ParsedStatsFile]
    ) -> pd.DataFrame:
        """Convert parsed stats files into canonical long-format rows.

        Args:
            subject_session: Session these files belong to.
            parsed: Successfully parsed files for the session.

        Returns:
            A canonical frame, empty if no recognised regions were present.
        """
        if not parsed:
            return pd.DataFrame()

        meta = self._session_metadata(subject_session)
        ingested_at = dt.datetime.now(dt.UTC).isoformat()

        # Hole counts and eTIV live in the aseg header but describe the whole
        # session, so they are hoisted onto every row of it.
        holes_lh = holes_rh = etiv = None
        fs_version: str | None = None
        for record in parsed:
            fs_version = fs_version or record.freesurfer_version
            if record.table_type is StatsTableType.ASEG:
                holes_lh = record.surface_holes_lh
                holes_rh = record.surface_holes_rh
                etiv = record.etiv

        rows: list[dict[str, Any]] = []
        for record in parsed:
            if record.table_type is StatsTableType.ASEG:
                rows.extend(self._aseg_rows(record))
            else:
                rows.extend(self._aparc_rows(record))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["dataset"] = self.dataset
        df["dataset_version"] = self.dataset_version
        df["subject_id"] = subject_session.subject_id
        df["session_id"] = subject_session.session_id
        df["etiv"] = etiv
        df["surface_holes_lh"] = holes_lh
        df["surface_holes_rh"] = holes_rh
        df["freesurfer_version"] = fs_version
        df["parser_version"] = PARSER_VERSION
        df["ingested_at"] = ingested_at
        for key, value in meta.items():
            df[key] = value
        return df

    def _aseg_rows(self, record: ParsedStatsFile) -> list[dict[str, Any]]:
        """Extract v1 subcortical volumes from one parsed aseg table."""
        rows: list[dict[str, Any]] = []
        for row in record.rows:
            struct_name = row.get("StructName")
            if not isinstance(struct_name, str):
                continue
            mapped = ASEG_STRUCT_MAP.get(struct_name)
            if mapped is None:
                continue
            structure, hemisphere = mapped
            volume = row.get("Volume_mm3")
            if not isinstance(volume, float):
                continue
            rows.append(
                {
                    "region": region_key(structure, hemisphere),
                    "hemisphere": hemisphere.value,
                    "measure_type": str(MeasureType.VOLUME),
                    "value": volume,
                    "unit": "mm^3",
                    "source_file": str(record.source_file),
                    "source_file_checksum": record.checksum,
                }
            )
        return rows

    def _aparc_rows(self, record: ParsedStatsFile) -> list[dict[str, Any]]:
        """Extract v1 cortical thicknesses from one parsed aparc table."""
        if record.hemisphere not in {"lh", "rh"}:
            return []
        hemisphere = Hemisphere(record.hemisphere)
        rows: list[dict[str, Any]] = []
        for row in record.rows:
            struct_name = row.get("StructName")
            if not isinstance(struct_name, str):
                continue
            structure = APARC_STRUCT_MAP.get(struct_name)
            if structure is None:
                continue
            thickness = row.get("ThickAvg")
            if not isinstance(thickness, float):
                continue
            rows.append(
                {
                    "region": region_key(structure, hemisphere),
                    "hemisphere": hemisphere.value,
                    "measure_type": str(MeasureType.THICKNESS),
                    "value": thickness,
                    "unit": "mm",
                    "source_file": str(record.source_file),
                    "source_file_checksum": record.checksum,
                }
            )
        return rows

    def _session_metadata(self, subject_session: SubjectSession) -> dict[str, Any]:
        """Look up acquisition and covariate metadata for one session."""
        blank: dict[str, Any] = {
            "site": None,
            "scanner_manufacturer": None,
            "scanner_model": None,
            "field_strength_tesla": None,
            "time_from_baseline_years": None,
            "age_at_session": None,
            "age_baseline": None,
            "sex": None,
            "dx_baseline": None,
            "dx_at_session": None,
            "etiv_baseline": None,
        }

        subjects = self.subjects
        sessions = self.sessions
        if subjects.empty or sessions.empty:
            return blank

        subj_match = subjects[subjects["subject_id"] == subject_session.subject_id]
        sess_match = sessions[
            (sessions["subject_id"] == subject_session.subject_id)
            & (sessions["session_id"] == subject_session.session_id)
        ]
        if subj_match.empty or sess_match.empty:
            return blank

        subj = subj_match.iloc[0]
        sess = sess_match.iloc[0]

        # The *acquisition* site is the session's, not the subject's: under
        # Regime B a subject's later sessions move to a different scanner, and
        # attributing them to the enrolling site would erase the very confound
        # the regime exists to create (§3.2).
        site = sess.get("acquisition_site", subj["site"])

        return {
            "site": as_str(site),
            "scanner_manufacturer": as_str(subj["scanner_manufacturer"]),
            "scanner_model": as_str(subj["scanner_model"]),
            "field_strength_tesla": as_float(subj["field_strength_tesla"]),
            "time_from_baseline_years": as_float(sess["time_from_baseline_years"]),
            "age_at_session": as_float(sess["age_at_session"]),
            "age_baseline": as_float(subj["age_baseline"]),
            "sex": as_str(subj["sex"]),
            "dx_baseline": as_str(subj["dx_baseline"]),
            # v1 models baseline diagnosis only. Time-varying diagnosis is a
            # post-baseline variable affected by the process being modeled —
            # conversion is partly a consequence of atrophy — so conditioning
            # on it invites collider bias (§2.5.1). Carried, never modeled.
            "dx_at_session": as_str(subj["dx_baseline"]),
            "etiv_baseline": as_float(subj["etiv_baseline"]),
        }
