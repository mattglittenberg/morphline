"""Adapter for ABIDE PCP FreeSurfer derivatives (BUILD_PLAN §1.3).

Track A's *parser* target: per-subject FreeSurfer 5.1 ``.stats`` files as
distributed by the Preprocessed Connectomes Project, laid out as::

    <root>/<SUBJECT>/stats/{aseg.stats,lh.aparc.stats,rh.aparc.stats}

``<SUBJECT>`` encodes the acquisition site as a prefix and the numeric
participant ID as a suffix, e.g. ``CMU_a_0050642``, ``UM_1_0050309``,
``UCLA_51233``.

**Cross-sectional.** ABIDE I has one structural session per participant, so a
single synthetic session ID keeps the canonical schema uniform, that session is
by definition baseline, and ``time_from_baseline_years`` is 0. It follows that
``age_baseline`` equals ``age_at_session`` and ``etiv_baseline`` equals
``etiv``. None of this makes ABIDE longitudinal — §1.3 is explicit that a
successful run here is *cross-sectional* real-data integration and nothing more.

**Demographics are not in the stats files.** Age, sex, and diagnosis live in
ABIDE's separate phenotypic table, so they are null unless ``phenotypic_csv``
is supplied. That is a real limitation for harmonization rather than a cosmetic
one: covariate-preserving ComBat needs the covariates, and ABIDE's diagnosis
distribution varies by site (§1.3), so without them a site effect and a
case-mix difference are indistinguishable.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from morphline.adapters.base import DatasetAdapter, SubjectSession
from morphline.adapters.freesurfer_rows import measurement_rows, session_globals
from morphline.config import DatasetConfig
from morphline.parsers import PARSER_VERSION, ParsedStatsFile
from morphline.schema import MissingnessCause

#: ABIDE I is cross-sectional; one session, and it is baseline.
SESSION_ID: Final = "1"

#: Tables read, by exact filename rather than a ``*.stats`` glob.
#:
#: This allowlist is load-bearing, not tidiness. The parser identifies tables by
#: filename, so ``lh.aparc.a2009s.stats`` is also an lh aparc table, and
#: ``lh.entorhinal_exvivo.stats`` reports a structure the Desikan-Killiany table
#: already reports. Globbing would emit two rows for the same
#: subject × session × region × measure and the canonical schema rejects
#: duplicate observation keys (§5.2) — correctly, since neither row is wrong,
#: they are answers to different questions.
CORE_TABLES: Final = ("aseg.stats", "lh.aparc.stats", "rh.aparc.stats")

#: Phenotypic coding published by ABIDE. Applied only on exact match; any other
#: value passes through verbatim rather than being coerced to a guess.
SEX_CODES: Final = {"1": "M", "2": "F"}

#: Mapped onto the canonical vocabulary the model formula expects
#: (``time:dx_baseline[T.patient]``), not ABIDE's own wording.
DX_CODES: Final = {"1": "patient", "2": "control"}

_NUMERIC_SUFFIX = re.compile(r"_(\d+)$")
_SUBSAMPLE_SUFFIX = re.compile(r"_(?:[a-z]|\d+)$")

_ID_COLUMNS: Final = ("sub_id", "subid", "subject_id", "participant_id", "subject")
_SITE_COLUMNS: Final = ("site_id", "site")
_AGE_COLUMNS: Final = ("age_at_scan", "age_at_session", "age")
_SEX_COLUMNS: Final = ("sex", "gender")
_DX_COLUMNS: Final = ("dx_group", "dx", "diagnosis", "group")


def site_from_subject_id(subject_id: str, *, collapse_subsample: bool = False) -> str | None:
    """Derive the acquisition site from an ABIDE subject directory name.

    The numeric participant ID is stripped from the end; what remains is the
    site. ABIDE splits several institutions into separately-acquired
    sub-samples — ``CMU_a``/``CMU_b``, ``Leuven_1``/``Leuven_2``,
    ``MaxMun_a``–``MaxMun_d``, ``UM_1``/``UM_2``, ``UCLA_1``/``UCLA_2`` — which
    is why the default keeps them distinct: they are different acquisitions, and
    the acquisition is what ComBat should treat as a batch.

    The cost is batch size. Splitting gives 25 batches where collapsing gives
    17, and several of the 25 fall below a ``min_batch_size`` of 20 (§2.3.3).
    That is a real trade-off between batch purity and estimability, so it is a
    caller's decision rather than a default buried here.

    Args:
        subject_id: Subject directory name, e.g. ``"CMU_a_0050642"``.
        collapse_subsample: Merge sub-samples into their institution, so
            ``CMU_a`` and ``CMU_b`` both become ``CMU``.

    Returns:
        The site label, or ``None`` if the name carries no numeric ID suffix and
        therefore does not follow the convention.
    """
    match = _NUMERIC_SUFFIX.search(subject_id)
    if match is None:
        return None
    site = subject_id[: match.start()]
    if collapse_subsample:
        site = _SUBSAMPLE_SUFFIX.sub("", site)
    return site or None


class AbidePcpAdapter(DatasetAdapter):
    """Reads an ABIDE PCP FreeSurfer tree into canonical observations.

    Args:
        root: Directory holding one subdirectory per subject.
        dataset_config: Dataset identity for provenance.
        phenotypic_csv: Optional ABIDE phenotypic table supplying site, age,
            sex, and diagnosis. Columns are matched case-insensitively against
            known aliases, and joined on the numeric participant ID so that
            zero-padding differences between the table and the directory names
            do not silently drop every row.
        collapse_site_subsample: See :func:`site_from_subject_id`.
    """

    def __init__(
        self,
        root: Path | str,
        dataset_config: DatasetConfig | None = None,
        *,
        phenotypic_csv: Path | str | None = None,
        collapse_site_subsample: bool = False,
    ) -> None:
        """Initialise the adapter against an ABIDE PCP root."""
        self.root = Path(root)
        self._config = dataset_config or DatasetConfig()
        self._phenotypic_csv = Path(phenotypic_csv) if phenotypic_csv is not None else None
        self._collapse = collapse_site_subsample
        self._phenotype: dict[int, dict[str, Any]] | None = None

    @property
    def dataset(self) -> str:
        """Short dataset identifier."""
        return self._config.name

    @property
    def dataset_version(self) -> str:
        """Derivative release the stats files came from."""
        return self._config.version

    # -- discovery -----------------------------------------------------------

    def discover(self) -> Iterator[SubjectSession]:
        """Walk the tree for per-subject stats directories.

        Yields:
            One :class:`SubjectSession` per subject directory, including those
            whose ``stats/`` directory is absent or empty. ABIDE PCP has nine
            such subjects (§1.3); dropping them here would remove them from the
            accounting funnel, turning an attributable loss into an unexplained
            one.
        """
        if not self.root.is_dir():
            return
        for subject_dir in sorted(self.root.iterdir()):
            if not subject_dir.is_dir():
                continue
            stats_dir = subject_dir / "stats"
            files = tuple(stats_dir / name for name in CORE_TABLES if (stats_dir / name).is_file())
            yield SubjectSession(
                subject_id=subject_dir.name,
                session_id=SESSION_ID,
                stats_files=files,
            )

    def expected_sessions(self) -> pd.DataFrame:
        """Sessions the dataset claims to contain, with a cause for each gap.

        With a phenotypic table the roster is authoritative, so a participant
        listed there with no directory on disk is ``missing_acquisition`` while
        a directory holding no usable tables is ``missing_derivative`` — the
        §2.5.4 distinction, drawn from evidence rather than assumed. Without
        one, the directories are all that is known, so only
        ``missing_derivative`` is detectable and no acquisition loss is
        *claimed*: silence is more honest than a zero here.

        Returns:
            A frame of ``subject_id``, ``session_id``, ``missing_cause``.
        """
        found = {session.subject_id: bool(session.stats_files) for session in self.discover()}
        rows: list[dict[str, Any]] = [
            {
                "subject_id": subject_id,
                "session_id": SESSION_ID,
                "missing_cause": None if has_files else str(MissingnessCause.DERIVATIVE),
            }
            for subject_id, has_files in found.items()
        ]

        for numeric_id, record in self._phenotype_table().items():
            subject_id = str(record.get("subject_id") or numeric_id)
            if subject_id in found:
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": SESSION_ID,
                    "missing_cause": str(MissingnessCause.ACQUISITION),
                }
            )

        if not rows:
            return pd.DataFrame(columns=["subject_id", "session_id", "missing_cause"])
        return pd.DataFrame(rows).sort_values("subject_id", ignore_index=True)

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

        rows = measurement_rows(parsed)
        if not rows:
            return pd.DataFrame()

        globals_ = session_globals(parsed)
        df = pd.DataFrame(rows)
        df["dataset"] = self.dataset
        df["dataset_version"] = self.dataset_version
        df["subject_id"] = subject_session.subject_id
        df["session_id"] = subject_session.session_id
        df["parser_version"] = PARSER_VERSION
        df["ingested_at"] = dt.datetime.now(dt.UTC).isoformat()
        for key, value in globals_.items():
            df[key] = value
        for key, value in self._session_metadata(subject_session.subject_id, globals_).items():
            df[key] = value
        return df

    def _session_metadata(self, subject_id: str, globals_: dict[str, Any]) -> dict[str, Any]:
        """Resolve acquisition and covariate metadata for one subject.

        Args:
            subject_id: Subject directory name.
            globals_: Session-level measures already hoisted from the aseg
                header, used because a single-session dataset's baseline eTIV
                *is* its session eTIV.

        Returns:
            The acquisition and covariate columns for every row of the session.
        """
        phenotype = self._phenotype_for(subject_id)
        site = phenotype.get("site") or site_from_subject_id(
            subject_id, collapse_subsample=self._collapse
        )
        age = phenotype.get("age")
        dx = phenotype.get("dx")
        etiv = globals_.get("etiv")

        return {
            "site": site,
            # Not recoverable from the stats files. ABIDE's own documentation
            # records scanner and field strength per site, but transcribing them
            # here would put unverifiable constants in the data path; a null
            # states the absence instead of asserting a value.
            "scanner_manufacturer": None,
            "scanner_model": None,
            "field_strength_tesla": None,
            "time_from_baseline_years": 0.0,
            "age_at_session": age,
            "age_baseline": age,
            "sex": phenotype.get("sex"),
            "dx_baseline": dx,
            "dx_at_session": dx,
            "etiv_baseline": etiv,
        }

    # -- phenotypic table ----------------------------------------------------

    def _phenotype_for(self, subject_id: str) -> dict[str, Any]:
        """Look up one subject's phenotypic record by numeric participant ID."""
        match = _NUMERIC_SUFFIX.search(subject_id)
        if match is None:
            return {}
        return self._phenotype_table().get(int(match.group(1)), {})

    def _phenotype_table(self) -> dict[int, dict[str, Any]]:
        """Load and normalise the phenotypic table, keyed by numeric ID.

        Returns:
            Empty when no table was supplied or it carries no recognisable
            participant ID column — an unusable table is reported as absent
            metadata rather than raised, so one bad sidecar cannot stop a run
            whose measurements are fine.
        """
        if self._phenotype is not None:
            return self._phenotype

        self._phenotype = {}
        if self._phenotypic_csv is None or not self._phenotypic_csv.is_file():
            return self._phenotype

        frame = pd.read_csv(self._phenotypic_csv)
        columns = {str(name).strip().lower(): name for name in frame.columns}
        id_column = _first_present(columns, _ID_COLUMNS)
        if id_column is None:
            return self._phenotype

        site_column = _first_present(columns, _SITE_COLUMNS)
        age_column = _first_present(columns, _AGE_COLUMNS)
        sex_column = _first_present(columns, _SEX_COLUMNS)
        dx_column = _first_present(columns, _DX_COLUMNS)

        for _, row in frame.iterrows():
            numeric_id = _as_int(row[id_column])
            if numeric_id is None:
                continue
            self._phenotype[numeric_id] = {
                "site": _as_text(row[site_column]) if site_column else None,
                "age": _as_number(row[age_column]) if age_column else None,
                "sex": _decode(_as_text(row[sex_column]) if sex_column else None, SEX_CODES),
                "dx": _decode(_as_text(row[dx_column]) if dx_column else None, DX_CODES),
            }
        return self._phenotype


def _first_present(columns: dict[str, Any], candidates: Sequence[str]) -> Any | None:
    """Return the original name of the first candidate column present."""
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _as_int(value: Any) -> int | None:
    """Coerce a phenotypic ID to ``int``, or ``None`` if it is not one."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_number(value: Any) -> float | None:
    """Coerce a phenotypic value to ``float``, or ``None`` if it is not one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _as_text(value: Any) -> str | None:
    """Coerce a phenotypic value to a trimmed string, or ``None`` if blank."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _decode(value: str | None, codes: dict[str, str]) -> str | None:
    """Translate a published ABIDE code, passing unknown values through."""
    if value is None:
        return None
    return codes.get(value, value)
