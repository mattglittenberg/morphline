"""FreeSurfer ``.stats`` file parser — structure only, no dataset knowledge.

Implements BUILD_PLAN.md §1.4's first half. This module knows the shape of
``aseg.stats`` and ``?h.aparc.stats`` and nothing else: not which dataset a
file came from, not how subject IDs are formed at a given site, not what a
session is. Resolving those is the :class:`~morphline.adapters.base.DatasetAdapter`'s
job. One parser, all datasets.

Concretely it handles ``# Measure`` header lines, ``# ColHeaders``
declarations, whitespace-delimited numeric rows, FreeSurfer 5.3 / 6 / 7
differences, extra and missing and reordered columns, and malformed input.

Failures never escape as exceptions. :meth:`FreeSurferStatsParser.parse`
returns either a :class:`ParsedStatsFile` or a
:class:`~morphline.parsers.errors.ParseFailure`, so a single bad file in a
10,000-file dataset is a reason-coded row in the accounting table rather than
a crashed run (§1.6).

Surface holes and the Euler number
----------------------------------
These are two different quantities and conflating them is a common error, so
the boundary is drawn here:

* ``lhSurfaceHoles`` / ``rhSurfaceHoles`` are *reported directly* in the
  ``aseg.stats`` header on FreeSurfer 6+. This module extracts them verbatim
  and applies no arithmetic.
* The Euler number is *derived* — ``2 - 2 * holes`` — by
  :func:`morphline.schema.euler_number`, downstream.
* FreeSurfer 5.3 does not emit hole counts. The parsed value is then ``None``,
  never ``0``. Zero would imply a flawless surface and would make the oldest
  data in a study look like its highest-quality data (§2.2).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from morphline.parsers.errors import ParseFailure, ParseFailureCode

PARSER_VERSION: Final = "1.0.0"

_MEASURE_RE: Final = re.compile(r"^#\s*Measure\s+(?P<body>.+)$")
_COLHEADERS_RE: Final = re.compile(r"^#\s*ColHeaders\s+(?P<cols>.+)$")
_KEYVAL_RE: Final = re.compile(r"^#\s*(?P<key>[A-Za-z][A-Za-z0-9_]*)\s+(?P<value>.*)$")

#: Header keys carrying the FreeSurfer version string, in preference order.
_VERSION_KEYS: Final = ("annotationfileversion", "cvs_version", "AnnotationFileVersion")

#: A declared version counts as a version only if it starts with a number.
#: FreeSurfer 5.1 declares a CVS ``$Id:`` blob here instead (see
#: :meth:`FreeSurferStatsParser._version_from_header`).
_VERSION_LIKE: Final = re.compile(r"^\d+(?:\.\d+)*")

#: Fallback columns for ``aseg.stats`` written without a ``# ColHeaders`` line.
_ASEG_FALLBACK_COLS: Final = (
    "Index",
    "SegId",
    "NVoxels",
    "Volume_mm3",
    "StructName",
    "normMean",
    "normStdDev",
    "normMin",
    "normMax",
    "normRange",
)


class StatsTableType(StrEnum):
    """Which FreeSurfer stats table a file represents."""

    ASEG = "aseg"
    APARC = "aparc"


@dataclass(frozen=True, slots=True)
class ParsedStatsFile:
    """The structural content of one ``.stats`` file.

    Deliberately close to the file's own vocabulary — ``StructName``,
    ``Volume_mm3``, ``lhSurfaceHoles``. Translating that vocabulary into the
    canonical schema is the adapter's job, not the parser's.

    Attributes:
        source_file: Path the content was read from.
        checksum: SHA-256 of the file bytes, for the provenance chain (§1.5).
        table_type: Whether this is an aseg or aparc table.
        hemisphere: ``"lh"`` / ``"rh"`` for aparc tables, ``None`` for aseg.
        header_measures: ``# Measure`` entries keyed by short name *and* by
            alias where the file declares one, since neither alone is stable
            across tables and FreeSurfer versions.
        header_fields: Other ``# key value`` header lines, verbatim.
        col_headers: Column names as declared by ``# ColHeaders``.
        rows: Data rows as dicts keyed by column name.
        freesurfer_version: Semantic version, if the header declared one that
            is actually a version. ``None`` on FreeSurfer 5.1, which declares a
            CVS revision instead.
        version_declaration: Whatever the header declared, verbatim, so a null
            ``freesurfer_version`` can still be traced to what the file said.
        warnings: Non-fatal structural oddities worth surfacing.
    """

    source_file: Path
    checksum: str
    table_type: StatsTableType
    hemisphere: str | None
    header_measures: dict[str, float]
    header_fields: dict[str, str]
    col_headers: tuple[str, ...]
    rows: tuple[dict[str, str | float], ...]
    freesurfer_version: str | None
    version_declaration: str | None = None
    warnings: tuple[str, ...] = field(default=())

    @property
    def surface_holes_lh(self) -> float | None:
        """Left-hemisphere hole count, extracted verbatim (``None`` on FS 5.3)."""
        return self.header_measures.get("lhSurfaceHoles")

    @property
    def surface_holes_rh(self) -> float | None:
        """Right-hemisphere hole count, extracted verbatim (``None`` on FS 5.3)."""
        return self.header_measures.get("rhSurfaceHoles")

    @property
    def etiv(self) -> float | None:
        """Estimated total intracranial volume, if the header reports it."""
        for key in ("EstimatedTotalIntraCranialVol", "eTIV", "IntraCranialVol"):
            if key in self.header_measures:
                return self.header_measures[key]
        return None


class FreeSurferStatsParser:
    """Parses FreeSurfer ``.stats`` files into :class:`ParsedStatsFile` records.

    Stateless and dataset-agnostic. Construct once and reuse.
    """

    version: Final = PARSER_VERSION

    def parse(self, path: Path | str) -> ParsedStatsFile | ParseFailure:
        """Parse one stats file.

        Args:
            path: Path to an ``aseg.stats`` or ``?h.aparc.stats`` file.

        Returns:
            A :class:`ParsedStatsFile` on success, or a
            :class:`~morphline.parsers.errors.ParseFailure` carrying a reason
            code. This method does not raise for malformed input.
        """
        source = Path(path)

        try:
            raw = source.read_bytes()
        except OSError as exc:
            return ParseFailure(source, ParseFailureCode.IO_ERROR, str(exc))

        if not raw.strip():
            return ParseFailure(source, ParseFailureCode.EMPTY_FILE, "file is empty or whitespace")

        checksum = hashlib.sha256(raw).hexdigest()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Real stats files occasionally carry latin-1 bytes in comment
            # fields (subject names, site notes). Retry before giving up:
            # comments must not be able to reject an otherwise valid table.
            try:
                text = raw.decode("latin-1")
            except UnicodeDecodeError:
                return ParseFailure(source, ParseFailureCode.ENCODING_ERROR, str(exc))

        table_type, hemisphere = self._identify(source, text)
        if table_type is None:
            return ParseFailure(
                source,
                ParseFailureCode.UNKNOWN_TABLE_TYPE,
                f"filename {source.name!r} does not match a known stats table",
            )

        return self._parse_text(source, checksum, text, table_type, hemisphere)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _identify(source: Path, text: str) -> tuple[StatsTableType | None, str | None]:
        """Determine table type and hemisphere from filename, then header."""
        name = source.name.lower()
        if "aparc" in name:
            hemi = "lh" if name.startswith("lh.") else "rh" if name.startswith("rh.") else None
            return StatsTableType.APARC, hemi
        if "aseg" in name:
            return StatsTableType.ASEG, None

        # Filename was unhelpful (renamed fixture, unusual layout). The header
        # names the generating command, so fall back to that.
        head = text[:4000].lower()
        if "mris_anatomical_stats" in head or "aparc.annot" in head:
            hemi = "lh" if " lh " in head or "hemi lh" in head else "rh" if " rh " in head else None
            return StatsTableType.APARC, hemi
        if "mri_segstats" in head or "aseg.mgz" in head:
            return StatsTableType.ASEG, None
        return None, None

    def _parse_text(
        self,
        source: Path,
        checksum: str,
        text: str,
        table_type: StatsTableType,
        hemisphere: str | None,
    ) -> ParsedStatsFile | ParseFailure:
        measures: dict[str, float] = {}
        header_fields: dict[str, str] = {}
        col_headers: tuple[str, ...] = ()
        data_lines: list[tuple[int, str]] = []
        warnings: list[str] = []

        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            if not stripped.startswith("#"):
                data_lines.append((lineno, stripped))
                continue

            if (m := _MEASURE_RE.match(stripped)) is not None:
                parsed = self._parse_measure(m.group("body"))
                if parsed is None:
                    warnings.append(f"line {lineno}: unparseable # Measure line")
                    continue
                keys, value = parsed
                for key in keys:
                    measures[key] = value
                continue

            if (m := _COLHEADERS_RE.match(stripped)) is not None:
                col_headers = tuple(m.group("cols").split())
                continue

            if (m := _KEYVAL_RE.match(stripped)) is not None:
                header_fields[m.group("key")] = m.group("value").strip()

        if not col_headers:
            if table_type is StatsTableType.ASEG:
                col_headers = _ASEG_FALLBACK_COLS
                warnings.append("no # ColHeaders; used aseg fallback column set")
            else:
                return ParseFailure(
                    source,
                    ParseFailureCode.NO_COLHEADERS,
                    "no # ColHeaders declaration and no fallback for aparc tables",
                )

        if not data_lines:
            return ParseFailure(
                source, ParseFailureCode.NO_DATA_ROWS, "headers parsed but table body is empty"
            )

        rows: list[dict[str, str | float]] = []
        for lineno, line in data_lines:
            fields = line.split()
            if len(fields) < len(col_headers):
                return ParseFailure(
                    source,
                    ParseFailureCode.TRUNCATED_ROW,
                    f"row has {len(fields)} fields, header declares {len(col_headers)}",
                    line_number=lineno,
                )
            if len(fields) > len(col_headers):
                return ParseFailure(
                    source,
                    ParseFailureCode.COLUMN_COUNT_MISMATCH,
                    f"row has {len(fields)} fields, header declares {len(col_headers)}",
                    line_number=lineno,
                )
            rows.append({name: _coerce(raw) for name, raw in zip(col_headers, fields, strict=True)})

        return ParsedStatsFile(
            source_file=source,
            checksum=checksum,
            table_type=table_type,
            hemisphere=hemisphere or self._hemisphere_from_header(header_fields),
            header_measures=measures,
            header_fields=header_fields,
            col_headers=col_headers,
            rows=tuple(rows),
            freesurfer_version=self._version_from_header(header_fields),
            version_declaration=self._version_declaration(header_fields),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _parse_measure(body: str) -> tuple[tuple[str, ...], float] | None:
        """Parse the body of a ``# Measure`` line into its keys and its value.

        The canonical shape is ``ShortName, Alias, LongDescription, value, unit``,
        but FreeSurfer 5.3 writes some measures with only three fields, and hole
        counts are written as bare ``# Measure lhSurfaceHoles, ..., 42, unitless``.
        Take the last numeric-looking field before the unit.

        Both ``ShortName`` and ``Alias`` are returned, because they are not
        interchangeable across tables. ``?h.aparc.stats`` reports every measure
        under the short name ``Cortex``, so keying on the first field alone
        collapses ``NumVert``, ``WhiteSurfArea``, and ``MeanThickness`` onto a
        single entry; while ``aseg.stats`` on FreeSurfer 5.1 carries the
        intracranial volume as ``IntraCranialVol, ICV`` and on 6+ as
        ``EstimatedTotalIntraCranialVol, eTIV``, so neither field alone resolves
        eTIV across versions. An alias is distinguished from a long description
        by containing no whitespace.
        """
        parts = [p.strip() for p in body.split(",")]
        if len(parts) < 2:
            return None
        key = parts[0]
        if not key:
            return None
        for offset, candidate in enumerate(reversed(parts[1:])):
            value = _to_float(candidate)
            if value is None:
                continue
            value_index = len(parts) - 1 - offset
            alias = parts[1] if value_index > 1 else ""
            if alias and alias != key and " " not in alias and _to_float(alias) is None:
                return (key, alias), value
            return (key,), value
        return None

    @staticmethod
    def _version_declaration(header_fields: dict[str, str]) -> str | None:
        """Return the version string the header declares, verbatim."""
        for key in _VERSION_KEYS:
            if key in header_fields:
                return header_fields[key]
        return None

    @staticmethod
    def _version_from_header(header_fields: dict[str, str]) -> str | None:
        """Derive a FreeSurfer version from the header's declaration.

        FreeSurfer 6 and 7 declare a semantic version (``# cvs_version 7.2.0``),
        but 5.1 declares the *source file's* CVS revision instead
        (``$Id: mri_segstats.c,v 1.75.2.2 ... $``), which names neither
        FreeSurfer nor its version — and differs between ``mri_segstats`` and
        ``mris_anatomical_stats``, so taking it at face value makes one release
        look like two.

        A declaration that is not version-like therefore yields ``None``, on the
        same rule as surface holes (§2.2): a null states that the file does not
        say, where a wrong value would be repeated into provenance as though it
        did. The verbatim declaration is kept on
        :attr:`ParsedStatsFile.version_declaration`, and the release a
        derivative distribution belongs to is the adapter's ``dataset_version``,
        which is knowledge about the download rather than about the file.
        """
        declaration = FreeSurferStatsParser._version_declaration(header_fields)
        if declaration is None:
            return None
        match = _VERSION_LIKE.match(declaration.strip())
        return match.group(0) if match else None

    @staticmethod
    def _hemisphere_from_header(header_fields: dict[str, str]) -> str | None:
        hemi = header_fields.get("hemi") or header_fields.get("Hemi")
        return hemi if hemi in {"lh", "rh"} else None


def _to_float(token: str) -> float | None:
    """Coerce a token to float, tolerating locale-flavoured decimal commas."""
    token = token.strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        pass
    # A stray decimal comma ("1234,5") shows up in files written on
    # comma-locale systems. Accept it only when unambiguous.
    if token.count(",") == 1 and token.replace(",", "", 1).replace("-", "", 1).isdigit():
        try:
            return float(token.replace(",", "."))
        except ValueError:
            return None
    return None


def _coerce(token: str) -> str | float:
    """Return the token as a float when it parses as one, else as a string."""
    value = _to_float(token)
    return token if value is None else value
