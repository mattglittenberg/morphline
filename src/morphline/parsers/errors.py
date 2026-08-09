"""Reason codes for parse failures.

BUILD_PLAN.md §1.6 requires that rejected files be reported *with reason
codes*, and §5.2 requires that every real-data parse failure carry one. So
failures are values, not exceptions: the parser returns a
:class:`ParseFailure` rather than raising, ingestion collects them, and the
accounting stage tabulates them by code.

Unexplained loss in the funnel is a bug, not a rounding error (§1.6) — which
is only enforceable if every dropped file says why it was dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ParseFailureCode(StrEnum):
    """Why a stats file could not be parsed."""

    EMPTY_FILE = "EMPTY_FILE"
    """File exists but has no content, or only whitespace."""

    NO_COLHEADERS = "NO_COLHEADERS"
    """No ``# ColHeaders`` declaration and no usable fallback column set."""

    MALFORMED_MEASURE = "MALFORMED_MEASURE"
    """A ``# Measure`` line did not have the expected comma-separated shape."""

    TRUNCATED_ROW = "TRUNCATED_ROW"
    """A data row has fewer fields than declared — file cut off mid-write."""

    COLUMN_COUNT_MISMATCH = "COLUMN_COUNT_MISMATCH"
    """A data row has more fields than the header declares."""

    UNPARSEABLE_NUMERIC = "UNPARSEABLE_NUMERIC"
    """A field expected to be numeric could not be coerced."""

    ENCODING_ERROR = "ENCODING_ERROR"
    """File could not be decoded as UTF-8 or the declared fallback."""

    UNKNOWN_TABLE_TYPE = "UNKNOWN_TABLE_TYPE"
    """Filename and header do not identify this as a known stats table."""

    NO_DATA_ROWS = "NO_DATA_ROWS"
    """Headers parsed but the table body is empty."""

    IO_ERROR = "IO_ERROR"
    """File could not be read at all (missing, permissions, not a file)."""


@dataclass(frozen=True, slots=True)
class ParseFailure:
    """A file the parser rejected, and why.

    Attributes:
        source_file: Path of the rejected file.
        code: Machine-readable reason, tabulated by the accounting stage.
        detail: Human-readable specifics, including line number where known.
        line_number: 1-indexed line the failure was detected on, if applicable.
    """

    source_file: Path
    code: ParseFailureCode
    detail: str
    line_number: int | None = None

    def as_record(self) -> dict[str, str | int | None]:
        """Return a flat record suitable for the accounting table.

        Returns:
            Mapping of column name to value.
        """
        return {
            "source_file": str(self.source_file),
            "failure_code": str(self.code),
            "failure_detail": self.detail,
            "line_number": self.line_number,
        }
