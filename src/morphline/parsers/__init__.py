"""Format parsers. Structure only — no dataset knowledge (BUILD_PLAN §1.4)."""

from __future__ import annotations

from morphline.parsers.errors import ParseFailure, ParseFailureCode
from morphline.parsers.freesurfer import (
    PARSER_VERSION,
    FreeSurferStatsParser,
    ParsedStatsFile,
    StatsTableType,
)

__all__ = [
    "PARSER_VERSION",
    "FreeSurferStatsParser",
    "ParseFailure",
    "ParseFailureCode",
    "ParsedStatsFile",
    "StatsTableType",
]
