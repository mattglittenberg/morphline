"""Pipeline stages.

Everything here reads only the canonical schema (BUILD_PLAN §1.4). No stage
module may import the parser, open a ``.stats`` file, or contain a
FreeSurfer-specific string — ``tests/test_architecture_boundary.py`` enforces
this by walking each module's AST.

``ingest`` is the exception that proves the rule: it *is* the ingestion
boundary, so it is permitted to drive the parser. Nothing downstream of it is.
"""

from __future__ import annotations
