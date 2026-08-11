"""Enforces BUILD_PLAN §1.4's architectural rule.

    Everything downstream of ingestion reads only the canonical schema. QC,
    harmonization, modeling, and reporting must never import the parser, never
    touch a `.stats` file, and never contain a FreeSurfer-specific string.
    Adding a fifth dataset should require zero changes downstream of the
    adapter layer.

The spec says this rule is "enforced by module boundaries". A docstring asking
nicely is not enforcement — it survives exactly until the first deadline. This
module walks each downstream module's AST and fails the build instead.

``stages/ingest.py`` is exempt: it *is* the ingestion boundary, so driving the
parser is its job. Nothing downstream of it is exempt.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "morphline"

#: Modules that live downstream of ingestion and must stay format-agnostic.
DOWNSTREAM_MODULES = (
    SRC / "stages" / "accounting.py",
    SRC / "stages" / "qc.py",
    SRC / "stages" / "harmonize.py",
    SRC / "stages" / "model.py",
    SRC / "stages" / "report.py",
    SRC / "regions.py",
    SRC / "schema.py",
)

#: Import prefixes a downstream module may not depend on.
FORBIDDEN_IMPORTS = ("morphline.parsers", "morphline.adapters", "morphline.fixtures")

#: Vocabulary that betrays knowledge of a specific derivative format or dataset.
FORBIDDEN_TOKENS = (
    "aseg",
    "aparc",
    ".stats",
    "ColHeaders",
    "SurfaceHoles",
    "StructName",
    "Volume_mm3",
    "ThickAvg",
    "SUBJECTS_DIR",
    "abide",
    "openneuro",
)

#: Tokens permitted inside comments and docstrings. Explaining *why* the
#: boundary exists requires naming what is on the other side of it; the ban is
#: on depending on that vocabulary, not on discussing it.
_PROSE_EXEMPT = True


def _imported_names(tree: ast.AST) -> list[str]:
    """Collect every module name imported anywhere in a module, including
    function-local imports."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _code_only_source(source: str, tree: ast.AST) -> str:
    """Return the module source with docstrings and comments stripped.

    Only *executable* content is checked for format vocabulary. A module may
    explain the boundary in prose; it may not depend on the vocabulary.
    """
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                continue
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                first = body[0]
                if first.lineno and first.end_lineno:
                    docstring_lines.update(range(first.lineno, first.end_lineno + 1))

    kept: list[str] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in docstring_lines:
            continue
        code = line.split("#", 1)[0]
        kept.append(code)
    return "\n".join(kept)


@pytest.mark.parametrize("module_path", DOWNSTREAM_MODULES, ids=lambda p: p.name)
def test_downstream_module_does_not_import_ingestion(module_path: Path) -> None:
    """No stage downstream of ingestion may import the parser or adapters."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    offending = [
        name
        for name in _imported_names(tree)
        if any(name.startswith(prefix) for prefix in FORBIDDEN_IMPORTS)
    ]
    assert not offending, (
        f"{module_path.name} imports ingestion-layer modules {offending}. "
        "Everything downstream of ingestion must read only the canonical schema."
    )


@pytest.mark.parametrize("module_path", DOWNSTREAM_MODULES, ids=lambda p: p.name)
def test_downstream_module_has_no_format_specific_vocabulary(module_path: Path) -> None:
    """No stage downstream of ingestion may contain FreeSurfer vocabulary in code."""
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    code = _code_only_source(source, tree).lower()

    offending = [token for token in FORBIDDEN_TOKENS if token.lower() in code]
    assert not offending, (
        f"{module_path.name} contains format-specific tokens {offending} in executable code. "
        "Adding a fifth dataset must require zero changes downstream of the adapter layer."
    )


def test_ingest_is_the_only_stage_permitted_to_import_the_parser() -> None:
    """The exemption is deliberate and narrow: verify it still holds."""
    ingest = SRC / "stages" / "ingest.py"
    tree = ast.parse(ingest.read_text(encoding="utf-8"))
    imports = _imported_names(tree)
    assert any(name.startswith("morphline.parsers") for name in imports), (
        "ingest.py is the ingestion boundary and is expected to drive the parser; "
        "if that changed, this test and the architecture docs need revisiting."
    )


def test_forbidden_token_list_would_actually_catch_a_violation() -> None:
    """Guard against the enforcement itself silently becoming a no-op.

    A boundary test that cannot fail is worse than none, because it reads as
    protection while providing none.
    """
    violating = "value = row['StructName']\n"
    tree = ast.parse(violating)
    code = _code_only_source(violating, tree).lower()
    assert any(token.lower() in code for token in FORBIDDEN_TOKENS)


def test_prose_mentioning_freesurfer_is_permitted() -> None:
    """Docstrings may explain the boundary without tripping the check."""
    prose = '"""This module never reads aseg.stats files."""\nx = 1\n'
    tree = ast.parse(prose)
    code = _code_only_source(prose, tree).lower()
    assert not any(token.lower() in code for token in FORBIDDEN_TOKENS)
