"""Hypothesis property tests over generated header/row structures (§3.1).

The parser will meet real files in week 4 and BUILD_PLAN §8 lists "real stats
files break the parser" as a standing risk, mitigated by property-based tests
and version-tolerant parsing from day 1. Example-based tests only cover the
malformations someone thought of; these cover the shape of the input space.

The invariant throughout: **the parser always returns, never raises.** A
malformed file is a reason-coded value, because one bad file must not be able
to crash a run over thousands.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from morphline.parsers import FreeSurferStatsParser, ParsedStatsFile, ParseFailure

SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

column_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=12,
).filter(lambda s: not s[0].isdigit())

numeric_tokens = st.one_of(
    st.integers(min_value=-10_000, max_value=1_000_000).map(str),
    st.floats(min_value=-1e5, max_value=1e7, allow_nan=False, allow_infinity=False).map(
        lambda f: f"{f:.4f}"
    ),
)

comment_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")), min_size=0, max_size=60
)


@given(
    columns=st.lists(column_names, min_size=1, max_size=10, unique=True),
    rows=st.lists(st.lists(numeric_tokens, min_size=1, max_size=10), min_size=0, max_size=6),
    comment=comment_text,
)
@SETTINGS
def test_parser_never_raises_on_arbitrary_table(
    tmp_path_factory: object, columns: list[str], rows: list[list[str]], comment: str
) -> None:
    """Whatever the header/row shape, the parser returns a value."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aseg.stats"
        lines = [
            "# Title Segmentation Statistics",
            "# cvs_version 6.0.0",
            f"# comment {comment}",
            "# ColHeaders " + " ".join(columns),
        ]
        lines.extend("  ".join(row) for row in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = FreeSurferStatsParser().parse(path)
        assert isinstance(result, ParsedStatsFile | ParseFailure)


@given(data=st.binary(min_size=0, max_size=800))
@SETTINGS
def test_parser_never_raises_on_arbitrary_bytes(data: bytes) -> None:
    """Including bytes that are not valid text in any encoding."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aseg.stats"
        path.write_bytes(data)
        result = FreeSurferStatsParser().parse(path)
        assert isinstance(result, ParsedStatsFile | ParseFailure)


@given(
    columns=st.lists(column_names, min_size=2, max_size=8, unique=True),
    values=st.lists(numeric_tokens, min_size=2, max_size=8),
)
@SETTINGS
def test_row_field_count_must_match_header_or_be_rejected(
    columns: list[str], values: list[str]
) -> None:
    """A row is either parsed with every declared column, or rejected.

    Silently truncating or padding a mismatched row is how a column offset
    turns into a plausible-looking wrong number.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aseg.stats"
        path.write_text(
            "\n".join(
                [
                    "# Title Segmentation Statistics",
                    "# ColHeaders " + " ".join(columns),
                    "  ".join(values),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = FreeSurferStatsParser().parse(path)

        if len(values) == len(columns):
            assert isinstance(result, ParsedStatsFile)
            assert set(result.rows[0]) == set(columns)
        else:
            assert isinstance(result, ParseFailure)


@given(holes_lh=st.integers(min_value=0, max_value=500), holes_rh=st.integers(0, 500))
@SETTINGS
def test_surface_holes_roundtrip_exactly(holes_lh: int, holes_rh: int) -> None:
    """Whatever hole count the header declares comes back unmodified."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aseg.stats"
        path.write_text(
            "\n".join(
                [
                    "# Title Segmentation Statistics",
                    "# cvs_version 6.0.0",
                    f"# Measure lhSurfaceHoles, lhSurfaceHoles, holes, {holes_lh}, unitless",
                    f"# Measure rhSurfaceHoles, rhSurfaceHoles, holes, {holes_rh}, unitless",
                    "# ColHeaders Index StructName Volume_mm3",
                    "  1  Left-Hippocampus  4000.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = FreeSurferStatsParser().parse(path)
        assert isinstance(result, ParsedStatsFile)
        assert result.surface_holes_lh == holes_lh
        assert result.surface_holes_rh == holes_rh


@given(truncate_at=st.integers(min_value=1, max_value=400))
@SETTINGS
def test_truncation_at_any_point_is_handled(truncate_at: int) -> None:
    """A file cut off mid-write at any byte must not crash the parser."""
    import tempfile

    full = "\n".join(
        [
            "# Title Segmentation Statistics",
            "# cvs_version 6.0.0",
            "# Measure EstimatedTotalIntraCranialVol, eTIV, vol, 1500000.0, mm^3",
            "# ColHeaders Index SegId NVoxels Volume_mm3 StructName",
            "  1  17  4100  4200.5  Left-Hippocampus",
            "  2  53  4050  4150.2  Right-Hippocampus",
        ]
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aseg.stats"
        path.write_text(full[:truncate_at], encoding="utf-8")
        result = FreeSurferStatsParser().parse(path)
        assert isinstance(result, ParsedStatsFile | ParseFailure)
