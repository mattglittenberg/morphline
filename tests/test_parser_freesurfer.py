"""Parser tests, including the three surface-hole tests BUILD_PLAN §2.2 names.

The hole/Euler distinction is called out in the spec as commonly got subtly
wrong, so it gets explicit tests rather than incidental coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from morphline.parsers import FreeSurferStatsParser, ParseFailure, ParseFailureCode
from morphline.schema import euler_number

ASEG_FS6 = """\
# Title Segmentation Statistics
# cvs_version 6.0.0
# subjectname sub-0001_ses-01
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1500000.0, mm^3
# Measure lhSurfaceHoles, lhSurfaceHoles, Number of defect holes in lh surfaces, 42, unitless
# Measure rhSurfaceHoles, rhSurfaceHoles, Number of defect holes in rh surfaces, 17, unitless
# Measure SurfaceHoles, SurfaceHoles, Total number of defect holes in surfaces, 59, unitless
# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1  17  4100  4200.5  Left-Hippocampus  93.1  6.7  54  134  80
  2  53  4050  4150.2  Right-Hippocampus  92.4  6.9  53  133  80
"""

# FreeSurfer 5.3 emits no SurfaceHoles measures at all.
ASEG_FS53 = """\
# Title Segmentation Statistics
# cvs_version 5.3.0
# subjectname sub-0002_ses-01
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1490000.0, mm^3
# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1  17  4100  4200.5  Left-Hippocampus  93.1  6.7  54  134  80
"""

APARC_REORDERED = """\
# Table of FreeSurfer cortical parcellation anatomical statistics
# cvs_version 7.2.0
# hemi lh
# ColHeaders StructName NumVert SurfArea ThickAvg GrayVol ThickStd MeanCurv GausCurv FoldInd CurvInd
entorhinal  2100  1400  3.3100  8200  0.5100  0.1200  0.0300  12  2.5000
precuneus  4100  2900  2.3500  11000  0.4800  0.1100  0.0280  18  3.1000
"""


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_surface_holes_extracted_verbatim_from_header(tmp_path: Path) -> None:
    """Header says lhSurfaceHoles 42 -> field equals 42, no arithmetic applied."""
    result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", ASEG_FS6))
    assert not isinstance(result, ParseFailure)
    assert result.surface_holes_lh == 42
    assert result.surface_holes_rh == 17
    # The header's total SurfaceHoles is the summed hole count and must not be
    # confused for a Euler number.
    assert result.header_measures["SurfaceHoles"] == 59


def test_euler_number_derived_correctly(tmp_path: Path) -> None:
    """lhSurfaceHoles 42 -> euler_lh == -82."""
    result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", ASEG_FS6))
    assert not isinstance(result, ParseFailure)
    assert euler_number(result.surface_holes_lh) == -82.0
    assert euler_number(result.surface_holes_rh) == -32.0


def test_surface_holes_absent_in_fs53_fixture_yields_null_not_zero(tmp_path: Path) -> None:
    """Missing metric must be null; 0 would silently mean "perfect surface".

    A default of zero produces a Euler number of 2 — a flawless surface —
    which would make old data look like the highest-quality data in the study.
    """
    result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", ASEG_FS53))
    assert not isinstance(result, ParseFailure)
    assert result.surface_holes_lh is None
    assert result.surface_holes_rh is None
    assert euler_number(result.surface_holes_lh) is None
    assert euler_number(result.surface_holes_lh) != 2.0


def test_etiv_extracted(tmp_path: Path) -> None:
    result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", ASEG_FS6))
    assert not isinstance(result, ParseFailure)
    assert result.etiv == pytest.approx(1_500_000.0)


def test_freesurfer_version_extracted(tmp_path: Path) -> None:
    result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", ASEG_FS6))
    assert not isinstance(result, ParseFailure)
    assert result.freesurfer_version == "6.0.0"


def test_reordered_columns_parse_by_name(tmp_path: Path) -> None:
    """Column order must come from ColHeaders, not from position."""
    result = FreeSurferStatsParser().parse(write(tmp_path, "lh.aparc.stats", APARC_REORDERED))
    assert not isinstance(result, ParseFailure)
    assert result.hemisphere == "lh"
    rows = {r["StructName"]: r for r in result.rows}
    # ThickAvg sits in position 4 here, where GrayVol usually is. Reading by
    # position would silently return 8200 as a cortical thickness.
    assert rows["entorhinal"]["ThickAvg"] == pytest.approx(3.31)
    assert rows["entorhinal"]["GrayVol"] == pytest.approx(8200)


def test_unicode_in_comments_does_not_reject_file(tmp_path: Path) -> None:
    content = ASEG_FS6.replace(
        "# subjectname sub-0001_ses-01",
        "# subjectname sub-0001_ses-01\n# comment sujet contrôlé — 被试者",
    )
    result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", content))
    assert not isinstance(result, ParseFailure)
    assert len(result.rows) == 2


def test_latin1_bytes_in_comment_do_not_reject_file(tmp_path: Path) -> None:
    """A comment field must not be able to reject an otherwise valid table."""
    path = tmp_path / "aseg.stats"
    path.write_bytes(ASEG_FS6.encode("utf-8").replace(b"# Title", b"# Caf\xe9\n# Title"))
    result = FreeSurferStatsParser().parse(path)
    assert not isinstance(result, ParseFailure)


class TestFailureCodes:
    """Every rejection carries a machine-readable reason code (§1.6)."""

    def test_empty_file(self, tmp_path: Path) -> None:
        result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", ""))
        assert isinstance(result, ParseFailure)
        assert result.code is ParseFailureCode.EMPTY_FILE

    def test_whitespace_only_file(self, tmp_path: Path) -> None:
        result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", "\n\n   \n"))
        assert isinstance(result, ParseFailure)
        assert result.code is ParseFailureCode.EMPTY_FILE

    def test_truncated_row(self, tmp_path: Path) -> None:
        content = ASEG_FS6 + "  3  18  400\n"
        result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", content))
        assert isinstance(result, ParseFailure)
        assert result.code is ParseFailureCode.TRUNCATED_ROW
        assert result.line_number is not None

    def test_column_count_mismatch(self, tmp_path: Path) -> None:
        content = ASEG_FS6 + "  3  18  400  410.0  Left-Amygdala  90  6  50  130  80  EXTRA\n"
        result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", content))
        assert isinstance(result, ParseFailure)
        assert result.code is ParseFailureCode.COLUMN_COUNT_MISMATCH

    def test_aparc_without_colheaders(self, tmp_path: Path) -> None:
        content = "\n".join(
            line for line in APARC_REORDERED.splitlines() if not line.startswith("# ColHeaders")
        )
        result = FreeSurferStatsParser().parse(write(tmp_path, "lh.aparc.stats", content))
        assert isinstance(result, ParseFailure)
        assert result.code is ParseFailureCode.NO_COLHEADERS

    def test_no_data_rows(self, tmp_path: Path) -> None:
        content = "\n".join(line for line in ASEG_FS6.splitlines() if line.startswith("#"))
        result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", content))
        assert isinstance(result, ParseFailure)
        assert result.code is ParseFailureCode.NO_DATA_ROWS

    def test_unknown_table_type(self, tmp_path: Path) -> None:
        result = FreeSurferStatsParser().parse(write(tmp_path, "mystery.stats", "1 2 3\n"))
        assert isinstance(result, ParseFailure)
        assert result.code is ParseFailureCode.UNKNOWN_TABLE_TYPE

    def test_missing_file(self, tmp_path: Path) -> None:
        result = FreeSurferStatsParser().parse(tmp_path / "aseg.stats")
        assert isinstance(result, ParseFailure)
        assert result.code is ParseFailureCode.IO_ERROR

    def test_failure_is_returned_not_raised(self, tmp_path: Path) -> None:
        """One bad file must not be able to crash a 10,000-file run."""
        result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", ""))
        assert isinstance(result, ParseFailure)
        assert result.as_record()["failure_code"] == "EMPTY_FILE"


def test_aseg_fallback_columns_when_colheaders_absent(tmp_path: Path) -> None:
    """aseg has a known column set, so a missing header is recoverable."""
    content = "\n".join(
        line for line in ASEG_FS6.splitlines() if not line.startswith("# ColHeaders")
    )
    result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", content))
    assert not isinstance(result, ParseFailure)
    assert result.rows[0]["StructName"] == "Left-Hippocampus"
    assert any("fallback" in w for w in result.warnings)


def test_malformed_measure_warns_but_does_not_reject(tmp_path: Path) -> None:
    content = ASEG_FS6.replace("# Title", "# Measure BrokenLineNoCommas\n# Title")
    result = FreeSurferStatsParser().parse(write(tmp_path, "aseg.stats", content))
    assert not isinstance(result, ParseFailure)
    assert any("Measure" in w for w in result.warnings)


def test_checksum_is_stable_and_content_dependent(tmp_path: Path) -> None:
    """Provenance depends on the checksum actually tracking content (§1.5)."""
    parser = FreeSurferStatsParser()
    a = parser.parse(write(tmp_path, "aseg.stats", ASEG_FS6))
    b = parser.parse(write(tmp_path, "aseg.stats", ASEG_FS6))
    c = parser.parse(write(tmp_path, "aseg.stats", ASEG_FS6.replace("4200.5", "4200.6")))
    assert not isinstance(a, ParseFailure)
    assert not isinstance(b, ParseFailure)
    assert not isinstance(c, ParseFailure)
    assert a.checksum == b.checksum
    assert a.checksum != c.checksum
