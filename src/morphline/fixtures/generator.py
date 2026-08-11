"""Writes synthetic FreeSurfer ``.stats`` files from injected ground truth.

Implements BUILD_PLAN §3.1. The goal is **structurally valid, representative**
FreeSurfer output — not byte-for-byte reproductions. Fixtures exist to exercise
the parser and the statistics realistically, not to clone ``mri_segstats``.

Writing real text files rather than synthesising canonical rows directly is the
whole point: it means the parser is on the critical path of every test and
every CI run, instead of being bypassed by fixtures that skip it.

Structural variation exercised here
-----------------------------------
FreeSurfer 5.3 / 6 / 7 header differences (including versions that omit
``SurfaceHoles``), extra columns, missing columns, reordered columns,
malformed and truncated headers, files truncated mid-row, Unicode in comment
fields, empty files, and locale-flavoured numeric formatting. Paired with
Hypothesis property tests over generated header/row structures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from morphline.adapters.freesurfer_regions import CANONICAL_TO_APARC, CANONICAL_TO_ASEG
from morphline.coerce import as_float, as_str
from morphline.config import FixtureConfig
from morphline.fixtures.truth import GroundTruth, generate_ground_truth
from morphline.schema import Hemisphere

#: Corruption modes applied to the ``malformed_file_fraction`` of files. Each
#: must be caught by a distinct parser reason code, so that the accounting
#: table can show *why* files were rejected rather than only how many.
CORRUPTIONS: Final = (
    "empty",
    "truncated_mid_row",
    "extra_column",
    "missing_colheaders",
    "malformed_measure",
    "unicode_comment",
)

_ASEG_COLUMNS: Final = (
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

_APARC_COLUMNS: Final = (
    "StructName",
    "NumVert",
    "SurfArea",
    "GrayVol",
    "ThickAvg",
    "ThickStd",
    "MeanCurv",
    "GausCurv",
    "FoldInd",
    "CurvInd",
)


def _fs_major(version: str) -> int:
    """Return the FreeSurfer major version number from a version string."""
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return 6


def _aseg_header(
    subject: str,
    session: str,
    version: str,
    etiv: float,
    holes_lh: float | None,
    holes_rh: float | None,
    *,
    unicode_comment: bool = False,
    malformed_measure: bool = False,
) -> list[str]:
    """Build ``aseg.stats`` header lines for one session.

    FreeSurfer 5.3 emits no ``SurfaceHoles`` measures at all. That omission is
    deliberately reproduced: it is what makes
    ``test_surface_holes_absent_in_fs53_fixture_yields_null_not_zero``
    meaningful, and a default of 0 would imply a flawless surface (§2.2).
    """
    lines = [
        "# Title Segmentation Statistics",
        "#",
        f"# generating_program mri_segstats (morphline synthetic fixture, FS {version})",
        f"# cvs_version {version}",
        f"# subjectname {subject}_{session}",
        "# anatomy_type volume",
    ]
    if unicode_comment:
        lines.append("# comment sujet contrôlé — æøå — 被试者")

    if malformed_measure:
        # No comma-separated body: exercises the MALFORMED_MEASURE warning path.
        lines.append("# Measure BrokenMeasureLineWithNoCommas")

    lines.append(
        "# Measure EstimatedTotalIntraCranialVol, eTIV, "
        f"Estimated Total Intracranial Volume, {etiv:.6f}, mm^3"
    )
    lines.append(
        f"# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, {etiv * 0.72:.6f}, mm^3"
    )

    if _fs_major(version) >= 6 and holes_lh is not None and holes_rh is not None:
        lines.append(
            f"# Measure lhSurfaceHoles, lhSurfaceHoles, "
            f"Number of defect holes in lh surfaces prior to fixing, {holes_lh:.0f}, unitless"
        )
        lines.append(
            f"# Measure rhSurfaceHoles, rhSurfaceHoles, "
            f"Number of defect holes in rh surfaces prior to fixing, {holes_rh:.0f}, unitless"
        )
        lines.append(
            f"# Measure SurfaceHoles, SurfaceHoles, "
            f"Total number of defect holes in surfaces prior to fixing, "
            f"{holes_lh + holes_rh:.0f}, unitless"
        )

    lines.append("# NRows 45")
    lines.append(f"# NTableCols {len(_ASEG_COLUMNS)}")
    return lines


def _aparc_header(
    subject: str,
    session: str,
    hemisphere: str,
    version: str,
    columns: tuple[str, ...],
    *,
    num_vert: int,
    white_surf_area: float,
    mean_thickness: float,
) -> list[str]:
    """Build ``?h.aparc.stats`` header lines for one hemisphere.

    Every FreeSurfer version reports all three cortical header measures under
    the *same* short name, ``Cortex``, distinguishing them only by alias. The
    fixtures reproduce that shape deliberately: a parser keyed on the short
    name alone silently keeps one of the three, and omitting these lines from
    the fixtures is what let that bug survive against real ABIDE files.
    """
    return [
        "# Table of FreeSurfer cortical parcellation anatomical statistics",
        "#",
        "# generating_program mris_anatomical_stats (morphline synthetic fixture)",
        f"# cvs_version {version}",
        f"# subjectname {subject}_{session}",
        f"# hemi {hemisphere}",
        "# annot aparc.annot",
        f"# Measure Cortex, NumVert, Number of Vertices, {num_vert}, unitless",
        f"# Measure Cortex, WhiteSurfArea, White Surface Total Area, {white_surf_area:.1f}, mm^2",
        f"# Measure Cortex, MeanThickness, Mean Thickness, {mean_thickness:.5f}, mm",
        f"# NTableCols {len(columns)}",
    ]


def _format_number(value: float, *, decimal_comma: bool = False) -> str:
    """Format a float, optionally with a locale-flavoured decimal comma."""
    text = f"{value:.4f}"
    return text.replace(".", ",") if decimal_comma else text


def _write_aseg(
    path: Path,
    subject: str,
    session: str,
    version: str,
    etiv: float,
    holes_lh: float | None,
    holes_rh: float | None,
    values: dict[tuple[str, str], float],
    rng: np.random.Generator,
    corruption: str | None,
) -> None:
    """Write one ``aseg.stats`` file, applying any corruption mode."""
    if corruption == "empty":
        path.write_text("", encoding="utf-8")
        return

    header = _aseg_header(
        subject,
        session,
        version,
        etiv,
        holes_lh,
        holes_rh,
        unicode_comment=corruption == "unicode_comment",
        malformed_measure=corruption == "malformed_measure",
    )

    columns = list(_ASEG_COLUMNS)
    if corruption == "extra_column":
        columns.append("ExtraDiagnosticCol")

    lines = list(header)
    if corruption != "missing_colheaders":
        lines.append("# ColHeaders " + " ".join(columns))

    body: list[str] = []
    index = 0
    for (structure, hemi_value), volume in sorted(values.items()):
        struct_name = CANONICAL_TO_ASEG.get((structure, Hemisphere(hemi_value)))
        if struct_name is None:
            continue
        index += 1
        norm_mean = float(rng.normal(85.0, 6.0))
        fields = [
            f"{index:3d}",
            f"{1000 + index:4d}",
            f"{int(volume / 1.02):7d}",
            f"{volume:.1f}",
            struct_name,
            f"{norm_mean:.4f}",
            f"{abs(rng.normal(8.0, 1.5)):.4f}",
            f"{max(0.0, norm_mean - 40):.0f}",
            f"{norm_mean + 40:.0f}",
            f"{80:.0f}",
        ]
        if corruption == "extra_column":
            fields.append("0.0000")
        body.append("  ".join(fields))

    if corruption == "truncated_mid_row" and body:
        # Drop trailing fields from the final row, as a file cut off mid-write.
        body[-1] = " ".join(body[-1].split()[:4])

    path.write_text("\n".join([*lines, *body]) + "\n", encoding="utf-8")


def _write_aparc(
    path: Path,
    subject: str,
    session: str,
    hemisphere: str,
    version: str,
    values: dict[str, float],
    rng: np.random.Generator,
    *,
    reorder_columns: bool,
    decimal_comma: bool,
) -> None:
    """Write one ``?h.aparc.stats`` file."""
    columns = list(_APARC_COLUMNS)
    if reorder_columns:
        # FreeSurfer 7 moved ThickAvg relative to GrayVol in some builds, and
        # real-world files do vary. The parser keys on ColHeaders, so a
        # reordered file must parse identically.
        columns = [
            "StructName",
            "NumVert",
            "SurfArea",
            "ThickAvg",
            "GrayVol",
            "ThickStd",
            "MeanCurv",
            "GausCurv",
            "FoldInd",
            "CurvInd",
        ]

    rows: list[str] = []
    num_vert_total = 0
    surf_area_total = 0
    thicknesses: list[float] = []
    for structure, thickness in sorted(values.items()):
        struct_name = CANONICAL_TO_APARC.get(structure)
        if struct_name is None:
            continue
        num_vert = int(rng.integers(1200, 6000))
        surf_area = int(rng.integers(800, 4200))
        cells = {
            "StructName": struct_name,
            "NumVert": f"{num_vert}",
            "SurfArea": f"{surf_area}",
            "GrayVol": f"{int(rng.integers(3000, 14000))}",
            "ThickAvg": _format_number(thickness, decimal_comma=decimal_comma),
            "ThickStd": _format_number(float(abs(rng.normal(0.5, 0.1)))),
            "MeanCurv": _format_number(float(abs(rng.normal(0.12, 0.02)))),
            "GausCurv": _format_number(float(abs(rng.normal(0.03, 0.01)))),
            "FoldInd": f"{int(rng.integers(5, 40))}",
            "CurvInd": _format_number(float(abs(rng.normal(2.5, 0.8)))),
        }
        rows.append("  ".join(cells[c] for c in columns))
        num_vert_total += num_vert
        surf_area_total += surf_area
        thicknesses.append(thickness)

    mean_thickness = sum(thicknesses) / len(thicknesses) if thicknesses else 0.0
    lines = _aparc_header(
        subject,
        session,
        hemisphere,
        version,
        tuple(columns),
        num_vert=num_vert_total,
        white_surf_area=float(surf_area_total),
        mean_thickness=mean_thickness,
    )
    lines.append("# ColHeaders " + " ".join(columns))
    lines.extend(rows)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixtures(cfg: FixtureConfig, outdir: Path | str) -> GroundTruth:
    """Generate ground truth and write it out as FreeSurfer stats files.

    Layout::

        <outdir>/derivatives/freesurfer/sub-XXXX/ses-YY/stats/aseg.stats
                                                          /lh.aparc.stats
                                                          /rh.aparc.stats
        <outdir>/truth/ground_truth.parquet
        <outdir>/truth/sessions.parquet
        <outdir>/truth/subjects.parquet
        <outdir>/truth/manifest.json

    Sessions with a ``missing_acquisition`` cause get no directory at all;
    sessions with ``missing_derivative`` get a session directory containing no
    usable stats output. The distinction is what lets the accounting stage
    report missingness *by cause* (§2.5.4) rather than as undifferentiated loss.

    Args:
        cfg: Fixture configuration.
        outdir: Destination root; created if absent.

    Returns:
        The :class:`~morphline.fixtures.truth.GroundTruth` that was written.
    """
    root = Path(outdir)
    deriv = root / "derivatives" / "freesurfer"
    truth_dir = root / "truth"
    deriv.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    truth = generate_ground_truth(cfg)
    rng = np.random.default_rng(cfg.seed + 1)

    subject_lookup = truth.subjects.set_index("subject_id")
    obs = truth.observations

    written_files: list[dict[str, Any]] = []

    for session in truth.sessions.itertuples():
        subject_id = str(session.subject_id)
        session_id = str(session.session_id)
        subj = subject_lookup.loc[subject_id]
        version = as_str(subj["freesurfer_version"])

        if session.missing_cause == "missing_acquisition":
            continue

        session_dir = deriv / subject_id / session_id / "stats"
        session_dir.mkdir(parents=True, exist_ok=True)

        if session.missing_cause == "missing_derivative":
            # Session happened, but FreeSurfer produced nothing usable. An
            # empty directory is exactly what this looks like on disk.
            continue

        rows = obs[(obs["subject_id"] == subject_id) & (obs["session_id"] == session_id)]
        if rows.empty:
            continue

        etiv = as_float(subj["etiv_baseline"]) + float(rng.normal(0.0, 6000.0))
        if session.planted_bad_etiv:
            etiv *= 2.4  # implausible on its face; the eTIV bounds check exists for this

        holes_lh: float | None
        holes_rh: float | None
        if _fs_major(version) >= 6:
            base_holes = 60.0 if session.planted_high_holes else 12.0
            holes_lh = float(max(0, rng.poisson(base_holes)))
            holes_rh = float(max(0, rng.poisson(base_holes)))
        else:
            holes_lh = holes_rh = None

        corruption: str | None = None
        if session.planted_malformed:
            corruption = str(rng.choice(CORRUPTIONS))

        subcortical = {
            (as_str(r.structure), as_str(r.hemisphere)): as_float(r.value)
            for r in rows.itertuples()
            if r.measure_type == "volume"
        }
        aseg_path = session_dir / "aseg.stats"
        _write_aseg(
            aseg_path,
            subject_id,
            session_id,
            version,
            etiv,
            holes_lh,
            holes_rh,
            subcortical,
            rng,
            corruption,
        )
        written_files.append({"path": str(aseg_path), "corruption": corruption, "table": "aseg"})

        for hemi in ("lh", "rh"):
            cortical = {
                as_str(r.structure): as_float(r.value)
                for r in rows.itertuples()
                if r.measure_type == "thickness" and r.hemisphere == hemi
            }
            aparc_path = session_dir / f"{hemi}.aparc.stats"
            _write_aparc(
                aparc_path,
                subject_id,
                session_id,
                hemi,
                version,
                cortical,
                rng,
                reorder_columns=_fs_major(version) >= 7,
                decimal_comma=False,
            )
            written_files.append({"path": str(aparc_path), "corruption": None, "table": "aparc"})

    truth.manifest["n_files_written"] = len(written_files)
    truth.manifest["n_files_corrupted"] = sum(
        1 for f in written_files if f["corruption"] is not None
    )
    truth.manifest["corruption_modes"] = (
        pd.Series([f["corruption"] for f in written_files if f["corruption"]])
        .value_counts()
        .to_dict()
    )

    truth.observations.to_parquet(truth_dir / "ground_truth.parquet", index=False)
    truth.sessions.to_parquet(truth_dir / "sessions.parquet", index=False)
    truth.subjects.to_parquet(truth_dir / "subjects.parquet", index=False)
    (truth_dir / "manifest.json").write_text(
        json.dumps(truth.manifest, indent=2, default=str), encoding="utf-8"
    )

    return truth
