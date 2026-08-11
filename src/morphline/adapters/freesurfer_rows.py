"""Shared canonicalization of parsed FreeSurfer tables into measurement rows.

Every adapter turns the same two table types into the same canonical rows; only
the *metadata* around them is dataset-specific. Keeping the row extraction here
rather than in each adapter is the same argument as one parser for all datasets
(§1.4): two copies of this logic would drift, and the drift would show up as
datasets disagreeing about what a thickness is.

The rows produced carry measurement and provenance columns only. Identity,
acquisition, and covariate columns are the adapter's job, because those are
exactly what the adapter knows and this module does not.
"""

from __future__ import annotations

from typing import Any

from morphline.adapters.freesurfer_regions import APARC_STRUCT_MAP, ASEG_STRUCT_MAP
from morphline.parsers import ParsedStatsFile, StatsTableType
from morphline.regions import (
    CORTICAL_REGIONS,
    LATERAL_HEMISPHERES,
    SUBCORTICAL_REGIONS,
    region_key,
)
from morphline.schema import Hemisphere, MeasureType


def aseg_volume_rows(record: ParsedStatsFile) -> list[dict[str, Any]]:
    """Extract v1 subcortical volumes from one parsed aseg table.

    Args:
        record: A successfully parsed aseg table.

    Returns:
        One row per recognised subcortical structure. Structures outside the
        §2.5.2 region set are skipped, not errors.
    """
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


def aparc_thickness_rows(record: ParsedStatsFile) -> list[dict[str, Any]]:
    """Extract v1 cortical thicknesses from one parsed aparc table.

    Args:
        record: A successfully parsed aparc table.

    Returns:
        One row per recognised cortical parcel, or an empty list if the table's
        hemisphere could not be resolved — an unlabelled hemisphere would
        otherwise be silently attributed to one side.
    """
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


def measurement_rows(parsed: list[ParsedStatsFile]) -> list[dict[str, Any]]:
    """Extract canonical measurement rows from every table in one session.

    Args:
        parsed: Successfully parsed tables belonging to a single session.

    Returns:
        Concatenated measurement rows across the session's tables.
    """
    rows: list[dict[str, Any]] = []
    for record in parsed:
        if record.table_type is StatsTableType.ASEG:
            rows.extend(aseg_volume_rows(record))
        else:
            rows.extend(aparc_thickness_rows(record))
    return rows


def regions_in_scope(parsed: list[ParsedStatsFile]) -> set[str]:
    """Return the canonical regions this session's parsed tables could report.

    Coverage, not content: an aparc table for one hemisphere puts that
    hemisphere's cortical parcels in scope whether or not the table actually
    lists them. Subtracting the regions produced from this set is what
    separates a region whose source was never read from one whose source was
    read and stayed silent — two facts that look identical in a row count.

    Args:
        parsed: Successfully parsed tables belonging to a single session.

    Returns:
        Canonical region names covered by the tables present.
    """
    scope: set[str] = set()
    for record in parsed:
        if record.table_type is StatsTableType.ASEG:
            scope.update(
                region_key(structure, hemisphere)
                for structure in SUBCORTICAL_REGIONS
                for hemisphere in LATERAL_HEMISPHERES
            )
        elif record.hemisphere in {"lh", "rh"}:
            hemisphere = Hemisphere(record.hemisphere)
            scope.update(region_key(structure, hemisphere) for structure in CORTICAL_REGIONS)
    return scope


def session_globals(parsed: list[ParsedStatsFile]) -> dict[str, Any]:
    """Hoist the whole-session measures that live in the aseg header.

    Hole counts and eTIV describe the session rather than any one region, so
    they are lifted onto every row of it.

    Args:
        parsed: Successfully parsed tables belonging to a single session.

    Returns:
        ``etiv``, ``surface_holes_lh``, ``surface_holes_rh``, and the first
        FreeSurfer version observed. Hole counts stay ``None`` when the version
        does not report them — never 0, which would claim a flawless surface
        (§2.2).
    """
    globals_: dict[str, Any] = {
        "etiv": None,
        "surface_holes_lh": None,
        "surface_holes_rh": None,
        "freesurfer_version": None,
    }
    for record in parsed:
        if globals_["freesurfer_version"] is None:
            globals_["freesurfer_version"] = record.freesurfer_version
        if record.table_type is StatsTableType.ASEG:
            globals_["etiv"] = record.etiv
            globals_["surface_holes_lh"] = record.surface_holes_lh
            globals_["surface_holes_rh"] = record.surface_holes_rh
    return globals_
