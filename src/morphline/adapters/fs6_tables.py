"""Reader for pre-aggregated FreeSurfer 6 tables (BUILD_PLAN §1.3).

**This is not a `DatasetAdapter` and must never become one.** It deliberately
does not implement the protocol and is not registered in
:func:`morphline.adapters.build_adapter`, so no configuration can point the
pipeline at these tables.

The reason is the distinction §1.3 exists to draw. These files are
``asegstats2table`` / ``aparcstats2table`` output — one row per subject, one
column per structure — produced by someone else's aggregation run. They bypass
:class:`~morphline.parsers.freesurfer.FreeSurferStatsParser` entirely. A
pipeline whose ingestion claim is "we parse FreeSurfer's native output" cannot
demonstrate that against a table that was already aggregated, and wiring these
in as an adapter would make that confusion available to anyone reading a config
file.

What they *are* good for is the independent cross-check: morphline's per-subject
FreeSurfer 5.1 numbers, aggregated, against numbers a different program derived
from a different FreeSurfer release. That is the only external validation of the
ingested values in the repo — everything else compares morphline to morphline or
to truth morphline generated.

Column vocabulary is shared with the real parser rather than re-derived::

    aseg_table.tsv                 Left-Thalamus-Proper   -> ASEG_STRUCT_MAP
    lh.aparc_table_thickness.tsv   lh_entorhinal_thickness -> APARC_STRUCT_MAP

Two things this module refuses to do. It does not read the ``aparc.a2009s``
tables, which parcellate the same cortex differently — comparing a Destrieux
"entorhinal" against a Desikan-Killiany one would report a definitional
difference as a version effect. And it does not reconcile subject identifiers,
because there is nothing to reconcile: the identifiers match exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from morphline.adapters.freesurfer_regions import APARC_STRUCT_MAP, ASEG_STRUCT_MAP
from morphline.regions import region_key
from morphline.schema import Hemisphere, MeasureType

#: Subcortical volumes, one row per subject.
ASEG_TABLE: Final = "stats/aseg_table.tsv"

#: Cortical thickness per hemisphere, matching the measure the v1 region set
#: uses for cortical structures.
APARC_THICKNESS_TABLES: Final[dict[Hemisphere, str]] = {
    Hemisphere.LEFT: "stats/lh.aparc_table_thickness.tsv",
    Hemisphere.RIGHT: "stats/rh.aparc_table_thickness.tsv",
}

#: The deposit's own name for the subject-identifier column, which differs per
#: table and is not a structure.
_ID_COLUMNS: Final = frozenset({"Measure:volume", "lh.aparc.thickness", "rh.aparc.thickness"})

#: Release the tables were derived from. Knowledge about the download, not
#: about any file — the same rule ``dataset_version`` follows.
DATASET_VERSION: Final = "dfsp-spirit-freesurfer-6"


def _subject_column(frame: pd.DataFrame) -> str:
    """Return the column holding subject identifiers.

    Args:
        frame: A loaded table.

    Returns:
        The column name.

    Raises:
        ValueError: If no known identifier column is present.
    """
    for column in frame.columns:
        if str(column) in _ID_COLUMNS:
            return str(column)
    raise ValueError(
        f"no subject-identifier column found; saw {list(frame.columns)[:5]}. "
        f"Expected one of {sorted(_ID_COLUMNS)}."
    )


def _read_aseg(path: Path) -> pd.DataFrame:
    """Read subcortical volumes into long format."""
    frame = pd.read_csv(path, sep="\t")
    subject = _subject_column(frame)

    rows: list[dict[str, object]] = []
    for struct_name, (structure, hemisphere) in ASEG_STRUCT_MAP.items():
        if struct_name not in frame.columns:
            continue
        for identifier, value in zip(frame[subject], frame[struct_name], strict=True):
            rows.append(
                {
                    "subject_id": str(identifier),
                    "region": region_key(structure, hemisphere),
                    "measure_type": str(MeasureType.VOLUME),
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def _read_aparc_thickness(path: Path, hemisphere: Hemisphere) -> pd.DataFrame:
    """Read cortical thickness for one hemisphere into long format.

    Columns are named ``{hemi}_{structure}_thickness``; the structure segment is
    the same token the parser sees as a ``StructName``, so the canonical mapping
    is shared rather than duplicated.
    """
    frame = pd.read_csv(path, sep="\t")
    subject = _subject_column(frame)
    prefix, suffix = f"{hemisphere.value}_", "_thickness"

    rows: list[dict[str, object]] = []
    for column in frame.columns:
        name = str(column)
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        struct_name = name[len(prefix) : -len(suffix)]
        structure = APARC_STRUCT_MAP.get(struct_name)
        if structure is None:
            continue
        for identifier, value in zip(frame[subject], frame[column], strict=True):
            rows.append(
                {
                    "subject_id": str(identifier),
                    "region": region_key(structure, hemisphere),
                    "measure_type": str(MeasureType.THICKNESS),
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def read_fs6_tables(root: Path | str) -> pd.DataFrame:
    """Load the FS6 aggregate tables into the canonical long shape.

    Long format with the same ``subject_id`` / ``region`` / ``measure_type`` /
    ``value`` keys the canonical schema uses, so a comparison is a join rather
    than a translation. It is deliberately *not* a canonical frame: these rows
    carry no provenance, no session, and no QC verdict, because no file was
    parsed to produce them.

    Args:
        root: The cloned deposit, from ``scripts/fetch_abide_fs6.sh``.

    Returns:
        One row per subject × region × measure, restricted to the v1 region set.

    Raises:
        FileNotFoundError: If a required table is absent.
    """
    base = Path(root)
    frames: list[pd.DataFrame] = []

    aseg = base / ASEG_TABLE
    if not aseg.is_file():
        raise FileNotFoundError(f"missing FS6 table: {aseg}")
    frames.append(_read_aseg(aseg))

    for hemisphere, relative in APARC_THICKNESS_TABLES.items():
        path = base / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing FS6 table: {path}")
        frames.append(_read_aparc_thickness(path, hemisphere))

    combined = pd.concat(frames, ignore_index=True)
    combined["source"] = DATASET_VERSION
    return combined
