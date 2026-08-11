"""Canonical observation schema — the contract between pipeline stages.

Implements BUILD_PLAN.md §1.5. Long format: one row per
``subject × session × region × measure``. Parquet is the persistence boundary
between every stage (§2.7), and this module is the single source of truth for
what those Parquet files contain.

Traceability requirement (§1.5): every row carries ``source_file`` and
``source_file_checksum``, so any coefficient in the final results table can be
walked back to the specific files that produced it. If that walk is not
possible, the provenance design has failed.

Downstream stages import :func:`read_canonical` / :func:`write_canonical` and
the column constants from here. They must never import the parser, touch a
``.stats`` file, or contain a FreeSurfer-specific string — see
``tests/test_architecture_boundary.py``, which enforces this.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION: Final = "1.0.0"


class MeasureType(StrEnum):
    """Kinds of morphometric measurement carried in the ``value`` column."""

    VOLUME = "volume"
    THICKNESS = "thickness"
    AREA = "area"
    CURVATURE = "curvature"


class Hemisphere(StrEnum):
    """Hemisphere a region belongs to.

    ``BILATERAL`` covers structures that are not lateralised (e.g. brainstem,
    third ventricle) and whole-brain global measures.
    """

    LEFT = "lh"
    RIGHT = "rh"
    BILATERAL = "bilateral"


class QCStatus(StrEnum):
    """Three-level QC status (§2.4.1).

    QC *identifies and classifies*; the analysis layer *decides* what to
    include. Keeping those separate means review-worthy observations stay
    visible instead of vanishing into an exclusion list.
    """

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class MissingnessCause(StrEnum):
    """Why an expected observation is absent (§2.5.4).

    The distinction matters because the causes carry different implications
    for the missing-at-random assumption the mixed model rests on.
    """

    ACQUISITION = "missing_acquisition"
    DERIVATIVE = "missing_derivative"
    EXCLUDED_QC = "excluded_qc"


class ModelExclusion(StrEnum):
    """Why a QC-passing observation did not reach the model.

    Deliberately separate from :class:`MissingnessCause`, because nothing is
    missing here: the observation exists and passed QC, it simply was not an
    input to a fit. The distinction matters because the reasons have opposite
    implications. A region outside the fitted set is a scope decision. A row
    dropped for incomplete covariates is a data limitation that biases the
    remaining sample if the incompleteness is not random.

    Collapsing them into one label lets the funnel reconcile while reporting a
    cause that is false — which is worse than not reconciling, since it looks
    like an answer.
    """

    OUTSIDE_REGION_SET = "outside_modeled_region_set"
    INCOMPLETE_COVARIATES = "incomplete_model_covariates"
    #: The count is known but the model's per-region inputs were not supplied,
    #: so no cause can be established. Naming the ignorance beats guessing.
    UNATTRIBUTED = "not_modeled_cause_unavailable"


# --- Column groups, mirroring the §1.5 layout -------------------------------

IDENTITY_COLUMNS: Final = (
    "dataset",
    "dataset_version",
    "subject_id",
    "session_id",
    "time_from_baseline_years",
)

ACQUISITION_COLUMNS: Final = (
    "site",
    "scanner_manufacturer",
    "scanner_model",
    "field_strength_tesla",
)

MEASUREMENT_COLUMNS: Final = (
    "region",
    "hemisphere",
    "measure_type",
    "value",
    "unit",
)

COVARIATE_COLUMNS: Final = (
    "age_at_session",
    "age_baseline",
    "sex",
    "dx_baseline",
    "dx_at_session",
)

GLOBAL_COLUMNS: Final = (
    "etiv",
    "etiv_baseline",
    "surface_holes_lh",
    "surface_holes_rh",
)

PROVENANCE_COLUMNS: Final = (
    "source_file",
    "source_file_checksum",
    "freesurfer_version",
    "parser_version",
    "ingested_at",
)

CANONICAL_COLUMNS: Final = (
    *IDENTITY_COLUMNS,
    *ACQUISITION_COLUMNS,
    *MEASUREMENT_COLUMNS,
    *COVARIATE_COLUMNS,
    *GLOBAL_COLUMNS,
    *PROVENANCE_COLUMNS,
)

#: Uniquely identifies an observation. Duplicates here are a bug (§5.2).
OBSERVATION_KEY: Final = ("dataset", "subject_id", "session_id", "region", "measure_type")

# --- QC columns -------------------------------------------------------------
# Added by the QC stage (§2.4.1). The full field shape ships in v0.1.0 even
# though the stub only ever emits PASS, so week 3 changes logic, not schema.

QC_COLUMNS: Final = (
    "qc_status",
    "qc_flags",
    "qc_score",
    "qc_notes",
    "analysis_included",
)


CANONICAL_SCHEMA: Final = pa.schema(
    [
        # identity
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("dataset_version", pa.string(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("time_from_baseline_years", pa.float64()),
        # acquisition
        pa.field("site", pa.string()),
        pa.field("scanner_manufacturer", pa.string()),
        pa.field("scanner_model", pa.string()),
        pa.field("field_strength_tesla", pa.float64()),
        # measurement
        pa.field("region", pa.string(), nullable=False),
        pa.field("hemisphere", pa.string()),
        pa.field("measure_type", pa.string(), nullable=False),
        pa.field("value", pa.float64()),
        pa.field("unit", pa.string()),
        # covariates
        pa.field("age_at_session", pa.float64()),
        pa.field("age_baseline", pa.float64()),
        pa.field("sex", pa.string()),
        pa.field("dx_baseline", pa.string()),
        pa.field("dx_at_session", pa.string()),
        # global measures
        pa.field("etiv", pa.float64()),
        pa.field("etiv_baseline", pa.float64()),
        # Nullable on purpose: FreeSurfer 5.3 does not emit hole counts, and a
        # default of 0 would imply a flawless surface (§2.2).
        pa.field("surface_holes_lh", pa.float64()),
        pa.field("surface_holes_rh", pa.float64()),
        # provenance
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_file_checksum", pa.string(), nullable=False),
        pa.field("freesurfer_version", pa.string()),
        pa.field("parser_version", pa.string(), nullable=False),
        pa.field("ingested_at", pa.string(), nullable=False),
        # QC (populated by the QC stage; null before it runs)
        pa.field("qc_status", pa.string()),
        pa.field("qc_flags", pa.list_(pa.string())),
        pa.field("qc_score", pa.float64()),
        pa.field("qc_notes", pa.string()),
        pa.field("analysis_included", pa.bool_()),
    ]
)


def euler_number(surface_holes: float | None) -> float | None:
    """Derive the Euler number from a surface hole count (§2.2).

    The two quantities are routinely conflated, so the distinction is worth
    stating: ``lhSurfaceHoles`` / ``rhSurfaceHoles`` are *reported directly* in
    the ``aseg.stats`` header on FreeSurfer 6+ and are extracted verbatim. The
    Euler number is *derived* from them as ``2 - 2 * holes``, and is a
    topological measure of surface defects where more negative means more
    defects. The header's ``SurfaceHoles`` field is the summed total hole count
    and must not be used as if it were a Euler number.

    Args:
        surface_holes: Hole count for one hemisphere, or ``None`` on FreeSurfer
            versions that do not report it.

    Returns:
        The Euler number, or ``None`` if the hole count was absent. ``None``
        propagates deliberately: substituting 0 would yield a Euler number of
        2 — a flawless surface — and make the oldest data in a study look like
        the highest quality.
    """
    if surface_holes is None or pd.isna(surface_holes):
        return None
    return 2.0 - 2.0 * float(surface_holes)


def empty_canonical() -> pd.DataFrame:
    """Return an empty frame carrying the full canonical column set and dtypes."""
    return cast("pd.DataFrame", CANONICAL_SCHEMA.empty_table().to_pandas())


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing canonical columns as null and order columns canonically.

    Stages build frames incrementally, so this is the single place that decides
    what a well-formed canonical frame looks like.

    Args:
        df: A frame containing at least the non-nullable canonical columns.

    Returns:
        A copy with every canonical and QC column present, in canonical order.

    Raises:
        ValueError: If a column required to be non-null is absent entirely.
    """
    required = [f.name for f in CANONICAL_SCHEMA if not f.nullable]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"canonical frame is missing required columns: {missing_required}")

    out = df.copy()
    for field in CANONICAL_SCHEMA:
        if field.name not in out.columns:
            out[field.name] = pd.Series([None] * len(out), dtype="object")
    ordered = [f.name for f in CANONICAL_SCHEMA]
    extras = [c for c in out.columns if c not in ordered]
    return out[[*ordered, *extras]]


def validate(df: pd.DataFrame) -> None:
    """Check a canonical frame's structural invariants.

    Args:
        df: Frame to check.

    Raises:
        ValueError: If required columns are missing, required values are null,
            or the observation key is duplicated. Duplicate
            ``subject × session × region × measure`` rows are called out
            explicitly in §5.2 as an output-sanity failure.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"canonical frame is missing columns: {missing}")

    if df.empty:
        return

    for field in CANONICAL_SCHEMA:
        if not field.nullable and df[field.name].isna().any():
            n = int(df[field.name].isna().sum())
            raise ValueError(f"column {field.name!r} is non-nullable but has {n} null values")

    dupes = df.duplicated(subset=list(OBSERVATION_KEY))
    if dupes.any():
        example = df.loc[dupes, list(OBSERVATION_KEY)].iloc[0].to_dict()
        raise ValueError(
            f"{int(dupes.sum())} duplicated observation keys; first example: {example}"
        )


def write_canonical(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a canonical frame to Parquet, conforming and validating it first.

    Args:
        df: Frame to persist.
        path: Destination ``.parquet`` path; parent directories are created.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conformed = conform(df)
    validate(conformed)
    table = pa.Table.from_pandas(
        conformed[[f.name for f in CANONICAL_SCHEMA]],
        schema=CANONICAL_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, target, compression="snappy")
    return target


def read_canonical(path: Path | str) -> pd.DataFrame:
    """Read one canonical Parquet file.

    Args:
        path: Source ``.parquet`` path.

    Returns:
        The frame, with canonical dtypes restored.
    """
    return cast("pd.DataFrame", pq.read_table(Path(path), schema=CANONICAL_SCHEMA).to_pandas())


def read_canonical_many(paths: list[Path] | list[str]) -> pd.DataFrame:
    """Concatenate several canonical Parquet files into one frame.

    This is how gather steps consume ``collect()``-ed channel paths: Nextflow
    hands over file paths, and the reading happens inside the consuming
    process (§2.7). Nothing ever puts a dataframe on a channel.

    Args:
        paths: Parquet paths to concatenate.

    Returns:
        The concatenated frame, or an empty canonical frame if ``paths`` is
        empty.
    """
    if not paths:
        return empty_canonical()
    frames = [read_canonical(p) for p in paths]
    return pd.concat(frames, ignore_index=True)
