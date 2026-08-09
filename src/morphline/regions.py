"""Canonical region vocabulary and the v1 hypothesis-driven region set.

Canonical region names only — no FreeSurfer strings live here, so downstream
stages can import this without violating the §1.4 boundary. The mapping from
FreeSurfer ``StructName`` values onto these names belongs to the ingestion
layer (:mod:`morphline.adapters.freesurfer_regions`).

The v1 default is 10–20 hypothesis-driven regions (§2.5.2), not an emergency
cut. The portfolio value is the ingestion → accounting → QC → harmonization →
modeling architecture, not the region count, and a focused set reduces
convergence failures, multiple-testing burden, and interpretation load while
demonstrating exactly the same engineering.
"""

from __future__ import annotations

from typing import Final

from morphline.schema import Hemisphere, MeasureType

#: Subcortical structures from the aseg table, AD/aging oriented (§2.5.2).
SUBCORTICAL_REGIONS: Final = (
    "hippocampus",
    "amygdala",
    "thalamus",
    "caudate",
    "putamen",
    "lateral-ventricle",
    "inferior-lateral-ventricle",
)

#: Cortical parcels from the aparc table, AD/aging oriented (§2.5.2).
CORTICAL_REGIONS: Final = (
    "entorhinal",
    "parahippocampal",
    "inferiorparietal",
    "precuneus",
    "posteriorcingulate",
    "middletemporal",
    "superiorfrontal",
)

#: 14 structures, bilateral.
V1_STRUCTURES: Final = (*SUBCORTICAL_REGIONS, *CORTICAL_REGIONS)

#: Measure type reported for each structure family.
MEASURE_FOR_STRUCTURE: Final = {
    **{r: MeasureType.VOLUME for r in SUBCORTICAL_REGIONS},
    **{r: MeasureType.THICKNESS for r in CORTICAL_REGIONS},
}

LATERAL_HEMISPHERES: Final = (Hemisphere.LEFT, Hemisphere.RIGHT)


def region_key(structure: str, hemisphere: Hemisphere | str) -> str:
    """Build the canonical region name for a structure in one hemisphere.

    Args:
        structure: Canonical structure name, e.g. ``"hippocampus"``.
        hemisphere: Hemisphere the structure belongs to.

    Returns:
        A name of the form ``"lh-hippocampus"``.
    """
    hemi = hemisphere.value if isinstance(hemisphere, Hemisphere) else hemisphere
    return f"{hemi}-{structure}"


def v1_region_set() -> tuple[tuple[str, str, MeasureType], ...]:
    """Return the v1 region set as ``(region, hemisphere, measure_type)`` triples.

    Returns:
        28 entries — 14 structures × 2 hemispheres.

    Note:
        Regions and *tests* are counted separately in the report. Hemispheres
        are independent tests and must be counted in the multiplicity family,
        so a "14-region analysis" is a 28-test family (§2.5.3). Stating the
        wrong one understates the correction burden.
    """
    return tuple(
        (region_key(structure, hemi), hemi.value, MEASURE_FOR_STRUCTURE[structure])
        for structure in V1_STRUCTURES
        for hemi in LATERAL_HEMISPHERES
    )


#: Size of the primary multiplicity family: the ``time:dx_baseline``
#: coefficient across the v1 region set (§2.5.3).
V1_TEST_COUNT: Final = len(V1_STRUCTURES) * len(LATERAL_HEMISPHERES)
