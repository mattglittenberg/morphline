"""FreeSurfer ``StructName`` → canonical region mapping.

Lives in the ingestion layer because it is FreeSurfer-specific vocabulary.
Every adapter that ingests FreeSurfer derivatives shares it; nothing
downstream of ingestion may import it (§1.4).
"""

from __future__ import annotations

from typing import Final

from morphline.regions import CORTICAL_REGIONS, SUBCORTICAL_REGIONS
from morphline.schema import Hemisphere

#: aseg ``StructName`` → (canonical structure, hemisphere).
ASEG_STRUCT_MAP: Final[dict[str, tuple[str, Hemisphere]]] = {
    "Left-Hippocampus": ("hippocampus", Hemisphere.LEFT),
    "Right-Hippocampus": ("hippocampus", Hemisphere.RIGHT),
    "Left-Amygdala": ("amygdala", Hemisphere.LEFT),
    "Right-Amygdala": ("amygdala", Hemisphere.RIGHT),
    "Left-Thalamus-Proper": ("thalamus", Hemisphere.LEFT),
    "Right-Thalamus-Proper": ("thalamus", Hemisphere.RIGHT),
    # FreeSurfer 7 dropped the "-Proper" suffix; both spellings occur in the
    # wild and both must map to the same canonical structure.
    "Left-Thalamus": ("thalamus", Hemisphere.LEFT),
    "Right-Thalamus": ("thalamus", Hemisphere.RIGHT),
    "Left-Caudate": ("caudate", Hemisphere.LEFT),
    "Right-Caudate": ("caudate", Hemisphere.RIGHT),
    "Left-Putamen": ("putamen", Hemisphere.LEFT),
    "Right-Putamen": ("putamen", Hemisphere.RIGHT),
    "Left-Lateral-Ventricle": ("lateral-ventricle", Hemisphere.LEFT),
    "Right-Lateral-Ventricle": ("lateral-ventricle", Hemisphere.RIGHT),
    "Left-Inf-Lat-Vent": ("inferior-lateral-ventricle", Hemisphere.LEFT),
    "Right-Inf-Lat-Vent": ("inferior-lateral-ventricle", Hemisphere.RIGHT),
}

#: aparc ``StructName`` → canonical structure. Hemisphere comes from the file.
APARC_STRUCT_MAP: Final[dict[str, str]] = {
    "entorhinal": "entorhinal",
    "parahippocampal": "parahippocampal",
    "inferiorparietal": "inferiorparietal",
    "precuneus": "precuneus",
    "posteriorcingulate": "posteriorcingulate",
    "middletemporal": "middletemporal",
    "superiorfrontal": "superiorfrontal",
}

#: Reverse lookups, used by the fixture writer to emit realistic files.
CANONICAL_TO_ASEG: Final[dict[tuple[str, Hemisphere], str]] = {
    ("hippocampus", Hemisphere.LEFT): "Left-Hippocampus",
    ("hippocampus", Hemisphere.RIGHT): "Right-Hippocampus",
    ("amygdala", Hemisphere.LEFT): "Left-Amygdala",
    ("amygdala", Hemisphere.RIGHT): "Right-Amygdala",
    ("thalamus", Hemisphere.LEFT): "Left-Thalamus-Proper",
    ("thalamus", Hemisphere.RIGHT): "Right-Thalamus-Proper",
    ("caudate", Hemisphere.LEFT): "Left-Caudate",
    ("caudate", Hemisphere.RIGHT): "Right-Caudate",
    ("putamen", Hemisphere.LEFT): "Left-Putamen",
    ("putamen", Hemisphere.RIGHT): "Right-Putamen",
    ("lateral-ventricle", Hemisphere.LEFT): "Left-Lateral-Ventricle",
    ("lateral-ventricle", Hemisphere.RIGHT): "Right-Lateral-Ventricle",
    ("inferior-lateral-ventricle", Hemisphere.LEFT): "Left-Inf-Lat-Vent",
    ("inferior-lateral-ventricle", Hemisphere.RIGHT): "Right-Inf-Lat-Vent",
}

CANONICAL_TO_APARC: Final[dict[str, str]] = {v: k for k, v in APARC_STRUCT_MAP.items()}


def _assert_coverage() -> None:
    """Guard that the maps cover the v1 region set exactly."""
    covered_subcortical = {structure for structure, _ in ASEG_STRUCT_MAP.values()}
    missing = set(SUBCORTICAL_REGIONS) - covered_subcortical
    if missing:
        raise RuntimeError(f"aseg map is missing v1 structures: {sorted(missing)}")
    missing_cortical = set(CORTICAL_REGIONS) - set(APARC_STRUCT_MAP.values())
    if missing_cortical:
        raise RuntimeError(f"aparc map is missing v1 structures: {sorted(missing_cortical)}")


_assert_coverage()
