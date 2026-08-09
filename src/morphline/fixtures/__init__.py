"""Synthetic fixture generation — the primary development substrate (§0.2)."""

from __future__ import annotations

from morphline.fixtures.generator import write_fixtures
from morphline.fixtures.truth import BASE_VALUES, GroundTruth, generate_ground_truth

__all__ = ["BASE_VALUES", "GroundTruth", "generate_ground_truth", "write_fixtures"]
