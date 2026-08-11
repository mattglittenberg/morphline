"""Dataset adapters. Dataset conventions only — no file-format parsing (§1.4)."""

from __future__ import annotations

from pathlib import Path

from morphline.adapters.abide_pcp import AbidePcpAdapter
from morphline.adapters.base import DatasetAdapter, SubjectSession
from morphline.adapters.filter import SubjectFilter
from morphline.adapters.synthetic import SyntheticAdapter
from morphline.config import DatasetConfig

__all__ = [
    "AbidePcpAdapter",
    "DatasetAdapter",
    "SubjectFilter",
    "SubjectSession",
    "SyntheticAdapter",
    "build_adapter",
]


def build_adapter(dataset_config: DatasetConfig, root: Path | str) -> DatasetAdapter:
    """Construct the adapter named by a dataset configuration.

    Args:
        dataset_config: Dataset identity and adapter selection.
        root: Dataset root directory.

    Returns:
        The configured adapter.

    Raises:
        ValueError: If the adapter name is not registered.
    """
    if dataset_config.adapter == "synthetic":
        return SyntheticAdapter(root, dataset_config)
    if dataset_config.adapter == "abide-pcp":
        return AbidePcpAdapter(
            root,
            dataset_config,
            phenotypic_csv=dataset_config.phenotypic_csv,
            collapse_site_subsample=dataset_config.collapse_site_subsample,
        )
    raise ValueError(f"unknown adapter: {dataset_config.adapter!r}")
