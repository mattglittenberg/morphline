"""Shared test fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from morphline.config import (
    AnalysisConfig,
    FixtureConfig,
    PlantedSpec,
    QCConfig,
    Regime,
    SiteSpec,
    load_config,
)

if TYPE_CHECKING:
    from morphline.fixtures.truth import GroundTruth

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FixtureBundle:
    """A written fixture tree beside the truth that generated it.

    Recovery tests need both halves and need them consistent: the injected
    effects are only meaningful against the observations they produced. Keeping
    them together stops a test from pairing one run's estimates with another
    run's truth, which fails in a way that looks like estimator bias.

    Attributes:
        root: The written fixture tree.
        truth: Injected ground truth, returned by ``write_fixtures`` directly
            rather than re-read from ``truth/ground_truth.parquet``.
        observations: QC-annotated canonical observations.
        ground_truth: Per-observation truth, joinable to ``observations`` on
            ``(subject_id, session_id, region)``.
    """

    root: Path
    truth: GroundTruth
    observations: pd.DataFrame
    ground_truth: pd.DataFrame


def build_fixture_bundle(
    config: FixtureConfig,
    root: Path,
    qc: QCConfig | None = None,
    analysis: AnalysisConfig | None = None,
) -> FixtureBundle:
    """Generate a fixture tree, ingest it, and apply QC.

    Args:
        config: Fixture generator configuration.
        root: Destination for the tree.
        qc: QC configuration. Defaults to :class:`QCConfig`.
        analysis: Analysis configuration. Defaults to :class:`AnalysisConfig`.

    Returns:
        The tree, the injected truth, and the QC-annotated observations.
    """
    from morphline.adapters import SyntheticAdapter
    from morphline.fixtures import write_fixtures
    from morphline.stages.ingest import ingest
    from morphline.stages.qc import apply_qc

    truth = write_fixtures(config, root)
    observations = apply_qc(
        ingest(SyntheticAdapter(root)).observations,
        qc or QCConfig(),
        analysis or AnalysisConfig(),
    )

    ground_truth = pd.read_parquet(root / "truth" / "ground_truth.parquet")
    ground_truth["session_id"] = ground_truth["session_id"].astype(str)
    ground_truth["subject_id"] = ground_truth["subject_id"].astype(str)

    return FixtureBundle(
        root=root, truth=truth, observations=observations, ground_truth=ground_truth
    )


def make_fixture_config(**overrides: object) -> FixtureConfig:
    """Build a small fixture config, with overrides applied."""
    base: dict[str, object] = {
        "seed": 1234,
        "n_sessions": 3,
        "sites": (
            SiteSpec(name="site-a", n_subjects=6, additive_effect=0.02, multiplicative_effect=1.03),
            SiteSpec(
                name="site-b", n_subjects=6, additive_effect=-0.03, multiplicative_effect=0.97
            ),
        ),
    }
    base.update(overrides)
    return FixtureConfig(**base)  # type: ignore[arg-type]


@pytest.fixture
def clean_fixture_config() -> FixtureConfig:
    """A fixture config with no planted problems at all.

    Used by specificity tests: an observation set with nothing wrong in it is
    the only way to measure a false-positive rate.
    """
    return make_fixture_config(
        planted=PlantedSpec(
            qc_high_holes_fraction=0.0,
            qc_bad_etiv_fraction=0.0,
            qc_extreme_change_fraction=0.0,
            missing_acquisition_fraction=0.0,
            missing_derivative_fraction=0.0,
            malformed_file_fraction=0.0,
        )
    )


@pytest.fixture
def lossy_fixture_config() -> FixtureConfig:
    """A fixture config that reliably plants losses of every kind."""
    return make_fixture_config(
        seed=20260800,
        sites=(
            SiteSpec(name="site-a", n_subjects=8, additive_effect=0.02, multiplicative_effect=1.03),
            SiteSpec(
                name="site-b", n_subjects=8, additive_effect=-0.03, multiplicative_effect=0.97
            ),
        ),
        planted=PlantedSpec(
            missing_acquisition_fraction=0.10,
            missing_derivative_fraction=0.08,
            malformed_file_fraction=0.08,
        ),
    )


@pytest.fixture
def confounded_fixture_config() -> FixtureConfig:
    """Regime B: site confounded with time."""
    return make_fixture_config(regime=Regime.B_CONFOUNDED, n_sessions=4)


@pytest.fixture
def fixture_tree(tmp_path: Path, clean_fixture_config: FixtureConfig) -> Path:
    """A written fixture tree with no planted problems."""
    from morphline.fixtures import write_fixtures

    root = tmp_path / "fx"
    write_fixtures(clean_fixture_config, root)
    return root


@pytest.fixture
def test_config_path() -> Path:
    """Path to the committed CI configuration."""
    return REPO_ROOT / "config" / "test.yaml"


@pytest.fixture
def test_run_config() -> object:
    """The committed CI configuration, loaded."""
    return load_config(REPO_ROOT / "config" / "test.yaml")
