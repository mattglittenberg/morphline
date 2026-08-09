"""Synthetic adapter tests, including the §1.5 traceability requirement."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_fixture_config
from morphline.adapters import SyntheticAdapter, build_adapter
from morphline.config import DatasetConfig, PlantedSpec, Regime
from morphline.fixtures import write_fixtures
from morphline.regions import V1_TEST_COUNT
from morphline.stages.ingest import ingest


def test_discover_finds_every_written_session(tmp_path: Path) -> None:
    cfg = make_fixture_config(
        planted=PlantedSpec(
            missing_acquisition_fraction=0.0,
            missing_derivative_fraction=0.0,
            malformed_file_fraction=0.0,
        )
    )
    root = tmp_path / "fx"
    write_fixtures(cfg, root)
    sessions = list(SyntheticAdapter(root).discover())
    assert len(sessions) == 12 * 3
    assert all(len(s.stats_files) == 3 for s in sessions)


def test_sessions_without_derivatives_are_still_discovered(tmp_path: Path) -> None:
    """An acquired session with no usable output is a different kind of loss
    from a session that never happened, and both must stay visible."""
    cfg = make_fixture_config(
        seed=20260800,
        planted=PlantedSpec(
            missing_acquisition_fraction=0.0,
            missing_derivative_fraction=0.4,
            malformed_file_fraction=0.0,
        ),
    )
    root = tmp_path / "fx"
    write_fixtures(cfg, root)
    sessions = list(SyntheticAdapter(root).discover())
    empty = [s for s in sessions if not s.stats_files]
    assert empty, "expected sessions present on disk with no stats files"


def test_canonical_rows_cover_the_v1_region_set(fixture_tree: Path) -> None:
    result = ingest(SyntheticAdapter(fixture_tree))
    assert result.observations["region"].nunique() == V1_TEST_COUNT


def test_every_row_is_traceable_to_a_source_file(fixture_tree: Path) -> None:
    """§1.5: any row must be traceable back to the file it came from.

    If you cannot answer "which files produced this coefficient", the
    provenance design has failed.
    """
    result = ingest(SyntheticAdapter(fixture_tree))
    obs = result.observations
    assert obs["source_file"].notna().all()
    assert obs["source_file_checksum"].notna().all()
    for path in obs["source_file"].unique():
        assert Path(path).is_file(), f"source_file does not exist on disk: {path}"


def test_checksum_matches_the_named_file(fixture_tree: Path) -> None:
    import hashlib

    result = ingest(SyntheticAdapter(fixture_tree))
    row = result.observations.iloc[0]
    actual = hashlib.sha256(Path(row["source_file"]).read_bytes()).hexdigest()
    assert row["source_file_checksum"] == actual


def test_volumes_and_thicknesses_get_correct_units(fixture_tree: Path) -> None:
    obs = ingest(SyntheticAdapter(fixture_tree)).observations
    volumes = obs[obs["measure_type"] == "volume"]
    thickness = obs[obs["measure_type"] == "thickness"]
    assert set(volumes["unit"]) == {"mm^3"}
    assert set(thickness["unit"]) == {"mm"}
    # Sanity: a thickness must not be in the thousands.
    assert thickness["value"].max() < 10
    assert volumes["value"].min() > 100


def test_hemispheres_are_resolved(fixture_tree: Path) -> None:
    obs = ingest(SyntheticAdapter(fixture_tree)).observations
    assert set(obs["hemisphere"]) == {"lh", "rh"}
    lh = obs[obs["hemisphere"] == "lh"]["region"].unique()
    assert all(r.startswith("lh-") for r in lh)


def test_covariates_are_attached(fixture_tree: Path) -> None:
    obs = ingest(SyntheticAdapter(fixture_tree)).observations
    for column in (
        "age_baseline",
        "age_at_session",
        "sex",
        "dx_baseline",
        "etiv_baseline",
        "time_from_baseline_years",
        "site",
    ):
        assert obs[column].notna().all(), f"{column} missing after canonicalization"


def test_baseline_covariates_are_constant_within_subject(fixture_tree: Path) -> None:
    """`etiv_baseline` and `dx_baseline` are fixed per subject by design (§2.5.1)."""
    obs = ingest(SyntheticAdapter(fixture_tree)).observations
    for column in ("etiv_baseline", "age_baseline", "dx_baseline", "sex"):
        counts = obs.groupby("subject_id")[column].nunique()
        assert (counts == 1).all(), f"{column} varies within subject"


def test_regime_b_assigns_acquisition_site_not_enrolling_site(tmp_path: Path) -> None:
    """Attributing later sessions to the enrolling site would erase the very
    confound Regime B exists to create."""
    root = tmp_path / "fx"
    write_fixtures(make_fixture_config(regime=Regime.B_CONFOUNDED, n_sessions=4), root)
    obs = ingest(SyntheticAdapter(root)).observations
    late = obs[obs["time_from_baseline_years"] > 1.5]
    assert late["site"].nunique() == 1, "late sessions should be on one scanner"


def test_no_duplicate_observation_keys(fixture_tree: Path) -> None:
    """§5.2 output sanity: no duplicated subject x session x region rows."""
    obs = ingest(SyntheticAdapter(fixture_tree)).observations
    dupes = obs.duplicated(subset=["subject_id", "session_id", "region", "measure_type"])
    assert not dupes.any()


def test_freesurfer_versions_are_observed_not_assumed(fixture_tree: Path) -> None:
    """Provenance reports versions found in the data, not configured ones."""
    result = ingest(SyntheticAdapter(fixture_tree))
    assert result.freesurfer_versions
    assert all(v[0].isdigit() for v in result.freesurfer_versions)


def test_fs53_sessions_have_null_surface_holes(tmp_path: Path) -> None:
    """The null-not-zero rule must survive canonicalization, not just parsing."""
    cfg = make_fixture_config(freesurfer_version_mix={"5.3.0": 1.0})
    root = tmp_path / "fx"
    write_fixtures(cfg, root)
    obs = ingest(SyntheticAdapter(root)).observations
    assert obs["surface_holes_lh"].isna().all()
    assert not (obs["surface_holes_lh"] == 0).any()


def test_fs6_sessions_have_populated_surface_holes(tmp_path: Path) -> None:
    cfg = make_fixture_config(freesurfer_version_mix={"6.0.0": 1.0})
    root = tmp_path / "fx"
    write_fixtures(cfg, root)
    obs = ingest(SyntheticAdapter(root)).observations
    assert obs["surface_holes_lh"].notna().all()


def test_build_adapter_rejects_unknown_adapter() -> None:
    cfg = DatasetConfig()
    object.__setattr__(cfg, "adapter", "nonexistent")
    with pytest.raises(ValueError, match="unknown adapter"):
        build_adapter(cfg, ".")


def test_empty_tree_yields_empty_observations(tmp_path: Path) -> None:
    result = ingest(SyntheticAdapter(tmp_path / "nothing"))
    assert result.observations.empty
    assert result.files_discovered == 0
