"""ABIDE PCP adapter tests (BUILD_PLAN §1.3).

The stats content here reproduces real ABIDE PCP FreeSurfer 5.1 files: no
``SurfaceHoles`` measures, intracranial volume as ``IntraCranialVol, ICV``, and
aparc header measures all sharing the short name ``Cortex``. Fabricating a
tidier FreeSurfer 6 shape would test the adapter against a dataset that does
not exist.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from morphline.adapters import AbidePcpAdapter, build_adapter
from morphline.adapters.abide_pcp import CORE_TABLES, SESSION_ID, site_from_subject_id
from morphline.config import DatasetConfig
from morphline.schema import MissingnessCause, validate, write_canonical
from morphline.stages.ingest import ingest

ASEG_FS51 = """\
# Title Segmentation Statistics
# cvs_version $Id: mri_segstats.c,v 1.75.2.2 2011/04/27 22:18:58 nicks Exp $
# subjectname {subject}
# Measure Cortex, CortexVol, Total cortical gray matter volume, 467564.035582, mm^3
# Measure IntraCranialVol, ICV, Intracranial Volume, 1549336.766171, mm^3
# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1  17  4100  4200.5  Left-Hippocampus  93.1  6.7  54  134  80
  2  53  4050  4150.2  Right-Hippocampus  92.4  6.9  53  133  80
  3  10  7300  7350.8  Left-Thalamus-Proper  98.2  8.1  60  140  80
  4  49  7280  7320.4  Right-Thalamus-Proper  97.9  8.3  59  139  80
"""

APARC_FS51 = """\
# Table of FreeSurfer cortical parcellation anatomical statistics
# cvs_version $Id: mris_anatomical_stats.c,v 1.72 2011/03/02 00:04:26 nicks Exp $
# subjectname {subject}
# hemi {hemi}
# annot aparc.annot
# Measure Cortex, NumVert, Number of Vertices, 133321, unitless
# Measure Cortex, WhiteSurfArea, White Surface Total Area, 87092.8, mm^2
# Measure Cortex, MeanThickness, Mean Thickness, 2.47864, mm
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
entorhinal  2100  1400  8200  3.3100  0.5100  0.1200  0.0300  12  2.5000
precuneus  4100  2900  11000  2.3500  0.4800  0.1100  0.0280  18  3.1000
"""

# Destrieux and exvivo tables sit alongside the core three under --all-tables.
# Both parse as aparc, and entorhinal_exvivo reports a structure the core table
# already reports.
APARC_A2009S = """\
# Table of FreeSurfer cortical parcellation anatomical statistics
# cvs_version 5.1.0
# hemi lh
# annot aparc.a2009s.annot
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
G_temp_sup-Lateral  2100  1400  8200  3.3100  0.5100  0.1200  0.0300  12  2.5000
"""

ENTORHINAL_EXVIVO = """\
# Table of FreeSurfer cortical parcellation anatomical statistics
# cvs_version 5.1.0
# hemi lh
# annot entorhinal_exvivo.label
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
entorhinal  1900  1300  7900  3.9900  0.5000  0.1200  0.0300  11  2.4000
"""


def make_subject(root: Path, subject: str, *, tables: tuple[str, ...] = CORE_TABLES) -> Path:
    """Write one ABIDE-shaped subject directory."""
    stats = root / subject / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    for name in tables:
        if name == "aseg.stats":
            body = ASEG_FS51.format(subject=subject)
        elif name.endswith("aparc.stats"):
            body = APARC_FS51.format(subject=subject, hemi=name.split(".")[0])
        elif "a2009s" in name:
            body = APARC_A2009S
        else:
            body = ENTORHINAL_EXVIVO
        (stats / name).write_text(body, encoding="utf-8")
    return stats


@pytest.fixture
def abide_root(tmp_path: Path) -> Path:
    """A small ABIDE PCP tree: two complete subjects, one empty, one lh-only."""
    root = tmp_path / "abide_pcp"
    make_subject(root, "CMU_a_0050642")
    make_subject(root, "UM_1_0050309")
    (root / "UCLA_51233" / "stats").mkdir(parents=True)
    make_subject(root, "UCLA_51244", tables=("lh.aparc.stats",))
    return root


def config() -> DatasetConfig:
    """Dataset identity for the adapter under test."""
    return DatasetConfig(name="abide-i-pcp", version="freesurfer-5.1", adapter="abide-pcp")


# -- site resolution ---------------------------------------------------------


@pytest.mark.parametrize(
    ("subject_id", "expected"),
    [
        ("CMU_a_0050642", "CMU_a"),
        ("UM_1_0050309", "UM_1"),
        ("Leuven_2_0050730", "Leuven_2"),
        ("MaxMun_d_0051350", "MaxMun_d"),
        ("UCLA_51233", "UCLA"),
        ("Pitt_0050003", "Pitt"),
        ("NoNumericSuffix", None),
    ],
)
def test_site_derived_from_subject_id(subject_id: str, expected: str | None) -> None:
    """Site is the subject ID with its numeric participant suffix removed."""
    assert site_from_subject_id(subject_id) == expected


@pytest.mark.parametrize(
    ("subject_id", "expected"),
    [
        ("CMU_a_0050642", "CMU"),
        ("UM_1_0050309", "UM"),
        ("MaxMun_d_0051350", "MaxMun"),
        ("UCLA_51233", "UCLA"),
        ("Pitt_0050003", "Pitt"),
    ],
)
def test_collapsing_subsamples_merges_sites(subject_id: str, expected: str) -> None:
    """Sub-samples collapse to their institution when asked (§2.3.3 trade-off)."""
    assert site_from_subject_id(subject_id, collapse_subsample=True) == expected


def test_subsample_split_and_collapse_differ_in_batch_count() -> None:
    """The trade-off is real: collapsing genuinely reduces the batch count."""
    subjects = ["CMU_a_1", "CMU_b_2", "MaxMun_a_3", "MaxMun_b_4", "Pitt_5"]
    split = {site_from_subject_id(s) for s in subjects}
    collapsed = {site_from_subject_id(s, collapse_subsample=True) for s in subjects}
    assert len(split) == 5
    assert len(collapsed) == 3


# -- discovery ---------------------------------------------------------------


def test_discovery_yields_every_subject_including_the_empty_one(abide_root: Path) -> None:
    """An empty stats dir must still be discovered, or the funnel loses it."""
    sessions = list(AbidePcpAdapter(abide_root, config()).discover())
    assert [s.subject_id for s in sessions] == [
        "CMU_a_0050642",
        "UCLA_51233",
        "UCLA_51244",
        "UM_1_0050309",
    ]
    by_id = {s.subject_id: s for s in sessions}
    assert by_id["UCLA_51233"].stats_files == ()
    assert len(by_id["UCLA_51244"].stats_files) == 1
    assert len(by_id["CMU_a_0050642"].stats_files) == 3
    assert all(s.session_id == SESSION_ID for s in sessions)


def test_discovery_ignores_non_core_tables(tmp_path: Path) -> None:
    """Only the allowlisted three tables are read, by exact filename."""
    root = tmp_path / "abide_pcp"
    make_subject(
        root,
        "CMU_a_0050642",
        tables=(*CORE_TABLES, "lh.aparc.a2009s.stats", "lh.entorhinal_exvivo.stats"),
    )
    session = next(iter(AbidePcpAdapter(root, config()).discover()))
    assert sorted(p.name for p in session.stats_files) == sorted(CORE_TABLES)


def test_missing_root_discovers_nothing(tmp_path: Path) -> None:
    """A root that does not exist yields no sessions rather than raising."""
    assert list(AbidePcpAdapter(tmp_path / "absent", config()).discover()) == []


# -- canonicalization --------------------------------------------------------


def test_canonical_frame_is_valid_and_cross_sectional(abide_root: Path) -> None:
    """The adapter's output satisfies the canonical contract (§1.5)."""
    result = ingest(AbidePcpAdapter(abide_root, config()))
    validate(result.observations)
    df = result.observations
    assert not df.empty
    assert set(df["session_id"]) == {SESSION_ID}
    assert set(df["time_from_baseline_years"]) == {0.0}
    assert set(df["dataset"]) == {"abide-i-pcp"}
    assert set(df["dataset_version"]) == {"freesurfer-5.1"}


def test_all_tables_present_does_not_duplicate_observation_keys(tmp_path: Path) -> None:
    """The allowlist prevents duplicate subject × region × measure rows.

    ``--all-tables`` puts lh.aparc.a2009s.stats and lh.entorhinal_exvivo.stats
    on disk. Both parse as lh aparc tables and the exvivo one reports
    entorhinal, which the core table also reports. Globbing would emit two rows
    for one observation key and §5.2 calls that an output-sanity failure.
    """
    root = tmp_path / "abide_pcp"
    make_subject(
        root,
        "CMU_a_0050642",
        tables=(*CORE_TABLES, "lh.aparc.a2009s.stats", "lh.entorhinal_exvivo.stats"),
    )
    result = ingest(AbidePcpAdapter(root, config()))
    validate(result.observations)
    entorhinal = result.observations[result.observations["region"] == "lh-entorhinal"]
    assert len(entorhinal) == 1
    assert entorhinal.iloc[0]["value"] == pytest.approx(3.31)


def test_persisting_to_parquet_round_trips(abide_root: Path, tmp_path: Path) -> None:
    """write_canonical conforms and validates; the adapter must satisfy it."""
    result = ingest(AbidePcpAdapter(abide_root, config()))
    path = write_canonical(result.observations, tmp_path / "observations.parquet")
    assert pd.read_parquet(path).shape[0] == result.observations.shape[0]


def test_surface_holes_are_null_on_freesurfer_51(abide_root: Path) -> None:
    """FS 5.1 emits no hole counts; null, never 0 (§2.2)."""
    result = ingest(AbidePcpAdapter(abide_root, config()))
    assert result.observations["surface_holes_lh"].isna().all()
    assert result.observations["surface_holes_rh"].isna().all()


def test_etiv_resolves_from_the_51_naming_and_seeds_baseline(abide_root: Path) -> None:
    """FS 5.1 writes IntraCranialVol/ICV, and one session means baseline eTIV."""
    result = ingest(AbidePcpAdapter(abide_root, config()))
    df = result.observations
    subject = df[df["subject_id"] == "CMU_a_0050642"]
    assert subject["etiv"].iloc[0] == pytest.approx(1549336.766171)
    assert subject["etiv_baseline"].iloc[0] == pytest.approx(1549336.766171)


def test_site_is_populated_without_a_phenotypic_table(abide_root: Path) -> None:
    """Site comes from the directory name, so it never needs the sidecar."""
    result = ingest(AbidePcpAdapter(abide_root, config()))
    sites = dict(zip(result.observations["subject_id"], result.observations["site"], strict=True))
    assert sites["CMU_a_0050642"] == "CMU_a"
    assert sites["UM_1_0050309"] == "UM_1"


def test_demographics_are_null_without_a_phenotypic_table(abide_root: Path) -> None:
    """Age, sex, and diagnosis are not in the stats files; null, not invented."""
    df = ingest(AbidePcpAdapter(abide_root, config())).observations
    for column in ("age_at_session", "age_baseline", "sex", "dx_baseline", "dx_at_session"):
        assert df[column].isna().all()


def test_scanner_fields_are_null_rather_than_asserted(abide_root: Path) -> None:
    """Field strength is not in the files; a null states that, a constant lies."""
    df = ingest(AbidePcpAdapter(abide_root, config())).observations
    assert df["field_strength_tesla"].isna().all()
    assert df["scanner_manufacturer"].isna().all()


# -- phenotypic sidecar ------------------------------------------------------


def write_phenotypic(path: Path) -> Path:
    """Write a phenotypic table using ABIDE's own column names and codes."""
    pd.DataFrame(
        {
            "SUB_ID": [50642, 50309, 51233, 99999],
            "SITE_ID": ["CMU_a", "UM_1", "UCLA", "GHOST"],
            "AGE_AT_SCAN": [24.5, 13.2, 31.0, 20.0],
            "SEX": [1, 2, 1, 2],
            "DX_GROUP": [1, 2, 1, 2],
        }
    ).to_csv(path, index=False)
    return path


def test_phenotypic_join_survives_zero_padding(abide_root: Path, tmp_path: Path) -> None:
    """Directory ``CMU_a_0050642`` must match phenotypic ``SUB_ID`` 50642."""
    csv = write_phenotypic(tmp_path / "pheno.csv")
    adapter = AbidePcpAdapter(abide_root, config(), phenotypic_csv=csv)
    df = ingest(adapter).observations
    rows = df[df["subject_id"] == "CMU_a_0050642"]
    assert rows["age_at_session"].iloc[0] == pytest.approx(24.5)
    assert rows["age_baseline"].iloc[0] == pytest.approx(24.5)


def test_phenotypic_codes_map_to_canonical_vocabulary(abide_root: Path, tmp_path: Path) -> None:
    """SEX 1/2 and DX_GROUP 1/2 become the values the model formula expects."""
    csv = write_phenotypic(tmp_path / "pheno.csv")
    df = ingest(AbidePcpAdapter(abide_root, config(), phenotypic_csv=csv)).observations
    by_subject = {
        subject: (sex, dx)
        for subject, sex, dx in zip(df["subject_id"], df["sex"], df["dx_baseline"], strict=True)
    }
    assert by_subject["CMU_a_0050642"] == ("M", "patient")
    assert by_subject["UM_1_0050309"] == ("F", "control")


def test_missing_phenotypic_file_is_tolerated(abide_root: Path, tmp_path: Path) -> None:
    """A sidecar path that does not exist yields null covariates, not a crash."""
    adapter = AbidePcpAdapter(abide_root, config(), phenotypic_csv=tmp_path / "nope.csv")
    df = ingest(adapter).observations
    assert df["age_at_session"].isna().all()
    assert not df.empty


def test_unrecognisable_phenotypic_table_is_tolerated(abide_root: Path, tmp_path: Path) -> None:
    """No usable ID column means absent metadata, not a failed run."""
    csv = tmp_path / "junk.csv"
    pd.DataFrame({"unrelated": [1, 2]}).to_csv(csv, index=False)
    df = ingest(AbidePcpAdapter(abide_root, config(), phenotypic_csv=csv)).observations
    assert df["age_at_session"].isna().all()
    assert not df.empty


# -- accounting --------------------------------------------------------------


def test_empty_subject_is_counted_as_a_derivative_loss(abide_root: Path) -> None:
    """The empty subject reaches the funnel as an attributable loss (§2.5.4)."""
    result = ingest(AbidePcpAdapter(abide_root, config()))
    assert result.sessions_discovered == 4
    assert result.sessions_without_files == 1


def test_expected_sessions_marks_derivative_loss(abide_root: Path) -> None:
    """A directory with no usable tables is missing_derivative."""
    expected = AbidePcpAdapter(abide_root, config()).expected_sessions()
    causes = dict(zip(expected["subject_id"], expected["missing_cause"], strict=True))
    assert causes["UCLA_51233"] == str(MissingnessCause.DERIVATIVE)
    assert pd.isna(causes["CMU_a_0050642"])


def test_phenotypic_roster_reveals_acquisition_loss(abide_root: Path, tmp_path: Path) -> None:
    """A participant listed phenotypically with no directory never acquired.

    Without the roster this loss is invisible, which is why expected_sessions
    claims no acquisition loss at all in that case rather than reporting zero.
    """
    csv = write_phenotypic(tmp_path / "pheno.csv")
    adapter = AbidePcpAdapter(abide_root, config(), phenotypic_csv=csv)
    expected = adapter.expected_sessions()
    causes = dict(zip(expected["subject_id"], expected["missing_cause"], strict=True))
    assert causes["99999"] == str(MissingnessCause.ACQUISITION)

    without = AbidePcpAdapter(abide_root, config()).expected_sessions()
    assert str(MissingnessCause.ACQUISITION) not in set(without["missing_cause"].dropna())


# -- factory -----------------------------------------------------------------


def test_build_adapter_constructs_the_abide_adapter(abide_root: Path, tmp_path: Path) -> None:
    """The adapter is reachable from configuration, not only from Python."""
    csv = write_phenotypic(tmp_path / "pheno.csv")
    cfg = DatasetConfig(
        name="abide-i-pcp",
        version="freesurfer-5.1",
        adapter="abide-pcp",
        phenotypic_csv=csv,
        collapse_site_subsample=True,
    )
    adapter = build_adapter(cfg, abide_root)
    assert isinstance(adapter, AbidePcpAdapter)
    df = ingest(adapter).observations
    assert set(df[df["subject_id"] == "CMU_a_0050642"]["site"]) == {"CMU_a"}


def test_collapse_applies_when_no_phenotypic_site_is_available(abide_root: Path) -> None:
    """Without a sidecar the collapsed site comes from the directory name."""
    adapter = AbidePcpAdapter(abide_root, config(), collapse_site_subsample=True)
    df = ingest(adapter).observations
    sites = dict(zip(df["subject_id"], df["site"], strict=True))
    assert sites["CMU_a_0050642"] == "CMU"
    assert sites["UM_1_0050309"] == "UM"


# -- configuration -----------------------------------------------------------


def test_real_data_config_needs_no_fixtures_block(abide_root: Path, tmp_path: Path) -> None:
    """A config pointing at real data must not have to carry fixture fiction."""
    from morphline.config import load_config

    path = tmp_path / "abide.yaml"
    path.write_text(
        "dataset:\n"
        "  name: abide-i-pcp\n"
        "  version: freesurfer-5.1\n"
        "  adapter: abide-pcp\n"
        f"  path: {abide_root}\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.fixtures is None
    assert cfg.dataset.adapter == "abide-pcp"


def test_fixtures_block_is_required_when_there_is_no_dataset_path(tmp_path: Path) -> None:
    """With nothing to read and nothing to generate, the config is unusable.

    Defaulting ``sites`` instead would fabricate the very site effects the
    harmonization tests exist to recover.
    """
    from pydantic import ValidationError

    from morphline.config import load_config

    path = tmp_path / "broken.yaml"
    path.write_text("dataset:\n  name: synthetic-v1\n  adapter: synthetic\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="fixtures are required"):
        load_config(path)


def test_real_data_run_records_no_fixture_seed(abide_root: Path, tmp_path: Path) -> None:
    """A seed that governed nothing must not appear in provenance (§2.8)."""
    import json

    from morphline.config import load_config
    from morphline.pipeline import run_pipeline

    path = tmp_path / "abide.yaml"
    path.write_text(
        "dataset:\n"
        "  name: abide-i-pcp\n"
        "  version: freesurfer-5.1\n"
        "  adapter: abide-pcp\n"
        f"  path: {abide_root}\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    run_pipeline(load_config(path), out)
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["random_seeds"] == {}
    assert provenance["freesurfer_versions"] == []
    assert len(provenance["freesurfer_version_declarations"]) == 2


def test_ingest_writes_a_versions_sidecar(abide_root: Path, tmp_path: Path) -> None:
    """The staged path reaches declarations by sidecar, not by schema change."""
    import json

    from morphline.stages.ingest import run_ingest

    out = tmp_path / "staged"
    run_ingest(AbidePcpAdapter(abide_root, config()), out)
    observed = json.loads((out / "ingest_versions.json").read_text(encoding="utf-8"))
    assert observed["freesurfer_versions"] == []
    assert len(observed["freesurfer_version_declarations"]) == 2
