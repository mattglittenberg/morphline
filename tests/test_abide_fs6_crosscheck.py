"""Cross-check morphline's ABIDE ingestion against the dfsp-spirit tables.

BUILD_PLAN §1.3 planned this as a *cross-version* check: our per-subject
FreeSurfer 5.1 numbers against a table someone else derived with FreeSurfer 6,
compared on rank correlation because the versions differ systematically.

**It is not that, because the tables are not FreeSurfer 6.** Measured here: all
28,976 shared observations — 1035 subjects × 28 regions, spanning volumes in mm³
and thickness in mm — are *bit-identical* to the ABIDE PCP FreeSurfer 5.1 values
morphline parses. Maximum absolute difference 0.0. Two ``recon-all`` runs at
different releases cannot do that; FreeSurfer 6 changed both the aseg
segmentation and the surface pipeline. The deposit's README states FreeSurfer
v6, and its own ``brainvol`` derivation carries the FreeSurfer 6 *column name*
``EstimatedTotalIntraCranialVol`` against the FreeSurfer 5.1 *value* our
``aseg.stats`` reports as ``IntraCranialVol, ICV``. See BUILD_PLAN revision 7.

That kills the cross-version claim and leaves a sharper check in its place.
``asegstats2table`` and ``aparcstats2table`` are FreeSurfer's own aggregation
tools, run by a third party over the same source files. So this validates
morphline's **parse → map → canonicalize** path against an independent
aggregation of identical inputs, and the expected answer is *exact equality*
rather than a correlation. That is the stronger test: a mis-read column, a
mis-mapped ``StructName``, or a swapped hemisphere shows up as an exact
mismatch instead of being absorbed into "well, the versions differ."

What it consequently does **not** validate: anything about FreeSurfer version
tolerance, and anything about the values themselves being right. Both sides
descend from the same recon-all run, so a segmentation error is present in both
and invisible here.

Requires both datasets on disk::

    scripts/fetch_abide_pcp.sh
    scripts/fetch_abide_fs6.sh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphline.adapters import build_adapter
from morphline.adapters.fs6_tables import read_fs6_tables
from morphline.config import load_config
from morphline.regions import V1_TEST_COUNT
from morphline.stages.ingest import ingest

pytestmark = pytest.mark.optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PCP_ROOT = REPO_ROOT / "data" / "abide_pcp"
FS6_ROOT = REPO_ROOT / "data" / "abide_fs6"

#: Subjects the PCP bucket has and the FS6 deposit does not. The deposit
#: describes itself as 1035 subjects against ABIDE I's 1112, and the difference
#: is one-directional.
EXPECTED_PCP_ONLY = 73

needs_data = pytest.mark.skipif(
    not (PCP_ROOT.is_dir() and FS6_ROOT.is_dir()),
    reason="needs data/abide_pcp and data/abide_fs6 (see scripts/fetch_abide_*.sh)",
)


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    """Morphline's ingested values joined to the aggregate tables."""
    config = load_config(REPO_ROOT / "config" / "abide.yaml")
    adapter = build_adapter(config.dataset, PCP_ROOT)
    ours = ingest(adapter).observations

    theirs = read_fs6_tables(FS6_ROOT)
    joined = (
        ours[["subject_id", "region", "measure_type", "value", "site"]]
        .rename(columns={"value": "morphline"})
        .merge(
            theirs[["subject_id", "region", "measure_type", "value"]].rename(
                columns={"value": "aggregate"}
            ),
            on=["subject_id", "region", "measure_type"],
            how="inner",
        )
    )
    return joined[np.isfinite(joined["morphline"]) & np.isfinite(joined["aggregate"])]


@needs_data
class TestIngestionAgreesExactly:
    """The parse → map → canonicalize path, against an independent aggregation."""

    def test_every_shared_observation_matches_exactly(self, comparison: pd.DataFrame) -> None:
        """Not approximately. The inputs are the same files.

        A tolerance here would be hiding the only thing worth knowing: whether
        morphline read the same number FreeSurfer's own tooling read.
        """
        difference = (comparison["morphline"] - comparison["aggregate"]).abs()

        assert len(comparison) > 28_000
        assert float(difference.max()) == 0.0
        assert int((difference > 0).sum()) == 0

    def test_both_measure_families_are_covered(self, comparison: pd.DataFrame) -> None:
        """Volumes in mm³ and thickness in mm exercise different parser paths —
        a whole-number ``Volume_mm3`` column against a decimal ``ThickAvg``."""
        by_measure = comparison.groupby("measure_type").size()

        assert set(by_measure.index) == {"volume", "thickness"}
        assert by_measure.min() > 10_000

    def test_the_full_region_set_is_compared(self, comparison: pd.DataFrame) -> None:
        assert comparison["region"].nunique() == V1_TEST_COUNT

    def test_hemispheres_are_not_swapped(self, comparison: pd.DataFrame) -> None:
        """The failure exact equality is most likely to catch.

        A left/right swap in ``ASEG_STRUCT_MAP`` would still produce plausible
        volumes, correlate near-perfectly across subjects, and survive every
        internal check in the repo.
        """
        for hemisphere in ("lh", "rh"):
            side = comparison[comparison["region"].str.startswith(f"{hemisphere}-")]
            assert not side.empty
            assert float((side["morphline"] - side["aggregate"]).abs().max()) == 0.0


@needs_data
class TestSubjectCoverage:
    """The count discrepancy is reported, not silently intersected away."""

    def test_the_aggregate_deposit_is_a_strict_subset(self) -> None:
        config = load_config(REPO_ROOT / "config" / "abide.yaml")
        ours = ingest(build_adapter(config.dataset, PCP_ROOT)).observations
        theirs = read_fs6_tables(FS6_ROOT)

        ingested = set(ours["subject_id"])
        aggregated = set(theirs["subject_id"])

        assert not aggregated - ingested, "aggregate tables carry subjects we never ingested"
        assert len(ingested - aggregated) == EXPECTED_PCP_ONLY


@needs_data
def test_the_tables_are_not_an_independent_freesurfer_6_derivation() -> None:
    """Pins the revision 7 finding so it cannot quietly revert.

    BUILD_PLAN originally treated this deposit as a FreeSurfer 6 source and
    expected systematic offsets. If this assertion ever fails, one of two things
    happened and both need a human: the deposit was corrected upstream and is
    now genuinely FreeSurfer 6 — in which case the cross-version check §1.3
    wanted becomes possible — or morphline's ingestion changed and the exact
    agreement above was lost.
    """
    config = load_config(REPO_ROOT / "config" / "abide.yaml")
    ours = ingest(build_adapter(config.dataset, PCP_ROOT)).observations
    theirs = read_fs6_tables(FS6_ROOT)

    joined = ours[["subject_id", "region", "measure_type", "value"]].merge(
        theirs[["subject_id", "region", "measure_type", "value"]],
        on=["subject_id", "region", "measure_type"],
        suffixes=("_ours", "_theirs"),
    )
    identical = float((joined["value_ours"] - joined["value_theirs"]).abs().max()) == 0.0

    assert identical, (
        "The aggregate tables no longer match ABIDE PCP's FreeSurfer 5.1 values exactly. "
        "Re-read BUILD_PLAN revision 7 before changing this test: it may mean the deposit "
        "is now a genuine FreeSurfer 6 derivation, which would make a cross-version "
        "comparison possible for the first time."
    )
