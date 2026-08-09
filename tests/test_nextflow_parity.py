"""The staged CLI path must produce the same results as the in-process path.

Nextflow is not installed on the development machine and cannot be run here,
so the *orchestration* is only ever verified in CI. What can be verified
locally is the contract Nextflow depends on: that each stage is independently
invocable, communicates purely through files, and — chained in the DAG's order
— produces exactly what running everything in one process produces.

If this test passes and the Nextflow run still fails, the bug is in the
workflow wiring, not in the stages. That is a much smaller place to look.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from morphline.config import load_config
from morphline.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config" / "test.yaml"


def morphline(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI exactly as a Nextflow process would."""
    result = subprocess.run(
        [sys.executable, "-m", "morphline", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"morphline {' '.join(args)} failed:\n{result.stderr}"
    return result


@pytest.fixture(scope="module")
def staged_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run every stage as a separate process, mirroring modules/local/*.nf."""
    work = tmp_path_factory.mktemp("staged")
    fixtures = work / "fixtures"
    cfg = str(CONFIG)

    # GENERATE_FIXTURES
    morphline("fixtures", "generate", "--config", cfg, "--outdir", str(fixtures))

    # PARSE_SUBJECT — fan out, one process per subject
    subjects = sorted(p.name for p in (fixtures / "derivatives" / "freesurfer").iterdir())
    per_subject = work / "sub"
    for subject in subjects:
        morphline(
            "ingest",
            "--config",
            cfg,
            "--indir",
            str(fixtures),
            "--subject",
            subject,
            "--outdir",
            str(per_subject / subject),
        )

    # COLLECT_CANONICAL — gathers PATHS, reads them inside the consumer
    parquets = sorted(str(p) for p in per_subject.glob("*/observations.parquet"))
    morphline("collect", *parquets, "--outdir", str(work))

    morphline(
        "qc",
        "--config",
        cfg,
        "--observations",
        str(work / "observations.parquet"),
        "--outdir",
        str(work),
    )
    morphline(
        "harmonize",
        "--config",
        cfg,
        "--observations",
        str(work / "qc_observations.parquet"),
        "--outdir",
        str(work),
    )
    morphline(
        "model",
        "--config",
        cfg,
        "--observations",
        str(work / "harmonized_observations.parquet"),
        "--outdir",
        str(work),
    )

    failure_args: list[str] = []
    for path in sorted(per_subject.glob("*/parse_failures.parquet")):
        failure_args += ["--failures", str(path)]

    morphline(
        "account",
        "--config",
        cfg,
        "--indir",
        str(fixtures),
        "--observations",
        str(work / "observations.parquet"),
        "--qc-observations",
        str(work / "qc_observations.parquet"),
        "--model-results",
        str(work / "model_results.parquet"),
        *failure_args,
        "--outdir",
        str(work),
    )
    morphline(
        "provenance",
        "--config",
        cfg,
        "--observations",
        str(work / "qc_observations.parquet"),
        "--outdir",
        str(work),
    )
    morphline("report", "--config", cfg, "--indir", str(work), "--outdir", str(work))
    return work


@pytest.fixture(scope="module")
def inprocess_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    outdir = tmp_path_factory.mktemp("inprocess")
    run_pipeline(load_config(CONFIG), outdir)
    return outdir


def test_staged_run_produces_every_artifact(staged_run: Path) -> None:
    for name in (
        "observations.parquet",
        "qc_observations.parquet",
        "harmonized_observations.parquet",
        "model_results.parquet",
        "accounting_funnel.parquet",
        "accounting.json",
        "harmonization.json",
        "model_results.json",
        "provenance.json",
        "report.html",
    ):
        assert (staged_run / name).is_file(), f"staged run did not produce {name}"


def test_staged_funnel_matches_inprocess_funnel(staged_run: Path, inprocess_run: Path) -> None:
    """The headline artifact must not depend on how the stages were driven."""
    staged = pd.read_parquet(staged_run / "accounting_funnel.parquet")
    inproc = pd.read_parquet(inprocess_run / "accounting_funnel.parquet")
    pd.testing.assert_frame_equal(staged, inproc)


def test_staged_funnel_reconciles(staged_run: Path) -> None:
    funnel = pd.read_parquet(staged_run / "accounting_funnel.parquet")
    assert funnel["unexplained"].sum() == 0
    assert funnel["lost"].sum() > 0, "nothing was lost, so reconciliation proves nothing"


def test_staged_observations_match_inprocess(staged_run: Path, inprocess_run: Path) -> None:
    """Per-subject fan-out then gather must equal a single sequential pass."""
    key = ["subject_id", "session_id", "region", "measure_type"]
    staged = pd.read_parquet(staged_run / "observations.parquet").sort_values(key)
    inproc = pd.read_parquet(inprocess_run / "observations.parquet").sort_values(key)

    assert len(staged) == len(inproc)
    pd.testing.assert_series_equal(
        staged["value"].reset_index(drop=True),
        inproc["value"].reset_index(drop=True),
    )


def test_staged_model_results_match_inprocess(staged_run: Path, inprocess_run: Path) -> None:
    staged = pd.read_parquet(staged_run / "model_results.parquet")
    inproc = pd.read_parquet(inprocess_run / "model_results.parquet")
    assert staged["region"].tolist() == inproc["region"].tolist()
    assert staged["estimate"].iloc[0] == pytest.approx(inproc["estimate"].iloc[0], rel=1e-9)


def test_staged_report_has_the_provenance_block(staged_run: Path) -> None:
    html = (staged_run / "report.html").read_text(encoding="utf-8")
    for label in ("pipeline_version", "parser_version", "Fully resolved configuration"):
        assert label in html


def test_staged_provenance_records_observed_freesurfer_versions(staged_run: Path) -> None:
    """The Nextflow path must not lose provenance the in-process path captures."""
    data = json.loads((staged_run / "provenance.json").read_text(encoding="utf-8"))
    assert set(data["freesurfer_versions"]) == {"5.3.0", "6.0.0", "7.2.0"}


def test_account_exits_nonzero_when_the_funnel_does_not_reconcile(tmp_path: Path) -> None:
    """Unexplained loss must fail a pipeline, not print a warning nobody reads.

    Pointing accounting at a dataset root whose truth tables describe more
    sessions than the observations account for produces exactly that.
    """
    fixtures = tmp_path / "fx"
    morphline("fixtures", "generate", "--config", str(CONFIG), "--outdir", str(fixtures))
    morphline(
        "ingest", "--config", str(CONFIG), "--indir", str(fixtures), "--outdir", str(tmp_path)
    )

    # Drop half the observations without recording any cause.
    obs_path = tmp_path / "observations.parquet"
    obs = pd.read_parquet(obs_path)
    obs.head(len(obs) // 2).to_parquet(obs_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "morphline",
            "account",
            "--config",
            str(CONFIG),
            "--indir",
            str(fixtures),
            "--observations",
            str(obs_path),
            "--outdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 1
    assert "does not reconcile" in result.stderr
