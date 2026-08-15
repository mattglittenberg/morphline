"""End-to-end tests — the week-2 exit criterion.

A cold clone must run the whole pipeline on generated fixtures with no
external data present, and the report must carry a complete provenance block.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from morphline.cli import app
from morphline.config import RunConfig, load_config
from morphline.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def run_output(tmp_path_factory: pytest.TempPathFactory) -> object:
    cfg = load_config(REPO_ROOT / "config" / "test.yaml")
    outdir = tmp_path_factory.mktemp("e2e")
    return run_pipeline(cfg, outdir)


def test_pipeline_runs_end_to_end_with_no_external_data(run_output: object) -> None:
    assert run_output.report_path.is_file()  # type: ignore[attr-defined]


def test_every_stage_wrote_its_parquet(run_output: object) -> None:
    outdir: Path = run_output.outdir  # type: ignore[attr-defined]
    for name in (
        "observations.parquet",
        "parse_failures.parquet",
        "qc_observations.parquet",
        "harmonized_observations.parquet",
        "model_results.parquet",
        "accounting_funnel.parquet",
    ):
        assert (outdir / name).is_file(), f"{name} was not written"


def test_funnel_reconciles_exactly(run_output: object) -> None:
    assert run_output.accounting.reconcile() == []  # type: ignore[attr-defined]


def test_funnel_actually_had_losses_to_attribute(run_output: object) -> None:
    """Otherwise reconciliation is vacuous."""
    funnel = run_output.accounting.funnel_frame()  # type: ignore[attr-defined]
    assert funnel["lost"].sum() > 0


def test_report_is_self_contained(run_output: object) -> None:
    """No external asset fetches — the report must survive being archived.

    Asserted against *references that cause a fetch*, not against the substring
    ``http``. The inlined Plotly bundle legitimately contains XML namespace
    identifiers (``http://www.w3.org/2000/svg``) and map-tile attribution URLs
    for chart types this report never builds; neither is a request. A substring
    check cannot tell those from a real one, and the version of this test that
    tried made "self-contained" mean "contains no URL-shaped text", which is a
    different and unmeetable claim.
    """
    html = run_output.report_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    fetching = re.findall(
        r"""<script[^>]+\ssrc\s*=|<link[^>]+\shref\s*=\s*["']https?://"""
        r"""|<img[^>]+\ssrc\s*=\s*["']https?://|url\(\s*["']?https?://""",
        html,
        flags=re.IGNORECASE,
    )
    assert not fetching, f"report fetches external assets: {fetching[:5]}"


def test_report_carries_the_full_provenance_block(run_output: object) -> None:
    """§2.8: a reader holding only the HTML must be able to reconstruct the run."""
    html = run_output.report_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    for label in (
        "pipeline_version",
        "git_sha",
        "python_version",
        "parser_version",
        "freesurfer_version",
        "random_seed",
        "Fully resolved configuration",
    ):
        assert label in html, f"provenance block is missing {label}"


def test_provenance_records_the_seed(run_output: object) -> None:
    prov = run_output.provenance  # type: ignore[attr-defined]
    assert prov.random_seeds["fixtures"] == 20260800


def test_provenance_records_observed_freesurfer_versions(run_output: object) -> None:
    """Versions are observed in the data, not read from config."""
    prov = run_output.provenance  # type: ignore[attr-defined]
    assert prov.freesurfer_versions
    assert set(prov.freesurfer_versions) <= {"5.3.0", "6.0.0", "7.2.0"}


def test_provenance_json_is_written_and_parseable(run_output: object) -> None:
    outdir: Path = run_output.outdir  # type: ignore[attr-defined]
    data = json.loads((outdir / "provenance.json").read_text(encoding="utf-8"))
    assert data["pipeline_version"]
    assert "run_parameters" in data


def test_resolved_config_includes_defaults_not_just_written_keys(run_output: object) -> None:
    """§2.8: if a parameter changed the output, it appears in the block."""
    prov = run_output.provenance  # type: ignore[attr-defined]
    params = prov.run_parameters
    # Never written in config/test.yaml, but governs QC behaviour.
    assert params["qc"]["thresholds"]["euler_min"] is not None
    assert params["analysis"]["fdr_alpha"] == 0.05


def test_report_sections_appear_in_the_order_the_spec_requires(run_output: object) -> None:
    """§2.8 fixes the order: provenance -> accounting -> QC -> harmonization
    -> model -> limitations."""
    html = run_output.report_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    order = [
        "Provenance",
        "Data accounting funnel",
        "Quality control",
        "Harmonization diagnostics",
        "Model results",
        "Limitations",
    ]
    positions = [html.index(section) for section in order]
    assert positions == sorted(positions), "report sections are out of order"


def test_report_states_the_sensitivity_versus_inference_distinction(
    run_output: object,
) -> None:
    """§2.3.1 demands those words, not a euphemism."""
    html = run_output.report_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    assert "Sensitivity analysis, not validated biological inference" in html


def test_report_states_the_mar_assumption(run_output: object) -> None:
    html = run_output.report_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    assert "missing-at-random" in html
    assert "MNAR" in html


def test_report_declares_the_multiplicity_family(run_output: object) -> None:
    """§2.5.3: the family is stated numerically, and secondary families are
    shown as separate from the primary one rather than pooled with it."""
    html = run_output.report_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    assert "primary family" in html.lower()
    assert "28 regional tests" in html
    assert "secondary families" in html.lower()
    assert "corrected separately" in html.lower()


class TestCLI:
    """The CLI is the local dev path and the fast CI e2e check."""

    def test_run_command_succeeds(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app,
            ["run", "--config", str(REPO_ROOT / "config" / "test.yaml"), "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert "funnel reconciles exactly" in result.output

    def test_fixtures_generate_command(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app,
            [
                "fixtures",
                "generate",
                "--config",
                str(REPO_ROOT / "config" / "test.yaml"),
                "--outdir",
                str(tmp_path / "fx"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "fx" / "truth" / "manifest.json").is_file()

    def test_stagewise_invocation_matches_the_nextflow_path(self, tmp_path: Path) -> None:
        """Each stage reads paths and writes paths, so Nextflow can chain them."""
        runner = CliRunner()
        cfg = str(REPO_ROOT / "config" / "test.yaml")
        fx = tmp_path / "fx"
        out = tmp_path / "out"

        assert (
            runner.invoke(
                app, ["fixtures", "generate", "--config", cfg, "--outdir", str(fx)]
            ).exit_code
            == 0
        )
        assert (
            runner.invoke(
                app, ["ingest", "--config", cfg, "--indir", str(fx), "--outdir", str(out)]
            ).exit_code
            == 0
        )
        assert (
            runner.invoke(
                app,
                [
                    "qc",
                    "--config",
                    cfg,
                    "--observations",
                    str(out / "observations.parquet"),
                    "--outdir",
                    str(out),
                ],
            ).exit_code
            == 0
        )
        assert (
            runner.invoke(
                app,
                [
                    "harmonize",
                    "--config",
                    cfg,
                    "--observations",
                    str(out / "qc_observations.parquet"),
                    "--outdir",
                    str(out),
                ],
            ).exit_code
            == 0
        )
        result = runner.invoke(
            app,
            [
                "model",
                "--config",
                cfg,
                "--observations",
                str(out / "harmonized_observations.parquet"),
                "--outdir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "model_results.parquet").is_file()

    def test_missing_config_exits_cleanly(self) -> None:
        result = CliRunner().invoke(app, ["run", "--config", "/nonexistent.yaml"])
        assert result.exit_code == 2
        assert "not found" in result.output

    def test_version_command(self) -> None:
        result = CliRunner().invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.output.strip() == "0.1.0"


class TestCommittedConfigs:
    """Every committed config must load and run."""

    @pytest.mark.parametrize("name", ["test.yaml", "recovery.yaml", "confounded.yaml"])
    def test_config_loads(self, name: str) -> None:
        cfg = load_config(REPO_ROOT / "config" / name)
        assert isinstance(cfg, RunConfig)

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        """A typo in a threshold name must not quietly leave the default."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "fixtures:\n  seed: 1\n  sites:\n    - {name: a, n_subjects: 2}\n  typo_key: true\n",
            encoding="utf-8",
        )
        with pytest.raises(Exception, match=r"typo_key|Extra inputs"):
            load_config(bad)
