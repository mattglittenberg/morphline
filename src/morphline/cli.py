"""Command-line interface.

The local development path, and what the fast end-to-end CI job runs. Each
stage is also exposed individually so Nextflow can call them as separate
processes that communicate through Parquet files (§2.7) — every subcommand
reads paths and writes paths, never in-memory tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from morphline import __version__
from morphline.adapters import SubjectFilter, build_adapter
from morphline.config import RunConfig, load_config
from morphline.fixtures import write_fixtures
from morphline.pipeline import run_pipeline
from morphline.provenance import Provenance, versions_yml
from morphline.schema import read_canonical, read_canonical_many
from morphline.stages.accounting import build_accounting, run_accounting
from morphline.stages.ingest import run_ingest
from morphline.stages.model import run_model
from morphline.stages.qc import run_qc

app = typer.Typer(
    name="morphline",
    help="BIDS-aware longitudinal neuroimaging derivatives pipeline.",
    no_args_is_help=True,
    add_completion=False,
)
fixtures_app = typer.Typer(help="Generate synthetic fixtures.", no_args_is_help=True)
app.add_typer(fixtures_app, name="fixtures")

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", help="Path to the YAML run configuration.")
]
OutdirOption = Annotated[Path, typer.Option("--outdir", "-o", help="Output directory.")]


def _load(config: Path) -> RunConfig:
    """Load a configuration, exiting cleanly on user error."""
    try:
        return load_config(config)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def version() -> None:
    """Print the morphline version."""
    typer.echo(__version__)


@fixtures_app.command("generate")
def fixtures_generate(
    config: ConfigOption,
    outdir: OutdirOption = Path("work/fixtures"),
) -> None:
    """Generate synthetic FreeSurfer stats fixtures with injected ground truth."""
    cfg = _load(config)
    truth = write_fixtures(cfg.fixtures, outdir)
    typer.echo(
        f"wrote {truth.manifest['n_files_written']} stats files "
        f"({truth.manifest['n_files_corrupted']} deliberately corrupted) "
        f"for {truth.manifest['n_subjects']} subjects to {outdir}"
    )
    typer.echo(f"regime {truth.manifest['regime']}, seed {truth.manifest['seed']}")


@app.command()
def ingest(
    config: ConfigOption,
    indir: Annotated[Path, typer.Option("--indir", help="Dataset root.")],
    outdir: OutdirOption = Path("results"),
    subject: Annotated[
        str | None,
        typer.Option("--subject", help="Restrict to one subject, for per-subject parallelism."),
    ] = None,
) -> None:
    """Parse a dataset into canonical observations.

    ``--subject`` is what lets Nextflow fan out: one task per subject, each
    writing its own Parquet and emitting the path.
    """
    cfg = _load(config)
    adapter = build_adapter(cfg.dataset, indir)
    if subject is not None:
        adapter = SubjectFilter(adapter, subject)
    result = run_ingest(adapter, outdir)
    typer.echo(
        f"discovered {result.files_discovered} files across "
        f"{result.sessions_discovered} sessions; "
        f"{len(result.failures)} parse failures; "
        f"{len(result.observations)} canonical observations"
    )


@app.command()
def qc(
    config: ConfigOption,
    observations: Annotated[Path, typer.Option("--observations", help="Canonical Parquet.")],
    outdir: OutdirOption = Path("results"),
) -> None:
    """Apply QC status and the analysis inclusion policy."""
    cfg = _load(config)
    annotated = run_qc(read_canonical(observations), cfg.qc, cfg.analysis, outdir)
    typer.echo(f"annotated {len(annotated)} observations")


@app.command()
def harmonize(
    config: ConfigOption,
    observations: Annotated[Path, typer.Option("--observations", help="QC-annotated Parquet.")],
    outdir: OutdirOption = Path("results"),
) -> None:
    """Run scanner harmonization and the scanner/time confound diagnostics."""
    from morphline.stages.harmonize import run_harmonize

    cfg = _load(config)
    result = run_harmonize(read_canonical(observations), cfg.harmonization, outdir)
    for note in result.notes:
        typer.echo(f"- {note}")


@app.command()
def model(
    config: ConfigOption,
    observations: Annotated[Path, typer.Option("--observations", help="Harmonized Parquet.")],
    outdir: OutdirOption = Path("results"),
) -> None:
    """Fit the longitudinal mixed-effects model."""
    cfg = _load(config)
    results = run_model(read_canonical(observations), cfg.analysis, outdir)
    typer.echo(
        f"fitted {len(results.fits)} region(s); "
        f"convergence {results.convergence_rate:.0%}; "
        f"{results.n_modeled_observations} modeled observations"
    )


@app.command()
def account(
    config: ConfigOption,
    indir: Annotated[Path, typer.Option("--indir", help="Dataset root.")],
    observations: Annotated[Path, typer.Option("--observations", help="Canonical Parquet.")],
    outdir: OutdirOption = Path("results"),
    failures: Annotated[
        list[Path] | None,
        typer.Option("--failures", help="Per-subject parse-failure Parquet files."),
    ] = None,
    qc_observations: Annotated[
        Path | None, typer.Option("--qc-observations", help="QC-annotated Parquet.")
    ] = None,
    model_results: Annotated[
        Path | None, typer.Option("--model-results", help="Model results Parquet.")
    ] = None,
) -> None:
    """Build the data accounting funnel and check that it reconciles.

    Exits non-zero when the funnel does not reconcile: unexplained loss is a
    bug, not a rounding error (§1.6), so it must fail a pipeline rather than
    print a warning nobody reads.
    """
    import pandas as pd

    cfg = _load(config)
    adapter = build_adapter(cfg.dataset, indir)
    obs = read_canonical(observations)

    failure_paths = list(failures or [])
    if not failure_paths:
        default = Path(observations).parent / "parse_failures.parquet"
        if default.is_file():
            failure_paths = [default]
    failure_frames = [pd.read_parquet(p) for p in failure_paths if Path(p).is_file()]
    failure_df = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames
        else pd.DataFrame(columns=["source_file", "failure_code"])
    )

    discovered = list(adapter.discover())
    sessions_discovered = len(discovered)
    sessions_without_files = sum(1 for s in discovered if not s.stats_files)
    files_discovered = sum(len(s.stats_files) for s in discovered)

    # Exact loss counts come from ingestion, which is the only stage that can
    # observe them. Absent the sidecar they stay zero, so any resulting loss
    # is reported as unexplained rather than quietly absorbed.
    counters: dict[str, int] = {}
    counters_path = Path(observations).parent / "ingest_counters.json"
    if counters_path.is_file():
        counters = json.loads(counters_path.read_text(encoding="utf-8"))
        sessions_without_files = counters.get("sessions_without_files", sessions_without_files)
        files_discovered = counters.get("files_discovered", files_discovered)

    qc_obs = read_canonical(qc_observations) if qc_observations else None
    modeled = None
    fits = None
    if model_results and Path(model_results).is_file():
        fits = pd.read_parquet(model_results)
        modeled = int(fits["n_observations"].sum())

    report = build_accounting(
        observations=obs,
        parse_failures=failure_df,
        expected_sessions=adapter.expected_sessions(),
        files_discovered=files_discovered,
        sessions_discovered=sessions_discovered,
        sessions_without_files=sessions_without_files,
        sessions_all_files_rejected=counters.get("sessions_all_files_rejected", 0),
        sessions_no_recognised_regions=counters.get("sessions_no_recognised_regions", 0),
        qc_observations=qc_obs,
        modeled_observations=modeled,
        model_fits=fits,
    )
    run_accounting(report, outdir)

    for line in report.funnel_frame().to_string(index=False).splitlines():
        typer.echo(line)

    errors = report.reconcile()
    if errors:
        typer.secho("funnel does not reconcile:", fg=typer.colors.RED, err=True)
        for err in errors:
            typer.secho(f"  {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho("funnel reconciles exactly", fg=typer.colors.GREEN)


@app.command()
def report(
    config: ConfigOption,
    indir: Annotated[Path, typer.Option("--indir", help="Directory holding stage JSON sidecars.")],
    outdir: OutdirOption = Path("results"),
) -> None:
    """Render the HTML report from the JSON sidecars each stage wrote.

    Reading files rather than live objects is what lets this run as its own
    Nextflow process (§2.7).
    """
    from morphline.stages.report import ReportInputs, run_report

    cfg = _load(config)
    source = Path(indir)
    inputs = ReportInputs.from_files(
        title=cfg.report.title,
        provenance_json=source / "provenance.json",
        accounting_json=source / "accounting.json",
        harmonization_json=source / "harmonization.json",
        model_json=source / "model_results.json",
    )
    path = run_report(inputs, outdir)
    typer.echo(f"report: {path}")


@app.command("run")
def run_all(
    config: ConfigOption,
    outdir: OutdirOption = Path("results"),
    fixtures: Annotated[
        Path | None, typer.Option("--fixtures", help="Existing fixture root to reuse.")
    ] = None,
) -> None:
    """Run every stage end to end, generating fixtures if none are supplied."""
    cfg = _load(config)
    outputs = run_pipeline(
        cfg,
        outdir,
        fixtures_dir=fixtures,
        generate_fixtures=fixtures is None,
    )

    typer.echo("")
    for line in outputs.accounting.funnel_frame().to_string(index=False).splitlines():
        typer.echo(line)
    typer.echo("")

    errors = outputs.accounting.reconcile()
    if errors:
        typer.secho("funnel does not reconcile:", fg=typer.colors.RED, err=True)
        for err in errors:
            typer.secho(f"  {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("funnel reconciles exactly", fg=typer.colors.GREEN)
    typer.echo(f"report: {outputs.report_path}")
    typer.echo(f"duration: {outputs.provenance.duration_seconds:.2f}s")


@app.command()
def provenance(
    config: ConfigOption,
    outdir: OutdirOption = Path("results"),
    observations: Annotated[
        Path | None,
        typer.Option("--observations", help="Canonical Parquet, for observed tool versions."),
    ] = None,
) -> None:
    """Emit the provenance block and a versions.yml fragment.

    ``--observations`` matters more than it looks: FreeSurfer versions in the
    block are the ones *observed in the input data*, not a configured value
    (§2.8). Without it the block would claim less than it could.
    """
    cfg = _load(config)
    block = Provenance.capture(
        dataset=cfg.dataset.name,
        dataset_version=cfg.dataset.version,
        run_parameters=cfg.resolved(),
        random_seeds={"fixtures": cfg.fixtures.seed},
    )
    if observations and Path(observations).is_file():
        obs = read_canonical(observations)
        block.freesurfer_versions = sorted(
            {str(v) for v in obs["freesurfer_version"].dropna().unique()}
        )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "provenance.json").write_text(
        json.dumps(block.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    (outdir / "versions.yml").write_text(
        versions_yml(
            "MORPHLINE",
            {"morphline": __version__, "python": block.python_version},
        ),
        encoding="utf-8",
    )
    typer.echo(f"wrote provenance to {outdir}")


@app.command()
def collect(
    inputs: Annotated[list[Path], typer.Argument(help="Canonical Parquet files to merge.")],
    outdir: OutdirOption = Path("results"),
) -> None:
    """Concatenate canonical Parquet files into one.

    This is the gather step Nextflow calls after ``collect()``-ing per-subject
    paths. The channel carries paths; the reading happens here, inside the
    consuming process.
    """
    from morphline.schema import write_canonical

    merged = read_canonical_many(list(inputs))
    outdir.mkdir(parents=True, exist_ok=True)
    path = write_canonical(merged, outdir / "observations.parquet")
    typer.echo(f"merged {len(inputs)} files into {path} ({len(merged)} observations)")


if __name__ == "__main__":
    app()
