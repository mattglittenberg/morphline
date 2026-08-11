"""End-to-end pipeline driver.

Chains every stage in-process. This is the local development path and the fast
CI end-to-end check; Nextflow runs the same stages as separate processes
communicating through Parquet files on disk (§2.7), so the two paths exercise
identical stage code and differ only in orchestration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from morphline.adapters import build_adapter
from morphline.config import RunConfig
from morphline.fixtures import write_fixtures
from morphline.provenance import Provenance
from morphline.stages.accounting import AccountingReport, build_accounting, run_accounting
from morphline.stages.harmonize import HarmonizationResult, run_harmonize
from morphline.stages.ingest import IngestResult, run_ingest
from morphline.stages.model import ModelResults, run_model
from morphline.stages.qc import run_qc
from morphline.stages.report import ReportInputs, run_report


@dataclass(slots=True)
class PipelineOutputs:
    """Everything a full run produced.

    Attributes:
        outdir: Root output directory.
        report_path: Path to the rendered HTML report.
        ingest: Ingestion result.
        accounting: Data accounting report.
        harmonization: Harmonization result.
        model: Model results.
        provenance: The provenance block, with duration filled in.
    """

    outdir: Path
    report_path: Path
    ingest: IngestResult
    accounting: AccountingReport
    harmonization: HarmonizationResult
    model: ModelResults
    provenance: Provenance


def run_pipeline(
    config: RunConfig,
    outdir: Path | str,
    *,
    fixtures_dir: Path | str | None = None,
    generate_fixtures: bool = True,
) -> PipelineOutputs:
    """Run every stage end to end.

    Args:
        config: Validated run configuration.
        outdir: Root output directory.
        fixtures_dir: Where fixtures live or should be written. Defaults to
            ``<outdir>/fixtures``.
        generate_fixtures: Whether to generate fixtures first. Set ``False`` to
            run against an existing tree.

    Returns:
        The outputs of the run, all already written to disk.
    """
    started = time.perf_counter()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    fixtures_root = Path(fixtures_dir) if fixtures_dir else out / "fixtures"
    dataset_root = config.dataset.path or fixtures_root

    if generate_fixtures and config.dataset.path is None and config.fixtures is not None:
        write_fixtures(config.fixtures, fixtures_root)

    provenance = Provenance.capture(
        dataset=config.dataset.name,
        dataset_version=config.dataset.version,
        run_parameters=config.resolved(),
        # A real-data run has no fixture seed. Recording one anyway would put a
        # number in the provenance block that governed nothing.
        random_seeds={"fixtures": config.fixtures.seed} if config.fixtures else {},
        input_path=str(dataset_root),
    )

    adapter = build_adapter(config.dataset, dataset_root)

    ingested = run_ingest(adapter, out)
    provenance.freesurfer_versions = ingested.freesurfer_versions
    provenance.freesurfer_version_declarations = ingested.freesurfer_version_declarations

    qc_observations = run_qc(ingested.observations, config.qc, config.analysis, out)
    harmonized = run_harmonize(qc_observations, config.harmonization, out)
    model_results = run_model(harmonized.observations, config.analysis, out)

    accounting = build_accounting(
        observations=ingested.observations,
        parse_failures=ingested.failures_frame(),
        expected_sessions=adapter.expected_sessions(),
        files_discovered=ingested.files_discovered,
        sessions_discovered=ingested.sessions_discovered,
        sessions_without_files=ingested.sessions_without_files,
        sessions_all_files_rejected=ingested.sessions_all_files_rejected,
        sessions_no_recognised_regions=ingested.sessions_no_recognised_regions,
        qc_observations=qc_observations,
        modeled_observations=model_results.n_modeled_observations,
        model_fits=model_results.to_frame(),
    )
    run_accounting(accounting, out)

    provenance.duration_seconds = time.perf_counter() - started
    (out / "provenance.json").write_text(
        json.dumps(provenance.as_dict(), indent=2, default=str), encoding="utf-8"
    )

    report_path = run_report(
        ReportInputs(
            title=config.report.title,
            provenance=provenance.as_dict(),
            resolved_config=config.resolved(),
            accounting=accounting.as_dict(),
            harmonization=harmonized.as_dict(),
            model=model_results.as_dict(),
            qc_summary=accounting.qc_summary,
        ),
        out,
    )

    return PipelineOutputs(
        outdir=out,
        report_path=report_path,
        ingest=ingested,
        accounting=accounting,
        harmonization=harmonized,
        model=model_results,
        provenance=provenance,
    )
