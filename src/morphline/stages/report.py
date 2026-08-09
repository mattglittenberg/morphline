"""HTML report generation — a reproducibility artifact, not a chart dump (§2.8).

The rule: **a reader holding only the HTML file should be able to reconstruct
the run.** Every report embeds a provenance block, and if a parameter changed
the output it appears in that block.

Section order is fixed by §2.8: provenance → data accounting funnel → QC
summary → harmonization diagnostics → model results → limitations. Output is a
self-contained single file with everything inlined and no external asset
fetches, so it survives being emailed, archived, or opened offline.

Inputs are plain dictionaries rather than stage objects. That is what lets the
Nextflow path work: each upstream stage persists a JSON sidecar next to its
Parquet, and this stage reconstructs its inputs from files rather than from
live objects held in one process's memory (§2.7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).parent.parent / "report" / "templates"


@dataclass(slots=True)
class ReportInputs:
    """Everything the report template needs, as serialisable data.

    Attributes:
        title: Report title.
        provenance: The §2.8 provenance block.
        resolved_config: Fully resolved run configuration.
        accounting: Data accounting report, from ``AccountingReport.as_dict``.
        harmonization: Harmonization result and confound diagnostics.
        model: Model results.
        qc_summary: QC status distribution.
    """

    title: str
    provenance: dict[str, Any]
    resolved_config: dict[str, Any]
    accounting: dict[str, Any]
    harmonization: dict[str, Any]
    model: dict[str, Any]
    qc_summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_files(
        cls,
        *,
        title: str,
        provenance_json: Path | str,
        accounting_json: Path | str,
        harmonization_json: Path | str,
        model_json: Path | str,
    ) -> ReportInputs:
        """Rebuild report inputs from the JSON sidecars each stage wrote.

        Args:
            title: Report title.
            provenance_json: Path to ``provenance.json``.
            accounting_json: Path to ``accounting.json``.
            harmonization_json: Path to ``harmonization.json``.
            model_json: Path to ``model_results.json``.

        Returns:
            Inputs equivalent to those the in-process pipeline builds.
        """

        def load(path: Path | str) -> dict[str, Any]:
            data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
            return data

        provenance = load(provenance_json)
        accounting = load(accounting_json)
        return cls(
            title=title,
            provenance=provenance,
            resolved_config=provenance.get("run_parameters", {}),
            accounting=accounting,
            harmonization=load(harmonization_json),
            model=load(model_json),
            qc_summary=accounting.get("qc_summary", {}),
        )


def _dynamic_limitations(inputs: ReportInputs) -> list[dict[str, str]]:
    """Limitations that depend on what this particular run found.

    The standing limitations — the derivative-indexing caveat, the
    sensitivity-versus-inference distinction, the MAR assumption, the
    synthetic-data scope — are presentation text and live in the template,
    where they cannot drag input-format vocabulary into a downstream stage
    module (§1.4). Only run-dependent findings are assembled here.
    """
    items: list[dict[str, str]] = []

    confound = inputs.harmonization.get("diagnostics")
    if confound and not confound.get("interpretable", True):
        items.append(
            {
                "title": f"Scanner/time confounding is {confound.get('severity')} in this run",
                "body": str(confound.get("message", "")),
            }
        )

    fits = inputs.model.get("fits", [])
    failed = [f["region"] for f in fits if not f.get("converged")]
    if failed:
        items.append(
            {
                "title": "Some models did not converge",
                "body": (
                    f"{len(failed)} of {len(fits)} fits failed to converge and are "
                    f"reported rather than dropped: {', '.join(failed)}."
                ),
            }
        )

    dropped = [f["region"] for f in fits if f.get("random_slope_dropped")]
    if dropped:
        items.append(
            {
                "title": "Random slope dropped for some regions",
                "body": (
                    "To achieve convergence, the random slope on time was dropped for "
                    f"{', '.join(dropped)}. A random-intercept-only model does not "
                    "represent between-subject variation in rate of change, so those "
                    "estimates answer a slightly different question."
                ),
            }
        )

    errors = inputs.accounting.get("reconciliation_errors", [])
    if errors:
        items.append(
            {
                "title": "Data accounting does not reconcile",
                "body": (
                    "Observations were lost without a stated cause. Treat every number "
                    "in this report as suspect until this is resolved: " + "; ".join(errors)
                ),
            }
        )
    return items


def _crosstab_html(confound: dict[str, Any] | None) -> str | None:
    """Render the site × time crosstab, which §2.3.1 requires reporting."""
    if not confound:
        return None
    crosstab = confound.get("crosstab")
    if not crosstab:
        return None
    try:
        frame = pd.DataFrame(crosstab)
    except ValueError:
        return None
    if frame.empty:
        return None
    return frame.to_html(classes="data", border=0)


def render_report(inputs: ReportInputs) -> str:
    """Render the HTML report.

    Args:
        inputs: Everything the template needs.

    Returns:
        Self-contained HTML.
    """
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.html.j2")

    accounting = inputs.accounting
    harmonization = inputs.harmonization
    confound = harmonization.get("diagnostics")

    model_rows = [
        {
            "region": f.get("region"),
            "measure_type": f.get("measure_type"),
            "n_observations": f.get("n_observations"),
            "n_subjects": f.get("n_subjects"),
            "converged": f.get("converged"),
            "estimate": (f.get("coefficients") or {}).get("time:dx_baseline[T.patient]"),
            "std_error": (f.get("std_errors") or {}).get("time:dx_baseline[T.patient]"),
            "p_value": (f.get("p_values") or {}).get("time:dx_baseline[T.patient]"),
            "q_value": f.get("q_value"),
        }
        for f in inputs.model.get("fits", [])
    ]

    return template.render(
        title=inputs.title,
        provenance=inputs.provenance,
        resolved_config=json.dumps(inputs.resolved_config, indent=2, default=str),
        funnel=accounting.get("funnel", []),
        reconciliation_errors=accounting.get("reconciliation_errors", []),
        parse_failures=accounting.get("parse_failures_by_code", {}),
        metadata_coverage=accounting.get("metadata_coverage", {}),
        batch_sizes=accounting.get("batch_sizes", {}),
        sessions_per_subject=accounting.get("sessions_per_subject", {}),
        missingness=accounting.get("missingness", {}),
        accounting_notes=accounting.get("notes", []),
        qc_summary=inputs.qc_summary,
        harmonization_notes=harmonization.get("notes", []),
        harmonization_applied=harmonization.get("applied", False),
        confound=confound,
        confound_crosstab=_crosstab_html(confound),
        small_batches=harmonization.get("small_batches", {}),
        model=inputs.model,
        model_rows=model_rows,
        dynamic_limitations=_dynamic_limitations(inputs),
    )


def run_report(inputs: ReportInputs, outdir: Path | str) -> Path:
    """Render the report and write it to disk.

    Args:
        inputs: Everything the template needs.
        outdir: Destination directory.

    Returns:
        Path to ``report.html``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.html"
    path.write_text(render_report(inputs), encoding="utf-8")
    return path
