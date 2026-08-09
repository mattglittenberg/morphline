"""Quality control — three-level status, not a binary exclusion list (§2.4.1).

**v0.1.0 stub.** Every observation is marked ``PASS``. The *field structure* is
final, so week 3 changes logic rather than schema.

The design commitment that survives the stub: QC **identifies and classifies**
problems; the analysis layer **decides** what to include. Keeping those
separate means review-worthy observations stay visible instead of vanishing
into an exclusion list — which is why ``analysis_included`` is set here from an
explicit configured policy rather than being conflated with ``qc_status``.

Week 3 adds (§2.4.2): Euler / surface holes evaluated *within site*, robust
median/MAD outliers within site, eTIV plausibility bounds, left-right asymmetry
outliers, and the longitudinal suspicious-change flag. The last of those is a
*review criterion, not an automatic failure* — large apparent change can
reflect genuine biology, a scanner change, registration differences,
segmentation error, or ordinary noise, and the QC layer cannot distinguish
these, so it must not claim to (§2.4.3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from morphline.config import AnalysisConfig, QCConfig
from morphline.schema import QCStatus, conform, write_canonical


def apply_qc(
    observations: pd.DataFrame,
    qc_config: QCConfig,
    analysis_config: AnalysisConfig,
) -> pd.DataFrame:
    """Assign QC status and the analysis inclusion decision.

    Args:
        observations: Canonical observations.
        qc_config: QC thresholds and acceptance criteria. Unused by the stub
            beyond its ``enabled`` flag; wired now so the signature does not
            change in week 3.
        analysis_config: Inclusion policy — which statuses reach the model.

    Returns:
        A copy carrying ``qc_status``, ``qc_flags``, ``qc_score``,
        ``qc_notes``, and ``analysis_included``.
    """
    out = conform(observations).copy()
    if out.empty:
        return out

    if qc_config.enabled:
        # Stub behaviour: no check has been implemented yet, so nothing can
        # honestly be flagged. Marking everything PASS is the *dumbest correct
        # thing* (§0.1) — as distinct from marking everything PASS while
        # claiming checks ran.
        out["qc_status"] = str(QCStatus.PASS)
        out["qc_notes"] = "v0.1.0 stub: no QC checks implemented"
    else:
        out["qc_status"] = str(QCStatus.PASS)
        out["qc_notes"] = "QC disabled by configuration"

    out["qc_flags"] = [[] for _ in range(len(out))]
    out["qc_score"] = None

    included = {str(s) for s in analysis_config.include_qc_status}
    out["analysis_included"] = out["qc_status"].isin(included)
    return out


def run_qc(
    observations: pd.DataFrame,
    qc_config: QCConfig,
    analysis_config: AnalysisConfig,
    outdir: Path | str,
) -> pd.DataFrame:
    """Apply QC and persist the annotated observations.

    Args:
        observations: Canonical observations.
        qc_config: QC configuration.
        analysis_config: Inclusion policy.
        outdir: Destination directory.

    Returns:
        The annotated observations, already written to
        ``qc_observations.parquet``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    annotated = apply_qc(observations, qc_config, analysis_config)
    write_canonical(annotated, out / "qc_observations.parquet")
    return annotated
