"""Quality control — three-level status, not a binary exclusion list (§2.4.1).

QC **identifies and classifies** problems; the analysis layer **decides** what
to include. Keeping those separate means review-worthy observations stay
visible instead of vanishing into an exclusion list, which is why
``analysis_included`` is set from an explicit configured policy rather than
being conflated with ``qc_status``.

Four checks are implemented (§2.4.2):

* **eTIV plausibility** against absolute bounds. A head size outside them is
  implausible on its face, so this is the one check that yields ``FAIL``.
* **Euler number**, derived from surface hole counts as ``2 - 2 * holes``
  (§2.2). Evaluated both against an absolute floor and as a robust outlier
  *within site*, because site distributions genuinely differ and a single
  global floor either misses bad sites or condemns good ones. Sessions whose
  FreeSurfer version reports no hole counts are **not evaluated** — the value
  is null, and a null is not a pass.
* **Robust regional outliers**, median/MAD z-scores per region within site.
* **Left-right asymmetry outliers**, on the asymmetry index per structure.

The **longitudinal suspicious-change flag (§2.4.3) is deliberately not
implemented here.** It is a review criterion rather than an automatic failure
and needs interval handling, expected biological variability, and
population-level distributions of annualized change to mean anything. Leaving
it out is visible in the confusion matrix rather than hidden: planted extreme
changes are reported as a known miss by the validation suite, not quietly
counted as clean.

Euler is **one input among several**, never a sole determinant, and none of
these checks can distinguish segmentation failure from biology. They classify;
they do not diagnose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from morphline.config import AnalysisConfig, QCConfig, QCThresholds
from morphline.schema import QCStatus, conform, write_canonical

#: Scale factor making the median absolute deviation a consistent estimator of
#: the standard deviation under normality.
_MAD_TO_SIGMA: Final = 1.4826

FLAG_ETIV: Final = "etiv_implausible"
FLAG_EULER: Final = "euler_low"
FLAG_REGION_OUTLIER: Final = "region_outlier"
FLAG_ASYMMETRY: Final = "asymmetry_outlier"

#: Flags severe enough to fail an observation outright. Everything else is a
#: WARNING: review-worthy, not disqualifying, and the analysis layer decides.
FAIL_FLAGS: Final = frozenset({FLAG_ETIV})

ALL_FLAGS: Final = (FLAG_ETIV, FLAG_EULER, FLAG_REGION_OUTLIER, FLAG_ASYMMETRY)

_SESSION_KEY: Final = ["subject_id", "session_id"]


def euler_number(surface_holes: pd.Series) -> pd.Series:
    """Derive Euler numbers from surface hole counts (§2.2).

    Hole counts are *reported* by FreeSurfer; the Euler number is *derived*
    from them. Conflating the two is a common and subtle error.

    Args:
        surface_holes: Hole counts, which may be null.

    Returns:
        ``2 - 2 * holes``, preserving nulls. A null stays null rather than
        becoming 2, which would claim a topologically flawless surface and rank
        the versions that report nothing as the highest quality data present.
    """
    return 2.0 - 2.0 * pd.to_numeric(surface_holes, errors="coerce")


def robust_z(values: pd.Series) -> pd.Series:
    """Return median/MAD z-scores, which outliers cannot inflate.

    Args:
        values: Values to score.

    Returns:
        Robust z-scores, zero where the MAD is zero or the group is too small
        to estimate a spread. A degenerate group yields "not an outlier"
        rather than a division by zero or a spurious infinity.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(0.0, index=values.index)
    return (numeric - median) / (mad * _MAD_TO_SIGMA)


def _grouped_robust_z(df: pd.DataFrame, value_column: str, by: list[str]) -> pd.Series:
    """Robust z-scores computed within each group, aligned to ``df``'s index."""
    if df.empty:
        return pd.Series(dtype="float64", index=df.index)
    return df.groupby(by, dropna=False, group_keys=False)[value_column].transform(robust_z)


def _session_frame(observations: pd.DataFrame) -> pd.DataFrame:
    """Collapse observations to one row per session, for session-level checks."""
    columns = ["site", "etiv", "surface_holes_lh", "surface_holes_rh"]
    present = [c for c in columns if c in observations.columns]
    return observations.groupby(_SESSION_KEY, dropna=False)[present].first().reset_index()


def _etiv_flags(sessions: pd.DataFrame, thresholds: QCThresholds) -> pd.Series:
    """Flag sessions whose eTIV falls outside the plausible range."""
    if "etiv" not in sessions.columns:
        return pd.Series(False, index=sessions.index)
    etiv = pd.to_numeric(sessions["etiv"], errors="coerce")
    outside = (etiv < thresholds.etiv_min) | (etiv > thresholds.etiv_max)
    return outside.fillna(False)


def _euler_flags(sessions: pd.DataFrame, thresholds: QCThresholds) -> pd.Series:
    """Flag sessions with a low Euler number, absolutely or within their site.

    Both tests are needed. The absolute floor catches a surface that is bad by
    any standard; the within-site robust test catches one that is bad *for the
    site it came from*, which a single global floor cannot do without either
    missing the worst subjects at good sites or condemning whole bad ones.
    """
    columns = {"surface_holes_lh", "surface_holes_rh"}
    if not columns <= set(sessions.columns):
        return pd.Series(False, index=sessions.index)

    euler = pd.concat(
        [euler_number(sessions["surface_holes_lh"]), euler_number(sessions["surface_holes_rh"])],
        axis=1,
    ).min(axis=1)

    scored = sessions.assign(_euler=euler)
    below_floor = euler < thresholds.euler_min
    within_site = _grouped_robust_z(scored, "_euler", ["site"]) < -thresholds.euler_mad_z

    # Null hole counts mean the check did not run. Neither branch may treat
    # that as a pass, and neither may treat it as a failure.
    return (below_floor | within_site).where(euler.notna(), False).fillna(False)


def _region_outlier_flags(observations: pd.DataFrame, thresholds: QCThresholds) -> pd.Series:
    """Flag observations that are robust outliers for their region and site."""
    z = _grouped_robust_z(observations, "value", ["site", "region", "measure_type"])
    return z.abs() > thresholds.outlier_mad_z


def _asymmetry_flags(observations: pd.DataFrame, thresholds: QCThresholds) -> pd.Series:
    """Flag structures whose left-right asymmetry is an outlier within site.

    The asymmetry index is scale-free, so structures of very different absolute
    size are comparable. A structure present in only one hemisphere has no
    asymmetry to speak of and is left unflagged rather than treated as maximally
    asymmetric.
    """
    flags = pd.Series(False, index=observations.index)
    needed = {"hemisphere", "region", "measure_type", "value", "site"}
    if not needed <= set(observations.columns):
        return flags

    df = observations.copy()
    df["_structure"] = [
        region.removeprefix(f"{hemi}-")
        if isinstance(region, str) and isinstance(hemi, str)
        else None
        for region, hemi in zip(df["region"], df["hemisphere"], strict=True)
    ]

    key = [*_SESSION_KEY, "_structure", "measure_type"]
    wide = (
        df[df["hemisphere"].isin(["lh", "rh"])]
        .pivot_table(index=key, columns="hemisphere", values="value", aggfunc="first")
        .reset_index()
    )
    if not {"lh", "rh"} <= set(wide.columns):
        return flags

    paired = wide.dropna(subset=["lh", "rh"])
    if paired.empty:
        return flags

    mean = (paired["lh"] + paired["rh"]) / 2.0
    paired = paired.assign(_ai=((paired["lh"] - paired["rh"]) / mean).where(mean != 0))
    paired = paired.merge(
        df.groupby(_SESSION_KEY, dropna=False)["site"].first().reset_index(),
        on=_SESSION_KEY,
        how="left",
    )
    paired["_z"] = _grouped_robust_z(paired, "_ai", ["site", "_structure", "measure_type"])

    flagged = paired.loc[paired["_z"].abs() > thresholds.asymmetry_mad_z, key]
    if flagged.empty:
        return flags

    marked = df.merge(flagged.assign(_flagged=True), on=key, how="left")
    marked.index = df.index
    return marked["_flagged"].fillna(False).astype(bool)


def apply_qc(
    observations: pd.DataFrame,
    qc_config: QCConfig,
    analysis_config: AnalysisConfig,
) -> pd.DataFrame:
    """Assign QC status, flags, and the analysis inclusion decision.

    Args:
        observations: Canonical observations.
        qc_config: QC thresholds and the enabled flag.
        analysis_config: Inclusion policy — which statuses reach the model.

    Returns:
        A copy carrying ``qc_status``, ``qc_flags``, ``qc_score``,
        ``qc_notes``, and ``analysis_included``.
    """
    out = conform(observations).copy()
    if out.empty:
        return out

    if not qc_config.enabled:
        out["qc_status"] = str(QCStatus.PASS)
        out["qc_notes"] = "QC disabled by configuration"
        out["qc_flags"] = [[] for _ in range(len(out))]
        out["qc_score"] = None
        included = {str(s) for s in analysis_config.include_qc_status}
        out["analysis_included"] = out["qc_status"].isin(included)
        return out

    thresholds = qc_config.thresholds
    sessions = _session_frame(out)
    sessions["_etiv"] = _etiv_flags(sessions, thresholds)
    sessions["_euler"] = _euler_flags(sessions, thresholds)

    session_flags = out[_SESSION_KEY].merge(
        sessions[[*_SESSION_KEY, "_etiv", "_euler"]], on=_SESSION_KEY, how="left"
    )
    session_flags.index = out.index

    per_flag = {
        FLAG_ETIV: session_flags["_etiv"].fillna(False).astype(bool),
        FLAG_EULER: session_flags["_euler"].fillna(False).astype(bool),
        FLAG_REGION_OUTLIER: _region_outlier_flags(out, thresholds),
        FLAG_ASYMMETRY: _asymmetry_flags(out, thresholds),
    }

    flags: list[list[str]] = [[] for _ in range(len(out))]
    for code, mask in per_flag.items():
        for position in np.flatnonzero(np.asarray(mask, dtype=bool)):
            flags[position].append(code)

    failed = np.zeros(len(out), dtype=bool)
    for code in FAIL_FLAGS:
        failed |= np.asarray(per_flag[code], dtype=bool)
    warned = np.array([bool(f) for f in flags]) & ~failed

    status = np.full(len(out), str(QCStatus.PASS), dtype=object)
    status[warned] = str(QCStatus.WARNING)
    status[failed] = str(QCStatus.FAIL)

    out["qc_status"] = status
    out["qc_flags"] = flags
    out["qc_score"] = (
        _grouped_robust_z(out, "value", ["site", "region", "measure_type"]).abs().to_numpy()
    )
    out["qc_notes"] = ["; ".join(codes) if codes else "no QC flag triggered" for codes in flags]

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
