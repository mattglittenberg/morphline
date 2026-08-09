"""Scanner harmonization — identity transform in v0.1.0 (BUILD_PLAN §2.3).

**v0.1.0 stub.** The stage is toggleable and runs both ways, but the transform
is the identity. Week 4 replaces the body with an in-repo empirical-Bayes
ComBat; nothing about the stage's interface changes.

The confound diagnostics below are *not* stubbed, deliberately. §7's cut order
puts the harmonization implementation first on the chopping block while
insisting the §2.3.1 written analysis and the confound diagnostics survive —
"the paragraph explaining non-identifiability is worth more to a reviewer than
working ComBat code." So the diagnostics ship in v0.1.0 even though the
estimator does not.

The identifiability problem
---------------------------
Standard ComBat assumes independent observations, which longitudinal data
violates. The deeper issue in aging cohorts is that **scanner is frequently
confounded with time**: subjects are scanned on an older scanner early in a
study and a newer one later.

When that confounding is strong, **the biological longitudinal effect and the
scanner effect are not identifiable from the observed data alone.** No
harmonization method recovers a unique answer from confounded data; the
separation depends on assumptions the data cannot check. This is a property of
the study design, not a deficiency in the software.

It follows that running harmonized and unharmonized models is a **sensitivity
analysis, not a solution**. It shows how conclusions depend on the
harmonization assumption; it does not establish which is correct. The report
must label these results as sensitivity analysis and distinguish them from
validated biological inference — in those words.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from morphline.config import HarmonizationConfig
from morphline.schema import conform, write_canonical


@dataclass(slots=True)
class ConfoundDiagnostics:
    """How badly scanner is confounded with time (§2.3.1).

    Attributes:
        crosstab: Site × time-bin observation counts.
        correlation: Correlation between site membership and time from
            baseline, as a quantitative confounding measure.
        max_site_time_r2: Proportion of variance in time explained by site.
            High values mean site and time are close to interchangeable.
        severity: ``none`` | ``mild`` | ``moderate`` | ``severe``.
        interpretable: Whether longitudinal estimates can be read as biology.
        message: Plain-language summary for the report.
    """

    crosstab: pd.DataFrame
    correlation: float
    max_site_time_r2: float
    severity: str
    interpretable: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        """Return the diagnostics as a plain mapping for templating."""
        return {
            "crosstab": {
                str(k): {str(i): int(v) for i, v in col.items()}
                for k, col in self.crosstab.to_dict().items()
            },
            "correlation": self.correlation,
            "max_site_time_r2": self.max_site_time_r2,
            "severity": self.severity,
            "interpretable": self.interpretable,
            "message": self.message,
        }


@dataclass(slots=True)
class HarmonizationResult:
    """Outcome of the harmonization stage.

    Attributes:
        observations: Observations after harmonization (or unchanged).
        applied: Whether any transform was actually applied.
        diagnostics: The scanner × time confound assessment.
        small_batches: Batches below ``min_batch_size``, with their sizes.
        notes: Messages destined for the report.
    """

    observations: pd.DataFrame
    applied: bool
    diagnostics: ConfoundDiagnostics | None = None
    small_batches: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the result as a plain mapping for templating and JSON."""
        return {
            "applied": self.applied,
            "diagnostics": self.diagnostics.as_dict() if self.diagnostics else None,
            "small_batches": self.small_batches,
            "notes": self.notes,
        }


def assess_confounding(
    observations: pd.DataFrame, batch_column: str = "site"
) -> ConfoundDiagnostics:
    """Quantify how far scanner is confounded with time.

    Where sites contribute across the full time range, harmonization is on
    much firmer ground; where a site's sessions cluster at one end of
    follow-up, the site effect and the time effect cannot be told apart. The
    point of reporting this per dataset is to **distinguish these cases rather
    than treating the whole dataset uniformly** (§2.3.1).

    Args:
        observations: Canonical observations.
        batch_column: Column holding the batch/site label.

    Returns:
        Diagnostics including a severity grade and whether longitudinal
        estimates should be read as biology at all.
    """
    empty = pd.DataFrame()
    if (
        observations.empty
        or batch_column not in observations.columns
        or "time_from_baseline_years" not in observations.columns
    ):
        return ConfoundDiagnostics(
            crosstab=empty,
            correlation=float("nan"),
            max_site_time_r2=float("nan"),
            severity="unknown",
            interpretable=True,
            message="Insufficient data to assess scanner/time confounding.",
        )

    df = observations[[batch_column, "time_from_baseline_years"]].dropna()
    if df.empty or df[batch_column].nunique() < 2:
        return ConfoundDiagnostics(
            crosstab=empty,
            correlation=0.0,
            max_site_time_r2=0.0,
            severity="none",
            interpretable=True,
            message="Single batch: scanner and time cannot be confounded.",
        )

    time_bins = pd.cut(df["time_from_baseline_years"], bins=4)
    crosstab = pd.crosstab(df[batch_column], time_bins)

    # One-way R²: how much of the variance in time is explained by knowing the
    # site. 1.0 means site *is* time, and no method can separate them.
    grand_mean = df["time_from_baseline_years"].mean()
    total_ss = float(((df["time_from_baseline_years"] - grand_mean) ** 2).sum())
    between_ss = float(
        sum(
            len(group) * (group["time_from_baseline_years"].mean() - grand_mean) ** 2
            for _, group in df.groupby(batch_column, observed=True)
        )
    )
    r2 = 0.0 if total_ss == 0 else between_ss / total_ss

    codes = pd.Categorical(df[batch_column]).codes
    correlation = (
        0.0
        if len(set(codes)) < 2
        else float(np.corrcoef(codes, df["time_from_baseline_years"])[0, 1])
    )

    if r2 >= 0.5:
        severity, interpretable = "severe", False
        message = (
            f"Site explains {r2:.0%} of the variance in time from baseline. The "
            "biological longitudinal effect and the scanner effect are NOT "
            "identifiable from these data. Longitudinal estimates must not be "
            "interpreted as biology."
        )
    elif r2 >= 0.2:
        severity, interpretable = "moderate", False
        message = (
            f"Site explains {r2:.0%} of the variance in time from baseline. "
            "Scanner and time are substantially confounded; treat harmonized "
            "longitudinal estimates as sensitivity analysis, not inference."
        )
    elif r2 >= 0.05:
        severity, interpretable = "mild", True
        message = (
            f"Site explains {r2:.0%} of the variance in time from baseline. Mild "
            "confounding; interpret longitudinal estimates with the caveat stated."
        )
    else:
        severity, interpretable = "none", True
        message = (
            f"Site explains {r2:.0%} of the variance in time from baseline. Sites "
            "contribute across the time range, so harmonization is on firm ground."
        )

    return ConfoundDiagnostics(
        crosstab=crosstab,
        correlation=correlation,
        max_site_time_r2=r2,
        severity=severity,
        interpretable=interpretable,
        message=message,
    )


def find_small_batches(observations: pd.DataFrame, config: HarmonizationConfig) -> dict[str, int]:
    """Return batches with fewer subjects than ``min_batch_size``.

    All batches below threshold are **reported**, never dropped silently
    (§2.3.3). The threshold itself is a conservative engineering default for
    stability, not a methodological law: adequacy actually depends on the
    number of batches, covariate balance within batch, outcome variance, the
    magnitude of the batch effect relative to that variance, and the
    complexity of the harmonization model.

    Args:
        observations: Canonical observations.
        config: Harmonization configuration.

    Returns:
        Batch name to subject count, for batches below the threshold.
    """
    if observations.empty or config.batch_column not in observations.columns:
        return {}
    sizes = observations.groupby(config.batch_column, dropna=False)["subject_id"].nunique()
    return {str(k): int(v) for k, v in sizes.items() if int(v) < config.min_batch_size}


def harmonize(observations: pd.DataFrame, config: HarmonizationConfig) -> HarmonizationResult:
    """Run the harmonization stage.

    Args:
        observations: Canonical observations.
        config: Harmonization configuration.

    Returns:
        The result. In v0.1.0 ``observations`` is returned unchanged and
        ``applied`` is ``False``, but the diagnostics are real.
    """
    df = conform(observations).copy()
    diagnostics = assess_confounding(df, config.batch_column)
    small = find_small_batches(df, config)

    notes: list[str] = [
        "v0.1.0: harmonization is an identity transform. The ComBat estimator "
        "lands in week 4; the confound diagnostics below are already real.",
        diagnostics.message,
    ]

    if small:
        listed = ", ".join(f"{k} (n={v})" for k, v in sorted(small.items()))
        notes.append(
            f"Batches below min_batch_size={config.min_batch_size}: {listed}. "
            f"Policy: {config.small_batch_policy}. Reported, not silently dropped."
        )

    if not config.enabled:
        notes.append("Harmonization disabled by configuration (unharmonized arm).")

    return HarmonizationResult(
        observations=df,
        applied=False,
        diagnostics=diagnostics,
        small_batches=small,
        notes=notes,
    )


def run_harmonize(
    observations: pd.DataFrame, config: HarmonizationConfig, outdir: Path | str
) -> HarmonizationResult:
    """Harmonize and persist the results.

    Args:
        observations: Canonical observations.
        config: Harmonization configuration.
        outdir: Destination directory.

    Returns:
        The harmonization result, already written to
        ``harmonized_observations.parquet``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result = harmonize(observations, config)
    write_canonical(result.observations, out / "harmonized_observations.parquet")
    (out / "harmonization.json").write_text(
        json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    return result
