"""Scanner harmonization with ComBat (BUILD_PLAN §2.3).

The stage is toggleable and runs both ways. The estimator is empirical-Bayes
ComBat, implemented in-repo in :mod:`morphline.combat` rather than taken from
``neuroHarmonize``, which pins ``numpy==1.26.4`` and would hold the project on
numpy 1.x (Build.md deviation 1). This module owns the *policy* around it —
which rows estimate the batch terms, what happens to a batch too small to
estimate, and what the report is told — while the numerical work stays a pure
function over a frame.

The confound diagnostics below are deliberately independent of all of that.
§7's cut order puts the harmonization implementation first on the chopping
block while insisting the §2.3.1 written analysis and the confound diagnostics
survive — "the paragraph explaining non-identifiability is worth more to a
reviewer than working ComBat code." They shipped before the estimator did and
would outlive it.

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

from morphline.combat import ComBatFit, run_combat
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
        applied: Whether the estimator ran and changed at least one value.
            Deliberately not "the stage executed" — ``n_values_changed`` makes
            the claim checkable rather than declarative.
        diagnostics: The scanner × time confound assessment.
        small_batches: Batches below ``min_batch_size``, with their sizes.
        small_batch_composition: Per small batch, the covariate composition
            §2.3.3 requires be reported alongside the size.
        fit: The fitted ComBat parameters, or ``None`` if nothing was fitted.
        pooling: Which batches were merged under the ``pool`` policy.
        estimation: How the batch terms were estimated, for the report.
        n_values_changed: How many values differ from the input.
        notes: Messages destined for the report.
    """

    observations: pd.DataFrame
    applied: bool
    diagnostics: ConfoundDiagnostics | None = None
    small_batches: dict[str, int] = field(default_factory=dict)
    small_batch_composition: dict[str, dict[str, Any]] = field(default_factory=dict)
    fit: ComBatFit | None = None
    pooling: dict[str, Any] | None = None
    estimation: dict[str, Any] = field(default_factory=dict)
    n_values_changed: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the result as a plain mapping for templating and JSON."""
        return {
            "applied": self.applied,
            "n_values_changed": self.n_values_changed,
            "diagnostics": self.diagnostics.as_dict() if self.diagnostics else None,
            "small_batches": self.small_batches,
            "small_batch_composition": self.small_batch_composition,
            "estimation": self.estimation,
            "pooling": self.pooling,
            "combat": self.fit.as_dict() if self.fit else None,
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

    # Zero variance in time makes every association with it zero *by
    # construction*, and the R² branches below would read that as a clean
    # design. It is not a finding: nothing was assessed. A cross-sectional
    # dataset reaches this on every run, and reporting "harmonization is on
    # firm ground" there is indistinguishable in the report from the same
    # sentence earned by a genuinely well-spread longitudinal cohort.
    if df["time_from_baseline_years"].nunique(dropna=True) < 2:
        return ConfoundDiagnostics(
            crosstab=empty,
            correlation=float("nan"),
            max_site_time_r2=float("nan"),
            severity="not_assessable",
            interpretable=True,
            message=(
                "Time from baseline is constant across every observation, so "
                "scanner/time confounding cannot be assessed — there is no time "
                "variance for site to explain. This is the expected result for a "
                "cross-sectional dataset and is not evidence of a clean design. "
                "Cross-sectional site effects may still be present; what cannot "
                "be evaluated here is their entanglement with time."
            ),
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
    # Either variable being constant makes the correlation undefined rather than
    # zero, but no association is *detectable* either way, which is what 0.0
    # already reports for a single-site run. A cross-sectional dataset holds
    # time at 0 for every row and reaches this by construction.
    time_varies = bool(df["time_from_baseline_years"].nunique(dropna=True) > 1)
    correlation = (
        0.0
        if len(set(codes)) < 2 or not time_varies
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


def describe_small_batches(
    observations: pd.DataFrame, small: dict[str, int], config: HarmonizationConfig
) -> dict[str, dict[str, Any]]:
    """Summarize the covariate composition of each below-threshold batch.

    §2.3.3 requires small batches be reported "with their sizes **and covariate
    composition**". The size alone cannot say whether a batch is small but
    balanced or small and entirely one diagnosis, and those have very different
    implications for whether its batch term means anything.

    Args:
        observations: Canonical observations.
        small: Batch name to subject count, from :func:`find_small_batches`.
        config: Harmonization configuration.

    Returns:
        Batch name to a summary of its size and covariate composition.
    """
    if not small or config.batch_column not in observations.columns:
        return {}

    composition: dict[str, dict[str, Any]] = {}
    for batch in sorted(small):
        rows = observations[observations[config.batch_column].astype("string") == batch]
        subjects = rows.drop_duplicates(subset=["subject_id"])
        entry: dict[str, Any] = {
            "n_subjects": int(subjects["subject_id"].nunique()),
            "n_observations": len(rows),
        }
        for covariate in ("sex", "dx_baseline"):
            if covariate in subjects.columns:
                counts = subjects[covariate].value_counts(dropna=False)
                entry[covariate] = {str(k): int(v) for k, v in counts.items()}
        if "age_baseline" in subjects.columns:
            ages = pd.to_numeric(subjects["age_baseline"], errors="coerce")
            if bool(ages.notna().any()):
                entry["age_baseline_mean"] = float(ages.mean())
                entry["age_baseline_sd"] = float(ages.std(ddof=1)) if len(ages) > 1 else 0.0
        composition[batch] = entry
    return composition


def estimation_mask(observations: pd.DataFrame, config: HarmonizationConfig) -> pd.Series:
    """Return which rows may contribute to estimating the batch terms.

    Exposed rather than inlined so the validation suite can select exactly the
    rows the estimator used. A test that reimplements this rule would be
    asserting against its own copy of it.

    Args:
        observations: Canonical observations.
        config: Harmonization configuration.

    Returns:
        A boolean mask over ``observations``.
    """
    if config.estimation_set == "all" or "analysis_included" not in observations.columns:
        return pd.Series(True, index=observations.index)

    included = observations["analysis_included"]
    if bool(included.isna().all()):
        return pd.Series(True, index=observations.index)
    return included.fillna(False).astype(bool)


def _pool_small_batches(
    observations: pd.DataFrame, small: dict[str, int], config: HarmonizationConfig
) -> tuple[pd.Series, dict[str, Any], list[str]]:
    """Merge below-threshold batches that share an acquisition setup.

    §2.3.3 permits pooling "only where scientifically defensible (e.g. same
    scanner model and protocol at one institution)". Defensibility is not
    checkable from the data, so the closest available proxy is agreement on
    manufacturer, model, and field strength — and batches that disagree fall
    back to exclusion rather than being merged on the strength of both being
    small, which is not a shared property of the scanners.

    Args:
        observations: Canonical observations.
        small: Batch name to subject count.
        config: Harmonization configuration.

    Returns:
        The batch labels to estimate with, the pooling record for provenance,
        and any notes.
    """
    labels = observations[config.batch_column].astype("string")
    keys = ("scanner_manufacturer", "scanner_model", "field_strength_tesla")

    groups: dict[tuple[str, ...], list[str]] = {}
    for batch in sorted(small):
        rows = observations[labels == batch]
        signature = tuple(
            str(rows[key].iloc[0]) if key in rows.columns and not rows.empty else "unknown"
            for key in keys
        )
        groups.setdefault(signature, []).append(batch)

    merged: list[dict[str, Any]] = []
    notes: list[str] = []
    for signature, members in sorted(groups.items()):
        if len(members) < 2:
            notes.append(
                f"Batch {members[0]} is below min_batch_size and shares its acquisition "
                f"setup with no other small batch, so it was excluded from estimation "
                f"rather than pooled with an unlike scanner."
            )
            continue
        label = "pooled:" + "+".join(members)
        labels = labels.mask(labels.isin(members), label)
        merged.append(
            {
                "label": label,
                "members": members,
                "scanner_manufacturer": signature[0],
                "scanner_model": signature[1],
                "field_strength_tesla": signature[2],
                "n_subjects": sum(small[m] for m in members),
            }
        )
        notes.append(
            f"Pooled {', '.join(members)} into one estimation batch on matching "
            f"{signature[0]} {signature[1]} at {signature[2]}T. Pooling asserts these "
            f"batches are exchangeable; that claim is not checkable from the data, and "
            f"it is recorded here and in provenance because it changed the estimate."
        )

    return labels, {"merged": merged} if merged else {}, notes


def _resolve_batches(
    observations: pd.DataFrame,
    small: dict[str, int],
    config: HarmonizationConfig,
    mask: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, Any] | None, list[str]]:
    """Apply the small-batch policy to the batch labels and estimation mask.

    Args:
        observations: Canonical observations.
        small: Batch name to subject count.
        config: Harmonization configuration.
        mask: The estimation mask before the policy is applied.

    Returns:
        The batch labels, the adjusted estimation mask, the pooling record,
        and any notes.
    """
    labels = observations[config.batch_column].astype("string")
    if not small:
        return labels, mask, None, []

    listed = ", ".join(f"{k} (n={v})" for k, v in sorted(small.items()))
    notes = [
        f"Batches below min_batch_size={config.min_batch_size}: {listed}. "
        f"Policy: {config.small_batch_policy}. Reported, not silently dropped."
    ]

    if config.small_batch_policy == "passthrough":
        notes.append(
            "WARNING: passthrough harmonizes below-threshold batches like any other. "
            "Their batch terms are estimated from few observations and are correspondingly "
            "unstable; empirical-Bayes shrinkage limits the damage but does not remove it."
        )
        return labels, mask, None, notes

    if config.small_batch_policy == "pool":
        pooled, record, pool_notes = _pool_small_batches(observations, small, config)
        return pooled, mask, record or None, notes + pool_notes

    excluded = labels.isin(sorted(small))
    notes.append(
        "Excluded from harmonization, not from the dataset: these rows stay in the "
        "output carrying their original values. Dropping them here would reappear at "
        "the modeling boundary under a cause that is false, which is worse than not "
        "reconciling because it looks like an answer."
    )
    return labels, mask & ~excluded, None, notes


def harmonize(observations: pd.DataFrame, config: HarmonizationConfig) -> HarmonizationResult:
    """Run the harmonization stage.

    Args:
        observations: Canonical observations, normally QC-annotated.
        config: Harmonization configuration.

    Returns:
        The result, carrying the adjusted observations, the fitted batch
        parameters, and the confound diagnostics.
    """
    df = conform(observations).copy()
    diagnostics = assess_confounding(df, config.batch_column)
    small = find_small_batches(df, config)
    composition = describe_small_batches(df, small, config)
    notes: list[str] = [diagnostics.message]

    def result(**overrides: Any) -> HarmonizationResult:
        base: dict[str, Any] = {
            "observations": df,
            "applied": False,
            "diagnostics": diagnostics,
            "small_batches": small,
            "small_batch_composition": composition,
            "notes": notes,
        }
        base.update(overrides)
        return HarmonizationResult(**base)

    if not config.enabled:
        notes.append(
            "Harmonization disabled by configuration (unharmonized arm). Values are "
            "unchanged; the confound diagnostics above still describe the design."
        )
        return result()

    if config.batch_column not in df.columns or df[config.batch_column].nunique(dropna=True) < 2:
        notes.append(
            f"Fewer than two distinct values of {config.batch_column!r}, so there is no "
            f"batch contrast to estimate and nothing was harmonized."
        )
        return result()

    mask = estimation_mask(df, config)
    if (
        config.estimation_set == "analysis_included"
        and "analysis_included" in df.columns
        and bool(df["analysis_included"].isna().all())
    ):
        notes.append(
            "No QC verdict was present, so the batch terms were estimated on every "
            "row rather than on an empty set."
        )
    labels, mask, pooling, policy_notes = _resolve_batches(df, small, config, mask)
    notes.extend(policy_notes)

    if labels[mask].nunique(dropna=True) < 2:
        notes.append(
            f"After applying the {config.small_batch_policy} policy, fewer than two "
            f"batches remain estimable, so there is no batch contrast left and nothing "
            f"was harmonized. This is the policy working as configured, not a failure: "
            f"with every batch below min_batch_size={config.min_batch_size} there is no "
            f"subset of the data large enough to estimate a batch effect from."
        )
        return result(pooling=pooling)

    working = df.copy()
    working["__batch__"] = labels

    combat = run_combat(
        working,
        batch_column="__batch__",
        covariates=config.covariates,
        estimation_mask=mask,
        empirical_bayes=config.empirical_bayes,
        max_iterations=config.eb_max_iterations,
        tolerance=config.eb_tolerance,
    )

    original = df["value"].astype("float64").to_numpy()
    adjusted = combat.values.to_numpy(dtype="float64")
    changed = int(
        np.count_nonzero(~np.isclose(original, adjusted, rtol=1e-12, atol=0.0, equal_nan=True))
    )
    df["value"] = combat.values

    estimation = {
        "method": ("ComBat (Johnson et al. 2007), parametric empirical Bayes, implemented in-repo"),
        "empirical_bayes": config.empirical_bayes,
        "tolerance": config.eb_tolerance,
        "max_iterations": config.eb_max_iterations,
        "estimation_set": config.estimation_set,
        "estimation_set_rationale": (
            "Batch terms are estimated on QC-passing rows only and applied to every row: "
            "letting known-bad reconstructions set the batch mean they are then corrected "
            "by would launder an artifact into a correction."
            if config.estimation_set == "analysis_included"
            else "Batch terms are estimated on every observation regardless of QC verdict."
        ),
        "batch_column": config.batch_column,
        "n_estimation_rows": int(mask.sum()),
        "n_rows": len(df),
        "n_groups_harmonized": combat.fit.n_groups_harmonized,
        "n_groups_skipped": combat.fit.n_groups_skipped,
    }

    if combat.fit.covariates_dropped:
        listed = ", ".join(f"{k} ({v})" for k, v in sorted(combat.fit.covariates_dropped.items()))
        notes.append(
            f"Covariates left out of the design matrix: {listed}. Their variation is not "
            f"protected from being absorbed into the batch terms."
        )
    if combat.fit.skipped:
        notes.append(
            f"{combat.fit.n_groups_skipped} region(s) were not harmonized and kept their "
            f"original values; see the per-region reason codes."
        )

    applied = changed > 0
    notes.insert(
        0,
        (
            f"ComBat adjusted {changed} of {len(df)} values across "
            f"{combat.fit.n_groups_harmonized} region(s)."
            if applied
            else "The estimator ran but changed no values."
        ),
    )

    return result(
        applied=applied,
        fit=combat.fit,
        pooling=pooling,
        estimation=estimation,
        n_values_changed=changed,
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
