"""Longitudinal mixed-effects modeling (BUILD_PLAN §2.5).

**v0.1.0 walking skeleton.** Fits the real specification with statsmodels
MixedLM, but on a single region, and applies no multiplicity correction — the
FDR family is declared but has one member. Week 5 extends the same code across
the 28-test region set.

Specification (§2.5.1), fully disambiguated::

    value ~ age_baseline + time + dx_baseline + time:dx_baseline
            + sex + etiv_baseline
    random: intercept + slope on time, grouped by subject_id

Why each term is what it is:

* ``age_baseline`` captures **between-subject** age differences; ``time``
  captures **within-subject** change. Splitting them is the point of the
  parameterization — it separates the cross-sectional age gradient, which is
  confounded with cohort effects, from the rate of change a longitudinal
  design exists to measure. Age-at-session must never appear alongside time;
  they are collinear by construction.
* ``time:dx_baseline`` is **the primary hypothesis** — differential rate of
  change by group.
* ``etiv_baseline``, not time-varying eTIV: within-subject eTIV should be
  approximately constant, so observed fluctuation is mostly measurement
  variability, and a noisy time-varying regressor lets covariate measurement
  error leak into the slope estimate.
* ``dx_baseline``, not time-varying diagnosis: conversion is partly a
  *consequence* of atrophy, so conditioning on post-baseline diagnosis invites
  collider bias in the very slope being estimated.

**Head-size adjustment is mandatory.** Regional volumes without it is the most
common error in this literature and it will be noticed immediately. Covariate
adjustment is the v1 choice.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from morphline.config import AnalysisConfig
from morphline.regions import V1_TEST_COUNT, v1_region_set

#: The model formula, fixed effects only. Random effects are specified
#: separately via ``re_formula``.
FIXED_EFFECTS_FORMULA = (
    "value ~ age_baseline + time + dx_baseline + time:dx_baseline + sex + etiv_baseline"
)

#: Random intercept and random slope on time, grouped by subject.
RANDOM_EFFECTS_FORMULA = "~ time"

#: The coefficient carrying the primary hypothesis, as statsmodels names it.
PRIMARY_TERM = "time:dx_baseline[T.patient]"

#: Optimizers tried in order before the model is simplified.
#:
#: statsmodels' MixedLM reports ``converged=False`` from ``lbfgs`` fairly
#: readily even where the estimate is stable — the flag reflects the
#: optimizer's own stopping criterion, not the quality of the optimum. Rather
#: than trusting one optimizer or quietly ignoring the flag, escalate through
#: more robust methods first.
_OPTIMIZERS = ("lbfgs", "powell", "nm")


@dataclass(slots=True)
class RegionFit:
    """Model results for one region.

    Attributes:
        region: Canonical region name.
        measure_type: What was measured.
        n_observations: Observations entering the fit.
        n_subjects: Subjects contributing.
        converged: Whether the optimizer converged. Reported per region
            because some fits will fail, and failing loudly beats silently
            reporting a non-converged model (§2.5.1).
        estimable: Whether the design can identify the specification at all.
            A cross-sectional dataset has no within-subject time variance, so
            ``time`` and ``time:dx_baseline`` are collinear with the intercept
            and no optimizer can help. Kept distinct from ``converged``
            because a design that cannot answer the question is not a model
            that failed to fit, and pooling them would report a dataset
            limitation as a numerical one.
        optimizer: Which optimizer produced the reported fit.
        random_slope_dropped: Whether the random slope on time had to be
            dropped for the fit to converge. A random-intercept-only fit
            answers a slightly different question, so it is never silent.
        coefficients: Fixed-effect estimates by term.
        std_errors: Standard errors by term.
        p_values: Raw p-values by term.
        q_value: BH-adjusted p-value for the primary term, filled in by
            :func:`apply_fdr`.
        message: Diagnostic message when the fit failed.
    """

    region: str
    measure_type: str
    n_observations: int
    n_subjects: int
    converged: bool
    estimable: bool = True
    coefficients: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    q_value: float | None = None
    optimizer: str = ""
    random_slope_dropped: bool = False
    message: str = ""

    @property
    def primary_estimate(self) -> float | None:
        """The ``time × diagnosis`` interaction estimate, if the fit converged."""
        return self.coefficients.get(PRIMARY_TERM)

    @property
    def primary_p(self) -> float | None:
        """Raw p-value for the primary term."""
        return self.p_values.get(PRIMARY_TERM)


@dataclass(slots=True)
class ModelResults:
    """Results across all fitted regions.

    Attributes:
        fits: Per-region fits.
        family_size: Number of tests in the primary multiplicity family.
        fdr_alpha: False discovery rate applied within that family.
        n_modeled_observations: Total observations that entered any model.
        notes: Messages for the report.
    """

    fits: list[RegionFit] = field(default_factory=list)
    family_size: int = 0
    fdr_alpha: float = 0.05
    n_modeled_observations: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_estimable(self) -> int:
        """Regions whose specification the design can identify at all."""
        return sum(1 for f in self.fits if f.estimable)

    @property
    def convergence_rate(self) -> float:
        """Fraction of *estimable* fits that converged.

        Regions the design cannot identify are outside the denominator. A
        cross-sectional dataset would otherwise report 0% convergence, which
        reads as a broken model rather than as a dataset that cannot answer the
        question asked of it — so :attr:`n_estimable` must be read alongside
        this, and the report states both.
        """
        estimable = [f for f in self.fits if f.estimable]
        if not estimable:
            return 0.0
        return sum(1 for f in estimable if f.converged) / len(estimable)

    def to_frame(self) -> pd.DataFrame:
        """Return per-region results as a flat frame."""
        return pd.DataFrame(
            [
                {
                    "region": f.region,
                    "measure_type": f.measure_type,
                    "n_observations": f.n_observations,
                    "n_subjects": f.n_subjects,
                    "converged": f.converged,
                    "estimable": f.estimable,
                    "optimizer": f.optimizer,
                    "random_slope_dropped": f.random_slope_dropped,
                    "estimate": f.primary_estimate,
                    "std_error": f.std_errors.get(PRIMARY_TERM),
                    "p_value": f.primary_p,
                    "q_value": f.q_value,
                    "message": f.message,
                }
                for f in self.fits
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        """Return results as a plain mapping for templating."""
        return {
            "fits": [asdict(f) for f in self.fits],
            "family_size": self.family_size,
            "fdr_alpha": self.fdr_alpha,
            "n_modeled_observations": self.n_modeled_observations,
            "convergence_rate": self.convergence_rate,
            "n_estimable": self.n_estimable,
            "notes": self.notes,
        }


def prepare_model_frame(observations: pd.DataFrame, region: str) -> pd.DataFrame:
    """Build the analysis frame for one region.

    Args:
        observations: QC-annotated canonical observations.
        region: Region to extract.

    Returns:
        One row per subject × session, with model terms named as the formula
        expects. ``time`` is renamed from ``time_from_baseline_years`` so the
        formula stays readable.
    """
    df = observations[observations["region"] == region].copy()
    if df.empty:
        return df

    df = df.rename(columns={"time_from_baseline_years": "time"})
    needed = [
        "value",
        "time",
        "age_baseline",
        "dx_baseline",
        "sex",
        "etiv_baseline",
        "subject_id",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df.dropna(subset=needed)
    # eTIV is on the order of 1.5e6 while thickness is ~2.5. Left unscaled, the
    # optimizer struggles on a design matrix spanning six orders of magnitude.
    df["etiv_baseline"] = df["etiv_baseline"] / 1_000_000.0
    return df


def fit_region(observations: pd.DataFrame, region: str, measure_type: str) -> RegionFit:
    """Fit the mixed model for a single region.

    Args:
        observations: QC-annotated canonical observations.
        region: Region to fit.
        measure_type: Measure type, carried through to the results.

    Returns:
        The fit, with ``converged=False`` and a message if the fit failed.
        Non-convergence is a reported outcome, never a raised exception.
    """
    df = prepare_model_frame(observations, region)
    if df.empty or df["subject_id"].nunique() < 3:
        return RegionFit(
            region=region,
            measure_type=measure_type,
            n_observations=len(df),
            n_subjects=int(df["subject_id"].nunique()) if not df.empty else 0,
            converged=False,
            message="insufficient data: fewer than 3 subjects with complete covariates",
        )

    # Checked before any optimizer runs, because no optimizer can fix it. With
    # a single session per subject, `time` is constant, so `time` and
    # `time:dx_baseline` are collinear with the intercept and the random slope
    # has nothing to vary over. statsmodels reports this as a singular matrix,
    # which reads as a numerical accident rather than as the design fact it is.
    if df["time"].nunique(dropna=True) < 2:
        return RegionFit(
            region=region,
            measure_type=measure_type,
            n_observations=0,
            n_subjects=int(df["subject_id"].nunique()),
            converged=False,
            estimable=False,
            message=(
                "not estimable: time has zero variance across "
                f"{df['subject_id'].nunique()} subjects, so the longitudinal "
                "specification is not identifiable on this cross-sectional dataset"
            ),
        )

    n_obs = len(df)
    n_subjects = int(df["subject_id"].nunique())

    def attempt(re_formula: str | None, method: str) -> tuple[Any, str] | str:
        """Fit once. Returns the result, or a message describing the failure."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    FIXED_EFFECTS_FORMULA,
                    data=df,
                    groups=df["subject_id"],
                    re_formula=re_formula,
                )
                return model.fit(method=method, maxiter=1000), method
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def build(result: Any, method: str, *, slope_dropped: bool, message: str) -> RegionFit:
        return RegionFit(
            region=region,
            measure_type=measure_type,
            n_observations=n_obs,
            n_subjects=n_subjects,
            converged=bool(getattr(result, "converged", False)),
            coefficients={k: float(v) for k, v in result.params.items()},
            std_errors={k: float(v) for k, v in result.bse.items()},
            p_values={k: float(v) for k, v in result.pvalues.items()},
            optimizer=method,
            random_slope_dropped=slope_dropped,
            message=message,
        )

    last_error = ""
    best: RegionFit | None = None

    # Stage 1: the full random intercept + slope model, escalating optimizers.
    for method in _OPTIMIZERS:
        outcome = attempt(RANDOM_EFFECTS_FORMULA, method)
        if isinstance(outcome, str):
            last_error = outcome
            continue
        result, used = outcome
        fit = build(result, used, slope_dropped=False, message="")
        if fit.converged:
            if used != _OPTIMIZERS[0]:
                fit.message = f"converged on fallback optimizer {used!r}"
            return fit
        best = best or fit

    # Stage 2: §8's stated mitigation — fall back to random intercepts only.
    # A random-intercept-only model answers a slightly different question, so
    # this is recorded on the fit rather than applied silently.
    for method in _OPTIMIZERS:
        outcome = attempt(None, method)
        if isinstance(outcome, str):
            last_error = outcome
            continue
        result, used = outcome
        fit = build(
            result,
            used,
            slope_dropped=True,
            message=(
                "random slope on time dropped to achieve convergence; "
                "between-subject slope variation is not modeled for this region"
            ),
        )
        if fit.converged:
            return fit
        best = best or fit

    if best is not None:
        best.message = (
            "no optimizer converged, including the random-intercept-only fallback; "
            "estimates are reported but must not be interpreted"
        )
        return best

    return RegionFit(
        region=region,
        measure_type=measure_type,
        n_observations=n_obs,
        n_subjects=n_subjects,
        converged=False,
        message=f"every fit attempt raised; last error was {last_error}",
    )


def apply_fdr(fits: list[RegionFit], alpha: float) -> None:
    """Apply Benjamini-Hochberg FDR within the primary family, in place.

    The family is **the ``time:dx_baseline`` coefficient across the regions in
    the region set — and only that**. Secondary effects (main effects of
    ``time``, ``dx_baseline``, ``age_baseline``) each form their own separate
    family and are labelled exploratory (§2.5.3).

    Hemispheres are separate tests and count toward family size: a 14-region
    analysis is a 28-test family. Stating the wrong one understates the
    correction burden.

    Args:
        fits: Region fits to annotate with ``q_value``.
        alpha: Target false discovery rate.
    """
    testable = [f for f in fits if f.converged and f.primary_p is not None]
    if not testable:
        return

    p_values = np.array([f.primary_p for f in testable], dtype=float)
    order = np.argsort(p_values)
    n = len(p_values)
    ranked = p_values[order]

    q = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward, per Benjamini-Hochberg.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    for position, fit_index in enumerate(order):
        testable[fit_index].q_value = float(q[position])


def fit_model(observations: pd.DataFrame, analysis_config: AnalysisConfig) -> ModelResults:
    """Fit the longitudinal model across the configured region set.

    v0.1.0 fits **one region** — the walking skeleton proves the wiring, not
    the science. Week 5 removes the truncation and the same code fits all 28.

    Args:
        observations: QC-annotated canonical observations.
        analysis_config: Inclusion policy, region set, and FDR level.

    Returns:
        Model results including per-region convergence status.
    """
    included = observations
    if "analysis_included" in observations.columns:
        included = observations[observations["analysis_included"].fillna(False)]

    region_specs = list(v1_region_set())
    skeleton_note = (
        f"v0.1.0 walking skeleton: fitting 1 of {V1_TEST_COUNT} regions. The full "
        "region set and its multiplicity family land in week 5."
    )
    region_specs = region_specs[:1]

    fits = [
        fit_region(included, region, str(measure_type))
        for region, _hemisphere, measure_type in region_specs
    ]
    apply_fdr(fits, analysis_config.fdr_alpha)

    modeled = sum(f.n_observations for f in fits)
    notes = [
        skeleton_note,
        (
            f"Primary family: the {PRIMARY_TERM} coefficient across "
            f"{len(fits)} regional test(s). BH-FDR is applied within this family "
            "and only this family. At full scale the family is "
            f"{V1_TEST_COUNT} tests (14 structures x 2 hemispheres)."
        ),
    ]
    unidentifiable = [f.region for f in fits if not f.estimable]
    if unidentifiable:
        notes.append(
            "This dataset carries no within-subject time variance, so the "
            f"{PRIMARY_TERM} coefficient is not identifiable for {len(unidentifiable)} "
            "of the attempted region(s). These are reported as not estimable rather "
            "than as failed fits: the limitation is in the study design, not in the "
            "optimizer, and a cross-sectional dataset cannot validate a longitudinal "
            "specification (§1.3)."
        )
    failed = [f.region for f in fits if f.estimable and not f.converged]
    if failed:
        notes.append(f"Non-converged regions reported rather than dropped: {failed}")

    return ModelResults(
        fits=fits,
        family_size=len(fits),
        fdr_alpha=analysis_config.fdr_alpha,
        n_modeled_observations=modeled,
        notes=notes,
    )


def run_model(
    observations: pd.DataFrame, analysis_config: AnalysisConfig, outdir: Path | str
) -> ModelResults:
    """Fit the model and persist per-region results.

    Args:
        observations: QC-annotated canonical observations.
        analysis_config: Analysis configuration.
        outdir: Destination directory.

    Returns:
        The results, already written to ``model_results.parquet``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    results = fit_model(observations, analysis_config)
    results.to_frame().to_parquet(out / "model_results.parquet", index=False)
    (out / "model_results.json").write_text(
        json.dumps(results.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    return results
