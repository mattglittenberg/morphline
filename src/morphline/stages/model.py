"""Longitudinal mixed-effects modeling (BUILD_PLAN §2.5).

Fits the §2.5.1 specification with statsmodels MixedLM across the v1 region
set — 14 structures × 2 hemispheres, so **28 tests, not 14** (§2.5.2).

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
from typing import Any, Final

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from morphline.config import AnalysisConfig
from morphline.regions import V1_STRUCTURES, V1_TEST_COUNT, v1_region_set

#: The model formula, fixed effects only. Random effects are specified
#: separately via ``re_formula``.
FIXED_EFFECTS_FORMULA = (
    "value ~ age_baseline + time + dx_baseline + time:dx_baseline + sex + etiv_baseline"
)

#: Random intercept and random slope on time, grouped by subject.
RANDOM_EFFECTS_FORMULA = "~ time"

#: The coefficient carrying the primary hypothesis, as statsmodels names it.
PRIMARY_TERM = "time:dx_baseline[T.patient]"

#: Secondary effects, each of which forms its **own** multiplicity family
#: (§2.5.3). They are corrected separately from the primary family and from
#: each other; pooling them into one family, or into the primary family, would
#: misstate the correction burden in both directions.
SECONDARY_TERMS: Final = ("time", "dx_baseline[T.patient]", "age_baseline")

#: Optimizers tried in order before the model is simplified.
#:
#: statsmodels' MixedLM reports ``converged=False`` from ``lbfgs`` fairly
#: readily even where the estimate is stable — the flag reflects the
#: optimizer's own stopping criterion, not the quality of the optimum. Rather
#: than trusting one optimizer or quietly ignoring the flag, escalate through
#: more robust methods first.
_OPTIMIZERS = ("lbfgs", "powell", "nm")


def term_slug(term: str) -> str:
    """Return a column-safe name for a statsmodels term.

    Args:
        term: A term as statsmodels names it, e.g. ``dx_baseline[T.patient]``.

    Returns:
        The term without its treatment-contrast suffix.
    """
    return term.split("[")[0]


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
        q_values: BH-adjusted p-values by term, filled in by
            :func:`apply_multiplicity`. Keyed by term because each term belongs
            to a different family (§2.5.3) — a single ``q`` could only describe
            one of them, and which one it described would be invisible.
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
    q_values: dict[str, float] = field(default_factory=dict)
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

    @property
    def q_value(self) -> float | None:
        """BH-adjusted p-value for the primary term."""
        return self.q_values.get(PRIMARY_TERM)


@dataclass(slots=True)
class ArmComparison:
    """One region's primary estimate under both harmonization arms.

    Attributes:
        region: Canonical region name.
        harmonized: Primary-term estimate from the harmonized arm.
        unharmonized: Primary-term estimate from the unharmonized arm.
        difference: ``harmonized - unharmonized``, or ``None`` when either arm
            produced no estimate.
        sign_flipped: Whether the two arms disagree about the *direction* of
            the effect. The loudest possible form of conclusion-dependence.
        harmonized_q: BH-adjusted p from the harmonized arm.
        unharmonized_q: BH-adjusted p from the unharmonized arm.
        significance_changed: Whether the arms disagree about crossing the FDR
            threshold. ``False`` when either q is missing, in which case
            :attr:`comparable` is what says so.
        comparable: Whether both arms produced an estimate at all.
    """

    region: str
    harmonized: float | None
    unharmonized: float | None
    difference: float | None
    sign_flipped: bool
    harmonized_q: float | None
    unharmonized_q: float | None
    significance_changed: bool
    comparable: bool


@dataclass(slots=True)
class SensitivityComparison:
    """Harmonized versus unharmonized estimates (§2.3.1, §4 week 5).

    **This is a sensitivity analysis, not a resolution.** It shows how far the
    conclusions move when the harmonization assumption is changed; it cannot
    establish which arm is correct. Where scanner is confounded with time the
    two are not distinguishable from the observed data at all — a property of
    the study design, not of the software — and a small difference between arms
    is not evidence that the confound is benign.

    Attributes:
        applicable: Whether a genuine comparison was possible. ``False`` when
            harmonization changed no values, which makes the two arms the same
            run reported twice.
        rows: Per-region comparison.
        n_sign_flips: Regions where the arms disagree on direction.
        n_significance_changes: Regions where the arms disagree on crossing the
            FDR threshold.
        n_comparable: Regions where both arms produced an estimate.
        notes: Messages for the report.
    """

    applicable: bool
    rows: list[ArmComparison] = field(default_factory=list)
    n_sign_flips: int = 0
    n_significance_changes: int = 0
    n_comparable: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the comparison as a plain mapping for templating and JSON."""
        return {
            "applicable": self.applicable,
            "rows": [asdict(r) for r in self.rows],
            "n_sign_flips": self.n_sign_flips,
            "n_significance_changes": self.n_significance_changes,
            "n_comparable": self.n_comparable,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ModelResults:
    """Results across all fitted regions.

    Attributes:
        fits: Per-region fits.
        family_size: Number of tests *declared* in the primary multiplicity
            family — one per region attempted. :attr:`family_sizes` reports how
            many were actually corrected, which is smaller wherever a fit did
            not converge.
        fdr_alpha: False discovery rate applied within each family.
        n_modeled_observations: Total observations that entered any model.
        sensitivity: Harmonized-versus-unharmonized comparison, when a second
            arm was supplied. ``None`` means the comparison was not run, which
            the report must state rather than leave to be inferred from an
            absent table.
        notes: Messages for the report.
    """

    fits: list[RegionFit] = field(default_factory=list)
    family_size: int = 0
    fdr_alpha: float = 0.05
    n_modeled_observations: int = 0
    sensitivity: SensitivityComparison | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def family_sizes(self) -> dict[str, int]:
        """Tests actually corrected within each family, keyed by term."""
        return family_sizes(self.fits)

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
        """Return per-region results as a flat frame.

        The primary family keeps the unqualified ``estimate`` / ``p_value`` /
        ``q_value`` names; secondary families are suffixed by term. Raw *p* and
        *q* appear for every test in every family (§2.5.3).
        """
        by_region = {r.region: r for r in (self.sensitivity.rows if self.sensitivity else [])}
        rows: list[dict[str, Any]] = []
        for f in self.fits:
            arm = by_region.get(f.region)
            row: dict[str, Any] = {
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
                # Always present, null when no second arm was run, so the
                # frame's schema does not depend on how it was invoked.
                "estimate_unharmonized": arm.unharmonized if arm else None,
                "sensitivity_difference": arm.difference if arm else None,
                "sensitivity_sign_flipped": arm.sign_flipped if arm else None,
            }
            for term in SECONDARY_TERMS:
                slug = term_slug(term)
                row[f"estimate_{slug}"] = f.coefficients.get(term)
                row[f"std_error_{slug}"] = f.std_errors.get(term)
                row[f"p_value_{slug}"] = f.p_values.get(term)
                row[f"q_value_{slug}"] = f.q_values.get(term)
            rows.append(row)
        return pd.DataFrame(rows)

    def as_dict(self) -> dict[str, Any]:
        """Return results as a plain mapping for templating."""
        return {
            # ``q_value`` is a property, so ``asdict`` does not carry it.
            "fits": [{**asdict(f), "q_value": f.q_value} for f in self.fits],
            "family_size": self.family_size,
            "family_sizes": self.family_sizes,
            "primary_term": PRIMARY_TERM,
            "secondary_terms": list(SECONDARY_TERMS),
            "fdr_alpha": self.fdr_alpha,
            "n_modeled_observations": self.n_modeled_observations,
            "convergence_rate": self.convergence_rate,
            "n_estimable": self.n_estimable,
            "sensitivity": self.sensitivity.as_dict() if self.sensitivity else None,
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


def apply_fdr(fits: list[RegionFit], alpha: float, term: str = PRIMARY_TERM) -> None:
    """Apply Benjamini-Hochberg FDR within one family, in place.

    A *family* is one coefficient across the regions in the region set. The
    primary family is **the ``time:dx_baseline`` coefficient and only that**;
    each secondary effect forms its own family and is corrected separately
    (§2.5.3). This function corrects exactly one of them, so the caller cannot
    accidentally pool two families by passing both at once.

    Hemispheres are separate tests and count toward family size: a 14-region
    analysis is a 28-test family. Stating the wrong one understates the
    correction burden.

    Args:
        fits: Region fits to annotate with a ``q_values`` entry.
        alpha: Target false discovery rate.
        term: The coefficient whose family is being corrected.
    """
    testable = [f for f in fits if f.converged and f.p_values.get(term) is not None]
    if not testable:
        return

    p_values = np.array([f.p_values[term] for f in testable], dtype=float)
    order = np.argsort(p_values)
    n = len(p_values)
    ranked = p_values[order]

    q = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward, per Benjamini-Hochberg.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    for position, fit_index in enumerate(order):
        testable[fit_index].q_values[term] = float(q[position])


def apply_multiplicity(fits: list[RegionFit], alpha: float) -> None:
    """Correct the primary family and every secondary family, separately.

    Each term is corrected against its own family, never against the union:
    the primary family answers the declared hypothesis, and the secondary
    families are exploratory. Correcting them together would inflate every
    family to four times its true size and understate the primary result;
    correcting the primary term against the union would let exploratory tests
    determine the significance of the declared one.

    Args:
        fits: Region fits to annotate.
        alpha: Target false discovery rate, applied within each family.
    """
    for term in (PRIMARY_TERM, *SECONDARY_TERMS):
        apply_fdr(fits, alpha, term)


def family_sizes(fits: list[RegionFit]) -> dict[str, int]:
    """Return the number of tests actually corrected within each family.

    Distinct from the *declared* family size: a fit that did not converge
    carries no p-value and is excluded, so it must not inflate the family and
    dilute the correction for the regions that did converge.

    Args:
        fits: Region fits, after fitting.

    Returns:
        Term to number of tests entering that term's BH correction.
    """
    return {
        term: sum(1 for f in fits if f.converged and f.p_values.get(term) is not None)
        for term in (PRIMARY_TERM, *SECONDARY_TERMS)
    }


def fit_model(observations: pd.DataFrame, analysis_config: AnalysisConfig) -> ModelResults:
    """Fit the longitudinal model across the configured region set.

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

    fits = [
        fit_region(included, region, str(measure_type))
        for region, _hemisphere, measure_type in region_specs
    ]
    apply_multiplicity(fits, analysis_config.fdr_alpha)

    sizes = family_sizes(fits)
    modeled = sum(f.n_observations for f in fits)
    notes = [
        (
            f"Primary family: the {PRIMARY_TERM} coefficient across "
            f"{len(fits)} regional tests ({len(V1_STRUCTURES)} structures x 2 "
            f"hemispheres = {V1_TEST_COUNT}). BH-FDR at alpha="
            f"{analysis_config.fdr_alpha} is applied within this family and only "
            f"this family; {sizes[PRIMARY_TERM]} tests carried a p-value and were "
            "corrected."
        ),
        (
            "Secondary families, each corrected separately and labelled "
            "exploratory (§2.5.3): "
            + ", ".join(f"{term} ({sizes[term]} tests)" for term in SECONDARY_TERMS)
            + ". A secondary q-value must never be read as if it belonged to the "
            "primary family."
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


#: Observation identity, used to align the two arms before comparing values.
_OBSERVATION_KEY: Final = ("subject_id", "session_id", "region", "measure_type")


def _arms_differ(harmonized: pd.DataFrame, unharmonized: pd.DataFrame) -> bool:
    """Return whether harmonization actually changed any value.

    Comparing the frames rather than trusting a configuration flag: a run with
    harmonization enabled that ends up changing nothing — every batch too small
    to estimate, say — must not present the same numbers twice as though they
    were a comparison.
    """
    key = [c for c in _OBSERVATION_KEY if c in harmonized.columns and c in unharmonized.columns]
    if not key or "value" not in harmonized.columns or "value" not in unharmonized.columns:
        return True
    left = harmonized.sort_values(key)["value"].to_numpy(dtype=float)
    right = unharmonized.sort_values(key)["value"].to_numpy(dtype=float)
    if left.shape != right.shape:
        return True
    return not bool(np.allclose(left, right, equal_nan=True))


def compare_arms(
    harmonized: ModelResults,
    unharmonized: ModelResults,
    alpha: float,
    *,
    applicable: bool = True,
) -> SensitivityComparison:
    """Compare the primary estimate across harmonization arms (§2.3.1).

    Args:
        harmonized: Results from the harmonized arm, which is the primary run.
        unharmonized: Results from the same specification on unharmonized
            values.
        alpha: FDR threshold, used only to ask whether the arms disagree about
            crossing it.
        applicable: Whether the two arms are genuinely different runs.

    Returns:
        The per-region comparison and its headline counts.
    """
    unharmonized_by_region = {f.region: f for f in unharmonized.fits}

    rows: list[ArmComparison] = []
    for fit in harmonized.fits:
        other = unharmonized_by_region.get(fit.region)
        left = fit.primary_estimate if fit.converged else None
        right = other.primary_estimate if other is not None and other.converged else None

        difference: float | None = None
        sign_flipped = False
        if left is not None and right is not None:
            difference = left - right
            # Only a genuine direction disagreement counts. An estimate of
            # exactly zero has no direction to disagree about.
            sign_flipped = left * right < 0

        left_q = fit.q_value
        right_q = other.q_value if other is not None else None
        significance_changed = (
            left_q is not None and right_q is not None and (left_q <= alpha) != (right_q <= alpha)
        )

        rows.append(
            ArmComparison(
                region=fit.region,
                harmonized=left,
                unharmonized=right,
                difference=difference,
                sign_flipped=sign_flipped,
                harmonized_q=left_q,
                unharmonized_q=right_q,
                significance_changed=significance_changed,
                comparable=left is not None and right is not None,
            )
        )

    n_sign_flips = sum(1 for r in rows if r.sign_flipped)
    n_significance_changes = sum(1 for r in rows if r.significance_changed)
    n_comparable = sum(1 for r in rows if r.comparable)

    notes: list[str] = []
    if not applicable:
        notes.append(
            "Harmonization changed no values, so both arms are the same run. No "
            "sensitivity comparison is possible and none should be read from the "
            "table below."
        )
    elif n_comparable == 0:
        notes.append(
            f"No region produced an estimate under both arms, so of {len(rows)} regions "
            "none could be compared and the sensitivity question is unanswered here. "
            "This is not agreement between the arms."
        )
    else:
        notes.append(
            f"Sensitivity analysis, not inference: {n_comparable} of {len(rows)} regions "
            "produced an estimate under both arms. The comparison shows how far "
            "conclusions move when the harmonization assumption changes; it cannot "
            "establish which arm is correct."
        )
    if n_sign_flips:
        notes.append(
            f"{n_sign_flips} region(s) change the *direction* of the estimated effect "
            "between arms. Where that happens the data alone do not determine the "
            "sign, and no directional claim about those regions is supportable."
        )
    if n_significance_changes:
        notes.append(
            f"{n_significance_changes} region(s) cross the q<={alpha} threshold in one "
            "arm but not the other, so their apparent significance is a consequence of "
            "the harmonization choice rather than of the data."
        )

    return SensitivityComparison(
        applicable=applicable,
        rows=rows,
        n_sign_flips=n_sign_flips,
        n_significance_changes=n_significance_changes,
        n_comparable=n_comparable,
        notes=notes,
    )


def fit_with_sensitivity(
    harmonized: pd.DataFrame,
    unharmonized: pd.DataFrame,
    analysis_config: AnalysisConfig,
) -> ModelResults:
    """Fit both arms and attach the comparison to the primary results.

    The **harmonized** arm is primary; the unharmonized arm exists to show how
    much the conclusions depend on that choice. Neither is promoted to
    inference where the scanner/time confound is present (§2.3.1).

    Args:
        harmonized: Harmonized, QC-annotated observations.
        unharmonized: The same observations before harmonization.
        analysis_config: Inclusion policy, region set, and FDR level.

    Returns:
        The harmonized arm's results, carrying :attr:`ModelResults.sensitivity`.
    """
    primary = fit_model(harmonized, analysis_config)
    comparison_arm = fit_model(unharmonized, analysis_config)
    primary.sensitivity = compare_arms(
        primary,
        comparison_arm,
        analysis_config.fdr_alpha,
        applicable=_arms_differ(harmonized, unharmonized),
    )
    return primary


def run_model(
    observations: pd.DataFrame,
    analysis_config: AnalysisConfig,
    outdir: Path | str,
    *,
    unharmonized: pd.DataFrame | None = None,
) -> ModelResults:
    """Fit the model and persist per-region results.

    Args:
        observations: QC-annotated canonical observations. Harmonized, where
            harmonization ran.
        analysis_config: Analysis configuration.
        outdir: Destination directory.
        unharmonized: The same observations before harmonization. Supplying
            them adds the §2.3.1 sensitivity arm; omitting them means the
            comparison was not run, which the report says explicitly.

    Returns:
        The results, already written to ``model_results.parquet``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    results = (
        fit_model(observations, analysis_config)
        if unharmonized is None
        else fit_with_sensitivity(observations, unharmonized, analysis_config)
    )
    results.to_frame().to_parquet(out / "model_results.parquet", index=False)
    (out / "model_results.json").write_text(
        json.dumps(results.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    return results
