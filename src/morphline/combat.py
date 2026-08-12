"""In-repo empirical-Bayes ComBat (BUILD_PLAN §2.3.4, Build.md deviation 1).

Why the estimator is written here rather than taken from ``neuroHarmonize`` or
``neuroCombat``: ``neuroHarmonize`` pins ``numpy==1.26.4``, which would hold the
whole project on numpy 1.x and drag pandas, scipy, and statsmodels back with it.
Writing it in numpy removes the pin, and it converts the §2.3.2 validation suite
from "does the library run" into "does *our* estimator recover the injected
batch parameter" — which is the stronger claim.

The method is the parametric empirical Bayes of Johnson, Li & Rabinovic (2007).
For batch *i*, region *r*, and observation *j*::

    y_irj = alpha_r + X_irj B_r + gamma_ir + delta_ir * eps_irj

``alpha_r`` is the region's grand mean, ``X B_r`` the preserved biological
covariate block, ``gamma_ir`` the batch's additive shift, and ``delta_ir`` its
residual scale. The estimator standardizes each region, shrinks the per-batch
moments toward a normal prior on ``gamma`` and an inverse-gamma prior on
``delta**2``, then back-transforms.

**Shrinkage pools across regions within a batch, not across batches within a
region.** This is the direction Johnson et al. specify and it is easy to get
backwards: the prior for site *i* is estimated from the distribution of that
site's effect across all 28 regions, so a batch with few subjects borrows
strength from the regions it also appears in. Pooling the other way would form
a prior from as many values as there are sites — three, on the fixture configs —
which is not a distribution. The pooling is legitimate precisely *because* the
standardization step already divided each region by its own residual SD, so
``gamma`` and ``delta`` are dimensionless and comparable across regions that
differ by three orders of magnitude.

One consequence worth knowing: a region's adjusted values depend on the other
regions in the frame. Harmonizing one region alone is well defined but gets no
shrinkage, because a single value has no variance to form a prior from.

**Covariate preservation is the whole point** (§2.3.4). Biological covariates
enter the design matrix, so their variation is accounted for before the batch
moments are taken and is restored afterwards. A harmonizer that omits them
absorbs biology into the batch term and deletes the effect the study exists to
measure — silently, and while passing any check that only asks whether site
means converged.

Scope, stated rather than implied
---------------------------------
* **Parametric priors only.** The non-parametric variant is not implemented.
* **Standard, not longitudinal, ComBat.** It assumes independent observations,
  which longitudinal data violates; Beer et al. (2020) is a v1.1 target. Where
  batch is confounded with time the two effects are not identifiable from the
  data at all — a property of the study design that no estimator repairs. That
  analysis lives in :mod:`morphline.stages.harmonize`.
* **``gamma`` absorbs a multiplicative mean shift.** Where the true batch effect
  scales values rather than offsetting them, the additive term picks up the
  induced change in the mean. A recovery test must therefore compare ``gamma``
  against the *realized* mean shift, not against a configured additive offset.

Nothing here raises on bad input. Consistent with the parser's reason-coded
parse failures, a region that cannot be estimated is recorded on
:attr:`ComBatFit.skipped` with a :class:`ComBatSkipCode` and its values pass
through unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

#: A batch needs at least this many observations before it has a variance worth
#: estimating. Below it the batch is excluded from estimation and its rows pass
#: through unadjusted rather than being scaled by a one-sample "variance".
MIN_ROWS_PER_BATCH: Final = 2

#: Standard ComBat is a two-batch-minimum method: with one batch there is no
#: contrast to estimate and the grand mean absorbs everything.
MIN_BATCHES: Final = 2

#: Residual spread below this fraction of the region's own scale counts as no
#: spread at all. The test is relative because the v1 region set spans mm and
#: mm³ — an absolute floor would be far too coarse for thickness and far too
#: fine for volume. A constant region reaches least squares with a residual of
#: ~1e-13 rather than exactly 0, and standardizing by that would turn rounding
#: error into a batch effect several orders of magnitude larger than the data.
RELATIVE_VARIANCE_FLOOR: Final = 1e-9

_VALUE: Final = "value"
_REGION: Final = "region"
_MEASURE_TYPE: Final = "measure_type"


class ComBatSkipCode(StrEnum):
    """Why a region was left unharmonized.

    Reason-coded rather than raised, on the same rule as
    :class:`~morphline.parsers.errors.ParseFailureCode`: a region that cannot be
    estimated is a fact to report, not an exception to propagate out of a stage
    that has 27 other regions to process.
    """

    SINGLE_BATCH = "single_batch"
    """Fewer than two batches had enough observations to estimate."""

    INSUFFICIENT_ROWS = "insufficient_rows"
    """Too few usable observations to fit the design matrix."""

    ZERO_VARIANCE = "zero_variance"
    """Residual variance is zero, so standardization is undefined."""

    SINGULAR_DESIGN = "singular_design"
    """The design matrix stayed rank-deficient after dropping covariates."""

    NONFINITE_VALUES = "nonfinite_values"
    """Values contained NaN or infinity where a number was required."""


class CovariateDropCode(StrEnum):
    """Why a declared biological covariate was left out of the design matrix.

    Dropping a covariate weakens the preservation guarantee, so each drop is
    recorded and surfaced rather than being handled quietly. Two of these are
    routine on real data: ``dx_baseline`` is all-null on healthy-control-only
    datasets, and ``time_from_baseline_years`` is constant on every
    cross-sectional one, ABIDE included.
    """

    ABSENT = "absent_from_observations"
    ALL_NULL = "all_null"
    CONSTANT = "constant"
    COLLINEAR_WITH_BATCH = "collinear_with_batch"


@dataclass(frozen=True, slots=True)
class BatchParameters:
    """Fitted ComBat parameters for one region.

    Attributes:
        region: Canonical region name.
        measure_type: The measure the parameters were fitted on.
        grand_mean: ``alpha``, the batch-size-weighted grand mean.
        pooled_sd: Pooled residual standard deviation used to standardize.
        gamma_star: Shrunk additive batch term per batch, on the standardized
            scale. Multiply by ``pooled_sd`` for native units.
        delta_star: Shrunk residual **standard deviation ratio** per batch — the
            divisor in the back-transform, not the variance. A value near 1.0
            means the batch's residual spread matches the pooled spread.
        n_per_batch: Observations per batch that entered estimation.
        covariate_terms: Design-matrix column names for the covariate block.
        covariate_coefficients: Fitted coefficient per covariate term.
        n_adjusted: Observations whose values were adjusted.
        n_unadjusted: Observations left unchanged because their batch was not
            estimable or their covariates were incomplete.
        converged: Whether the empirical-Bayes iteration met the tolerance.
        n_iterations: Iterations taken.
        shrinkage_applied: Whether empirical-Bayes shrinkage was used. ``False``
            means the per-batch moments were taken at face value, either by
            configuration or because the prior was degenerate.
    """

    region: str
    measure_type: str
    grand_mean: float
    pooled_sd: float
    gamma_star: dict[str, float]
    delta_star: dict[str, float]
    n_per_batch: dict[str, int]
    covariate_terms: tuple[str, ...]
    covariate_coefficients: dict[str, float]
    n_adjusted: int
    n_unadjusted: int
    converged: bool
    n_iterations: int
    shrinkage_applied: bool

    def native_gamma(self) -> dict[str, float]:
        """Return the additive batch terms in the measure's native units.

        ``gamma_star`` is estimated on the standardized scale, which is not
        comparable across regions whose values differ by three orders of
        magnitude. Rescaling by the pooled SD puts it back in mm or mm³.

        Returns:
            Batch name to additive shift in native units.
        """
        return {batch: value * self.pooled_sd for batch, value in self.gamma_star.items()}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view for the report and provenance.

        Returns:
            The parameters as plain Python types.
        """
        return {
            "region": self.region,
            "measure_type": self.measure_type,
            "grand_mean": self.grand_mean,
            "pooled_sd": self.pooled_sd,
            "gamma_star": self.gamma_star,
            "gamma_native": self.native_gamma(),
            "delta_star": self.delta_star,
            "n_per_batch": self.n_per_batch,
            "covariate_terms": list(self.covariate_terms),
            "covariate_coefficients": self.covariate_coefficients,
            "n_adjusted": self.n_adjusted,
            "n_unadjusted": self.n_unadjusted,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "shrinkage_applied": self.shrinkage_applied,
        }


@dataclass(frozen=True, slots=True)
class ComBatFit:
    """Everything estimated across every region.

    Attributes:
        parameters: ``(region, measure_type)`` to its fitted parameters.
        skipped: ``(region, measure_type)`` to the reason it was not harmonized.
        covariates_requested: Covariates the caller asked to preserve.
        covariates_dropped: Covariate name to the reason it was dropped. A
            covariate dropped in one region but usable in another is recorded
            here once, because a partial preservation guarantee is the thing
            worth surfacing.
    """

    parameters: dict[tuple[str, str], BatchParameters]
    skipped: dict[tuple[str, str], str]
    covariates_requested: tuple[str, ...]
    covariates_dropped: dict[str, str]

    @property
    def n_groups_harmonized(self) -> int:
        """How many regions were adjusted."""
        return len(self.parameters)

    @property
    def n_groups_skipped(self) -> int:
        """How many regions were left unharmonized."""
        return len(self.skipped)

    @property
    def covariates_used(self) -> tuple[str, ...]:
        """Covariates that survived into at least one design matrix."""
        return tuple(c for c in self.covariates_requested if c not in self.covariates_dropped)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view for the report and provenance.

        Returns:
            The fit as plain Python types, with tuple keys flattened to strings
            so the result survives a JSON round trip.
        """
        return {
            "n_groups_harmonized": self.n_groups_harmonized,
            "n_groups_skipped": self.n_groups_skipped,
            "covariates_requested": list(self.covariates_requested),
            "covariates_used": list(self.covariates_used),
            "covariates_dropped": self.covariates_dropped,
            "parameters": [p.as_dict() for p in self.parameters.values()],
            "skipped": {f"{region}:{measure}": r for (region, measure), r in self.skipped.items()},
        }


@dataclass(frozen=True, slots=True)
class ComBatResult:
    """Adjusted values alongside the parameters that produced them.

    Attributes:
        values: Adjusted values, indexed to match the input frame. Rows in
            skipped regions carry their original value unchanged.
        fit: The fitted parameters and per-region skip reasons.
    """

    values: pd.Series
    fit: ComBatFit


def _is_numeric(series: pd.Series) -> bool:
    """Whether a column should enter the design as a number rather than dummies."""
    return bool(pd.api.types.is_numeric_dtype(series)) and not bool(
        pd.api.types.is_bool_dtype(series)
    )


def _encode_covariates(
    frame: pd.DataFrame, covariates: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build the covariate design block over every row of a region.

    Encoding once over the full region — rather than separately over the
    estimation subset — is what keeps the design columns identical between
    fitting and application. A categorical level seen only outside the
    estimation set would otherwise produce a column the fit never saw.

    Levels are sorted and the first is the reference, so the encoding is
    deterministic across the in-process and staged paths. Dummy columns are
    named ``covariate[T.level]``, matching how ``stages.model`` names the same
    contrast, so a reader meets one vocabulary rather than two.

    Args:
        frame: Rows for one region.
        covariates: Covariate column names to preserve.

    Returns:
        The design block as float64 columns, and the reason each dropped
        covariate was dropped.
    """
    blocks: list[pd.DataFrame] = []
    dropped: dict[str, str] = {}

    for name in covariates:
        if name not in frame.columns:
            dropped[name] = str(CovariateDropCode.ABSENT)
            continue

        series = frame[name]
        if bool(series.isna().all()):
            dropped[name] = str(CovariateDropCode.ALL_NULL)
            continue

        if _is_numeric(series):
            values = pd.to_numeric(series, errors="coerce").astype("float64")
            if float(values.std(skipna=True)) == 0.0 or bool(values.isna().all()):
                dropped[name] = str(CovariateDropCode.CONSTANT)
                continue
            blocks.append(values.to_frame(name))
            continue

        labels = series.astype("string")
        levels = sorted({str(v) for v in labels.dropna().unique()})
        if len(levels) < 2:
            dropped[name] = str(CovariateDropCode.CONSTANT)
            continue

        dummies = pd.DataFrame(index=frame.index)
        for level in levels[1:]:
            dummies[f"{name}[T.{level}]"] = (labels == level).astype("float64")
        dummies[labels.isna().to_numpy()] = np.nan
        blocks.append(dummies)

    if not blocks:
        return pd.DataFrame(index=frame.index), dropped
    return pd.concat(blocks, axis=1), dropped


def _moment_priors(delta2_hat: npt.NDArray[np.float64]) -> tuple[float, float] | None:
    """Match an inverse-gamma prior to one batch's variance estimates.

    The estimates are that batch's residual variance in each region it appears
    in, so the prior describes how much a site's variance inflation varies
    across the brain.

    Args:
        delta2_hat: The batch's residual variance per region.

    Returns:
        The ``(shape, scale)`` hyperparameters, or ``None`` when the moments are
        degenerate — one region, or identical variances everywhere, leaves the
        prior undefined and shrinkage has nothing to pull toward.
    """
    if delta2_hat.size < 2:
        return None
    mean = float(delta2_hat.mean())
    variance = float(delta2_hat.var(ddof=1))
    if not np.isfinite(mean) or not np.isfinite(variance) or variance <= 0.0 or mean <= 0.0:
        return None
    shape = (2.0 * variance + mean**2) / variance
    scale = (mean * variance + mean**3) / variance
    if not np.isfinite(shape) or not np.isfinite(scale):
        return None
    return shape, scale


@dataclass(slots=True)
class _RegionState:
    """What the standardization pass produces for one region.

    Attributes:
        key: ``(region, measure_type)``.
        values: Original values for every row in the region.
        batches: Batch label per row.
        estimable: Batches with enough observations to estimate, ordered.
        adjust_rows: Rows eligible to be transformed.
        design: Encoded covariate block over every row in the region.
        terms: Retained covariate design column names.
        cov_coefficients: Fitted covariate coefficients.
        grand_mean: The region's grand mean.
        pooled_sd: The region's pooled residual standard deviation.
        codes_fit: Batch position per estimation row.
        z_fit: Standardized estimation values.
        counts: Estimation observations per batch.
        gamma_hat: Unshrunk additive batch term per batch.
        delta2_hat: Unshrunk residual variance per batch.
    """

    key: tuple[str, str]
    values: pd.Series
    batches: pd.Series
    estimable: list[str]
    adjust_rows: pd.Series
    design: pd.DataFrame
    terms: tuple[str, ...]
    cov_coefficients: npt.NDArray[np.float64]
    grand_mean: float
    pooled_sd: float
    codes_fit: npt.NDArray[np.intp]
    z_fit: npt.NDArray[np.float64]
    counts: npt.NDArray[np.float64]
    gamma_hat: npt.NDArray[np.float64]
    delta2_hat: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _BatchPrior:
    """Hyperparameters for one batch, pooled across the regions it appears in.

    Attributes:
        gamma_bar: Prior mean of the additive term.
        tau2: Prior variance of the additive term.
        shape: Inverse-gamma shape for the residual variance.
        scale: Inverse-gamma scale for the residual variance.
        n_regions: Regions the hyperparameters were estimated from.
    """

    gamma_bar: float
    tau2: float
    shape: float
    scale: float
    n_regions: int


def _drop_collinear(
    onehot: npt.NDArray[np.float64],
    covariates: npt.NDArray[np.float64],
    terms: tuple[str, ...],
    dropped: dict[str, str],
) -> tuple[npt.NDArray[np.float64], tuple[str, ...]]:
    """Drop covariate columns that the batch indicators already explain.

    A covariate perfectly predicted by batch — a site that recruited only
    patients, say — makes the design rank-deficient, and the least-squares
    solution would split the shared variation between the two arbitrarily. The
    batch indicators are kept and the covariate goes, because a batch term the
    method cannot estimate is worse than a covariate it cannot preserve.

    Args:
        onehot: Batch indicator block.
        covariates: Covariate design block.
        terms: Column names for the covariate block.
        dropped: Drop reasons, extended in place.

    Returns:
        The retained covariate block and its column names.
    """
    keep: list[int] = []
    kept_terms: list[str] = []
    current = onehot
    for position, term in enumerate(terms):
        candidate = np.hstack([current, covariates[:, [position]]])
        if np.linalg.matrix_rank(candidate) == candidate.shape[1]:
            keep.append(position)
            kept_terms.append(term)
            current = candidate
        else:
            dropped.setdefault(term.split("[")[0], str(CovariateDropCode.COLLINEAR_WITH_BATCH))
    if not keep:
        return np.zeros((onehot.shape[0], 0), dtype=np.float64), ()
    return covariates[:, keep], tuple(kept_terms)


def _standardize_region(
    frame: pd.DataFrame,
    *,
    key: tuple[str, str],
    batch_column: str,
    covariates: Sequence[str],
    estimation_mask: pd.Series,
) -> tuple[_RegionState | None, str | None, dict[str, str]]:
    """Fit the design for one region and take its raw per-batch moments.

    This is everything that can be done without knowing about the other
    regions. The shrinkage that follows needs all of them, so it happens in a
    second pass.

    Args:
        frame: Every row for this region.
        key: ``(region, measure_type)``.
        batch_column: Column holding the batch label.
        covariates: Biological covariates to preserve.
        estimation_mask: Which rows may contribute to estimation.

    Returns:
        ``(state, skip_reason, covariates_dropped)``. Exactly one of ``state``
        and ``skip_reason`` is set.
    """
    values = frame[_VALUE].astype("float64")
    design, dropped = _encode_covariates(frame, covariates)

    complete = values.notna()
    if not design.empty:
        complete &= design.notna().all(axis=1)
    batches = frame[batch_column].astype("string")
    complete &= batches.notna()

    usable = complete & np.isfinite(values.fillna(np.nan).to_numpy())
    eligible = usable & estimation_mask.reindex(frame.index, fill_value=False).fillna(False)

    if not bool(eligible.any()):
        return None, str(ComBatSkipCode.INSUFFICIENT_ROWS), dropped

    sizes = batches[eligible].value_counts()
    estimable = sorted(str(b) for b, n in sizes.items() if int(n) >= MIN_ROWS_PER_BATCH)
    if len(estimable) < MIN_BATCHES:
        return None, str(ComBatSkipCode.SINGLE_BATCH), dropped

    in_batch = batches.isin(estimable)
    fit_rows = eligible & in_batch
    if int(fit_rows.sum()) <= len(estimable) + design.shape[1]:
        return None, str(ComBatSkipCode.INSUFFICIENT_ROWS), dropped

    index = {name: position for position, name in enumerate(estimable)}
    codes_fit = np.asarray([index[str(b)] for b in batches[fit_rows]], dtype=np.intp)
    y_fit = values[fit_rows].to_numpy(dtype=np.float64)
    counts = np.bincount(codes_fit, minlength=len(estimable)).astype(np.float64)

    onehot = np.zeros((y_fit.size, len(estimable)), dtype=np.float64)
    onehot[np.arange(y_fit.size), codes_fit] = 1.0

    terms = tuple(str(c) for c in design.columns)
    cov_fit = (
        design.loc[fit_rows].to_numpy(dtype=np.float64)
        if terms
        else np.zeros((y_fit.size, 0), dtype=np.float64)
    )
    cov_fit, terms = _drop_collinear(onehot, cov_fit, terms, dropped)

    x_fit = np.hstack([onehot, cov_fit])
    if np.linalg.matrix_rank(x_fit) < x_fit.shape[1]:
        return None, str(ComBatSkipCode.SINGULAR_DESIGN), dropped

    coefficients, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    if not bool(np.isfinite(coefficients).all()):
        return None, str(ComBatSkipCode.NONFINITE_VALUES), dropped

    batch_coefficients = coefficients[: len(estimable)]
    cov_coefficients = coefficients[len(estimable) :]
    grand_mean = float((counts / counts.sum()) @ batch_coefficients)

    residuals = y_fit - x_fit @ coefficients
    pooled_sd = float(np.sqrt(float(residuals @ residuals) / y_fit.size))
    scale = max(abs(grand_mean), float(np.max(np.abs(y_fit))), 1.0)
    if not np.isfinite(pooled_sd) or pooled_sd <= RELATIVE_VARIANCE_FLOOR * scale:
        return None, str(ComBatSkipCode.ZERO_VARIANCE), dropped

    z_fit = (y_fit - grand_mean - cov_fit @ cov_coefficients) / pooled_sd
    gamma_hat = np.bincount(codes_fit, weights=z_fit, minlength=len(estimable)) / counts
    delta2_hat = np.asarray(
        [float(np.var(z_fit[codes_fit == position], ddof=1)) for position in range(len(estimable))],
        dtype=np.float64,
    )
    delta2_hat = np.where(delta2_hat > 0.0, delta2_hat, 1.0)

    state = _RegionState(
        key=key,
        values=values,
        batches=batches,
        estimable=estimable,
        adjust_rows=usable & in_batch,
        design=design,
        terms=terms,
        cov_coefficients=cov_coefficients,
        grand_mean=grand_mean,
        pooled_sd=pooled_sd,
        codes_fit=codes_fit,
        z_fit=z_fit,
        counts=counts,
        gamma_hat=gamma_hat,
        delta2_hat=delta2_hat,
    )
    return state, None, dropped


def _batch_priors(states: list[_RegionState]) -> dict[str, _BatchPrior]:
    """Estimate each batch's hyperparameters across the regions it appears in.

    This is the pooling direction Johnson et al. specify. A batch appearing in
    only one region gets no prior — a single value has no variance — and is
    left unshrunk rather than shrunk toward itself.

    Args:
        states: Standardized regions.

    Returns:
        Batch name to its hyperparameters, omitting batches without one.
    """
    collected: dict[str, list[tuple[float, float]]] = {}
    for state in states:
        for position, batch in enumerate(state.estimable):
            collected.setdefault(batch, []).append(
                (float(state.gamma_hat[position]), float(state.delta2_hat[position]))
            )

    priors: dict[str, _BatchPrior] = {}
    for batch, entries in collected.items():
        if len(entries) < 2:
            continue
        gamma = np.asarray([g for g, _d in entries], dtype=np.float64)
        delta2 = np.asarray([d for _g, d in entries], dtype=np.float64)
        moments = _moment_priors(delta2)
        tau2 = float(gamma.var(ddof=1))
        if moments is None or not np.isfinite(tau2):
            continue
        shape, scale = moments
        priors[batch] = _BatchPrior(
            gamma_bar=float(gamma.mean()),
            tau2=tau2,
            shape=shape,
            scale=scale,
            n_regions=len(entries),
        )
    return priors


def _solve_batch(
    prior: _BatchPrior,
    gamma_hat: npt.NDArray[np.float64],
    delta2_hat: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    residual_sums: list[npt.NDArray[np.float64]],
    max_iterations: int,
    tolerance: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool, int]:
    """Iterate the conditional posterior means for one batch across its regions.

    Args:
        prior: The batch's hyperparameters.
        gamma_hat: Unshrunk additive term per region.
        delta2_hat: Unshrunk residual variance per region.
        counts: Estimation observations per region.
        residual_sums: Standardized values for this batch, per region.
        max_iterations: Iteration cap.
        tolerance: Maximum relative change accepted as converged.

    Returns:
        ``(gamma_star, delta2_star, converged, n_iterations)``.
    """
    gamma_old = gamma_hat.copy()
    delta2_old = delta2_hat.copy()

    for iteration in range(1, max_iterations + 1):
        numerator = prior.tau2 * counts * gamma_hat + delta2_old * prior.gamma_bar
        denominator = prior.tau2 * counts + delta2_old
        gamma_new = np.where(denominator > 0.0, numerator / denominator, gamma_hat)

        sum2 = np.asarray(
            [float(((z - gamma_new[i]) ** 2).sum()) for i, z in enumerate(residual_sums)],
            dtype=np.float64,
        )
        delta2_new = (0.5 * sum2 + prior.scale) / (counts / 2.0 + prior.shape - 1.0)
        delta2_new = np.where(delta2_new > 0.0, delta2_new, delta2_old)

        change = max(
            float(np.max(np.abs(gamma_new - gamma_old) / (np.abs(gamma_old) + 1e-12))),
            float(np.max(np.abs(delta2_new - delta2_old) / (np.abs(delta2_old) + 1e-12))),
        )
        gamma_old, delta2_old = gamma_new, delta2_new
        if change < tolerance:
            return gamma_old, delta2_old, True, iteration

    return gamma_old, delta2_old, False, max_iterations


def _shrink(
    states: list[_RegionState], max_iterations: int, tolerance: float
) -> tuple[dict[tuple[tuple[str, str], str], tuple[float, float]], dict[str, tuple[bool, int]]]:
    """Run the empirical-Bayes solve for every batch across every region.

    Args:
        states: Standardized regions.
        max_iterations: Iteration cap for each batch's solve.
        tolerance: Convergence tolerance.

    Returns:
        ``((region_key, batch) -> (gamma_star, delta2_star))`` and
        ``batch -> (converged, iterations)`` for the batches that were shrunk.
    """
    priors = _batch_priors(states)
    shrunk: dict[tuple[tuple[str, str], str], tuple[float, float]] = {}
    status: dict[str, tuple[bool, int]] = {}

    for batch, prior in priors.items():
        members = [(s, s.estimable.index(batch)) for s in states if batch in s.estimable]
        gamma_hat = np.asarray([s.gamma_hat[i] for s, i in members], dtype=np.float64)
        delta2_hat = np.asarray([s.delta2_hat[i] for s, i in members], dtype=np.float64)
        counts = np.asarray([s.counts[i] for s, i in members], dtype=np.float64)
        residuals = [s.z_fit[s.codes_fit == i] for s, i in members]

        gamma_star, delta2_star, converged, iterations = _solve_batch(
            prior, gamma_hat, delta2_hat, counts, residuals, max_iterations, tolerance
        )
        status[batch] = (converged, iterations)
        for position, (state, _index) in enumerate(members):
            shrunk[(state.key, batch)] = (
                float(gamma_star[position]),
                float(delta2_star[position]),
            )

    return shrunk, status


def _adjust_region(
    state: _RegionState,
    gamma_star: npt.NDArray[np.float64],
    delta_star: npt.NDArray[np.float64],
    *,
    converged: bool,
    iterations: int,
    shrinkage_applied: bool,
) -> tuple[pd.Series, BatchParameters]:
    """Back-transform one region and package its parameters.

    Args:
        state: The standardized region.
        gamma_star: Additive term per batch, in ``state.estimable`` order.
        delta_star: Residual SD ratio per batch, in ``state.estimable`` order.
        converged: Whether the shrinkage solve converged.
        iterations: Iterations the solve took.
        shrinkage_applied: Whether any of the region's batches were shrunk.

    Returns:
        The adjusted values and the region's fitted parameters.
    """
    index = {name: position for position, name in enumerate(state.estimable)}
    adjusted = state.values.copy()
    rows = state.adjust_rows

    if bool(rows.any()):
        codes = np.asarray([index[str(b)] for b in state.batches[rows]], dtype=np.intp)
        y = state.values[rows].to_numpy(dtype=np.float64)
        cov = (
            state.design.loc[rows, list(state.terms)].to_numpy(dtype=np.float64)
            if state.terms
            else np.zeros((y.size, 0), dtype=np.float64)
        )
        preserved = state.grand_mean + cov @ state.cov_coefficients
        z = (y - preserved) / state.pooled_sd
        standardized = (z - gamma_star[codes]) / delta_star[codes]
        adjusted.loc[rows] = standardized * state.pooled_sd + preserved

    parameters = BatchParameters(
        region=state.key[0],
        measure_type=state.key[1],
        grand_mean=state.grand_mean,
        pooled_sd=state.pooled_sd,
        gamma_star={name: float(gamma_star[position]) for name, position in index.items()},
        delta_star={name: float(delta_star[position]) for name, position in index.items()},
        n_per_batch={name: int(state.counts[position]) for name, position in index.items()},
        covariate_terms=state.terms,
        covariate_coefficients={
            term: float(state.cov_coefficients[position])
            for position, term in enumerate(state.terms)
        },
        n_adjusted=int(rows.sum()),
        n_unadjusted=int(len(state.values) - int(rows.sum())),
        converged=converged,
        n_iterations=iterations,
        shrinkage_applied=shrinkage_applied,
    )
    return adjusted, parameters


def run_combat(
    observations: pd.DataFrame,
    *,
    batch_column: str = "site",
    covariates: Sequence[str] = (),
    estimation_mask: pd.Series | None = None,
    empirical_bayes: bool = True,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> ComBatResult:
    """Harmonize canonical observations with empirical-Bayes ComBat.

    The design matrix, grand mean, and pooled residual SD are fitted **per
    region** — the batch effect is per site per region, and the v1 region set
    mixes volumes in mm³ with thicknesses in mm, so one pooled fit would be
    dimensionally incoherent. Long format makes that a group-by, which is the
    canonical schema paying for itself.

    The empirical-Bayes shrinkage then runs **per batch across regions**, which
    is the direction Johnson et al. specify: a site's prior is estimated from
    its own effect across all regions, so a thinly-sampled site borrows strength
    from everywhere it appears. Because standardization has already divided each
    region by its own residual SD, those terms are dimensionless and pool
    legitimately.

    Estimation and application are separable through ``estimation_mask``:
    parameters are fitted on the rows the caller trusts, and applied to every
    row in an estimable batch. Letting known-bad reconstructions set the batch
    mean they are then corrected by is a way to launder an artifact into a
    correction.

    Args:
        observations: Canonical observations. Requires ``region``,
            ``measure_type``, ``value``, and ``batch_column``.
        batch_column: Column holding the batch label, normally ``site``.
        covariates: Biological covariates preserved in the design matrix.
        estimation_mask: Boolean mask over ``observations`` selecting rows that
            may contribute to estimation. Defaults to every row.
        empirical_bayes: Whether to shrink per-batch moments toward the priors.
            ``False`` estimates each batch independently, which is useful for
            testing exact recovery but throws away the method's main defence
            against small batches.
        max_iterations: Iteration cap for the shrinkage solve.
        tolerance: Maximum relative change accepted as converged.

    Returns:
        Adjusted values indexed to match ``observations``, and the fitted
        parameters. Regions that could not be estimated keep their original
        values and are recorded on :attr:`ComBatFit.skipped`.
    """
    requested = tuple(str(c) for c in covariates)
    required = {_REGION, _MEASURE_TYPE, _VALUE, batch_column}

    if observations.empty or not required.issubset(observations.columns):
        empty = (
            observations[_VALUE].astype("float64")
            if _VALUE in observations.columns
            else pd.Series(dtype="float64", index=observations.index)
        )
        return ComBatResult(
            values=empty,
            fit=ComBatFit(
                parameters={}, skipped={}, covariates_requested=requested, covariates_dropped={}
            ),
        )

    if estimation_mask is None:
        mask = pd.Series(True, index=observations.index)
    else:
        aligned = estimation_mask.reindex(observations.index, fill_value=False)
        mask = aligned.fillna(False).astype(bool)

    adjusted = observations[_VALUE].astype("float64").copy()
    states: list[_RegionState] = []
    skipped: dict[tuple[str, str], str] = {}
    dropped_overall: dict[str, str] = {}

    for (region, measure_type), group in observations.groupby(
        [_REGION, _MEASURE_TYPE], dropna=False, sort=True
    ):
        key = (str(region), str(measure_type))
        state, reason, dropped = _standardize_region(
            group,
            key=key,
            batch_column=batch_column,
            covariates=requested,
            estimation_mask=mask,
        )
        for name, code in dropped.items():
            dropped_overall.setdefault(name, code)
        if state is None:
            skipped[key] = reason or str(ComBatSkipCode.INSUFFICIENT_ROWS)
        else:
            states.append(state)

    shrunk: dict[tuple[tuple[str, str], str], tuple[float, float]] = {}
    status: dict[str, tuple[bool, int]] = {}
    if empirical_bayes:
        shrunk, status = _shrink(states, max_iterations, tolerance)

    parameters: dict[tuple[str, str], BatchParameters] = {}
    for state in states:
        gamma = state.gamma_hat.copy()
        delta2 = state.delta2_hat.copy()
        converged = True
        iterations = 0
        applied = False

        for position, batch in enumerate(state.estimable):
            entry = shrunk.get((state.key, batch))
            if entry is None:
                continue
            gamma[position], delta2[position] = entry
            batch_converged, batch_iterations = status[batch]
            converged = converged and batch_converged
            iterations = max(iterations, batch_iterations)
            applied = True

        values, params = _adjust_region(
            state,
            gamma,
            np.sqrt(delta2),
            converged=converged,
            iterations=iterations,
            shrinkage_applied=applied,
        )
        adjusted.loc[values.index] = values
        parameters[state.key] = params

    return ComBatResult(
        values=adjusted,
        fit=ComBatFit(
            parameters=parameters,
            skipped=skipped,
            covariates_requested=requested,
            covariates_dropped=dropped_overall,
        ),
    )
