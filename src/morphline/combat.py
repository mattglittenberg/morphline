"""In-repo empirical-Bayes ComBat (BUILD_PLAN §2.3.4, Build.md deviation 1).

Why the estimator is written here rather than taken from ``neuroHarmonize`` or
``neuroCombat``: ``neuroHarmonize`` pins ``numpy==1.26.4``, which would hold the
whole project on numpy 1.x and drag pandas, scipy, and statsmodels back with it.
Writing it in numpy removes the pin, and it converts the §2.3.2 validation suite
from "does the library run" into "does *our* estimator recover the injected
batch parameter" — which is the stronger claim.

The method is the parametric empirical Bayes of Johnson, Li & Rabinovic (2007).
For batch *i* and observation *j* within one region::

    y_ij = alpha + X_ij B + gamma_i + delta_i * eps_ij

``alpha`` is the grand mean, ``X B`` the preserved biological covariate block,
``gamma_i`` the batch's additive shift, and ``delta_i`` its residual scale. The
estimator standardizes, pools the per-batch moments toward a normal prior on
``gamma`` and an inverse-gamma prior on ``delta**2``, then back-transforms.
Shrinkage is what lets a small batch borrow strength from the others instead of
chasing its own noise.

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
    """Match an inverse-gamma prior to the per-batch variance estimates.

    Args:
        delta2_hat: Per-batch residual variance estimates.

    Returns:
        The ``(shape, scale)`` hyperparameters, or ``None`` when the moments are
        degenerate — a single batch, or identical variances across batches,
        leaves the prior undefined and shrinkage has nothing to pull toward.
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


def _solve_empirical_bayes(
    z: npt.NDArray[np.float64],
    codes: npt.NDArray[np.intp],
    gamma_hat: npt.NDArray[np.float64],
    delta2_hat: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    max_iterations: int,
    tolerance: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool, int, bool]:
    """Iterate the conditional posterior means for gamma and delta squared.

    Args:
        z: Standardized values for the region.
        codes: Batch index per observation.
        gamma_hat: Per-batch means of ``z``.
        delta2_hat: Per-batch variances of ``z``.
        counts: Observations per batch.
        max_iterations: Iteration cap.
        tolerance: Maximum relative change accepted as converged.

    Returns:
        ``(gamma_star, delta2_star, converged, n_iterations, shrinkage_applied)``.
    """
    priors = _moment_priors(delta2_hat)
    if priors is None:
        return gamma_hat, delta2_hat, True, 0, False

    shape, scale = priors
    gamma_bar = float(gamma_hat.mean())
    tau2 = float(gamma_hat.var(ddof=1))

    gamma_star = gamma_hat.copy()
    delta2_star = delta2_hat.copy()

    for iteration in range(1, max_iterations + 1):
        numerator = counts * tau2 * gamma_hat + delta2_star * gamma_bar
        denominator = counts * tau2 + delta2_star
        gamma_next = np.where(denominator > 0.0, numerator / denominator, gamma_hat)

        residuals = z - gamma_next[codes]
        sum_squares = np.bincount(codes, weights=residuals**2, minlength=gamma_hat.size)
        delta2_next = (scale + 0.5 * sum_squares) / (counts / 2.0 + shape - 1.0)
        delta2_next = np.where(delta2_next > 0.0, delta2_next, delta2_star)

        change = max(
            float(np.max(np.abs(gamma_next - gamma_star) / (np.abs(gamma_star) + 1e-12))),
            float(np.max(np.abs(delta2_next - delta2_star) / (np.abs(delta2_star) + 1e-12))),
        )
        gamma_star, delta2_star = gamma_next, delta2_next
        if change < tolerance:
            return gamma_star, delta2_star, True, iteration, True

    return gamma_star, delta2_star, False, max_iterations, True


def _fit_region(
    frame: pd.DataFrame,
    *,
    region: str,
    measure_type: str,
    batch_column: str,
    covariates: Sequence[str],
    estimation_mask: pd.Series,
    empirical_bayes: bool,
    max_iterations: int,
    tolerance: float,
) -> tuple[pd.Series, BatchParameters | None, str | None, dict[str, str]]:
    """Estimate and apply ComBat within one region.

    Args:
        frame: Every row for this region, adjusted or not.
        region: Canonical region name.
        measure_type: The measure being harmonized.
        batch_column: Column holding the batch label.
        covariates: Biological covariates to preserve.
        estimation_mask: Which rows may contribute to estimation.
        empirical_bayes: Whether to shrink the per-batch moments.
        max_iterations: Iteration cap for the shrinkage solve.
        tolerance: Convergence tolerance for the shrinkage solve.

    Returns:
        ``(adjusted_values, parameters, skip_reason, covariates_dropped)``. On a
        skip the values are returned unchanged and ``parameters`` is ``None``.
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
        return values, None, str(ComBatSkipCode.INSUFFICIENT_ROWS), dropped

    sizes = batches[eligible].value_counts()
    estimable = sorted(str(b) for b, n in sizes.items() if int(n) >= MIN_ROWS_PER_BATCH)
    if len(estimable) < MIN_BATCHES:
        return values, None, str(ComBatSkipCode.SINGLE_BATCH), dropped

    in_batch = batches.isin(estimable)
    fit_rows = eligible & in_batch
    if int(fit_rows.sum()) <= len(estimable) + design.shape[1]:
        return values, None, str(ComBatSkipCode.INSUFFICIENT_ROWS), dropped

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
        return values, None, str(ComBatSkipCode.SINGULAR_DESIGN), dropped

    coefficients, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    if not bool(np.isfinite(coefficients).all()):
        return values, None, str(ComBatSkipCode.NONFINITE_VALUES), dropped

    batch_coefficients = coefficients[: len(estimable)]
    cov_coefficients = coefficients[len(estimable) :]
    grand_mean = float((counts / counts.sum()) @ batch_coefficients)

    residuals = y_fit - x_fit @ coefficients
    pooled_sd = float(np.sqrt(float(residuals @ residuals) / y_fit.size))
    scale = max(abs(grand_mean), float(np.max(np.abs(y_fit))), 1.0)
    if not np.isfinite(pooled_sd) or pooled_sd <= RELATIVE_VARIANCE_FLOOR * scale:
        return values, None, str(ComBatSkipCode.ZERO_VARIANCE), dropped

    z_fit = (y_fit - grand_mean - cov_fit @ cov_coefficients) / pooled_sd
    gamma_hat = np.bincount(codes_fit, weights=z_fit, minlength=len(estimable)) / counts
    delta2_hat = np.asarray(
        [float(np.var(z_fit[codes_fit == position], ddof=1)) for position in range(len(estimable))],
        dtype=np.float64,
    )
    delta2_hat = np.where(delta2_hat > 0.0, delta2_hat, 1.0)

    if empirical_bayes:
        gamma_star, delta2_star, converged, iterations, shrunk = _solve_empirical_bayes(
            z_fit, codes_fit, gamma_hat, delta2_hat, counts, max_iterations, tolerance
        )
    else:
        gamma_star, delta2_star, converged, iterations, shrunk = (
            gamma_hat,
            delta2_hat,
            True,
            0,
            False,
        )

    delta_star = np.sqrt(delta2_star)
    adjust_rows = usable & in_batch
    adjusted = values.copy()

    if bool(adjust_rows.any()):
        codes_all = np.asarray([index[str(b)] for b in batches[adjust_rows]], dtype=np.intp)
        y_all = values[adjust_rows].to_numpy(dtype=np.float64)
        cov_all = (
            design.loc[adjust_rows, list(terms)].to_numpy(dtype=np.float64)
            if terms
            else np.zeros((y_all.size, 0), dtype=np.float64)
        )
        preserved = grand_mean + cov_all @ cov_coefficients
        z_all = (y_all - preserved) / pooled_sd
        standardized = (z_all - gamma_star[codes_all]) / delta_star[codes_all]
        adjusted.loc[adjust_rows] = standardized * pooled_sd + preserved

    parameters = BatchParameters(
        region=region,
        measure_type=measure_type,
        grand_mean=grand_mean,
        pooled_sd=pooled_sd,
        gamma_star={name: float(gamma_star[position]) for name, position in index.items()},
        delta_star={name: float(delta_star[position]) for name, position in index.items()},
        n_per_batch={name: int(counts[position]) for name, position in index.items()},
        covariate_terms=terms,
        covariate_coefficients={
            term: float(cov_coefficients[position]) for position, term in enumerate(terms)
        },
        n_adjusted=int(adjust_rows.sum()),
        n_unadjusted=int(len(frame) - int(adjust_rows.sum())),
        converged=converged,
        n_iterations=iterations,
        shrinkage_applied=shrunk,
    )
    return adjusted, parameters, None, dropped


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

    Estimation runs **per region**, because the batch effect is per site per
    region and because the v1 region set mixes volumes in mm³ with thicknesses
    in mm — one pooled fit would be dimensionally incoherent. Long format makes
    that a group-by, which is the canonical schema paying for itself.

    Estimation and application are deliberately separable through
    ``estimation_mask``: parameters are fitted on the rows the caller trusts,
    and applied to every row in an estimable batch. Letting known-bad
    reconstructions set the batch mean they are then corrected by is a way to
    launder an artifact into a correction.

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
    parameters: dict[tuple[str, str], BatchParameters] = {}
    skipped: dict[tuple[str, str], str] = {}
    dropped_overall: dict[str, str] = {}

    for (region, measure_type), group in observations.groupby(
        [_REGION, _MEASURE_TYPE], dropna=False, sort=True
    ):
        values, fitted, reason, dropped = _fit_region(
            group,
            region=str(region),
            measure_type=str(measure_type),
            batch_column=batch_column,
            covariates=requested,
            estimation_mask=mask,
            empirical_bayes=empirical_bayes,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        adjusted.loc[values.index] = values
        for name, code in dropped.items():
            dropped_overall.setdefault(name, code)
        key = (str(region), str(measure_type))
        if fitted is None:
            skipped[key] = reason or str(ComBatSkipCode.INSUFFICIENT_ROWS)
        else:
            parameters[key] = fitted

    return ComBatResult(
        values=adjusted,
        fit=ComBatFit(
            parameters=parameters,
            skipped=skipped,
            covariates_requested=requested,
            covariates_dropped=dropped_overall,
        ),
    )
