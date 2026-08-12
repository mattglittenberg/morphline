"""Run configuration, loaded from YAML and validated with pydantic.

The *fully resolved* configuration — defaults included, not just the keys the
user wrote — is dumped into the report's provenance block. BUILD_PLAN §2.8's
rule is that a reader holding only the HTML file should be able to reconstruct
the run, so if a parameter changed the output it appears in the block.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from morphline.schema import QCStatus


class Regime(StrEnum):
    """Fixture regime governing the site/time relationship (§3.2).

    Both are exercised in CI. Regime B failing loudly is the *correct* result:
    demonstrating the failure mode is a stronger artifact than a suite where
    everything passes.
    """

    A_INDEPENDENT = "A"
    """Site independent of time. Harmonization should work cleanly."""

    B_CONFOUNDED = "B"
    """Site confounded with time. Harmonization should visibly attenuate the
    longitudinal effect, and the test asserts that it does."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SiteSpec(_Base):
    """One acquisition site in a synthetic dataset."""

    name: str
    scanner_manufacturer: str = "Siemens"
    scanner_model: str = "Prisma"
    field_strength_tesla: float = 3.0
    n_subjects: int = Field(ge=1)
    #: Additive site offset, in native units of each region's measure.
    additive_effect: float = 0.0
    #: Multiplicative site scaling. 1.0 is no effect.
    multiplicative_effect: float = 1.0


class EffectSpec(_Base):
    """Injected biological effects, as fractional change per unit (§3.2).

    Every coefficient is a *fraction of the region's baseline value*, so they
    are comparable across regions of very different absolute size. The truth
    model equation is documented in :mod:`morphline.fixtures.truth`.
    """

    #: Fractional change per decade of baseline age (between-subject).
    age_per_decade: float = -0.015
    #: Fractional offset for the patient group at baseline.
    dx_baseline: float = -0.030
    #: Fractional change per year, control group (within-subject).
    time_per_year: float = -0.005
    #: Additional fractional change per year for the patient group.
    #: This is the primary modeled hypothesis (§2.5.1).
    dx_by_time_per_year: float = -0.015
    #: Fractional offset for female subjects.
    sex_effect: float = 0.0
    #: SD of subject random intercepts, as a fraction.
    random_intercept_sd: float = 0.040
    #: SD of subject random slopes, as a fraction per year.
    random_slope_sd: float = 0.004
    #: SD of measurement noise, as a fraction.
    noise_sd: float = 0.020


class PlantedSpec(_Base):
    """Rates of deliberately planted problems, for QC and accounting tests.

    Fixtures carry known-bad *and* known-clean observations, because recall
    alone is not a validation criterion — flagging everything achieves recall
    1.0 (§2.4.4).
    """

    #: Fraction of sessions given inflated surface hole counts.
    qc_high_holes_fraction: float = Field(default=0.06, ge=0.0, le=1.0)
    #: Fraction of sessions given an implausible eTIV.
    qc_bad_etiv_fraction: float = Field(default=0.03, ge=0.0, le=1.0)
    #: Fraction of follow-up sessions given an extreme longitudinal jump.
    qc_extreme_change_fraction: float = Field(default=0.04, ge=0.0, le=1.0)
    #: Sessions the subject never attended (§2.5.4 ``missing_acquisition``).
    missing_acquisition_fraction: float = Field(default=0.06, ge=0.0, le=1.0)
    #: Sessions acquired but with no usable FreeSurfer output
    #: (§2.5.4 ``missing_derivative``).
    missing_derivative_fraction: float = Field(default=0.03, ge=0.0, le=1.0)
    #: Fraction of written files deliberately corrupted, to exercise the
    #: parser's reason codes end to end.
    malformed_file_fraction: float = Field(default=0.02, ge=0.0, le=1.0)


class FixtureConfig(_Base):
    """Synthetic dataset generation parameters (§3)."""

    seed: int = 20260808
    regime: Regime = Regime.A_INDEPENDENT
    n_sessions: int = Field(default=3, ge=1)
    session_interval_years: float = Field(default=1.0, gt=0.0)
    #: Fraction of subjects assigned the patient diagnosis.
    patient_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    baseline_age_mean: float = 70.0
    baseline_age_sd: float = 7.0
    etiv_mean: float = 1_500_000.0
    etiv_sd: float = 150_000.0
    sites: tuple[SiteSpec, ...]
    effects: EffectSpec = EffectSpec()
    planted: PlantedSpec = PlantedSpec()
    #: FreeSurfer version strings and their sampling weights. Including 5.3 is
    #: deliberate: it omits surface hole counts, which is what makes the
    #: null-not-zero behaviour testable (§2.2).
    freesurfer_version_mix: dict[str, float] = Field(
        default_factory=lambda: {"5.3.0": 0.2, "6.0.0": 0.5, "7.2.0": 0.3}
    )

    @property
    def n_subjects(self) -> int:
        """Total subjects across all sites."""
        return sum(site.n_subjects for site in self.sites)


class QCThresholds(_Base):
    """QC thresholds. Week 2 ships the structure; week 3 uses the values."""

    #: Euler numbers below this are suspect. Evaluated *within site*, because
    #: site distributions genuinely differ (§2.4.2).
    euler_min: float = -200.0
    #: Robust z below which a Euler number is an outlier *for its own site*.
    #: The absolute floor alone cannot flag a surface that is bad for a good
    #: site without also condemning an entire poor one.
    euler_mad_z: float = 4.0
    etiv_min: float = 1_000_000.0
    etiv_max: float = 2_100_000.0
    #: Median/MAD robust z-score beyond which a region is an outlier.
    outlier_mad_z: float = 4.0
    asymmetry_mad_z: float = 4.0
    #: Annualized percentage change flagged for review (§2.4.3).
    annual_change_warn_pct: float = 8.0
    #: Annualized percentage change treated as physiologically impossible.
    annual_change_fail_pct: float = 25.0
    #: Below this inter-session interval, percentage-per-year explodes from
    #: noise alone, so the change flag is suppressed (§2.4.3).
    min_interval_years: float = 0.5


class QCConfig(_Base):
    """QC stage configuration (§2.4)."""

    enabled: bool = True
    thresholds: QCThresholds = QCThresholds()
    #: Acceptance criteria for the fixture validation suite (§2.4.4).
    target_recall: float = 0.95
    target_false_positive_rate: float = 0.05
    target_precision: float = 0.80


class HarmonizationConfig(_Base):
    """ComBat configuration (§2.3)."""

    enabled: bool = True
    batch_column: str = "site"
    #: Biological covariates preserved in the design matrix so they are not
    #: absorbed as batch effects (§2.3.4).
    covariates: tuple[str, ...] = (
        "age_baseline",
        "sex",
        "dx_baseline",
        "time_from_baseline_years",
    )
    #: Conservative engineering default, **not** a methodological threshold.
    #: There is no universal minimum n per batch; adequacy depends on the
    #: number of batches, covariate balance, outcome variance, and effect
    #: magnitude (§2.3.3).
    min_batch_size: int = 20
    #: What happens to a batch below ``min_batch_size``. ``report_and_exclude``
    #: excludes it from *harmonization*, never from the dataset — dropping rows
    #: here would resurface at the modeling boundary under a cause that is
    #: false. ``pool`` merges below-threshold batches into one estimation label,
    #: which asserts they are exchangeable and is defensible only where they
    #: share a scanner and protocol (§2.3.3). ``passthrough`` harmonizes them
    #: like any other batch and warns.
    small_batch_policy: Literal["report_and_exclude", "pool", "passthrough"] = "report_and_exclude"

    #: Empirical-Bayes shrinkage of the per-batch terms. ``False`` estimates
    #: each batch independently, which isolates whether a recovery failure sits
    #: in the shrinkage or in the standardization, and throws away the method's
    #: main defence against small batches.
    empirical_bayes: bool = True
    eb_tolerance: float = Field(default=1e-4, gt=0.0)
    eb_max_iterations: int = Field(default=100, ge=1)
    #: Which rows estimate the batch terms. ``analysis_included`` fits on
    #: QC-passing rows only and applies the transform to every row: letting
    #: known-bad reconstructions set the batch mean they are then corrected by
    #: launders an artifact into a correction. Recorded here rather than
    #: hardcoded so the choice reaches the provenance block.
    estimation_set: Literal["analysis_included", "all"] = "analysis_included"

    #: Acceptance criteria for the §2.3.2 fixture validation suite, on the same
    #: rule as :class:`QCConfig`'s targets: thresholds live in config and the
    #: tests assert against them, so tightening a criterion is a config edit
    #: rather than a hunt through assertions.
    target_batch_recovery_tolerance: float = 0.15
    target_biological_preservation_tolerance: float = 0.35
    target_site_association_reduction: float = 0.50
    target_slope_attenuation_max: float = 0.25


class AnalysisConfig(_Base):
    """Which observations reach the model, and how they are modeled.

    QC identifies and classifies; *this* layer decides what to include
    (§2.4.1). The separation is why the two lists live here and not in
    :class:`QCConfig`.
    """

    include_qc_status: tuple[QCStatus, ...] = (QCStatus.PASS,)
    sensitivity_include: tuple[QCStatus, ...] = (QCStatus.PASS, QCStatus.WARNING)
    region_set: Literal["v1", "all"] = "v1"
    #: False discovery rate for the primary family (§2.5.3).
    fdr_alpha: float = 0.05


class ReportConfig(_Base):
    """HTML report configuration (§2.8)."""

    title: str = "morphline pipeline report"
    #: Self-contained single HTML with everything inlined, no external fetches.
    inline_assets: bool = True


class DatasetConfig(_Base):
    """Which dataset is being processed, and by which adapter."""

    name: str = "synthetic-v1"
    version: str = "0.1.0"
    adapter: Literal["synthetic", "abide-pcp"] = "synthetic"
    #: Root path for a real dataset. ``None`` means fixtures are generated.
    path: Path | None = None
    #: Phenotypic sidecar supplying covariates the stats files do not carry.
    #: Ignored by adapters that resolve their own metadata.
    phenotypic_csv: Path | None = None
    #: Treat separately-acquired sub-samples of one institution as one site.
    #: Trades batch purity for batch size (§2.3.3).
    collapse_site_subsample: bool = False


class RunConfig(_Base):
    """Top-level run configuration."""

    dataset: DatasetConfig = DatasetConfig()
    #: Required only when fixtures are generated. A real-data run reads
    #: ``dataset.path`` and never touches the generator, so demanding an unused
    #: ``fixtures:`` block would make every real config carry fiction.
    fixtures: FixtureConfig | None = None
    qc: QCConfig = QCConfig()
    harmonization: HarmonizationConfig = HarmonizationConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    report: ReportConfig = ReportConfig()

    @model_validator(mode="after")
    def _require_fixtures_when_generating(self) -> RunConfig:
        """Demand a ``fixtures:`` block exactly when one will be used.

        Absent ``dataset.path`` there is no data to read, so a missing block is
        a config error rather than something to paper over with defaults —
        ``FixtureConfig.sites`` in particular has no sensible default, and
        silently inventing sites would fabricate the site effects the
        harmonization tests exist to recover.

        Raises:
            ValueError: If fixtures would be generated but none were configured.
        """
        if self.dataset.path is None and self.fixtures is None:
            raise ValueError(
                "fixtures are required when dataset.path is unset: with no dataset root "
                "there is nothing to ingest. Set dataset.path for a real dataset, or add "
                "a fixtures block to generate one."
            )
        return self

    def resolved(self) -> dict[str, Any]:
        """Return the fully resolved config for the provenance block (§2.8).

        Returns:
            A JSON-serialisable mapping including every default.
        """
        return self.model_dump(mode="json")


def load_config(path: Path | str) -> RunConfig:
    """Load and validate a YAML run configuration.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the file does not satisfy the schema.
            Unknown keys are rejected rather than silently ignored — a typo in
            a threshold name must not quietly leave the default in place.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"config file not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return RunConfig.model_validate(raw)
