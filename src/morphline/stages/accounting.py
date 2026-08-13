"""Data accounting — counts and losses at every boundary (BUILD_PLAN §1.6).

The cheapest defence against silent data loss, and what makes a real-data run
auditable. Built for real in v0.1.0 rather than stubbed, because §7 lists the
accounting funnel among the things that are never cut.

The headline artifact is a single funnel::

    raw files → parsed files → canonical observations
              → QC-passing observations → modeled observations

**Every drop in that funnel must have a stated cause. Unexplained loss is a
bug, not a rounding error.** :func:`reconcile` is what turns that sentence from
an aspiration into a check that fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

from morphline.schema import MissingnessCause, ModelExclusion, QCStatus


@dataclass(slots=True)
class FunnelStage:
    """One row of the accounting funnel.

    Attributes:
        boundary: Name of the boundary, e.g. ``"parsing"``.
        unit: What is being counted — files, sessions, or observations.
        count: How many survived this boundary.
        lost: How many were lost at it.
        causes: Loss counts by reason code. Must sum to ``lost``.
    """

    boundary: str
    unit: str
    count: int
    lost: int = 0
    causes: dict[str, int] = field(default_factory=dict)

    def unexplained(self) -> int:
        """Return loss at this boundary with no stated cause."""
        return self.lost - sum(self.causes.values())


@dataclass(slots=True)
class AccountingReport:
    """The full accounting output (§1.6).

    Attributes:
        funnel: Ordered funnel stages.
        parse_failures_by_code: Rejected file counts by reason code.
        sessions_per_subject: Distribution of session counts per subject.
        regions_per_session: Distribution of regions observed per session.
        metadata_coverage: Value distributions for site, scanner, field
            strength, and FreeSurfer version.
        batch_sizes: Observations and subjects per site.
        qc_summary: PASS/WARNING/FAIL counts, overall and by site.
        measures_per_file: Distribution of header measure counts per parsed
            file. Reported, never assumed: a silently lost header measure
            changes neither the row count nor the failure rate (§5.2).
        missingness: Counts by cause (§2.5.4).
        missingness_by_site: Missingness causes broken down by site.
        missingness_by_timepoint: Missingness causes broken down by session.
        completers: Baseline comparison of completers against non-completers,
            the §2.5.4 check on whether the MAR assumption is load-bearing.
        notes: Free-text observations worth surfacing in the report.
    """

    funnel: list[FunnelStage] = field(default_factory=list)
    parse_failures_by_code: dict[str, int] = field(default_factory=dict)
    sessions_per_subject: dict[str, int] = field(default_factory=dict)
    regions_per_session: dict[str, int] = field(default_factory=dict)
    metadata_coverage: dict[str, dict[str, int]] = field(default_factory=dict)
    batch_sizes: dict[str, dict[str, int]] = field(default_factory=dict)
    qc_summary: dict[str, Any] = field(default_factory=dict)
    measures_per_file: dict[str, int] = field(default_factory=dict)
    missingness: dict[str, int] = field(default_factory=dict)
    missingness_by_site: dict[str, dict[str, int]] = field(default_factory=dict)
    missingness_by_timepoint: dict[str, dict[str, int]] = field(default_factory=dict)
    completers: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def funnel_frame(self) -> pd.DataFrame:
        """Return the funnel as a frame for persistence and templating."""
        return pd.DataFrame(
            [
                {
                    "boundary": s.boundary,
                    "unit": s.unit,
                    "count": s.count,
                    "lost": s.lost,
                    "causes": "; ".join(f"{k}={v}" for k, v in sorted(s.causes.items())),
                    "unexplained": s.unexplained(),
                }
                for s in self.funnel
            ]
        )

    def reconcile(self) -> list[str]:
        """Check that every drop in the funnel has a stated cause.

        Returns:
            One message per boundary whose losses are not fully attributed.
            An empty list means the funnel reconciles exactly, which is the
            week-2 exit criterion.
        """
        return [
            f"{s.boundary}: {s.unexplained()} of {s.lost} lost {s.unit} have no stated cause"
            for s in self.funnel
            if s.unexplained() != 0
        ]

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a plain mapping for templating and JSON."""
        return {
            "funnel": self.funnel_frame().to_dict(orient="records"),
            "parse_failures_by_code": self.parse_failures_by_code,
            "sessions_per_subject": self.sessions_per_subject,
            "regions_per_session": self.regions_per_session,
            "metadata_coverage": self.metadata_coverage,
            "batch_sizes": self.batch_sizes,
            "qc_summary": self.qc_summary,
            "measures_per_file": self.measures_per_file,
            "missingness": self.missingness,
            "missingness_by_site": self.missingness_by_site,
            "missingness_by_timepoint": self.missingness_by_timepoint,
            "completers": self.completers,
            "notes": self.notes,
            "reconciles": not self.reconcile(),
            "reconciliation_errors": self.reconcile(),
        }


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Return value counts for a column, tolerating absence and nulls."""
    if column not in df.columns or df.empty:
        return {}
    counts = df[column].astype("object").where(df[column].notna(), "<null>").value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def build_accounting(
    *,
    observations: pd.DataFrame,
    parse_failures: pd.DataFrame,
    expected_sessions: pd.DataFrame,
    files_discovered: int,
    sessions_discovered: int,
    sessions_without_files: int,
    sessions_all_files_rejected: int = 0,
    sessions_no_recognised_regions: int = 0,
    observations_expected: int = 0,
    observation_losses: dict[str, int] | None = None,
    regions_per_session: dict[str, int] | None = None,
    measures_per_file: dict[str, int] | None = None,
    measures_overwritten: int = 0,
    qc_observations: pd.DataFrame | None = None,
    modeled_observations: int | None = None,
    model_fits: pd.DataFrame | None = None,
) -> AccountingReport:
    """Assemble the accounting report for one run.

    Args:
        observations: Canonical observations after ingestion.
        parse_failures: Reason-coded parse failures.
        expected_sessions: Sessions the dataset claims to contain.
        files_discovered: Stats files located on disk.
        sessions_discovered: Subject-sessions located on disk.
        sessions_without_files: Sessions present but holding no stats files.
        sessions_all_files_rejected: Sessions whose every file failed to parse.
        sessions_no_recognised_regions: Sessions that parsed but yielded no
            canonical regions.
        observations_expected: Observations the sessions that produced any
            should have produced. Zero means ingestion did not supply it, and
            the canonicalization boundary then reports no loss.
        observation_losses: Region-level loss counts by reason code.
        regions_per_session: Distribution of regions observed per session.
        measures_per_file: Distribution of header measure counts per file.
        measures_overwritten: Header measures lost to key collisions.
        qc_observations: Observations after QC, carrying ``qc_status``.
        modeled_observations: Observations that actually entered the model.
        model_fits: Per-region fit summary carrying ``region`` and
            ``n_observations``. Without it the shortfall at the modeling
            boundary can be counted but not explained, so it is reported as
            :attr:`~morphline.schema.ModelExclusion.UNATTRIBUTED` rather than
            attributed to a guess.

    Returns:
        The assembled report. Call :meth:`AccountingReport.reconcile` to verify
        it balances.
    """
    report = AccountingReport()

    # --- Discovery ---------------------------------------------------------
    expected_total = len(expected_sessions)
    planned_missing: dict[str, int] = {}
    if "missing_cause" in expected_sessions.columns:
        planned_missing = {
            str(k): int(v)
            for k, v in expected_sessions["missing_cause"].dropna().value_counts().items()
        }

    report.funnel.append(
        FunnelStage(
            boundary="expected sessions",
            unit="sessions",
            count=expected_total,
        )
    )

    # Sessions the subject never attended leave no directory at all, so the
    # gap between expected and discovered is exactly the acquisition-level
    # missingness — provided the dataset told us what to expect.
    acquisition_missing = planned_missing.get(str(MissingnessCause.ACQUISITION), 0)
    discovery_lost = max(0, expected_total - sessions_discovered)
    discovery_causes: dict[str, int] = {}
    if discovery_lost:
        discovery_causes[str(MissingnessCause.ACQUISITION)] = min(
            acquisition_missing, discovery_lost
        )
        remainder = discovery_lost - discovery_causes[str(MissingnessCause.ACQUISITION)]
        if remainder:
            discovery_causes["undocumented_absence"] = remainder
            report.notes.append(
                f"{remainder} expected sessions were absent from disk without a "
                "recorded cause; investigate before trusting the funnel."
            )

    report.funnel.append(
        FunnelStage(
            boundary="discovered sessions",
            unit="sessions",
            count=sessions_discovered,
            lost=discovery_lost,
            causes=discovery_causes,
        )
    )

    # --- Parsing -----------------------------------------------------------
    failure_counts: dict[str, int] = {}
    if not parse_failures.empty and "failure_code" in parse_failures.columns:
        failure_counts = {
            str(k): int(v) for k, v in parse_failures["failure_code"].value_counts().items()
        }
    report.parse_failures_by_code = failure_counts

    files_parsed = files_discovered - len(parse_failures)
    report.funnel.append(
        FunnelStage(
            boundary="parsed files",
            unit="files",
            count=files_parsed,
            lost=len(parse_failures),
            causes=failure_counts,
        )
    )

    # --- Entity resolution -------------------------------------------------
    sessions_with_observations = (
        int(observations.groupby(["subject_id", "session_id"], dropna=False).ngroups)
        if not observations.empty
        else 0
    )
    resolution_lost = sessions_discovered - sessions_with_observations
    # Each cause is counted exactly by the ingestion stage. Nothing is
    # attributed by subtraction: a remainder-absorbing catch-all would make
    # this boundary reconcile by construction, and a check that cannot fail is
    # worse than no check, because it reads as assurance while providing none.
    resolution_causes: dict[str, int] = {}
    if resolution_lost:
        for cause, count in (
            (str(MissingnessCause.DERIVATIVE), sessions_without_files),
            ("all_files_rejected", sessions_all_files_rejected),
            ("no_recognised_regions", sessions_no_recognised_regions),
        ):
            if count:
                resolution_causes[cause] = count

    report.funnel.append(
        FunnelStage(
            boundary="sessions with observations",
            unit="sessions",
            count=sessions_with_observations,
            lost=resolution_lost,
            causes=resolution_causes,
        )
    )

    if not observations.empty:
        per_subject = observations.groupby("subject_id")["session_id"].nunique()
        report.sessions_per_subject = {
            str(k): int(v) for k, v in per_subject.value_counts().sort_index().items()
        }

    # A session that yields *some* of its regions passes every session-level
    # counter above intact, so region-level loss is invisible unless this
    # boundary looks for it. The expected count and its causes are measured at
    # ingestion; absent them the boundary reports no loss rather than inventing
    # an expectation it cannot justify.
    observations_lost = max(0, observations_expected - len(observations))
    report.regions_per_session = dict(regions_per_session or {})
    report.funnel.append(
        FunnelStage(
            boundary="canonical observations",
            unit="observations",
            count=len(observations),
            lost=observations_lost,
            causes=dict(observation_losses or {}),
        )
    )

    # --- Metadata coverage -------------------------------------------------
    report.metadata_coverage = {
        column: _value_counts(observations, column)
        for column in (
            "site",
            "scanner_manufacturer",
            "scanner_model",
            "field_strength_tesla",
            "freesurfer_version",
        )
    }

    if not observations.empty and "site" in observations.columns:
        report.batch_sizes = {
            "observations_per_site": _value_counts(observations, "site"),
            "subjects_per_site": {
                str(site): int(group["subject_id"].nunique())
                for site, group in observations.groupby("site", dropna=False)
            },
        }

    # --- QC ----------------------------------------------------------------
    qc_passing = len(observations)
    if qc_observations is not None and not qc_observations.empty:
        status_counts = _value_counts(qc_observations, "qc_status")
        report.qc_summary = {
            "by_status": status_counts,
            "by_site": {
                str(site): _value_counts(group, "qc_status")
                for site, group in qc_observations.groupby("site", dropna=False)
            },
            "by_flag": _flag_counts(qc_observations),
        }
        qc_passing = int(status_counts.get(str(QCStatus.PASS), 0))
        excluded = len(qc_observations) - qc_passing
        report.funnel.append(
            FunnelStage(
                boundary="QC-passing observations",
                unit="observations",
                count=qc_passing,
                lost=excluded,
                causes=(
                    {
                        str(MissingnessCause.EXCLUDED_QC): excluded,
                    }
                    if excluded
                    else {}
                ),
            )
        )

    # --- Analysis ----------------------------------------------------------
    if modeled_observations is not None:
        lost = qc_passing - modeled_observations
        report.funnel.append(
            FunnelStage(
                boundary="modeled observations",
                unit="observations",
                count=modeled_observations,
                lost=max(0, lost),
                causes=_model_exclusion_causes(
                    lost,
                    qc_observations if qc_observations is not None else observations,
                    model_fits,
                ),
            )
        )

    report.measures_per_file = dict(measures_per_file or {})
    if measures_overwritten:
        report.notes.append(
            f"{measures_overwritten} header measure(s) were overwritten by a later "
            "# Measure line and no longer appear in the output. This is a parser "
            "defect rather than a property of the data — a declared measurement "
            "vanished without changing the row count or the failure rate."
        )

    report.missingness = planned_missing
    report.missingness_by_site = _missingness_by(expected_sessions, observations, "site")
    report.missingness_by_timepoint = _missingness_by(expected_sessions, observations, "session_id")
    report.completers = completer_comparison(expected_sessions, observations)
    return report


#: Baseline covariates compared between completers and non-completers, split
#: by how a difference in them is summarised.
_CONTINUOUS_BASELINE: Final = ("age_baseline", "etiv_baseline")
_CATEGORICAL_BASELINE: Final = ("sex", "dx_baseline", "site")


def _baseline_rows(observations: pd.DataFrame) -> pd.DataFrame:
    """Return one row per subject carrying its baseline covariates."""
    columns = [
        c for c in (*_CONTINUOUS_BASELINE, *_CATEGORICAL_BASELINE) if c in observations.columns
    ]
    if not columns or "subject_id" not in observations.columns:
        return pd.DataFrame()
    return observations.groupby("subject_id", dropna=False)[columns].first()


def _continuous_summary(values: pd.Series) -> dict[str, float | int | None]:
    """Summarise one continuous baseline covariate for one group."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"n": 0, "mean": None, "sd": None}
    return {
        "n": int(numeric.size),
        "mean": float(numeric.mean()),
        "sd": float(numeric.std(ddof=1)) if numeric.size > 1 else 0.0,
    }


def _standardized_difference(
    left: dict[str, float | int | None], right: dict[str, float | int | None]
) -> float | None:
    """Return the standardized mean difference between two group summaries.

    Reported instead of a p-value on purpose. A significance test here answers
    "is the sample large enough to detect a difference", which is not the
    question — the question is whether the groups differ enough for the MAR
    assumption to be doing real work (§2.5.4).
    """
    left_mean, right_mean = left.get("mean"), right.get("mean")
    left_sd, right_sd = left.get("sd"), right.get("sd")
    if left_mean is None or right_mean is None or left_sd is None or right_sd is None:
        return None
    pooled = ((float(left_sd) ** 2 + float(right_sd) ** 2) / 2.0) ** 0.5
    if pooled == 0.0:
        return None
    return float((float(left_mean) - float(right_mean)) / pooled)


def _categorical_summary(values: pd.Series) -> dict[str, float]:
    """Return level proportions for one categorical baseline covariate."""
    known = values.dropna()
    if known.empty:
        return {}
    counts = known.astype(str).value_counts(normalize=True)
    return {str(level): float(share) for level, share in counts.items()}


def completer_comparison(
    expected_sessions: pd.DataFrame, observations: pd.DataFrame
) -> dict[str, Any]:
    """Compare baseline characteristics of completers and non-completers (§2.5.4).

    The pipeline's validity rests on a missing-at-random assumption, and that
    assumption is load-bearing exactly where dropout is *caused* by the process
    under study. This comparison is the cheap check on it: if the subjects who
    dropped out differ at baseline from those who did not, MAR is doing more
    work than the analysis can justify, and the report must say so.

    Subjects who produced no observations at all are counted but never
    compared. They are the group most likely to be informatively missing, so
    silently dropping them would bias the very comparison meant to detect bias.

    Args:
        expected_sessions: Sessions the dataset claims to contain.
        observations: Canonical observations, supplying baseline covariates.

    Returns:
        The comparison, or a mapping whose ``applicable`` is ``False`` and
        whose ``reason`` states why the question could not be asked.
    """
    if expected_sessions.empty or "subject_id" not in expected_sessions.columns:
        return {
            "applicable": False,
            "reason": "the dataset supplied no session roster, so dropout cannot be identified",
        }

    expected_per_subject = expected_sessions.groupby("subject_id", dropna=False).size()
    if int(expected_per_subject.max()) <= 1:
        return {
            "applicable": False,
            "reason": (
                "every subject has at most one expected session, so this dataset is "
                "cross-sectional and there is no dropout to compare. Completion is "
                "not a meaningful category here"
            ),
            "n_subjects": int(expected_per_subject.size),
        }

    if observations.empty or "subject_id" not in observations.columns:
        observed_per_subject = pd.Series(dtype=int)
    else:
        observed_per_subject = observations.groupby("subject_id", dropna=False)[
            "session_id"
        ].nunique()

    observed = observed_per_subject.reindex(expected_per_subject.index).fillna(0).astype(int)
    completers = observed.index[observed == expected_per_subject]
    partial = observed.index[(observed > 0) & (observed < expected_per_subject)]
    absent = observed.index[observed == 0]

    baseline = _baseline_rows(observations)
    left = baseline.reindex(completers).dropna(how="all") if not baseline.empty else baseline
    right = baseline.reindex(partial).dropna(how="all") if not baseline.empty else baseline

    continuous: dict[str, dict[str, Any]] = {}
    categorical: dict[str, dict[str, Any]] = {}
    if not baseline.empty:
        for column in _CONTINUOUS_BASELINE:
            if column not in baseline.columns:
                continue
            summaries = (_continuous_summary(left[column]), _continuous_summary(right[column]))
            continuous[column] = {
                "completers": summaries[0],
                "non_completers": summaries[1],
                "standardized_difference": _standardized_difference(*summaries),
            }
        for column in _CATEGORICAL_BASELINE:
            if column not in baseline.columns:
                continue
            categorical[column] = {
                "completers": _categorical_summary(left[column]),
                "non_completers": _categorical_summary(right[column]),
            }

    notes: list[str] = []
    if len(partial) == 0:
        notes.append(
            "Every subject with data completed every expected session, so there is no "
            "dropout to compare against. This is a property of the dataset, not "
            "evidence that missingness is ignorable."
        )
    if len(absent):
        notes.append(
            f"{len(absent)} subject(s) produced no observations at all and carry no "
            "baseline covariates, so they cannot enter the comparison. They are "
            "counted here because that group is the most likely to be informatively "
            "missing, and omitting it silently would bias the check itself."
        )
    notable = [
        f"{column} (SMD {stats['standardized_difference']:.2f})"
        for column, stats in continuous.items()
        if stats["standardized_difference"] is not None
        and abs(float(stats["standardized_difference"])) >= 0.1
    ]
    if notable:
        notes.append(
            "Completers and non-completers differ at baseline on "
            + ", ".join(notable)
            + ". A standardized difference of 0.1 or more is the conventional "
            "threshold for imbalance worth reporting, and it weakens the MAR "
            "assumption the longitudinal estimates rest on (§2.5.4)."
        )

    return {
        "applicable": True,
        "definition": (
            "a completer produced observations for every session the dataset expects "
            "of that subject; a non-completer produced at least one but not all"
        ),
        "n_completers": len(completers),
        "n_non_completers": len(partial),
        "n_without_baseline": len(absent),
        "continuous": continuous,
        "categorical": categorical,
        "notes": notes,
    }


def _missingness_by(
    expected_sessions: pd.DataFrame, observations: pd.DataFrame, column: str
) -> dict[str, dict[str, int]]:
    """Break missingness down by a grouping column (§5.2).

    A single overall rate hides the thing worth knowing. Loss concentrated in
    one site or one timepoint is a different finding from loss spread evenly,
    and only the breakdown distinguishes them.

    Args:
        expected_sessions: Sessions the dataset claims to contain, with causes.
        observations: Canonical observations, supplying each session's site.
        column: ``"site"`` or ``"session_id"``.

    Returns:
        Cause counts per group. Empty when the grouping is unavailable —
        sessions that never produced observations carry no site, so an absent
        breakdown is reported as absent rather than bucketed under a fabricated
        label.
    """
    if expected_sessions.empty or "missing_cause" not in expected_sessions.columns:
        return {}

    missing = expected_sessions[expected_sessions["missing_cause"].notna()].copy()
    if missing.empty:
        return {}

    if column == "site":
        # The roster is preferred over the observations because the sessions
        # being counted here are the ones that produced no observations, so
        # the observations cannot say where they came from.
        if "site" in missing.columns:
            missing["_group"] = missing["site"]
        elif not observations.empty and "site" in observations.columns:
            sites = observations.groupby("subject_id", dropna=False)["site"].first()
            missing["_group"] = missing["subject_id"].map(sites)
        else:
            return {}
    elif column in missing.columns:
        missing["_group"] = missing[column]
    else:
        return {}

    missing["_group"] = missing["_group"].fillna("unknown").astype(str)
    return {
        str(group): {str(k): int(v) for k, v in frame["missing_cause"].value_counts().items()}
        for group, frame in missing.groupby("_group", dropna=False)
    }


def _model_exclusion_causes(
    lost: int,
    observations: pd.DataFrame,
    model_fits: pd.DataFrame | None,
) -> dict[str, int]:
    """Attribute the shortfall at the modeling boundary to actual causes.

    Two things separate observations that passed QC from observations that were
    fitted, and they are counted rather than inferred from each other: regions
    the model never attempted, and rows a fitted region had to drop because a
    model term was null.

    Args:
        lost: Observations that passed QC but were not modeled.
        observations: The frame the modeling stage drew from, used to count how
            many observations each attempted region actually offered.
        model_fits: Per-region fit summary, or ``None`` when unavailable.

    Returns:
        Causes summing exactly to ``lost``. When ``model_fits`` is absent, or
        the arithmetic does not close, the residual is reported as
        unattributed — a funnel that reconciles on a fabricated cause is worse
        than one that admits the gap.
    """
    if lost <= 0:
        return {}
    if model_fits is None or model_fits.empty or "region" not in model_fits.columns:
        return {str(ModelExclusion.UNATTRIBUTED): lost}

    # Mirror the modeling stage's own inclusion filter. Counting against a
    # different denominator than the model used would push the difference into
    # the residual and report it as unattributed.
    considered = observations
    if "analysis_included" in observations.columns:
        considered = observations[observations["analysis_included"].fillna(False)]

    attempted = set(model_fits["region"].dropna())
    if "region" in considered.columns:
        available = int(considered["region"].isin(attempted).sum())
    else:
        available = 0

    # A region the design cannot identify offered its observations and had
    # nothing wrong with them. Folding those into the covariate bucket would
    # report a study-design limitation as a data-quality one — the two have
    # opposite implications, which is why they are separate codes at all.
    unidentifiable_regions: set[str] = set()
    if "estimable" in model_fits.columns:
        not_estimable = ~model_fits["estimable"].fillna(True).astype(bool)
        unidentifiable_regions = set(model_fits.loc[not_estimable, "region"].dropna())
    unidentifiable = (
        int(considered["region"].isin(unidentifiable_regions).sum())
        if unidentifiable_regions and "region" in considered.columns
        else 0
    )

    outside = max(0, len(considered) - available)
    fitted = (
        int(model_fits["n_observations"].fillna(0).sum())
        if "n_observations" in model_fits.columns
        else 0
    )
    incomplete = max(0, available - unidentifiable - fitted)

    causes = {
        str(ModelExclusion.OUTSIDE_REGION_SET): outside,
        str(ModelExclusion.DESIGN_NOT_IDENTIFIABLE): unidentifiable,
        str(ModelExclusion.INCOMPLETE_COVARIATES): incomplete,
    }
    causes = {code: count for code, count in causes.items() if count > 0}

    residual = lost - sum(causes.values())
    if residual > 0:
        causes[str(ModelExclusion.UNATTRIBUTED)] = residual
    return causes


def _flag_counts(qc_observations: pd.DataFrame) -> dict[str, int]:
    """Count how often each QC flag code fired."""
    if "qc_flags" not in qc_observations.columns:
        return {}
    counts: dict[str, int] = {}
    for flags in qc_observations["qc_flags"]:
        if flags is None:
            continue
        for flag in flags:
            counts[str(flag)] = counts.get(str(flag), 0) + 1
    return counts


def run_accounting(report: AccountingReport, outdir: Path | str) -> Path:
    """Persist the accounting funnel to Parquet.

    Args:
        report: The assembled report.
        outdir: Destination directory.

    Returns:
        Path to the written funnel Parquet file.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "accounting_funnel.parquet"
    report.funnel_frame().to_parquet(path, index=False)
    (out / "accounting.json").write_text(
        json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    return path
