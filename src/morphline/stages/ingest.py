"""Ingestion — the only stage that knows what a FreeSurfer file looks like.

Drives the parser over an adapter's discovered files and writes canonical
Parquet. Parse failures are collected as reason-coded records rather than
raised, so one malformed file in a large dataset is a row in the accounting
table, not a crashed run (§1.6).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from morphline.adapters.base import DatasetAdapter
from morphline.parsers import FreeSurferStatsParser, ParsedStatsFile, ParseFailure
from morphline.regions import v1_region_set
from morphline.schema import ObservationLoss, conform, empty_canonical, write_canonical


@dataclass(slots=True)
class IngestResult:
    """Outcome of an ingestion run.

    Attributes:
        observations: Canonical observations.
        failures: Reason-coded parse failures.
        files_discovered: Count of stats files the adapter located.
        sessions_discovered: Count of subject-sessions the adapter located.
        sessions_without_files: Sessions that exist on disk but hold no stats
            files — ``missing_derivative`` in the §2.5.4 taxonomy.
        sessions_all_files_rejected: Sessions that had files, every one of
            which failed to parse.
        sessions_no_recognised_regions: Sessions that parsed cleanly but
            contained none of the regions in the canonical vocabulary.
        observations_expected: Observations the sessions that produced any
            should have produced — the region set times those sessions.
        observation_losses: Region-level loss counts by reason code, for
            sessions that produced *some* observations but not all of them.
        regions_per_session: Distribution of regions observed per session,
            keyed by count (§1.6's canonicalization boundary).
        measures_per_file: Distribution of ``# Measure`` lines per parsed file,
            keyed by count. Reported rather than assumed (§5.2).
        measures_overwritten: Header measures lost to key collisions across
            every parsed file. Non-zero is a parser defect, not a data
            property: it means a file declared a measurement that no longer
            exists anywhere in the output.
        freesurfer_versions: Versions actually observed in the data.
        freesurfer_version_declarations: Version strings the headers declared,
            verbatim. FreeSurfer 5.1 declares a CVS revision rather than a
            version, so this can be populated while
            ``freesurfer_versions`` is empty.

    Note:
        The three loss counters are tracked separately and exactly, rather
        than being derived as a remainder. A catch-all cause would make the
        accounting funnel reconcile by construction, which would defeat the
        point of checking that it reconciles at all (§1.6).
    """

    observations: pd.DataFrame
    failures: list[ParseFailure] = field(default_factory=list)
    files_discovered: int = 0
    sessions_discovered: int = 0
    sessions_without_files: int = 0
    sessions_all_files_rejected: int = 0
    sessions_no_recognised_regions: int = 0
    observations_expected: int = 0
    observation_losses: dict[str, int] = field(default_factory=dict)
    regions_per_session: dict[str, int] = field(default_factory=dict)
    measures_per_file: dict[str, int] = field(default_factory=dict)
    measures_overwritten: int = 0
    freesurfer_versions: list[str] = field(default_factory=list)
    freesurfer_version_declarations: list[str] = field(default_factory=list)

    def counters(self) -> dict[str, int | dict[str, int]]:
        """Return the loss counters, for the accounting stage's JSON sidecar."""
        return {
            "files_discovered": self.files_discovered,
            "sessions_discovered": self.sessions_discovered,
            "sessions_without_files": self.sessions_without_files,
            "sessions_all_files_rejected": self.sessions_all_files_rejected,
            "sessions_no_recognised_regions": self.sessions_no_recognised_regions,
            "observations_expected": self.observations_expected,
            "observation_losses": self.observation_losses,
            "regions_per_session": self.regions_per_session,
            "measures_per_file": self.measures_per_file,
            "measures_overwritten": self.measures_overwritten,
        }

    def observed_versions(self) -> dict[str, list[str]]:
        """Return the observed versions, for the provenance stage's sidecar.

        ``version_declaration`` is not a canonical column — the schema is an
        interface and this does not belong in it — so the staged path needs a
        sidecar to reach what the in-process path holds in memory.
        """
        return {
            "freesurfer_versions": self.freesurfer_versions,
            "freesurfer_version_declarations": self.freesurfer_version_declarations,
        }

    def failures_frame(self) -> pd.DataFrame:
        """Return parse failures as a frame for the accounting stage."""
        if not self.failures:
            return pd.DataFrame(
                columns=["source_file", "failure_code", "failure_detail", "line_number"]
            )
        return pd.DataFrame([f.as_record() for f in self.failures])


def merge_counters(paths: Iterable[Path | str]) -> dict[str, Any]:
    """Sum ingestion counter sidecars written by a fanned-out staged run.

    The in-process path holds one :class:`IngestResult` for the whole dataset;
    the staged path ingests one subject per process and writes one sidecar
    each. Without this, the accounting stage sees only whichever sidecar
    happened to sit beside the merged Parquet, and the staged funnel silently
    disagrees with the in-process one.

    Args:
        paths: Counter sidecar paths. Missing files are skipped.

    Returns:
        Counters summed across every sidecar found, integers added and nested
        distributions merged key by key.
    """
    totals: dict[str, Any] = {}
    for path in paths:
        source = Path(path)
        if not source.is_file():
            continue
        for key, value in json.loads(source.read_text(encoding="utf-8")).items():
            if isinstance(value, dict):
                bucket = totals.setdefault(key, {})
                for name, count in value.items():
                    bucket[name] = bucket.get(name, 0) + count
            else:
                totals[key] = totals.get(key, 0) + value
    return totals


def ingest(adapter: DatasetAdapter) -> IngestResult:
    """Parse every file an adapter discovers into canonical observations.

    Args:
        adapter: The dataset adapter to drive.

    Returns:
        Canonical observations plus everything the accounting stage needs to
        explain what was lost along the way.
    """
    parser = FreeSurferStatsParser()
    frames: list[pd.DataFrame] = []
    failures: list[ParseFailure] = []
    versions: set[str] = set()
    declarations: set[str] = set()
    files_discovered = 0
    sessions_discovered = 0
    sessions_without_files = 0
    sessions_all_files_rejected = 0
    sessions_no_recognised_regions = 0
    expected_regions = {region for region, _, _ in v1_region_set()}
    observation_losses: dict[str, int] = {}
    regions_per_session: Counter[int] = Counter()
    measures_per_file: Counter[int] = Counter()
    measures_overwritten = 0

    for subject_session in adapter.discover():
        sessions_discovered += 1
        if not subject_session.stats_files:
            sessions_without_files += 1
            continue

        parsed: list[ParsedStatsFile] = []
        for path in subject_session.stats_files:
            files_discovered += 1
            result = parser.parse(path)
            if isinstance(result, ParseFailure):
                failures.append(result)
                continue
            parsed.append(result)
            measures_per_file[result.measure_lines_declared] += 1
            measures_overwritten += result.measures_overwritten
            if result.freesurfer_version:
                versions.add(result.freesurfer_version)
            if result.version_declaration:
                declarations.add(result.version_declaration)

        if not parsed:
            sessions_all_files_rejected += 1
            continue

        frame = adapter.to_canonical(subject_session, parsed)
        if frame.empty:
            sessions_no_recognised_regions += 1
            continue
        frames.append(frame)

        produced = set(frame["region"])
        regions_per_session[len(produced)] += 1
        # Counted here because ingestion is the only stage that can see *why* a
        # region is absent: downstream, a session missing half its regions and
        # a session that never had them are the same handful of rows.
        in_scope = adapter.regions_in_scope(parsed)
        for region in expected_regions - produced:
            if in_scope is None:
                cause = ObservationLoss.UNATTRIBUTED
            elif region in in_scope:
                cause = ObservationLoss.ABSENT_FROM_SOURCE
            else:
                cause = ObservationLoss.SOURCE_UNAVAILABLE
            observation_losses[str(cause)] = observation_losses.get(str(cause), 0) + 1

    observations = conform(pd.concat(frames, ignore_index=True)) if frames else empty_canonical()

    return IngestResult(
        observations=observations,
        failures=failures,
        files_discovered=files_discovered,
        sessions_discovered=sessions_discovered,
        sessions_without_files=sessions_without_files,
        sessions_all_files_rejected=sessions_all_files_rejected,
        sessions_no_recognised_regions=sessions_no_recognised_regions,
        observations_expected=sum(regions_per_session.values()) * len(expected_regions),
        observation_losses=observation_losses,
        regions_per_session={str(k): v for k, v in sorted(regions_per_session.items())},
        measures_per_file={str(k): v for k, v in sorted(measures_per_file.items())},
        measures_overwritten=measures_overwritten,
        freesurfer_versions=sorted(versions),
        freesurfer_version_declarations=sorted(declarations),
    )


def run_ingest(adapter: DatasetAdapter, outdir: Path | str) -> IngestResult:
    """Ingest a dataset and persist the results to Parquet.

    Args:
        adapter: Dataset adapter to drive.
        outdir: Directory for ``observations.parquet`` and
            ``parse_failures.parquet``.

    Returns:
        The ingestion result, already written to disk.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result = ingest(adapter)
    write_canonical(result.observations, out / "observations.parquet")
    result.failures_frame().to_parquet(out / "parse_failures.parquet", index=False)
    # Counters travel with the data so the accounting stage can run as its own
    # Nextflow process without re-deriving them (and without guessing).
    (out / "ingest_counters.json").write_text(
        json.dumps(result.counters(), indent=2), encoding="utf-8"
    )
    (out / "ingest_versions.json").write_text(
        json.dumps(result.observed_versions(), indent=2), encoding="utf-8"
    )
    return result
