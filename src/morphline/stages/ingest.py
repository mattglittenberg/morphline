"""Ingestion — the only stage that knows what a FreeSurfer file looks like.

Drives the parser over an adapter's discovered files and writes canonical
Parquet. Parse failures are collected as reason-coded records rather than
raised, so one malformed file in a large dataset is a row in the accounting
table, not a crashed run (§1.6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from morphline.adapters.base import DatasetAdapter
from morphline.parsers import FreeSurferStatsParser, ParsedStatsFile, ParseFailure
from morphline.schema import conform, empty_canonical, write_canonical


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
        freesurfer_versions: Versions actually observed in the data.

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
    freesurfer_versions: list[str] = field(default_factory=list)

    def counters(self) -> dict[str, int]:
        """Return the loss counters, for the accounting stage's JSON sidecar."""
        return {
            "files_discovered": self.files_discovered,
            "sessions_discovered": self.sessions_discovered,
            "sessions_without_files": self.sessions_without_files,
            "sessions_all_files_rejected": self.sessions_all_files_rejected,
            "sessions_no_recognised_regions": self.sessions_no_recognised_regions,
        }

    def failures_frame(self) -> pd.DataFrame:
        """Return parse failures as a frame for the accounting stage."""
        if not self.failures:
            return pd.DataFrame(
                columns=["source_file", "failure_code", "failure_detail", "line_number"]
            )
        return pd.DataFrame([f.as_record() for f in self.failures])


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
    files_discovered = 0
    sessions_discovered = 0
    sessions_without_files = 0
    sessions_all_files_rejected = 0
    sessions_no_recognised_regions = 0

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
            if result.freesurfer_version:
                versions.add(result.freesurfer_version)

        if not parsed:
            sessions_all_files_rejected += 1
            continue

        frame = adapter.to_canonical(subject_session, parsed)
        if frame.empty:
            sessions_no_recognised_regions += 1
            continue
        frames.append(frame)

    observations = conform(pd.concat(frames, ignore_index=True)) if frames else empty_canonical()

    return IngestResult(
        observations=observations,
        failures=failures,
        files_discovered=files_discovered,
        sessions_discovered=sessions_discovered,
        sessions_without_files=sessions_without_files,
        sessions_all_files_rejected=sessions_all_files_rejected,
        sessions_no_recognised_regions=sessions_no_recognised_regions,
        freesurfer_versions=sorted(versions),
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
    return result
