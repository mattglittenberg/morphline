"""Provenance capture for the report block (BUILD_PLAN §2.8).

The governing rule: **a reader holding only the HTML file should be able to
reconstruct the run.** If a parameter changed the output, it appears in the
block. That is why the fully *resolved* configuration is captured rather than
the user's YAML — a default that silently changed between versions is exactly
the kind of thing that makes a run irreproducible.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

from morphline import __version__
from morphline.parsers import PARSER_VERSION


def _git(*args: str) -> str | None:
    """Run a git command, returning ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_sha() -> str | None:
    """Return the current commit SHA, or ``None`` outside a repository."""
    return _git("rev-parse", "HEAD")


def git_is_dirty() -> bool:
    """Return whether the working tree has uncommitted changes.

    A dirty flag in the provenance block is not a formality: it is the
    difference between "this run is reproducible from this SHA" and "this run
    included changes that exist on exactly one machine."
    """
    status = _git("status", "--porcelain")
    return bool(status)


@dataclass(slots=True)
class Provenance:
    """The §2.8 provenance block.

    Attributes:
        pipeline_version: morphline package version.
        git_sha: Commit the run was made from, if available.
        git_dirty: Whether uncommitted changes were present.
        container_image: Image reference, from the environment when
            containerised.
        container_digest: Immutable image digest, when known.
        python_version: Interpreter version.
        platform: Host platform string.
        nextflow_version: Nextflow version, when run under Nextflow.
        parser_version: Version of the stats parser used.
        freesurfer_versions: Versions actually observed in the input data —
            not a configured value, an empirical one. Empty when the inputs
            declare no semantic version, as FreeSurfer 5.1 does not.
        freesurfer_version_declarations: What the input headers declared,
            verbatim. Populated even when no semantic version could be derived,
            so an empty ``freesurfer_versions`` is distinguishable from inputs
            that said nothing at all.
        dataset: Dataset identifier.
        dataset_version: Dataset version, accession, or fixture seed.
        input_path: Dataset root as ingested.
        run_parameters: Fully resolved configuration.
        random_seeds: Seeds governing every stochastic step.
        run_timestamp: UTC start time, ISO 8601.
        duration_seconds: Wall-clock duration, filled in at the end.
        stage_versions: Tool versions reported per stage.
    """

    pipeline_version: str = __version__
    git_sha: str | None = None
    git_dirty: bool = False
    container_image: str | None = None
    container_digest: str | None = None
    python_version: str = ""
    platform: str = ""
    nextflow_version: str | None = None
    parser_version: str = PARSER_VERSION
    freesurfer_versions: list[str] = field(default_factory=list)
    freesurfer_version_declarations: list[str] = field(default_factory=list)
    dataset: str = ""
    dataset_version: str = ""
    input_path: str | None = None
    run_parameters: dict[str, Any] = field(default_factory=dict)
    random_seeds: dict[str, int] = field(default_factory=dict)
    run_timestamp: str = ""
    duration_seconds: float | None = None
    stage_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        *,
        dataset: str,
        dataset_version: str,
        run_parameters: dict[str, Any],
        random_seeds: dict[str, int] | None = None,
        input_path: str | None = None,
    ) -> Provenance:
        """Capture the current environment into a provenance block.

        Args:
            dataset: Dataset identifier.
            dataset_version: Dataset version or fixture seed.
            run_parameters: Fully resolved configuration.
            random_seeds: Seeds governing stochastic steps.
            input_path: Dataset root as ingested.

        Returns:
            A populated provenance block; ``duration_seconds`` and
            ``freesurfer_versions`` are filled in as the run proceeds.
        """
        return cls(
            git_sha=git_sha(),
            git_dirty=git_is_dirty(),
            container_image=os.environ.get("MORPHLINE_IMAGE"),
            container_digest=os.environ.get("MORPHLINE_IMAGE_DIGEST"),
            python_version=platform.python_version(),
            platform=f"{platform.system()} {platform.machine()}",
            nextflow_version=os.environ.get("NXF_VER"),
            dataset=dataset,
            dataset_version=dataset_version,
            input_path=input_path,
            run_parameters=run_parameters,
            random_seeds=random_seeds or {},
            run_timestamp=dt.datetime.now(dt.UTC).isoformat(),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the block as a plain mapping for templating and JSON."""
        return asdict(self)


def versions_yml(process: str, tools: dict[str, str]) -> str:
    """Render an nf-core style ``versions.yml`` fragment for one process.

    Args:
        process: Process name, used as the top-level key.
        tools: Tool name to version string.

    Returns:
        YAML text.
    """
    lines = [f'"{process}":']
    lines.extend(f"    {name}: {version}" for name, version in sorted(tools.items()))
    return "\n".join(lines) + "\n"
