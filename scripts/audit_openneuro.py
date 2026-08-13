"""Audit OpenNeuro for a longitudinal accession with FreeSurfer derivatives.

BUILD_PLAN §1.2's Track B access task, timeboxed to four hours. It answers one
question: **does a public OpenNeuro accession ship FreeSurfer `.stats` files
alongside ≥2 structural sessions per subject?** A hit makes longitudinal
real-data integration a v1.0 bonus; a miss is the §0.3 outcome and v1.0 ships
with the gap named. Either answer is useful; not having one is not.

§1.2 specifies `datalad clone` per candidate. This does not do that. Cloning
needs `datalad` and `git-annex` installed and pulls repository metadata for
every candidate, which is minutes each across hundreds of candidates. Two
public endpoints answer the same question without downloading anything:

* **OpenNeuro's GraphQL API** reports each dataset's session and subject labels
  directly, so the session filter costs one request per hundred datasets.
* **OpenNeuro's S3 bucket is anonymously listable**, exactly like the fcp-indi
  bucket ``fetch_abide_pcp.sh`` already reads, so file *paths* can be inspected
  without fetching file *contents*.

Stage 2 is where nearly everything dies, and that is expected: §1.3 already
warns most accessions ship raw images with no FreeSurfer derivatives at all.
The check greps for the filenames rather than for a ``derivatives/`` directory
on purpose — a derivatives directory holding MRIQC or fMRIPrep output is a
reject, and it is the most common near-miss.

What this cannot decide for you is criterion three, scanner or site variation.
§1.2 is explicit that failing *only* that is still worth taking: it makes the
accession a longitudinal-only target rather than a rejection. Survivors are
printed for a human to judge.

Usage::

    uv run python scripts/audit_openneuro.py
    uv run python scripts/audit_openneuro.py --min-sessions 3 --limit 400
    uv run python scripts/audit_openneuro.py --check ds000001 ds001234

Stdlib only, so it runs from a cold clone with nothing installed but Python.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

GRAPHQL_URL: Final = "https://openneuro.org/crn/graphql"
S3_ROOT: Final = "https://s3.amazonaws.com/openneuro.org"
S3_NAMESPACE: Final = "{http://s3.amazonaws.com/doc/2006-03-01/}"

#: The files that decide the question. A `derivatives/` directory alone proves
#: nothing — it is usually MRIQC or fMRIPrep output, neither of which morphline
#: can ingest.
STATS_FILENAMES: Final = ("aseg.stats", "lh.aparc.stats", "rh.aparc.stats")

#: Bound on the fallback whole-dataset walk. Large accessions carry hundreds of
#: thousands of keys, most of them raw images, and walking all of them to prove
#: a negative is not worth the requests. A scan that hits this cap without
#: finding anything is reported as ``truncated`` — never as ``no-stats``.
MAX_KEY_PAGES: Final = 40

#: Bound on the ``derivatives/`` walk, which is where FreeSurfer output lives in
#: BIDS and is a small fraction of a dataset's keys. Generous on purpose: this
#: is the scan whose counts get reported, so truncating it understates coverage.
DERIVATIVES_KEY_PAGES: Final = 200

#: Seconds between requests. Two public endpoints, one afternoon, no reason to
#: hammer either.
REQUEST_DELAY: Final = 0.2

#: Attempts per request before giving up. The GraphQL endpoint returns 500s
#: during deep pagination — observed around 800 datasets in — and an audit that
#: dies on the first one throws away everything scanned so far.
MAX_ATTEMPTS: Final = 4

#: Base for exponential backoff between attempts, in seconds.
RETRY_BACKOFF: Final = 2.0

#: Subject and session entities are matched independently, anywhere in the key,
#: because FreeSurfer's own convention and BIDS' disagree about the separator.
#: A ``SUBJECTS_DIR`` laid out for longitudinal BIDS derivatives is
#: ``sub-01_ses-02/stats/aseg.stats`` — one directory, underscore-joined — while
#: raw BIDS nests ``sub-01/ses-02/``. A pattern requiring the nested form reads
#: the underscore layout as single-session, which would classify the one dataset
#: this audit exists to find as ``stats-only`` and reject it.
_SUBJECT = re.compile(r"sub-([A-Za-z0-9]+)")
_SESSION = re.compile(r"ses-([A-Za-z0-9]+)")


@dataclass(slots=True)
class Candidate:
    """A dataset that cleared the session filter.

    Attributes:
        dataset_id: OpenNeuro accession, e.g. ``ds001234``.
        n_subjects: Subject labels reported by the API.
        n_sessions: Session labels reported by the API.
        modalities: Modalities reported by the API.
        publish_date: ISO publication date, for tie-breaking by recency.
    """

    dataset_id: str
    n_subjects: int
    n_sessions: int
    modalities: list[str]
    publish_date: str | None


@dataclass(slots=True)
class Finding:
    """What a key listing revealed about one candidate.

    Attributes:
        dataset_id: OpenNeuro accession.
        counts: Matching key count per stats filename.
        subjects_with_stats: Distinct ``sub-*`` entities owning a stats file.
        subjects_multi_session: Subjects with ≥2 distinct ``ses-*`` entities
            among their stats files — the criterion that actually matters, and
            the one the API's dataset-level session list cannot establish.
        sample_paths: A few matching keys, so a human can see the layout.
        keys_scanned: How many keys were inspected.
        truncated: Whether the scan hit :data:`MAX_KEY_PAGES`.
        error: Transport failure, if the listing could not be completed.
    """

    dataset_id: str
    counts: dict[str, int] = field(default_factory=dict)
    subjects_with_stats: int = 0
    subjects_multi_session: int = 0
    sample_paths: list[str] = field(default_factory=list)
    keys_scanned: int = 0
    truncated: bool = False
    error: str | None = None

    @property
    def has_stats(self) -> bool:
        """Whether any FreeSurfer stats file was found."""
        return sum(self.counts.values()) > 0

    @property
    def verdict(self) -> str:
        """A one-word summary for the report table.

        ``truncated`` exists because "we did not find any" and "we stopped
        looking" are different claims, and only the first is a rejection. A
        capped scan that found nothing gets flagged for a human rather than
        counted as a negative.
        """
        if self.error:
            return "ERROR"
        if self.has_stats:
            return "CANDIDATE" if self.subjects_multi_session >= 2 else "stats-only"
        return "truncated" if self.truncated else "no-stats"


def _read_with_retry(request: urllib.request.Request | str, timeout: int = 60) -> bytes:
    """Fetch a URL, retrying server-side failures with exponential backoff.

    Client errors are not retried — a 404 will still be a 404 — but 5xx and
    transport failures are, because both endpoints are public services that
    intermittently fail under paging and the alternative is losing a scan that
    has already cost minutes.

    Args:
        request: A prepared request, or a URL to GET.
        timeout: Per-attempt timeout in seconds.

    Returns:
        The response body.

    Raises:
        urllib.error.URLError: If every attempt failed.
    """
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(RETRY_BACKOFF**attempt)
    assert last is not None
    raise last


def _post_graphql(query: str) -> dict[str, Any]:
    """Execute a GraphQL query.

    Args:
        query: The query document.

    Returns:
        The ``data`` payload.

    Raises:
        RuntimeError: If the endpoint returns errors or an unusable response.
    """
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    payload = json.loads(_read_with_retry(request))
    data = payload.get("data")

    # GraphQL reports per-field failures alongside usable data: a resolver that
    # cannot find one dataset's snapshot nulls that field and records an error,
    # leaving the rest of the page intact. Treating any `errors` entry as fatal
    # made a single broken dataset a permanent wall — the page always failed, so
    # the cursor never advanced past it, and the audit stopped at the same 1100
    # datasets on every run while looking like a transient outage.
    if isinstance(data, dict) and data.get("datasets"):
        if "errors" in payload:
            messages = {str(e.get("message")) for e in payload["errors"]}
            print(
                f"  note: {len(payload['errors'])} field error(s) in this page "
                f"({', '.join(sorted(messages))}); affected datasets skipped",
                file=sys.stderr,
            )
        return data

    if "errors" in payload:
        raise RuntimeError(f"GraphQL error: {payload['errors']}")
    raise RuntimeError(f"unexpected GraphQL response: {payload!r}")


def list_candidates(
    min_sessions: int, limit: int | None, page_size: int = 100
) -> tuple[list[Candidate], int, bool]:
    """Page the dataset index, keeping those with enough sessions.

    The API reports *dataset-level* session labels, so this is a coarse filter:
    a dataset can declare two sessions while most subjects have one. That is
    fine — it is cheap, and :func:`inspect_dataset` establishes the per-subject
    truth for the handful that survive.

    Args:
        min_sessions: Minimum session labels a dataset must declare.
        limit: Stop after examining this many datasets, or ``None`` for all.
        page_size: Datasets per request.

    Returns:
        Candidates newest first, how many datasets were examined, and whether
        the index was exhausted. The last matters: a partial stage 1 that found
        nothing is not the same claim as a complete one that found nothing.
    """
    candidates: list[Candidate] = []
    cursor: str | None = None
    examined = 0
    complete = True

    while True:
        after = f', after: "{cursor}"' if cursor else ""
        query = (
            f"{{datasets(first: {page_size}{after}) {{"
            "pageInfo {hasNextPage endCursor} "
            "edges {node {id publishDate latestSnapshot {summary "
            "{modalities sessions subjects}}}}}}"
        )
        try:
            data = _post_graphql(query)
        except (urllib.error.URLError, RuntimeError) as exc:
            print(
                f"\n  stage 1 stopped early after {examined} datasets: {exc}\n"
                f"  Reporting what was scanned. This is a PARTIAL audit — re-run with\n"
                f"  --limit to resume coverage before concluding anything.",
                file=sys.stderr,
            )
            complete = False
            break
        block = data["datasets"]

        for edge in block["edges"] or []:
            # A field error nulls the whole edge, not just the failing field, so
            # every level here has to tolerate None. The dataset is counted as
            # examined and skipped: it has no session list to filter on.
            node = (edge or {}).get("node") or {}
            examined += 1
            if not node.get("id"):
                continue
            snapshot = node.get("latestSnapshot") or {}
            summary = snapshot.get("summary") or {}
            sessions = summary.get("sessions") or []
            subjects = summary.get("subjects") or []
            modalities = summary.get("modalities") or []

            if len(sessions) < min_sessions:
                continue
            if "mri" not in [str(m).lower() for m in modalities]:
                continue
            candidates.append(
                Candidate(
                    dataset_id=str(node["id"]),
                    n_subjects=len(subjects),
                    n_sessions=len(sessions),
                    modalities=[str(m) for m in modalities],
                    publish_date=node.get("publishDate"),
                )
            )

        print(
            f"  scanned {examined} datasets, {len(candidates)} with ≥{min_sessions} sessions",
            end="\r" if sys.stderr.isatty() else "\n",
            file=sys.stderr,
        )
        if limit is not None and examined >= limit:
            complete = False
            break
        if not block["pageInfo"]["hasNextPage"]:
            break
        cursor = block["pageInfo"]["endCursor"]
        time.sleep(REQUEST_DELAY)

    print(file=sys.stderr)
    candidates.sort(key=lambda c: c.publish_date or "", reverse=True)
    return candidates, examined, complete


def _list_keys(prefix: str, max_pages: int) -> tuple[list[str], bool]:
    """List bucket keys under a prefix.

    Args:
        prefix: Key prefix, e.g. ``ds001234/``.
        max_pages: Page cap.

    Returns:
        The keys, and whether the listing was cut short by the cap.
    """
    keys: list[str] = []
    token: str | None = None

    for _page in range(max_pages):
        url = f"{S3_ROOT}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token, safe='')}"
        tree = ElementTree.fromstring(_read_with_retry(url))

        for contents in tree.findall(f"{S3_NAMESPACE}Contents"):
            key = contents.findtext(f"{S3_NAMESPACE}Key")
            if key:
                keys.append(key)

        if tree.findtext(f"{S3_NAMESPACE}IsTruncated") != "true":
            return keys, False
        token = tree.findtext(f"{S3_NAMESPACE}NextContinuationToken")
        if not token:
            return keys, False
        time.sleep(REQUEST_DELAY)

    return keys, True


def inspect_dataset(dataset_id: str) -> Finding:
    """Look for FreeSurfer stats files in a dataset's key listing.

    Walks ``<id>/derivatives/`` first, because that is where BIDS puts
    FreeSurfer output and it is a small fraction of a dataset's keys. Falls
    through to a bounded walk of the whole prefix **only when no stats file was
    found there**, which catches non-standard layouts without paying for them in
    the common case.

    The ordering matters and both earlier versions of it were wrong. Stopping at
    ``derivatives/`` whenever it was non-empty missed everything else in
    ds000117, whose MEG derivatives ended the scan after 130 of 978 keys.
    Walking only the whole prefix hit the page cap on ds002785 and reported
    ``no-stats`` for a dataset whose ``derivatives/freesurfer/`` holds
    ``aseg.stats`` — a false negative in the one direction this audit exists to
    avoid. Hence: derivatives first, full scan as a fallback, and a cap that is
    reported rather than silently believed.

    Args:
        dataset_id: OpenNeuro accession.

    Returns:
        What was found, including transport errors rather than raising them.
    """
    finding = Finding(dataset_id=dataset_id)
    try:
        keys, truncated = _list_keys(f"{dataset_id}/derivatives/", DERIVATIVES_KEY_PAGES)
        if not any(k.rsplit("/", 1)[-1] in STATS_FILENAMES for k in keys):
            fallback, fallback_truncated = _list_keys(f"{dataset_id}/", MAX_KEY_PAGES)
            keys = sorted(set(keys) | set(fallback))
            truncated = truncated or fallback_truncated
    except (urllib.error.URLError, ElementTree.ParseError, TimeoutError) as exc:
        finding.error = f"{type(exc).__name__}: {exc}"
        return finding

    finding.keys_scanned = len(keys)
    finding.truncated = truncated

    sessions_by_subject: dict[str, set[str]] = {}
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        if name not in STATS_FILENAMES:
            continue
        finding.counts[name] = finding.counts.get(name, 0) + 1
        if len(finding.sample_paths) < 4:
            finding.sample_paths.append(key)
        subject_match = _SUBJECT.search(key)
        if subject_match:
            sessions = sessions_by_subject.setdefault(subject_match.group(1), set())
            session_match = _SESSION.search(key)
            if session_match:
                sessions.add(session_match.group(1))

    finding.subjects_with_stats = len(sessions_by_subject)
    finding.subjects_multi_session = sum(
        1 for sessions in sessions_by_subject.values() if len(sessions) >= 2
    )
    return finding


def _report(candidates: list[Candidate], findings: list[Finding]) -> None:
    """Print the audit result as a table, most promising first."""
    by_id = {c.dataset_id: c for c in candidates}
    rank = {"CANDIDATE": 0, "stats-only": 1, "truncated": 2, "ERROR": 3, "no-stats": 4}
    findings.sort(key=lambda f: (rank[f.verdict], -f.subjects_multi_session))

    print(
        f"\n{'dataset':<12} {'verdict':<11} {'subj':>5} {'sess':>5} "
        f"{'w/stats':>8} {'multi':>6}  sample"
    )
    print("-" * 96)
    for finding in findings:
        candidate = by_id.get(finding.dataset_id)
        sample = finding.sample_paths[0] if finding.sample_paths else ""
        if finding.truncated:
            sample = (sample + "  [scan truncated]").strip()
        if finding.error:
            sample = finding.error[:48]
        print(
            f"{finding.dataset_id:<12} {finding.verdict:<11} "
            f"{candidate.n_subjects if candidate else 0:>5} "
            f"{candidate.n_sessions if candidate else 0:>5} "
            f"{finding.subjects_with_stats:>8} {finding.subjects_multi_session:>6}  "
            f"{sample[:44]}"
        )

    hits = [f for f in findings if f.verdict == "CANDIDATE"]
    unresolved = [f for f in findings if f.verdict in {"truncated", "ERROR"}]
    print()
    if unresolved:
        print(
            f"{len(unresolved)} accession(s) were not resolved — the key scan hit its cap or\n"
            "failed. These are not rejections. Re-run them individually, where the whole\n"
            "page budget goes to one dataset:\n"
        )
        print(
            "  uv run python scripts/audit_openneuro.py --check "
            + " ".join(f.dataset_id for f in unresolved[:8])
        )
        print()
    if hits:
        print(f"{len(hits)} accession(s) cleared both automated criteria:")
        for finding in hits:
            print(f"  https://openneuro.org/datasets/{finding.dataset_id}")
        print(
            "\nNext, by hand (§1.2 criterion three, which no script settles): scanner or\n"
            "site variation, licence, and whether the repeat sessions are structural\n"
            "rather than functional-only. Failing only the scanner criterion is still a\n"
            "take — it makes the accession a longitudinal-only target, not a rejection."
        )
    else:
        print(
            "No accession cleared both criteria.\n\n"
            "That is BUILD_PLAN §0.3's documented outcome, not a failure: Track B stays\n"
            "synthetic-only, the longitudinal path keeps its injected-truth validation,\n"
            "and the README names the gap. Stop looking when the timebox expires."
        )


def main(argv: list[str] | None = None) -> int:
    """Run the audit.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status. Zero whether or not a candidate was found — a null
        result is an answer, and §1.2 treats it as an acceptable one.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="stop after N datasets")
    parser.add_argument("--check", nargs="+", metavar="ID", help="inspect these accessions only")
    parser.add_argument("--out", type=Path, default=Path("data/openneuro_audit.json"))
    args = parser.parse_args(argv)

    if args.check:
        candidates = [Candidate(d, 0, 0, [], None) for d in args.check]
        examined, complete = len(candidates), True
        print(f"inspecting {len(candidates)} accession(s)", file=sys.stderr)
    else:
        print("stage 1: filtering the dataset index by session count", file=sys.stderr)
        candidates, examined, complete = list_candidates(args.min_sessions, args.limit)
        print(f"stage 2: listing keys for {len(candidates)} candidate(s)", file=sys.stderr)

    findings: list[Finding] = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"  [{index}/{len(candidates)}] {candidate.dataset_id}", end="\r", file=sys.stderr)
        findings.append(inspect_dataset(candidate.dataset_id))
        time.sleep(REQUEST_DELAY)
    print(file=sys.stderr)

    _report(candidates, findings)
    if not complete:
        print(
            f"\nPARTIAL: stage 1 examined {examined} datasets without exhausting the\n"
            "index, so absence of a candidate here is not absence of a candidate.",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "min_sessions": args.min_sessions,
                "datasets_examined": examined,
                "index_exhausted": complete,
                "candidates": [asdict(c) for c in candidates],
                "findings": [asdict(f) for f in findings],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
