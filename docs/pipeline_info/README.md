# Nextflow run artifacts

Committed per BUILD_PLAN §2.7, which asks for `-with-report`, `-with-trace`, and
`-with-dag` output from a real run.

## Provenance

| | |
|---|---|
| Commit | `fb96512` |
| CI run | [31849677219](https://github.com/mattglittenberg/morphline/actions/runs/31849677219) |
| Command | `nextflow run . -profile test,docker --outdir results-nf` |
| Nextflow | 26.04.6 (pinned via `NXF_VER`) |
| Input | Generated synthetic fixtures — no external data |

This is the first run in which the `nextflow (docker)` job was **gating** rather
than `continue-on-error`, and the first in which the accounting funnel
reconciled using ingestion's exact counters rather than re-derived ones.

## Files

| File | What it is |
|---|---|
| `dag.html` | Workflow DAG, as a Mermaid diagram |
| `execution_trace.txt` | Per-task trace: status, exit code, duration, peak RSS |
| `execution_report.html` | Nextflow's own execution report |
| `versions.yml` | Merged per-process `versions.yml`, the nf-core convention |

`timeline.html` is deliberately not committed: BUILD_PLAN §2.7 does not name it,
and its content is largely duplicated inside `execution_report.html`.

## Regenerating

These are a snapshot, and a snapshot of a *workflow shape*. **Refresh them
whenever the DAG changes** — adding a process, or adding an input to one, makes
the committed diagram wrong, and a diagram that disagrees with `main.nf` is
worse than no diagram. An earlier attempt to commit these used the previous
run's artifacts, whose DAG showed `COLLECT_CANONICAL` with two inputs instead of
four; the mismatch is invisible unless you go looking for it.

The `nf` CI job uploads them as the `nextflow-run` artifact on every run, so a
refresh is a download rather than a local execution. Locally, `-profile local`
needs no container and no amd64:

```bash
nextflow run . -profile test,local --outdir results-nf \
  -with-report results-nf/pipeline_info/execution_report.html \
  -with-trace  results-nf/pipeline_info/execution_trace.txt \
  -with-dag    results-nf/pipeline_info/dag.html
```
