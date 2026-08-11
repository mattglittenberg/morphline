#!/usr/bin/env bash
#
# Fetch ABIDE PCP per-subject FreeSurfer 5.1 stats files (BUILD_PLAN §1.3).
#
# Track A's parser target: real per-subject .stats files, as opposed to the
# pre-aggregated FS6 tables which bypass the parser entirely. Anonymous HTTPS,
# no credentials, no requester-pays. ~36 MB for the three core files.
#
# Serial by design. Resumable: existing non-empty files are skipped, so an
# interrupted run can simply be re-run.
#
# Empty subject directories are created and kept. All 1112 subjects have a
# directory in the bucket; four hold no files. That empty directory is the
# evidence distinguishing "acquired, derivative missing" from "never acquired",
# and deleting it makes the adapter unable to tell the two apart.
#
# Usage:
#   scripts/fetch_abide_pcp.sh [--outdir DIR] [--subjects FILE] [--all-tables]
#
#   --outdir DIR      destination (default: data/abide_pcp)
#   --subjects FILE   subject list, one per line; generated if absent
#                     (default: data/subjects.txt)
#   --all-tables      fetch all 10 stats files per subject (~67 MB) instead of
#                     the three the 28-test region set needs

set -euo pipefail

S3_ROOT="https://s3.amazonaws.com/fcp-indi"
PREFIX="data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1"

OUTDIR="data/abide_pcp"
SUBJECTS="data/subjects.txt"
TABLES="aseg.stats lh.aparc.stats rh.aparc.stats"

# Verified 2026-08-10 by listing the bucket: 1112 subject directories, 1103
# complete, 9 incomplete. Hard-coding them lets the script tell an expected
# gap apart from a genuine download failure, which a bare error count cannot.
EXPECTED_SUBJECTS=1112
EMPTY_SUBJECTS="UCLA_51233 UCLA_51243 UCLA_51270 UM_2_0050423"
LH_ONLY_SUBJECTS="UCLA_51244 UCLA_51310 UM_1_0050309 UM_1_0050323 UM_1_0050328"

while [ $# -gt 0 ]; do
  case "$1" in
    --outdir)   OUTDIR="$2"; shift 2 ;;
    --subjects) SUBJECTS="$2"; shift 2 ;;
    --all-tables)
      TABLES="aseg.stats lh.aparc.stats rh.aparc.stats lh.aparc.a2009s.stats \
rh.aparc.a2009s.stats lh.BA.stats rh.BA.stats lh.entorhinal_exvivo.stats \
rh.entorhinal_exvivo.stats wmparc.stats"
      shift ;;
    -h|--help) sed -n '3,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

is_listed() {
  for candidate in $2; do
    [ "$candidate" = "$1" ] && return 0
  done
  return 1
}

# The data is CC BY-NC-SA and must not reach a public repo by accident.
if git rev-parse --git-dir >/dev/null 2>&1; then
  if ! git check-ignore -q "$OUTDIR" 2>/dev/null; then
    echo "WARNING: $OUTDIR is not gitignored." >&2
    echo "         ABIDE PCP is CC BY-NC-SA; add it before committing:" >&2
    echo "           printf '\\ndata/\\n' >> .gitignore" >&2
    echo >&2
  fi
fi

mkdir -p "$OUTDIR" "$(dirname "$SUBJECTS")"

if [ ! -s "$SUBJECTS" ]; then
  echo "Building subject list -> $SUBJECTS"
  python3 - "$S3_ROOT" "$PREFIX" > "$SUBJECTS" <<'PY'
import re
import sys
import urllib.parse
import urllib.request

root, prefix = sys.argv[1], sys.argv[2] + "/"
names, token = [], None
while True:
    query = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
    if token:
        query["continuation-token"] = token
    body = urllib.request.urlopen(f"{root}/?{urllib.parse.urlencode(query)}", timeout=60).read()
    text = body.decode()
    names += [
        p[len(prefix):].strip("/")
        for p in re.findall(r"<Prefix>(.*?)</Prefix>", text)
        if p != prefix
    ]
    match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", text)
    if not match:
        break
    token = match.group(1)
print("\n".join(names))
PY
fi

total_subjects=$(grep -c . "$SUBJECTS" || true)
echo "Subjects listed:  $total_subjects (expected $EXPECTED_SUBJECTS)"
if [ "$total_subjects" -ne "$EXPECTED_SUBJECTS" ]; then
  echo "NOTE: subject count differs from the verified figure; the bucket may have changed." >&2
fi
echo "Destination:      $OUTDIR"
echo "Tables:           $(echo "$TABLES" | wc -w | tr -d ' ') per subject"
echo

fetched=0
skipped=0
expected_miss=0
unexpected_miss=0
unexpected_list=""
processed=0

while IFS= read -r subject; do
  [ -n "$subject" ] || continue
  processed=$((processed + 1))
  printf '\r[%4d/%4d] %-24s' "$processed" "$total_subjects" "$subject"

  mkdir -p "$OUTDIR/$subject/stats"
  for table in $TABLES; do
    dest="$OUTDIR/$subject/stats/$table"
    if [ -s "$dest" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    if curl -sfS --retry 3 --retry-delay 1 --max-time 60 \
        "$S3_ROOT/$PREFIX/$subject/stats/$table" -o "$dest" 2>/dev/null; then
      fetched=$((fetched + 1))
    else
      rm -f "$dest"
      if is_listed "$subject" "$EMPTY_SUBJECTS"; then
        expected_miss=$((expected_miss + 1))
      elif is_listed "$subject" "$LH_ONLY_SUBJECTS" && [ "$table" != "lh.aparc.stats" ]; then
        expected_miss=$((expected_miss + 1))
      else
        unexpected_miss=$((unexpected_miss + 1))
        unexpected_list="$unexpected_list  $subject/$table"$'\n'
      fi
    fi
  done
done < "$SUBJECTS"

printf '\r%-40s\r' ''

echo "Fetched:          $fetched"
echo "Already present:  $skipped"
echo "Missing (known):  $expected_miss"
echo "Missing (other):  $unexpected_miss"
echo

aseg_count=$(find "$OUTDIR" -name 'aseg.stats' | wc -l | tr -d ' ')
lh_count=$(find "$OUTDIR" -name 'lh.aparc.stats' | wc -l | tr -d ' ')
rh_count=$(find "$OUTDIR" -name 'rh.aparc.stats' | wc -l | tr -d ' ')
echo "aseg.stats:       $aseg_count"
echo "lh.aparc.stats:   $lh_count"
echo "rh.aparc.stats:   $rh_count"
echo "On disk:          $(du -sh "$OUTDIR" | cut -f1)"

if [ "$unexpected_miss" -gt 0 ]; then
  echo
  echo "Unexpected misses — re-run to retry, these are not the known-incomplete 9:" >&2
  printf '%s' "$unexpected_list" >&2
  exit 1
fi

echo
echo "Complete. Every gap is one of the 9 known-incomplete subjects (BUILD_PLAN §1.3);"
echo "they belong in the accounting funnel with reason codes, not silently dropped."
