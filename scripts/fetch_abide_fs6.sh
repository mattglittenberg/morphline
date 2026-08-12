#!/usr/bin/env bash
#
# Fetch the dfsp-spirit ABIDE FreeSurfer 6 aggregate tables (BUILD_PLAN §1.3).
#
# These are NOT a parser target and must never be used as one. They are
# `asegstats2table` / `aparcstats2table` output — one row per subject, one
# column per structure — so they bypass FreeSurferStatsParser entirely. Their
# job here is the independent cross-check §1.3 asks for: morphline's
# per-subject FreeSurfer 5.1 numbers, aggregated, against a table produced by
# different code from a different FreeSurfer release.
#
# Expect systematic offsets. FS 5.1 and FS 6 differ in segmentation — the
# thalamus and the aparc parcellation notably — so the check is on rank
# correlation and site-level structure, never equality. A version difference
# that reads as disagreement is the expected result; a *rank* disagreement is
# the one worth investigating.
#
# Subject IDs match exactly between the two sources — verified 1035/1035, with
# the FS6 deposit a strict subset of the 1112 PCP directories. The 77 subjects
# PCP has and FS6 lacks are themselves a reportable finding, not an error here.
#
# **Licence: CC BY-NC-SA.** The destination must be gitignored; this repo is
# headed for public and the tables are not redistributable. The script refuses
# to write to a tracked path.
#
# Usage:
#   scripts/fetch_abide_fs6.sh [--outdir DIR]
#
#   --outdir DIR   destination (default: data/abide_fs6)

set -euo pipefail

REPO="https://github.com/dfsp-spirit/abide_preproc_smri_freesurfer6.git"
OUTDIR="data/abide_fs6"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir) OUTDIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# The tables are CC BY-NC-SA. Writing them somewhere git would track is how a
# non-redistributable dataset ends up in a public repository, so refuse rather
# than warn.
if git -C . rev-parse --git-dir >/dev/null 2>&1; then
    if ! git check-ignore -q "$OUTDIR" 2>/dev/null; then
        echo "refusing to write to '$OUTDIR': not gitignored." >&2
        echo "These tables are CC BY-NC-SA and must not be committed." >&2
        echo "Add the destination to .gitignore, or pass --outdir." >&2
        exit 1
    fi
fi

if [[ -d "$OUTDIR/.git" ]]; then
    echo "already present at $OUTDIR; pulling"
    git -C "$OUTDIR" pull --ff-only
else
    mkdir -p "$(dirname "$OUTDIR")"
    git clone --depth 1 "$REPO" "$OUTDIR"
fi

# The three tables the 28-test region set needs. The a2009s tables are a
# different parcellation of the same cortex and are deliberately not used:
# mixing them with Desikan-Killiany would compare two different definitions of
# "entorhinal" and call the difference a version effect.
REQUIRED=(
    "stats/aseg_table.tsv"
    "stats/lh.aparc_table_thickness.tsv"
    "stats/rh.aparc_table_thickness.tsv"
)

missing=0
for table in "${REQUIRED[@]}"; do
    if [[ -s "$OUTDIR/$table" ]]; then
        rows=$(( $(wc -l < "$OUTDIR/$table") - 1 ))
        printf '  %-40s %5d subjects\n' "$table" "$rows"
    else
        echo "  MISSING: $table" >&2
        missing=$((missing + 1))
    fi
done

if (( missing > 0 )); then
    echo "$missing required table(s) missing — the deposit layout may have changed." >&2
    exit 1
fi

echo "ok: $OUTDIR"
