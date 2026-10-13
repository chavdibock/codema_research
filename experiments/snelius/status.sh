#!/usr/bin/env bash
# DONE / running / failed / missing counts of an experiment, by scanning results/.
#   experiments/snelius/status.sh E1 [-- make_manifest filters]
set -euo pipefail
EXP="${1:?usage: status.sh EXP}"; shift || true
source "$(dirname "$0")/env.sh"
cdx_activate
python -m experiments.snelius.make_manifest --exp "$EXP" --out "$CDX_RESULTS" --status "$@"
if command -v squeue >/dev/null; then
    echo; echo "== queue =="
    squeue -u "$USER" -h -o "%.18i %.30j %.8T %.10M %.6D %R" | grep "cdx_${EXP}_" || echo "no jobs of $EXP in the queue"
fi
