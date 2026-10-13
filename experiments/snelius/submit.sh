#!/usr/bin/env bash
# Regenerate the manifest(s) of an experiment and submit packed GPU arrays.
#
#   experiments/snelius/submit.sh E1 [--pack P] [--time HH:MM:SS] [--max-parallel G]
#                                    [--mps] [--partition gpu_a100] [--kinds seq,eval]
#                                    [--dry-run] [--yes] [-- <make_manifest filters>]
#
# Prints manifest size, pack size, --time and the estimated GPU-hours (upper bound:
# array jobs x --time) and asks for confirmation before calling sbatch. Nothing is
# submitted with --dry-run, and nothing is submitted when this file is sourced.
set -euo pipefail
[[ "${BASH_SOURCE[0]}" != "$0" ]] && { echo "do not source submit.sh"; return 1; }

EXP="${1:?usage: submit.sh EXP [options]}"; shift
PACK_OVERRIDE=""; TIME_OVERRIDE=""; MPS=0; DRY=0; YES=0; KINDS="seq,mlp,eval,osrl"
MM_ARGS=()
source "$(dirname "$0")/env.sh"
G="$CDX_MAX_PARALLEL"; PART="$CDX_PARTITION"
while [ $# -gt 0 ]; do
    case "$1" in
        --pack) PACK_OVERRIDE="$2"; shift 2;;
        --time) TIME_OVERRIDE="$2"; shift 2;;
        --max-parallel) G="$2"; shift 2;;
        --partition) PART="$2"; shift 2;;
        --kinds) KINDS="$2"; shift 2;;
        --mps) MPS=1; shift;;
        --dry-run) DRY=1; shift;;
        --yes) YES=1; shift;;
        --) shift; MM_ARGS=("$@"); break;;
        *) echo "unknown option $1"; exit 2;;
    esac
done

cdx_activate
mkdir -p logs/slurm logs/runs
python -m experiments.snelius.make_manifest --exp "$EXP" --out "$CDX_RESULTS" "${MM_ARGS[@]}"

to_hours() { awk -F: '{ if (NF==3) print $1 + $2/60 + $3/3600; else if (NF==2) print $1/60 + $2/3600; else print $1/60 }' <<< "$1"; }

for kind in ${KINDS//,/ }; do
    MANIFEST="experiments/manifests/${EXP}.${kind}.txt"
    [ -s "$MANIFEST" ] || continue
    N=$(wc -l < "$MANIFEST")
    case "$kind" in
        seq)  P="${PACK_OVERRIDE:-$CDX_PACK_SEQ}";  T="${TIME_OVERRIDE:-$CDX_TIME_SEQ}";;
        mlp)  P="${PACK_OVERRIDE:-$CDX_PACK_MLP}";  T="${TIME_OVERRIDE:-$CDX_TIME_MLP}";;
        eval) P="${PACK_OVERRIDE:-$CDX_PACK_EVAL}"; T="${TIME_OVERRIDE:-$CDX_TIME_EVAL}";;
        osrl) P="${PACK_OVERRIDE:-$CDX_PACK_OSRL}"; T="${TIME_OVERRIDE:-$CDX_TIME_OSRL}";;
    esac
    NJOBS=$(( (N + P - 1) / P ))
    HOURS=$(to_hours "$T")
    EST=$(awk -v j="$NJOBS" -v h="$HOURS" 'BEGIN { printf "%.1f", j * h }')
    echo "------------------------------------------------------------------"
    echo "experiment     : $EXP ($kind)"
    echo "manifest       : $MANIFEST"
    echo "manifest size  : $N runs"
    echo "pack size      : $P runs per GPU slice  -> $NJOBS array jobs (%$G at once)"
    echo "--time         : $T   partition: $PART   MPS: $MPS"
    echo "est. GPU-hours : <= $EST  (array jobs x --time; 1 GPU = 1 A100 slice)"
    if [ -n "${CDX_SBU_PER_GPU_HOUR:-}" ]; then
        awk -v e="$EST" -v r="$CDX_SBU_PER_GPU_HOUR" 'BEGIN { printf "est. SBU       : <= %.0f\n", e * r }'
    fi
    if [ "$DRY" = "1" ]; then echo "(dry run: not submitted)"; continue; fi
    if [ "$YES" != "1" ]; then
        read -r -p "submit $NJOBS array jobs? [y/N] " ans
        [[ "$ans" =~ ^[Yy]$ ]] || { echo "skipped"; continue; }
    fi
    # array tasks read their lines when they start: submit an immutable snapshot so a
    # later regeneration of the manifest cannot change pending jobs
    mkdir -p experiments/manifests/submitted
    SNAP="experiments/manifests/submitted/${EXP}.${kind}.$(date +%Y%m%d-%H%M%S).txt"
    cp "$MANIFEST" "$SNAP"
    ACC=(); [ -n "$CDX_ACCOUNT" ] && ACC=(--account="$CDX_ACCOUNT")
    sbatch "${ACC[@]}" --job-name="cdx_${EXP}_${kind}" --partition="$PART" --time="$T" \
        --array="0-$(( NJOBS - 1 ))%${G}" \
        --export=ALL,MANIFEST="$SNAP",PACK="$P",USE_MPS="$MPS",CDX_ROOT="$CDX_ROOT" \
        experiments/snelius/array_gpu.sbatch
    sleep 1  # distinct snapshot names
done
