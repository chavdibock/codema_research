#!/usr/bin/env bash
# One command for the whole study: submits every remaining stage as a SLURM dependency
# chain and returns. Run it from the Experiments/ directory on the login node:
#
#   experiments/snelius/run_all.sh [--dry-run] [--yes] [--gate auto|manual] [--p3]
#        [--max-parallel G] [--time HH:MM:SS] [--rounds R] [--e0-time HH:MM:SS]
#        [--max-attempts N] [--partition P] [--mps] [--no-profile]
#   experiments/snelius/run_all.sh --status     # progress of every stage
#   experiments/snelius/run_all.sh --cancel     # cancel the jobs of the last submission
#
# It refuses to submit while jobs of a previous submission (cdx_*) are still queued.
#
# Stages (only the ones with work left are submitted; re-running is always safe):
#   bootstrap  1 slice: venv (if missing), datasets (if missing), checks, tests, cache
#   E0         G0 slices of work-queue workers (E0, E0prec, E0tune) + 1 job per E0pack
#              size + E0fid (informative, nothing waits for it)
#   main       G slices x R rounds of work-queue workers over E6full E1 E4 E2 E3 E5 E7
#              E8 E9 (+ E6long E7ext E11 with --p3). Round r+1 of slice i starts when
#              round r of slice i ends (aftercorr), so timeouts and late eval-only runs
#              are picked up without you. The first worker turns E0 into decisions.yaml
#              (--gate auto) or waits for yours (--gate manual: jobs are held).
#   profile    E6 on an exclusive full node, after main round 1 (E1 checkpoints)
#   finalize   status, E0 report, tables and figures into $CDX_PROJECT/summary
#
# Before calling sbatch it prints, per stage, the job count, --time and the GPU-hour
# upper bound (jobs x GPUs x --time; SLURM charges only the time actually used, and
# workers exit as soon as their queue is empty), then asks once for confirmation.
set -euo pipefail
[[ "${BASH_SOURCE[0]}" != "$0" ]] && { echo "do not source run_all.sh"; return 1; }
cd "$(dirname "$0")/../.."   # Experiments/
source experiments/snelius/env.sh

GATE=auto; P3=0; DRY=0; YES=0; MPS=0; PROFILE=1; ACTION=submit
G="$CDX_MAX_PARALLEL"; PART="$CDX_PARTITION"
T_MAIN="${CDX_TIME_WORKER:-12:00:00}"; T_E0="${CDX_TIME_E0:-04:00:00}"
ROUNDS="${CDX_ROUNDS:-4}"; MAX_ATT="${CDX_MAX_ATTEMPTS:-3}"
while [ $# -gt 0 ]; do
    case "$1" in
        --gate) GATE="$2"; shift 2;;
        --p3) P3=1; shift;;
        --max-parallel) G="$2"; shift 2;;
        --time) T_MAIN="$2"; shift 2;;
        --e0-time) T_E0="$2"; shift 2;;
        --rounds) ROUNDS="$2"; shift 2;;
        --max-attempts) MAX_ATT="$2"; shift 2;;
        --partition) PART="$2"; shift 2;;
        --mps) MPS=1; shift;;
        --no-profile) PROFILE=0; shift;;
        --dry-run) DRY=1; shift;;
        --yes) YES=1; shift;;
        --status) ACTION=status; shift;;
        --cancel) ACTION=cancel; shift;;
        *) echo "unknown option $1"; exit 2;;
    esac
done
[[ "$GATE" =~ ^(auto|manual)$ ]] || { echo "--gate auto|manual"; exit 2; }
mkdir -p logs/slurm logs/runs "$CDX_RESULTS"
LAST=logs/run_all.last

if [ "$ACTION" = cancel ]; then
    [ -s "$LAST" ] || { echo "no recorded submission ($LAST)"; exit 0; }
    ids=$(awk '{print $2}' "$LAST" | tr '\n' ' ')
    echo "cancelling: $ids"; scancel $ids; exit 0
fi

HAVE_VENV=0; [ -x "$CDX_VENV/bin/python" ] && HAVE_VENV=1
if [ "$HAVE_VENV" = 1 ]; then cdx_activate; fi

if [ "$ACTION" = status ]; then
    [ "$HAVE_VENV" = 1 ] || { echo "venv not built yet"; exit 0; }
    python -m experiments.snelius.pipeline status --out "$CDX_RESULTS" $([ $P3 = 1 ] && echo --p3)
    echo; squeue -u "$USER" -h -o "%.18i %.22j %.9T %.10M %.10l %R" | grep -E "cdx_" || echo "no cdx jobs queued"
    [ -f "$CDX_RESULTS/GATE_FAILED" ] && echo "E0 GATE FAILED: $CDX_RESULTS/GATE_FAILED"
    exit 0
fi

# a previous submission that is still queued or running would only duplicate workers
if command -v squeue >/dev/null && [ "$DRY" != 1 ]; then
    live=$(squeue -h -u "$USER" -o "%j" | grep -c '^cdx_' || true)
    if [ "$live" -gt 0 ]; then
        echo "$live cdx_* jobs are still queued/running: wait for them, or cancel them with"
        echo "  experiments/snelius/run_all.sh --cancel"
        exit 1
    fi
fi

# ---------------------------------------------------------------- what is left
PLAN_JSON=$(mktemp)
if [ "$HAVE_VENV" = 1 ]; then
    # datasets: download here (login node) if missing; compute nodes only as a fallback
    python -c "import sys; from experiments.common.data import TASKS, dataset_path; \
sys.exit(any(not dataset_path(t).exists() for t in TASKS))" || {
        echo "== downloading DSRL datasets into $DSRL_DATASET_DIR"
        python -m experiments.common.data --download --tasks all; }
    python -m experiments.snelius.pipeline plan --out "$CDX_RESULTS" --json "$PLAN_JSON" \
        $([ $P3 = 1 ] && echo --p3)
    # a previous failed gate is re-evaluated after E0 runs again
    [ -f "$CDX_RESULTS/decisions.yaml" ] || rm -f "$CDX_RESULTS/GATE_FAILED"
    pj() { python -c "import json,sys; v=json.load(open('$PLAN_JSON'))['$1']; \
print(' '.join(map(str,v)) if isinstance(v,list) else (int(v) if isinstance(v,bool) else v))"; }
    BOOT=$(pj bootstrap_needed); E0_SLICES=$(pj e0_slices); E0PACK=$(pj e0pack_sizes)
    E0FID=$(pj e0fid_needed); MAIN_LEFT=$(pj main_left); PROF=$(pj profile_needed)
    MAIN_EXPS=$(pj main_exps)
else
    echo "venv not built yet: everything is scheduled; bootstrap builds the venv first"
    BOOT=1; E0_SLICES="$G"; E0PACK="1 3 6 9"; E0FID=1; MAIN_LEFT=1; PROF=1
    MAIN_EXPS="E6full E1 E4 E2 E3 E5 E7 E8 E9$([ $P3 = 1 ] && echo ' E6long E7ext E11')"
fi
rm -f "$PLAN_JSON"
[ "$PROFILE" = 1 ] || PROF=0
[ "$E0_SLICES" -gt "$G" ] && E0_SLICES="$G"

# ---------------------------------------------------------------- the plan
secs() { awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4;
                          else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }' <<< "$1"; }
hrs() { awk -v s="$(secs "$1")" -v n="$2" 'BEGIN { printf "%.1f", s * n / 3600 }'; }
TOTAL=0
line() {  # stage, jobs, gpus per job, time
    local h; h=$(hrs "$4" $(( $2 * $3 )))
    TOTAL=$(awk -v a="$TOTAL" -v b="$h" 'BEGIN { print a + b }')
    printf "  %-34s %4s job(s) x %s GPU  --time %-9s <= %7s GPU-h\n" "$1" "$2" "$3" "$4" "$h"
}
echo "=================================================================="
echo "partition $PART   results $CDX_RESULTS   gate $GATE   max parallel slices $G"
echo "packs: seq ${CDX_PACK_SEQ} (or the E0pack decision)  mlp ${CDX_PACK_MLP}  eval ${CDX_PACK_EVAL}"
[ "$BOOT" = 1 ] && line "bootstrap" 1 1 03:00:00
[ "$E0_SLICES" -gt 0 ] && line "E0 workers (E0, E0prec, E0tune)" "$E0_SLICES" 1 "$T_E0"
for p in $E0PACK; do line "E0pack p$p" 1 1 "$T_E0"; done
[ "$E0FID" = 1 ] && line "E0fid (OSRL script)" 1 1 "$CDX_TIME_OSRL"
[ "$MAIN_LEFT" -gt 0 ] && line "main workers ($ROUNDS rounds x $G)" $(( ROUNDS * G )) 1 "$T_MAIN"
[ "$PROF" = 1 ] && line "E6 profiling (exclusive node)" 1 4 04:00:00
line "finalize (analysis)" 1 1 01:00:00
echo "  upper bound total: $TOTAL GPU-hours (charged: only the time actually used)"
if [ -n "${CDX_SBU_PER_GPU_HOUR:-}" ]; then
    awk -v e="$TOTAL" -v r="$CDX_SBU_PER_GPU_HOUR" 'BEGIN { printf "  upper bound SBU  : %.0f\n", e * r }'
fi
echo "  main queue: $MAIN_EXPS"
echo "=================================================================="
if [ "$DRY" = 1 ]; then echo "(dry run: nothing submitted)"; exit 0; fi
if [ "$YES" != 1 ]; then
    read -r -p "submit? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "nothing submitted"; exit 0; }
fi

# ---------------------------------------------------------------- submit
ACC=(); [ -n "$CDX_ACCOUNT" ] && ACC=(--account="$CDX_ACCOUNT")
COMMON=("${ACC[@]}" --parsable --partition="$PART" --kill-on-invalid-dep=yes)
: > "$LAST"
sub() {  # name, sbatch args...  -> prints job id, records it
    local name="$1"; shift
    local id; id=$(sbatch "${COMMON[@]}" "$@")
    id="${id%%;*}"
    echo "$name $id" >> "$LAST"
    echo "  submitted $name: $id" >&2
    echo "$id"
}
dep() {  # joins non-empty dependency specs; prints nothing when there are none
    local d="" x
    for x in "$@"; do if [ -n "$x" ]; then d="${d:+$d,}$x"; fi; done
    if [ -n "$d" ]; then echo "--dependency=$d"; fi
}
WEXP=(--export=ALL,USE_MPS="$MPS",CDX_MAX_ATTEMPTS="$MAX_ATT",CDX_ROOT="$CDX_ROOT")

BOOT_ID=""; AFTER_BOOT=""
if [ "$BOOT" = 1 ]; then
    BOOT_ID=$(sub bootstrap experiments/snelius/bootstrap.sbatch)
    AFTER_BOOT="afterok:$BOOT_ID"
fi

E0_IDS=()
if [ "$E0_SLICES" -gt 0 ]; then
    E0_IDS+=("$(sub e0 $(dep "$AFTER_BOOT") --job-name=cdx_E0 --time="$T_E0" \
        --array=0-$(( E0_SLICES - 1 )) "${WEXP[@]}",CDX_TIME_LIMIT_S="$(secs "$T_E0")" \
        experiments/snelius/worker.sbatch --exps E0 E0prec E0tune --gate off)")
fi
for p in $E0PACK; do
    E0_IDS+=("$(sub "e0pack_p$p" $(dep "$AFTER_BOOT") --job-name="cdx_E0pack_p$p" \
        --time="$T_E0" "${WEXP[@]}",CDX_TIME_LIMIT_S="$(secs "$T_E0")" \
        experiments/snelius/worker.sbatch --exps E0pack --variants "p$p" --pack-seq "$p" \
        --gate off)")
done
FID_ID=""
if [ "$E0FID" = 1 ]; then
    FID_ID=$(sub e0fid $(dep "$AFTER_BOOT") --job-name=cdx_E0fid --time="$CDX_TIME_OSRL" \
        "${WEXP[@]}",CDX_TIME_LIMIT_S="$(secs "$CDX_TIME_OSRL")" \
        experiments/snelius/worker.sbatch --exps E0fid --gate off)
fi

AFTER_E0=""; for id in "${E0_IDS[@]}"; do AFTER_E0="${AFTER_E0:+$AFTER_E0,}afterany:$id"; done
HOLD=(); NEED_RELEASE=0
if [ "$GATE" = manual ] && [ ! -f "$CDX_RESULTS/decisions.yaml" ]; then HOLD=(--hold); NEED_RELEASE=1; fi
GATEMODE=auto; [ "$GATE" = manual ] && GATEMODE=require
MAIN_IDS=()
if [ "$MAIN_LEFT" -gt 0 ]; then
    prev=""
    for r in $(seq 1 "$ROUNDS"); do
        d=$(dep "$AFTER_BOOT" "$AFTER_E0" ${prev:+aftercorr:$prev})
        h=(); [ "$r" = 1 ] && h=("${HOLD[@]}")
        prev=$(sub "main_r$r" $d "${h[@]}" --job-name=cdx_main --time="$T_MAIN" \
            --array=0-$(( G - 1 )) "${WEXP[@]}",CDX_TIME_LIMIT_S="$(secs "$T_MAIN")" \
            experiments/snelius/worker.sbatch --exps $MAIN_EXPS --gate "$GATEMODE")
        MAIN_IDS+=("$prev")
    done
fi

PROF_ID=""
if [ "$PROF" = 1 ]; then
    first="${MAIN_IDS[0]:-}"
    PROF_ID=$(sub profile $(dep "$AFTER_BOOT" ${first:+afterany:$first} "$AFTER_E0") \
        --export=ALL,CDX_ROOT="$CDX_ROOT" experiments/snelius/profile.sbatch)
fi

AFTER_ALL=""
for id in "${MAIN_IDS[@]}" $PROF_ID $FID_ID "${E0_IDS[@]}"; do
    AFTER_ALL="${AFTER_ALL:+$AFTER_ALL,}afterany:$id"
done
sub finalize ${AFTER_ALL:+--dependency=$AFTER_ALL} \
    --export=ALL,CDX_ROOT="$CDX_ROOT",P3="$([ $P3 = 1 ] && echo 1)" \
    experiments/snelius/finalize.sbatch > /dev/null

echo
echo "all stages submitted (ids in $LAST). Follow with:"
echo "  experiments/snelius/run_all.sh --status"
echo "  tail -f logs/slurm/cdx_*.out"
if [ "$NEED_RELEASE" = 1 ] && [ "${#MAIN_IDS[@]}" -gt 0 ]; then
    echo
    echo "manual gate: when E0 is done, check and write the decisions, then release:"
    echo "  python -m experiments.snelius.decisions --out \$CDX_RESULTS --write"
    echo "  scontrol release ${MAIN_IDS[0]}"
fi
