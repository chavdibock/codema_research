#!/bin/bash
# Packed SLURM array over a manifest (spec §8.3, S7). Array task i runs manifest lines
# [i*PACK, (i+1)*PACK) concurrently on one A100 slice and waits for all of them.
# Do not sbatch this directly; use experiments/snelius/submit.sh, which sets
#     --export=ALL,MANIFEST=<file>,PACK=<P>,USE_MPS=<0|1>  --array=0-N%G  --time=...
#SBATCH --job-name=cdx_array
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
set -uo pipefail
source "${CDX_ROOT:-$SLURM_SUBMIT_DIR}/experiments/snelius/env.sh"
cdx_activate

: "${MANIFEST:?MANIFEST not set (use submit.sh)}"
PACK="${PACK:-6}"
USE_MPS="${USE_MPS:-0}"
i="${SLURM_ARRAY_TASK_ID:-0}"
first=$(( i * PACK + 1 ))
last=$(( (i + 1) * PACK ))
mapfile -t LINES < <(sed -n "${first},${last}p" "$MANIFEST")
n=${#LINES[@]}
if [ "$n" -eq 0 ]; then
    echo "array task $i: no lines in $MANIFEST"; exit 0
fi
echo "array task $i: $n runs from $MANIFEST on $(hostname); GPU: $(nvidia-smi -L | head -1)"

if [ "$USE_MPS" = "1" ]; then
    export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_pipe_${SLURM_JOB_ID}"
    export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/mps_log_${SLURM_JOB_ID}"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d && echo "MPS started"
fi

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CDX_PACK="$PACK"
cpus="${SLURM_CPUS_PER_TASK:-18}"
export CDX_EVAL_WORKERS=$(( cpus / n > 2 ? cpus / n : 2 ))
mkdir -p logs/runs

pids=()
for k in "${!LINES[@]}"; do
    line="${LINES[$k]}"
    seed=$(sed -n 's/.*--seed \([0-9][0-9]*\).*/\1/p' <<< "$line")
    log="logs/runs/${SLURM_JOB_NAME:-cdx}_${SLURM_ARRAY_JOB_ID:-0}_${i}_${k}.log"
    echo "[$k] $line  > $log"
    ( export PYTHONHASHSEED="${seed:-0}"; eval "$line" ) > "$log" 2>&1 &
    pids+=($!)
done

fail=0
for k in "${!pids[@]}"; do
    if ! wait "${pids[$k]}"; then
        echo "run $k FAILED: ${LINES[$k]}"
        fail=$(( fail + 1 ))
    fi
done

if [ "$USE_MPS" = "1" ]; then
    echo quit | nvidia-cuda-mps-control || true
fi
echo "array task $i finished: $(( n - fail ))/$n ok"
[ "$fail" -eq 0 ]
