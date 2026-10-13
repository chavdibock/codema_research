#!/bin/bash
# One work-queue worker on one GPU slice (see worker.py). Submitted by run_all.sh as an
# array (one task per slice); arguments are passed to worker.py, e.g.
#     sbatch --array=0-7 --time=12:00:00 --export=ALL,CDX_TIME_LIMIT_S=43200 \
#            experiments/snelius/worker.sbatch --exps E1 E2 --gate auto
#SBATCH --job-name=cdx_worker
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
set -uo pipefail
export CDX_JOB_T0="$(date +%s)"   # the worker's deadline counts from the job start
source "${CDX_ROOT:-$SLURM_SUBMIT_DIR}/experiments/snelius/env.sh"
cdx_activate
echo "worker ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} on $(hostname): $(nvidia-smi -L | head -1)"

if [ "${USE_MPS:-0}" = "1" ]; then
    export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_pipe_${SLURM_JOB_ID}"
    export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/mps_log_${SLURM_JOB_ID}"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d && echo "MPS started"
fi

python -m experiments.snelius.worker --out "$CDX_RESULTS" "$@"
rc=$?

if [ "${USE_MPS:-0}" = "1" ]; then
    echo quit | nvidia-cuda-mps-control || true
fi
exit $rc
