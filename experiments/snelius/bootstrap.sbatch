#!/bin/bash
# Bootstrap stage of run_all.sh: environment (built here if missing), datasets, checks,
# tests, preprocessing cache. See bootstrap.py. Fails (exit 1) on the first fatal check,
# which cancels every dependent stage.
#SBATCH --job-name=cdx_bootstrap
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source "${CDX_ROOT:-$SLURM_SUBMIT_DIR}/experiments/snelius/env.sh"

if [ ! -x "$CDX_VENV/bin/python" ]; then
    echo "== venv missing: building it (setup_env.sh)"
    bash "$CDX_ROOT/experiments/snelius/setup_env.sh"
fi
cdx_activate
mkdir -p "$CDX_RESULTS" "$CDX_CACHE" "$DSRL_DATASET_DIR"

if ! python -c "import sys; from experiments.common.data import TASKS, dataset_path; \
sys.exit(any(not dataset_path(t).exists() for t in TASKS))"; then
    echo "== datasets missing: downloading from the compute node"
    CDX_ALLOW_DOWNLOAD=1 python -m experiments.common.data --download --tasks all
fi

export OMP_NUM_THREADS=1
python -m experiments.snelius.bootstrap --out "$CDX_RESULTS" --cache-dir "$CDX_CACHE" \
    ${PREPARE_EXPS:+--prepare-exps $PREPARE_EXPS}
