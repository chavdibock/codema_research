#!/bin/bash
# Build the preprocessing cache (S5) for every (task, data-config, seed) the listed
# experiments need. Datasets must already be downloaded on the login node:
#     python -m experiments.common.data --download --tasks all
# Submit:  sbatch experiments/snelius/prepare_data.sbatch [EXP ...]   (default: P0-P2)
#SBATCH --job-name=cdx_prepare
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source "${CDX_ROOT:-$SLURM_SUBMIT_DIR}/experiments/snelius/env.sh"
cdx_activate

EXPS=("$@")
if [ ${#EXPS[@]} -eq 0 ]; then
    EXPS=(E0 E0prec E0tune E1 E2 E3 E4 E6full E7 E8 E9 E10)
fi
export OMP_NUM_THREADS=1
# compute nodes must not download (CDX_ALLOW_DOWNLOAD unset => error if a file is missing)
python -m experiments.common.data --prepare --exp "${EXPS[@]}" \
    --cache-dir "$CDX_CACHE" --workers "${SLURM_CPUS_PER_TASK:-8}"
