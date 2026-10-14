#!/usr/bin/env bash
# One-time environment setup (spec §8.1). Run it on a GPU node so nvcc and the GPU
# are available for building mamba_ssm:
#
#   export CDX_PROJECT=/projects/0/<your-project>/cdx
#   srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=01:30:00 \
#        bash experiments/snelius/setup_env.sh
#
# Then: python experiments/snelius/check_env.py  (writes env_report.txt)
set -euo pipefail
source "$(dirname "$0")/env.sh"

set +u  # Lmod and venv activate scripts may reference unset variables
module purge || true
for m in $CDX_MODULES; do module load "$m"; done
echo "loaded modules: ${LOADEDMODULES:-?}"
which python nvcc || true

# a venv is complete only when the marker exists; a half-built one is rebuilt from scratch
if [ ! -f "$CDX_VENV/.cdx_ready" ]; then
    python -m venv --clear "$CDX_VENV"
fi
# shellcheck disable=SC1091
source "$CDX_VENV/bin/activate"
set -u
cd "$CDX_ROOT"
pip install --upgrade pip wheel setuptools packaging ninja

# torch first (CUDA 12 wheels), then the pinned stack. OSRL pins torch~=1.13: we
# ignore OSRL's pins and install its real dependencies explicitly.
pip install -r experiments/requirements-snelius.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e OSRL --no-deps

# mamba_ssm with Mamba-3 (Triton kernels; the package also compiles its Mamba-1
# selective-scan extension, which ``import mamba_ssm`` requires). Built from source
# against the loaded CUDA. causal-conv1d is optional for Mamba-3 (no short conv).
export MAMBA_FORCE_BUILD=TRUE
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
pip install --no-build-isolation "mamba-ssm==${CDX_MAMBA_VERSION:-2.3.2.post1}" || {
    echo "WARNING: mamba_ssm build failed. Training can fall back to the pure-PyTorch"
    echo "reference path: --override model.backbone_kwargs.mamba_impl=ref (report its"
    echo "timings as reference-implementation timings), or build the Apptainer image"
    echo "from experiments/snelius/cdx.def."
}
pip install --no-build-isolation "causal-conv1d>=1.4.0" || echo "causal-conv1d optional: skipped"

mkdir -p "$CDX_RESULTS" "$CDX_CACHE" "$DSRL_DATASET_DIR" "$CDX_ROOT/logs/slurm" "$CDX_ROOT/logs/runs"
pip freeze > "$CDX_ROOT/experiments/pip-freeze-snelius.txt"
touch "$CDX_VENV/.cdx_ready"
echo "done. next: python experiments/snelius/check_env.py"
