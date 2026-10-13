#!/usr/bin/env bash
# Shared settings for all Snelius scripts. Source it, do not execute it:
#     source experiments/snelius/env.sh
# Only CDX_PROJECT must be set by you (project space, never $HOME).

# ---- paths -------------------------------------------------------------------
CDX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the Experiments/ dir
export CDX_ROOT
if [ -z "${CDX_PROJECT:-}" ]; then
    echo "ERROR: export CDX_PROJECT=/projects/0/<your-project>/cdx (project space) first" >&2
    return 1 2>/dev/null || exit 1
fi
export CDX_VENV="${CDX_VENV:-$HOME/venvs/cdx}"
export CDX_RESULTS="${CDX_RESULTS:-$CDX_PROJECT/results}"
export CDX_CACHE="${CDX_CACHE:-$CDX_PROJECT/cache}"
# Verified in dsrl/offline_env.py: DSRL reads DSRL_DATASET_DIR (default ~/.dsrl/datasets).
export DSRL_DATASET_DIR="${DSRL_DATASET_DIR:-$CDX_PROJECT/dsrl_datasets}"
export OSRL_ROOT="${OSRL_ROOT:-$CDX_ROOT/OSRL}"

# ---- modules (VERIFY on Snelius: `module avail Python CUDA`) -------------------
export CDX_MODULES="${CDX_MODULES:-2023 Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1}"

# ---- SLURM defaults (GPU partitions only; spec §5.3, §8) ----------------------
export CDX_PARTITION="${CDX_PARTITION:-gpu_a100}"
export CDX_ACCOUNT="${CDX_ACCOUNT:-}"           # optional: --account
export CDX_MAX_PARALLEL="${CDX_MAX_PARALLEL:-8}"  # G in --array=0-N%G (spec §7: 8)
export CDX_PACK_SEQ="${CDX_PACK_SEQ:-6}"
export CDX_PACK_MLP="${CDX_PACK_MLP:-12}"
export CDX_PACK_EVAL="${CDX_PACK_EVAL:-4}"
export CDX_PACK_OSRL="${CDX_PACK_OSRL:-1}"
# --time per packed array job. Planning values from spec §7; REPLACE with the E0
# measurements (seconds/step x 100k x pack, plus evaluation) before E1.
export CDX_TIME_SEQ="${CDX_TIME_SEQ:-03:00:00}"
export CDX_TIME_MLP="${CDX_TIME_MLP:-02:00:00}"
export CDX_TIME_EVAL="${CDX_TIME_EVAL:-01:30:00}"
export CDX_TIME_OSRL="${CDX_TIME_OSRL:-10:00:00}"

# ---- runtime environment inside jobs (spec §8.3) -------------------------------
export WANDB_MODE=offline
export TQDM_DISABLE=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cdx_activate() {
    # Lmod and venv activate scripts can reference unset variables: run them without
    # `set -u` (the job scripts use it) and restore the caller's setting afterwards
    local had_u=0
    case $- in *u*) had_u=1; set +u;; esac
    # shellcheck disable=SC2086
    module purge >/dev/null 2>&1 || true
    for m in $CDX_MODULES; do module load "$m"; done
    # shellcheck disable=SC1091
    source "$CDX_VENV/bin/activate"
    [ "$had_u" = 1 ] && set -u
    cd "$CDX_ROOT" || exit 1
}
