#!/bin/bash
# Last stage of run_all.sh: status of every experiment, the E0 report, and the full
# analysis (tables + figures) into $CDX_PROJECT/summary. Cheap; needs no GPU work.
#SBATCH --job-name=cdx_finalize
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -uo pipefail
source "${CDX_ROOT:-$SLURM_SUBMIT_DIR}/experiments/snelius/env.sh"
cdx_activate
SUMMARY="${CDX_SUMMARY:-$CDX_PROJECT/summary}"
mkdir -p "$SUMMARY"
python -m experiments.snelius.pipeline status --out "$CDX_RESULTS" ${P3:+--p3} \
    ${SKIP_MODELS:+--exclude-models "$SKIP_MODELS"} | tee "$SUMMARY/status.txt"
python -m experiments.common.sanity --e0-report --out "$CDX_RESULTS" > "$SUMMARY/e0_report.txt" 2>&1
find "$CDX_RESULTS" -name FAILED -path '*/seed_*' | sort > "$SUMMARY/failed_runs.txt"
echo "failed runs: $(wc -l < "$SUMMARY/failed_runs.txt") (list in $SUMMARY/failed_runs.txt)"
make -f experiments/analysis/Makefile all RESULTS="$CDX_RESULTS" CACHE="$CDX_CACHE" SUMMARY="$SUMMARY"
echo "analysis written to $SUMMARY"
