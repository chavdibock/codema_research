#!/bin/bash
# E6 profiling (spec §6, E6): dedicated exclusive full gpu_a100 node (4 GPUs + all host
# cores), so nothing else shares GPU, CPU cores or memory bandwidth. Uses the trained
# E1 checkpoints (seed 0). 3 repetitions, each in a separate process.
# Submit (after E1 is DONE):  sbatch experiments/snelius/profile.sbatch
# Restrict with env vars, e.g. WHAT="infer mem" or MODELS="cdt cdmamba".
#SBATCH --job-name=cdx_profile
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -uo pipefail
source "${CDX_ROOT:-$SLURM_SUBMIT_DIR}/experiments/snelius/env.sh"
cdx_activate

MODELS=${MODELS:-"cdt cdt_light cdmamba cdconv"}
TASKS=${TASKS:-"OfflineAntRun-v0 OfflineCarCircle-v0 OfflineCarRun-v0 OfflineDroneCircle-v0 OfflineDroneRun-v0"}
TRAIN_TASKS=${TRAIN_TASKS:-"OfflineCarCircle-v0"}
REPS=${REPS:-"0 1 2"}
export CUDA_VISIBLE_DEVICES=0   # one fixed GPU of the exclusive node
export OMP_NUM_THREADS=8
P="python -m experiments.common.profiling --e6 --ckpt-from E1 --out $CDX_RESULTS"

for rep in $REPS; do
    # GPU: size, M1 (fp32 + bf16, batch 2048/256), M3; M2/M4 on every task
    $P --device cuda --rep "$rep" --models $MODELS --tasks $TRAIN_TASKS --what size train
    $P --device cuda --rep "$rep" --models $MODELS --tasks $TASKS --what infer mem
    # host CPU of the GPU node: 1 and 8 threads, cores pinned (M2, M4)
    CUDA_VISIBLE_DEVICES="" taskset -c 0 \
        $P --device cpu --threads 1 --rep "$rep" --models $MODELS --tasks $TASKS --what infer mem
    CUDA_VISIBLE_DEVICES="" taskset -c 0-7 \
        $P --device cpu --threads 8 --rep "$rep" --models $MODELS --tasks $TASKS --what infer mem
    # E6.6 context scaling (fresh weights): GPU training + inference, CPU inference
    $P --device cuda --rep "$rep" --models $MODELS --what scaling --allow-fresh
    CUDA_VISIBLE_DEVICES="" taskset -c 0 \
        $P --device cpu --threads 1 --rep "$rep" --models $MODELS --what scaling \
        --allow-fresh --batches
done
echo "efficiency rows: $(wc -l < "$CDX_RESULTS/E6/efficiency.csv")"
