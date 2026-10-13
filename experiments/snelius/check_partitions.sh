#!/usr/bin/env bash
# Print budget and the GPU partitions this account can use (run on the login node
# before the first submission). Read-only: submits nothing.
set -uo pipefail

echo "== accinfo (budget) =="
accinfo 2>/dev/null || echo "accinfo not available"

echo; echo "== partitions =="
sinfo -p gpu_a100,gpu_h100 -o "%12P %6a %11l %6D %10T %20G %N" 2>/dev/null || true

echo; echo "== associations for $USER =="
sacctmgr -n show assoc user="$USER" format=account%20,partition%15,qos%30 2>/dev/null || true

echo; echo "== partition details =="
for p in gpu_a100 gpu_h100; do
    scontrol show partition "$p" 2>/dev/null | grep -E "PartitionName|MaxTime|TRES|DefMemPerGPU|State" || echo "$p: not visible"
done

echo; echo "Check the SBU rate per GPU hour in the Snelius documentation and set"
echo "CDX_SBU_PER_GPU_HOUR so submit.sh can print SBU estimates."
