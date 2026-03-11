#!/bin/bash

# Submit all Qwen 2 ablation study RL experiments to SLURM
# This script submits all RL jobs (run after selector jobs complete)

WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR

# TID ranges
# Selector: 62400-62432
# RL: 63400-63432

# Experiment matrix
declare -a METHODS=("vplgp" "vplortho")
declare -a CLIENT_COUNTS=("10" "50" "100")

# Base TIDs
SELECTOR_BASE=62400
RL_BASE=63400

# Method offsets
declare -A METHOD_OFFSETS=(
    ["vplgp"]=0
    ["vplortho"]=10
)

# Client count offsets
declare -A CLIENT_OFFSETS=(
    ["10"]=0
    ["50"]=1
    ["100"]=2
)

echo "=========================================="
echo "Submitting Qwen 2 Ablation Study RL Experiments"
echo "=========================================="

# Submit RL jobs
echo ""
echo "Submitting Ablation RL Jobs..."
for method in "${METHODS[@]}"; do
    for client_count in "${CLIENT_COUNTS[@]}"; do
        method_offset=${METHOD_OFFSETS[$method]}
        client_offset=${CLIENT_OFFSETS[$client_count]}
        selector_tid=$((SELECTOR_BASE + method_offset + client_offset))
        rl_tid=$((RL_BASE + method_offset + client_offset))
        
        echo "  Submitting: $method, N=$client_count, RL_TID=$rl_tid, Selector_TID=$selector_tid"
        sbatch --job-name="ablation_rl_${method}_n${client_count}_${rl_tid}" \
               scripts/slurm/main_table/ablation/run_rl_ablation_qwen.sh \
               $method $client_count $rl_tid $selector_tid
        sleep 1
    done
done

echo ""
echo "Ablation RL jobs submitted."
