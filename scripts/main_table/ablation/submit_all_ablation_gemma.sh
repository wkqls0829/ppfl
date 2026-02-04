#!/bin/bash

# Submit all Gemma-2B ablation study experiments to SLURM
# This script submits all selector jobs

WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR

# TID ranges
# Selector: 62300-62332
# RL: 63300-63332

# Experiment matrix
declare -a METHODS=("vplgp" "vplortho")
declare -a CLIENT_COUNTS=("10" "50" "100")

# Base TIDs
SELECTOR_BASE=62300
RL_BASE=63300

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
echo "Submitting Gemma-2B Ablation Study Experiments"
echo "=========================================="

# Submit selector jobs
echo ""
echo "Submitting Ablation Selector Jobs..."
for method in "${METHODS[@]}"; do
    for client_count in "${CLIENT_COUNTS[@]}"; do
        method_offset=${METHOD_OFFSETS[$method]}
        client_offset=${CLIENT_OFFSETS[$client_count]}
        tid=$((SELECTOR_BASE + method_offset + client_offset))
        
        echo "  Submitting: $method, N=$client_count, TID=$tid"
        sbatch --job-name="ablation_sel_${method}_n${client_count}_${tid}" \
               scripts/main_table/ablation/run_selector_ablation_gemma.sh \
               $method $client_count $tid
        sleep 1
    done
done

echo ""
echo "Ablation selector jobs submitted. Waiting for completion before submitting RL jobs..."
echo "Please run submit_rl_ablation_gemma.sh after selector jobs complete."
