#!/bin/bash

# Submit all Qwen 2 main table experiments to SLURM
# This script submits all selector and RL jobs

WORK_DIR="/home/jbkoo/ppfl"
cd $WORK_DIR

# TID ranges
# Selector: 62200-62232
# RL: 63200-63232

# Experiment matrix
declare -a METHODS=("feddpo" "fedbiscuit" "fedvpl" "fedvpagp")
declare -a CLIENT_COUNTS=("10" "50" "100")

# Base TIDs
SELECTOR_BASE=62200
RL_BASE=63200

# Method offsets
declare -A METHOD_OFFSETS=(
    ["feddpo"]=0
    ["fedbiscuit"]=10
    ["fedvpl"]=20
    ["fedvpagp"]=30
)

# Client count offsets
declare -A CLIENT_OFFSETS=(
    ["10"]=0
    ["50"]=1
    ["100"]=2
)

echo "=========================================="
echo "Submitting Qwen 2 Main Table Experiments"
echo "=========================================="

# Submit selector jobs
echo ""
echo "Submitting Selector Jobs..."
for method in "${METHODS[@]}"; do
    for client_count in "${CLIENT_COUNTS[@]}"; do
        method_offset=${METHOD_OFFSETS[$method]}
        client_offset=${CLIENT_OFFSETS[$client_count]}
        tid=$((SELECTOR_BASE + method_offset + client_offset))
        
        echo "  Submitting: $method, N=$client_count, TID=$tid"
        sbatch --job-name="sel_${method}_n${client_count}_${tid}" \
               scripts/main_table/run_selector_qwen.sh \
               $method $client_count $tid
        sleep 1
    done
done

echo ""
echo "Selector jobs submitted. Waiting for completion before submitting RL jobs..."
echo "Please run submit_rl_qwen.sh after selector jobs complete."
