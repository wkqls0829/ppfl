#!/bin/bash

# Submit all Qwen 2 RL experiments to SLURM
# Run this after selector jobs complete

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
echo "Submitting Qwen 2 RL Experiments"
echo "=========================================="

# Submit RL jobs
echo ""
echo "Submitting RL Jobs..."
for method in "${METHODS[@]}"; do
    for client_count in "${CLIENT_COUNTS[@]}"; do
        method_offset=${METHOD_OFFSETS[$method]}
        client_offset=${CLIENT_OFFSETS[$client_count]}
        selector_tid=$((SELECTOR_BASE + method_offset + client_offset))
        rl_tid=$((RL_BASE + method_offset + client_offset))
        
        # Check if selector checkpoint exists
        CHECKPOINT_DIR="$WORK_DIR/checkpoints"
        SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_qwen2_fedbiscuit_u3_${method}_t${selector_tid}.ckpt"
        if [ ! -f "$SELECTOR_CKPT" ]; then
            SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_qwen2_fedbiscuit_u3_${method}_t${selector_tid}.ckpt"
        fi
        
        if [ ! -f "$SELECTOR_CKPT" ]; then
            echo "  WARNING: Selector checkpoint not found for $method, N=$client_count (TID=$selector_tid)"
            echo "  Skipping RL job..."
            continue
        fi
        
        echo "  Submitting: $method, N=$client_count, RL_TID=$rl_tid, Selector_TID=$selector_tid"
        sbatch --job-name="rl_${method}_n${client_count}_${rl_tid}" \
               --dependency=afterok:$(squeue -u $USER -n "sel_${method}_n${client_count}_${selector_tid}" -h -o %i 2>/dev/null || echo "") \
               scripts/main_table/run_rl_qwen.sh \
               $method $client_count $rl_tid $selector_tid
        sleep 1
    done
done

echo ""
echo "RL jobs submitted."
