#!/bin/bash

# Unseen Client Experiment: RL Training and Evaluation
# Tests whether VPL-GP can adapt to unknown clients during RL evaluation
# - Total 20 clients: 10 harmless + 10 helpful
# - First 10 clients (5 harmless + 5 helpful) participated in selector training
# - All 20 clients participate in RL evaluation

tid=70001
device=0  # Set GPU device number
SELECTOR_TID=70000  # Selector task ID from hhst_unseen.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Set working directory
WORK_DIR="$PROJECT_ROOT"
cd $WORK_DIR

# Load environment variables from .env file if it exists
if [ -f "$WORK_DIR/.env" ]; then
    export $(cat $WORK_DIR/.env | grep -v '^#' | xargs)
    echo "Loaded environment variables from .env file"
fi

# Set Hugging Face cache directory
export HF_HOME="$WORK_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORK_DIR/.cache/huggingface/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Check if Hugging Face token is set
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "WARNING: HF_TOKEN or HUGGING_FACE_HUB_TOKEN not set."
fi

# Set checkpoint path (local repo)
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p $CHECKPOINT_DIR

# Selector checkpoint
SELECTOR_CKPT="${CHECKPOINT_DIR}/hhrl_choice_gemma_unseen_vplgp_t${SELECTOR_TID}.ckpt"

echo "=========================================="
echo "Unseen Client Experiment: RL Training & Evaluation"
echo "=========================================="
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Total clients: 20 (10 harmless + 10 helpful)"
echo "Training clients (selector): 10 (5 harmless + 5 helpful)"
echo "Unseen clients: 10 (5 harmless + 5 helpful)"
echo "RL evaluation: All 20 clients participate"
echo ""
echo "Selector checkpoint: ${SELECTOR_CKPT} (from hhst_unseen.sh, tid=${SELECTOR_TID})"
echo "RLHF config: cfg/vpl-gp-unseen/hrl.yaml"
echo "RLHF checkpoint: ${CHECKPOINT_DIR}/hhrl_rlhf_gemma_unseen_vplgp_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo ""
echo "Note: All 20 clients will participate in RL evaluation"
echo "      to test adaptation to unseen clients."
echo "=========================================="
echo ""

# Check if selector checkpoint exists
if [ ! -f "${SELECTOR_CKPT}" ]; then
    echo "ERROR: Selector checkpoint not found: ${SELECTOR_CKPT}"
    echo "Please run scripts/unseen/hhst_unseen.sh first to generate the selector checkpoint."
    exit 1
fi
echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
echo ""

# Create temporary selector config with checkpoint path
temp_selector_config="/tmp/selector_config_unseen_${tid}_$$.yaml"
cp cfg/vpl-gp-unseen/hhst.yaml "${temp_selector_config}"
sed -i "s|save_to:.*|save_to: \"${SELECTOR_CKPT}\"|" "${temp_selector_config}"

# For RL, we need to create 20 clients but only use 1 for training (standalone RLHF)
# However, evaluation should use all 20 clients
# We'll set client_num=20 for evaluation, but training will use client_num=1 (standalone)
# Note: This may require code modification to support different client sets for train vs eval

nohup python -u federatedscope/llm/rlhf/main.py \
    --selector-cfg-file "${temp_selector_config}" \
    --cfg cfg/vpl-gp-unseen/hrl.yaml \
    device ${device} \
    federate.client_num 20 \
    data.root ${WORK_DIR}/data \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_rlhf_gemma_unseen_vplgp_t${tid}.ckpt \
    expname "unseen_vplgp_hrl_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"

# Clean up temp config file after job starts
(sleep 30 && rm -f "${temp_selector_config}") &
