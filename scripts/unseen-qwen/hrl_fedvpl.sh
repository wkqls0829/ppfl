#!/bin/bash

# Unseen Client Experiment: FedVPL RL Training (Qwen 2)
# TID: 70005
# GPU: 6
# Selector TID: 70002

tid=${1:-70005}
selector_tid=${2:-70002}
device=${3:-6}
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

# Determine checkpoint directory based on environment
if [ -d "/hdd/hdd3/kjb" ]; then
    CHECKPOINT_DIR="/hdd/hdd3/kjb/checkpoints"
else
    CHECKPOINT_DIR="$WORK_DIR/checkpoints"
fi
mkdir -p $CHECKPOINT_DIR

# Selector checkpoint
SELECTOR_CKPT="${CHECKPOINT_DIR}/final_hhrl_choice_qwen2_fedvpl_unseen_t${selector_tid}.ckpt"
if [ ! -f "${SELECTOR_CKPT}" ]; then
    SELECTOR_CKPT="${CHECKPOINT_DIR}/hhrl_choice_qwen2_fedvpl_unseen_t${selector_tid}.ckpt"
fi

echo "=========================================="
echo "Unseen Client Experiment: FedVPL RL Training (Qwen 2)"
echo "=========================================="
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Selector TID: ${selector_tid}"
echo "Total clients: 20 (all participate in evaluation)"
echo "Selector checkpoint: ${SELECTOR_CKPT}"
echo "Config: cfg/fedvpl-unseen/hrl.yaml"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_fedvpl_unseen_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo "=========================================="
echo ""

# Check if selector checkpoint exists
if [ ! -f "${SELECTOR_CKPT}" ]; then
    echo "ERROR: Selector checkpoint not found: ${SELECTOR_CKPT}"
    echo "Please run scripts/unseen-qwen/hhst_fedvpl.sh first."
    exit 1
fi
echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
echo ""

# Create temporary selector config
temp_selector_config="/tmp/selector_config_fedvpl_unseen_${tid}_$$.yaml"
cp cfg/fedvpl-unseen/hhst.yaml "${temp_selector_config}"
sed -i "s|save_to:.*|save_to: \"${SELECTOR_CKPT}\"|" "${temp_selector_config}"

nohup python -u federatedscope/llm/rlhf/main.py \
    --selector-cfg-file "${temp_selector_config}" \
    --cfg cfg/fedvpl-unseen/hrl.yaml \
    device ${device} \
    federate.client_num 20 \
    data.root ${WORK_DIR}/data \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_fedvpl_unseen_t${tid}.ckpt \
    expname "fedvpl_unseen_hrl_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"

# Clean up temp config file after job starts
(sleep 30 && rm -f "${temp_selector_config}") &
