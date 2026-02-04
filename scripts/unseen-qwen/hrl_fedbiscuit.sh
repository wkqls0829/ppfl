#!/bin/bash

# Unseen Client Experiment: FedBiscuit RL Training (Qwen 2)
# TID: 70004 (can be overridden)
# GPU: 6 (can be overridden)
# Selector TID: 70000

tid=${1:-70004}
selector_tid=${2:-70000}
device=${3:-6}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Set working directory
WORK_DIR="$PROJECT_ROOT"
cd $WORK_DIR

# Activate conda environment
source ~/.bashrc
conda activate biscuit || {
    echo "ERROR: Failed to activate conda environment 'biscuit'"
    echo "Please activate the conda environment manually: conda activate biscuit"
    exit 1
}

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
    DATA_ROOT="/hdd/hdd3/kjb"
else
    CHECKPOINT_DIR="$WORK_DIR/checkpoints"
    DATA_ROOT="$WORK_DIR/data"
fi
mkdir -p $CHECKPOINT_DIR

# Selector checkpoint (for unseen client evaluation)
SELECTOR_CKPT="${CHECKPOINT_DIR}/final_hhrl_choice_qwen2_fedbiscuit_unseen_t${selector_tid}.ckpt"
if [ ! -f "${SELECTOR_CKPT}" ]; then
    SELECTOR_CKPT="${CHECKPOINT_DIR}/hhrl_choice_qwen2_fedbiscuit_unseen_t${selector_tid}.ckpt"
fi

echo "=========================================="
echo "Unseen Client Experiment: FedBiscuit RL Training (Qwen 2)"
echo "=========================================="
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Selector TID: ${selector_tid}"
echo "Total clients: 20 (all participate in evaluation)"
echo "Selector checkpoint: ${SELECTOR_CKPT}"
echo "Config: cfg/fedbiscuit-unseen/hrl.yaml"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_fedbiscuit_unseen_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo "=========================================="
echo ""

# Check if selector checkpoint exists (for unseen client z computation)
if [ ! -f "${SELECTOR_CKPT}" ]; then
    echo "WARNING: Selector checkpoint not found: ${SELECTOR_CKPT}"
    echo "Unseen client z computation will be skipped."
    SELECTOR_CKPT=""
else
    echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
fi
echo ""

# Create temporary selector config if selector checkpoint exists
temp_selector_config=""
if [ -n "${SELECTOR_CKPT}" ]; then
    temp_selector_config="/tmp/selector_config_fedbiscuit_unseen_${tid}_$$.yaml"
    cp cfg/fedbiscuit-unseen/hhst.yaml "${temp_selector_config}"
    sed -i "s|save_to:.*|save_to: \"${SELECTOR_CKPT}\"|" "${temp_selector_config}"
fi

# Run RL training
if [ -n "${temp_selector_config}" ]; then
    nohup python -u federatedscope/llm/rlhf/main.py \
        --selector-cfg-file "${temp_selector_config}" \
        --cfg cfg/fedbiscuit-unseen/hrl.yaml \
        device ${device} \
        federate.client_num 20 \
        data.root ${DATA_ROOT} \
        federate.save_to ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_fedbiscuit_unseen_t${tid}.ckpt \
        train.optimizer.lr 0.00001 \
        expname "fedbiscuit_unseen_hrl_t${tid}" \
        > outputs/${tid}.log 2>&1 &
else
    nohup python -u federatedscope/llm/rlhf/main.py \
        --cfg cfg/fedbiscuit-unseen/hrl.yaml \
        device ${device} \
        federate.client_num 20 \
        data.root ${DATA_ROOT} \
        federate.save_to ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_fedbiscuit_unseen_t${tid}.ckpt \
        train.optimizer.lr 0.00001 \
        expname "fedbiscuit_unseen_hrl_t${tid}" \
        > outputs/${tid}.log 2>&1 &
fi

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"

# Clean up temp config file after job starts
if [ -n "${temp_selector_config}" ]; then
    (sleep 30 && rm -f "${temp_selector_config}") &
fi
