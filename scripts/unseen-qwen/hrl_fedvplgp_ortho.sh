#!/bin/bash

# Unseen Client Experiment: FedVPL-GP-Ortho RL Training (Qwen 2)
# TID: 70013 (can be overridden)
# GPU: 7 (can be overridden)
# Selector TID: 70003

tid=${1:-70013}
selector_tid=${2:-70003}
device=${3:-7}
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

# Activate conda environment
source ~/.bashrc
conda activate biscuit || {
    echo "ERROR: Failed to activate conda environment 'biscuit'"
    echo "Please activate the conda environment manually: conda activate biscuit"
    exit 1
}

# Determine checkpoint directory based on environment
if [ -d "/hdd/hdd3/kjb" ]; then
    CHECKPOINT_DIR="/hdd/hdd3/kjb/checkpoints"
    DATA_ROOT="/hdd/hdd3/kjb"
else
    CHECKPOINT_DIR="$WORK_DIR/checkpoints"
    DATA_ROOT="$WORK_DIR/data"
fi
mkdir -p $CHECKPOINT_DIR

# Selector checkpoint
SELECTOR_CKPT="${CHECKPOINT_DIR}/final_hhrl_choice_qwen2_fedvplgp_ortho_unseen_t${selector_tid}.ckpt"
if [ ! -f "${SELECTOR_CKPT}" ]; then
    SELECTOR_CKPT="${CHECKPOINT_DIR}/hhrl_choice_qwen2_fedvplgp_ortho_unseen_t${selector_tid}.ckpt"
fi

echo "=========================================="
echo "Unseen Client Experiment: FedVPL-GP-Ortho RL Training (Qwen 2)"
echo "=========================================="
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Selector TID: ${selector_tid}"
echo "Total clients: 20 (all participate in evaluation)"
echo "Selector checkpoint: ${SELECTOR_CKPT}"
echo "Config: cfg/fedvpl-gp-ortho-unseen/hrl.yaml"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_fedvplgp_ortho_unseen_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo "=========================================="
echo ""

# Check if selector checkpoint exists
if [ ! -f "${SELECTOR_CKPT}" ]; then
    echo "ERROR: Selector checkpoint not found: ${SELECTOR_CKPT}"
    echo "Please run scripts/unseen-qwen/hhst_fedvplgp_ortho.sh first."
    exit 1
fi
echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
echo ""

# Create temporary selector config
temp_selector_config="/tmp/selector_config_fedvplgp_ortho_unseen_${tid}_$$.yaml"
cp cfg/fedvpl-gp-ortho-unseen/hhst.yaml "${temp_selector_config}"
sed -i "s|save_to:.*|save_to: \"${SELECTOR_CKPT}\"|" "${temp_selector_config}"

nohup python -u federatedscope/llm/rlhf/main.py \
    --selector-cfg-file "${temp_selector_config}" \
    --cfg cfg/fedvpl-gp-ortho-unseen/hrl.yaml \
    device ${device} \
    federate.client_num 20 \
    data.root ${DATA_ROOT} \
    train.optimizer.lr 0.00001 \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_fedvplgp_ortho_unseen_t${tid}.ckpt \
    expname "fedvplgp_ortho_unseen_hrl_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"

# Clean up temp config file after job starts
(sleep 30 && rm -f "${temp_selector_config}") &
