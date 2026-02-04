#!/bin/bash

# Unseen Client Experiment: FedDPO RL Training (Qwen 2)
# TID: 70001
# GPU: 7
# Note: FedDPO does not require selector checkpoint

tid=70001
device=0
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

echo "=========================================="
echo "Unseen Client Experiment: FedDPO RL Training (Qwen 2)"
echo "=========================================="
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Total clients: 20 (all participate in evaluation)"
echo "Config: cfg/feddpo-unseen/hrl.yaml"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_feddpo_unseen_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo "Note: FedDPO does not require selector checkpoint"
echo "=========================================="
echo ""

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/feddpo-unseen/hrl.yaml \
    device ${device} \
    federate.client_num 20 \
    data.root ${WORK_DIR}/data \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_feddpo_unseen_t${tid}.ckpt \
    expname "feddpo_unseen_hrl_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"
