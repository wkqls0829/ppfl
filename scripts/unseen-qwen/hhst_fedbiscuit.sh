#!/bin/bash

# Unseen Client Experiment: FedBiscuit Selector Training (Qwen 2)
# TID: 70000
# GPU: 6

tid=70000
device=6
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
echo "Unseen Client Experiment: FedBiscuit Selector Training (Qwen 2)"
echo "=========================================="
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Total clients: 20 (10 harmless + 10 helpful)"
echo "Training clients: 10 (5 harmless + 5 helpful)"
echo "Unseen clients: 10 (5 harmless + 5 helpful)"
echo "Config: cfg/fedbiscuit-unseen/hhst.yaml"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_choice_qwen2_fedbiscuit_unseen_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo "=========================================="
echo ""

nohup python -u federatedscope/main.py \
    --cfg cfg/fedbiscuit-unseen/hhst.yaml \
    device ${device} \
    federate.client_num 20 \
    federate.sample_client_num 5 \
    data.root ${WORK_DIR}/data \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_choice_qwen2_fedbiscuit_unseen_t${tid}.ckpt \
    expname "fedbiscuit_unseen_hhst_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"
