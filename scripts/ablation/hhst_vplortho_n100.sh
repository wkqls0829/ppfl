#!/bin/bash

# Ablation Study: VPL + Ortho (Orthogonal loss, no GP prior)
# Binary selector training script for local server execution
# Client count: 100

tid=62311
device=3  # GPU 3
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

# Check if Hugging Face token is set (required for gated models like Gemma)
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "WARNING: HF_TOKEN or HUGGING_FACE_HUB_TOKEN not set."
    echo "Gemma-2B is a gated model and may require authentication."
fi

# Set checkpoint path (local repo)
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p $CHECKPOINT_DIR

echo "Starting Ablation: VPL + Ortho (no GP prior) - Selector Training"
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Client count: 100"
echo "Config: cfg/vpl-ortho/hhst.yaml"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_choice_gemma_ablation_vplortho_n100_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo ""

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-ortho/hhst.yaml \
    device ${device} \
    federate.client_num 100 \
    data.root ${WORK_DIR}/data \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_choice_gemma_ablation_vplortho_n100_t${tid}.ckpt \
    expname "vpl_ortho_hhst_n100_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"
