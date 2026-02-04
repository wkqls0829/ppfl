#!/bin/bash

# Ablation Study: VPL + Ortho (Orthogonal loss, no GP prior)
# Binary selector training script for local server execution
# Based on vpl-gp/hhst_c100.sh structure

tid=62310
device=1  # Set GPU device number
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

# Determine checkpoint directory based on environment
# Local server: /hdd/hdd3/kjb exists → use /hdd/hdd3/kjb/checkpoints
# Cluster (SLURM): /hdd/hdd3/kjb doesn't exist → use WORK_DIR/checkpoints
if [ -d "/hdd/hdd3/kjb" ]; then
    # Local server environment
    CHECKPOINT_DIR="/hdd/hdd3/kjb/checkpoints"
    echo "Local server environment detected. Using checkpoint dir: $CHECKPOINT_DIR"
else
    # Cluster environment
    CHECKPOINT_DIR="$WORK_DIR/checkpoints"
    echo "Cluster environment detected. Using checkpoint dir: $CHECKPOINT_DIR"
fi
mkdir -p $CHECKPOINT_DIR

echo "Starting Ablation: VPL + Ortho (no GP prior) - Selector Training"
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Config: cfg/vpl-ortho/hhst.yaml"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_choice_gemma_ablation_vplortho_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo ""

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-ortho/hhst.yaml \
    device ${device} \
    data.root ${WORK_DIR}/data \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_choice_gemma_ablation_vplortho_t${tid}.ckpt \
    expname "vpl_ortho_hhst_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"
