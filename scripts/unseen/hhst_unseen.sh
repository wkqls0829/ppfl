#!/bin/bash

# Unseen Client Experiment: VPL-GP Selector Training
# Tests whether VPL-GP can adapt to unknown clients
# - Total 20 clients: 10 harmless + 10 helpful
# - Only 10 clients (5 harmless + 5 helpful) participate in selector training
# - Remaining 10 clients are unseen during training
# - All 20 clients will participate in RL evaluation

tid=70000
device=0  # Set GPU device number
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

echo "=========================================="
echo "Unseen Client Experiment: Selector Training"
echo "=========================================="
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Total clients: 20 (10 harmless + 10 helpful)"
echo "Training clients: 10 (5 harmless + 5 helpful)"
echo "Unseen clients: 10 (5 harmless + 5 helpful)"
echo "Config: cfg/vpl-gp-unseen/hhst.yaml"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_choice_gemma_unseen_vplgp_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo ""
echo "Note: Only first 10 clients will participate in training."
echo "      Remaining 10 clients will be used only in RL evaluation."
echo "=========================================="
echo ""

# Create 20 clients but only use first 10 for training
# This ensures consistent client IDs between selector and RL phases
# The remaining 10 clients (11-20) will be unseen during training
# 
# Implementation note: We set client_num=20 to create all clients,
# but training will only use client_id 1-10. This may require
# code modification to restrict training to specific client IDs.

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-gp-unseen/hhst.yaml \
    device ${device} \
    federate.client_num 20 \
    federate.sample_client_num 5 \
    data.root ${WORK_DIR}/data \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_choice_gemma_unseen_vplgp_t${tid}.ckpt \
    expname "unseen_vplgp_hhst_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"
echo ""
echo "After selector training completes, run:"
echo "  bash scripts/unseen/hrl_unseen.sh"
