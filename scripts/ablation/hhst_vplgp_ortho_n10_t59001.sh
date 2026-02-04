#!/bin/bash

# VPL-GP-Ortho (Gumbel-Softmax Prior + Orthogonal Loss)
# Binary selector training script for local server execution
# Client count: 10
# Hyperparameters: Based on 52003 (kl_weight=0.1, orthogonal_weight=1.0, orthonorm_weight=0.02)

tid=59001
device=2  # GPU 2
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

echo "Starting VPL-GP-Ortho (GP Prior + Orthogonal Loss) - Selector Training"
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Client count: 10"
echo "Config: cfg/vpl-gp/hhst.yaml (with orthogonal loss enabled)"
echo "Hyperparameters: kl_weight=0.1, orthogonal_weight=1.0, orthonorm_weight=0.02 (based on 52003)"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_choice_gemma_vplgp_ortho_n10_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo ""

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-gp/hhst.yaml \
    device ${device} \
    federate.client_num 10 \
    llm.vpl_kl_weight 0.1 \
    llm.vpl_orthogonal_weight 1.0 \
    llm.vpl_orthogonal_orthonorm_weight 0.02 \
    llm.vpl_use_manual_orthogonal_labels True \
    llm.vpl_num_prototypes 2 \
    llm.vpl_prototype_scale 5.0 \
    data.root ${WORK_DIR}/data \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_choice_gemma_vplgp_ortho_n10_t${tid}.ckpt \
    expname "vplgp_ortho_hhst_n10_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"
