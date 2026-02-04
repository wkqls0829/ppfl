#!/bin/bash

# Ablation Study: VPL + GP (Gumbel-Softmax Prior, no orthogonal loss) - Qwen 2
# Binary selector training script for local server execution
# Client count: 100

tid=62402
device=1  # GPU 1
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

echo "Starting Ablation: VPL + GP (no orthogonal loss) - Qwen 2 - Selector Training"
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Client count: 100"
echo "Config: cfg/vpl-gp-no-ortho/hhst.yaml"
echo "Model: Qwen 2"
echo "Checkpoint: ${CHECKPOINT_DIR}/hhrl_choice_qwen2_ablation_vplgp_n100_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo ""

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-gp-no-ortho/hhst.yaml \
    device ${device} \
    federate.client_num 100 \
    model.type "Qwen/Qwen2-0.5B@huggingface_llm" \
    data.root ${WORK_DIR}/data \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_choice_qwen2_ablation_vplgp_n100_t${tid}.ckpt \
    expname "vplgp_no_ortho_qwen2_hhst_n100_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"
