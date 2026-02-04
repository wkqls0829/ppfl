#!/bin/bash

# Ablation Study: VPL-GP-Ortho (Gumbel-Softmax Prior + Orthogonal Loss) - RL Training
# RLHF training script for local server execution

tid=63330
device=5  # GPU 5 (same as selector)
SELECTOR_TID=59000  # Selector task ID from hhst_vplgp_ortho_n10.sh
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

# Determine data root and checkpoint directory based on environment
# Local server: /hdd/hdd3/kjb exists → use /hdd/hdd3/kjb
# Cluster (SLURM): /hdd/hdd3/kjb doesn't exist → use WORK_DIR
if [ -d "/hdd/hdd3/kjb" ]; then
    # Local server environment (ablation scripts run here)
    DATA_ROOT="/hdd/hdd3/kjb"
    CHECKPOINT_DIR="/hdd/hdd3/kjb/checkpoints"
    echo "Local server environment detected. Using data root: $DATA_ROOT"
else
    # Cluster environment (SLURM job, /hdd/hdd3 doesn't exist)
    DATA_ROOT="$WORK_DIR/data"
    CHECKPOINT_DIR="$WORK_DIR/checkpoints"
    echo "Cluster environment detected. Using data root: $DATA_ROOT"
fi
mkdir -p "$DATA_ROOT" "$CHECKPOINT_DIR"

# Selector checkpoint (try final_ prefix first)
SELECTOR_CKPT="${CHECKPOINT_DIR}/final_hhrl_choice_gemma_vplgp_ortho_n10_t${SELECTOR_TID}.ckpt"
if [ ! -f "${SELECTOR_CKPT}" ]; then
    SELECTOR_CKPT="${CHECKPOINT_DIR}/hhrl_choice_gemma_vplgp_ortho_n10_t${SELECTOR_TID}.ckpt"
    if [ ! -f "${SELECTOR_CKPT}" ]; then
        # Try 40_ checkpoint
        SELECTOR_CKPT="${CHECKPOINT_DIR}/40_hhrl_choice_gemma_vplgp_ortho_n10_t${SELECTOR_TID}.ckpt"
    fi
fi

echo "Starting Ablation: VPL-GP-Ortho (GP Prior + Orthogonal Loss) - RL Training"
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Selector checkpoint: ${SELECTOR_CKPT} (from hhst_vplgp_ortho_n10.sh, tid=${SELECTOR_TID})"
echo "RLHF config: cfg/vpl-gp/hrl.yaml (with orthogonal loss enabled)"
echo "RLHF checkpoint: ${CHECKPOINT_DIR}/hhrl_rlhf_gemma_vplgp_ortho_n10_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo ""

# Check if selector checkpoint exists
if [ ! -f "${SELECTOR_CKPT}" ]; then
    echo "ERROR: Selector checkpoint not found: ${SELECTOR_CKPT}"
    echo "Please run scripts/ablation/hhst_vplgp_ortho_n10.sh first to generate the selector checkpoint."
    exit 1
fi
echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
echo ""

# Create temporary selector config with checkpoint path
temp_selector_config="/tmp/selector_config_${tid}_$$.yaml"
cp cfg/vpl-gp/hhst.yaml "${temp_selector_config}"
sed -i "s|save_to:.*|save_to: \"${SELECTOR_CKPT}\"|" "${temp_selector_config}"

nohup python -u federatedscope/llm/rlhf/main.py \
    --selector-cfg-file "${temp_selector_config}" \
    --cfg cfg/vpl-gp/hrl.yaml \
    device ${device} \
    llm.vpl_kl_weight 0.02 \
    llm.vpl_orthogonal_weight 1.0 \
    llm.vpl_orthogonal_orthonorm_weight 0.0 \
    llm.vpl_use_manual_orthogonal_labels True \
    llm.vpl_num_prototypes 2 \
    llm.vpl_prototype_scale 5.0 \
    data.root ${DATA_ROOT} \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_rlhf_gemma_vplgp_ortho_n10_t${tid}.ckpt \
    expname "vplgp_ortho_hrl_n10_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"

# Clean up temp config file after job starts
(sleep 30 && rm -f "${temp_selector_config}") &
