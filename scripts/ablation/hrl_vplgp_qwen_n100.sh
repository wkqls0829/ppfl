#!/bin/bash

# Ablation Study (Qwen 2): VPL + GP (no orthogonal loss) - RL Training
# RLHF training script for local server execution
# Selector: 62402 (VPL+GP, N=100)

tid=63402
device=1          # GPU 1
SELECTOR_TID=62402
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
if [ -d "/hdd/hdd3/kjb" ]; then
    DATA_ROOT="/hdd/hdd3/kjb"
    CHECKPOINT_DIR="/hdd/hdd3/kjb/checkpoints"
    echo "Local server environment detected. Using data root: $DATA_ROOT"
else
    DATA_ROOT="$WORK_DIR/data"
    CHECKPOINT_DIR="$WORK_DIR/checkpoints"
    echo "Cluster environment detected. Using data root: $DATA_ROOT"
fi
mkdir -p "$DATA_ROOT" "$CHECKPOINT_DIR"

# Selector checkpoint (try final_ prefix first, then regular, then 40_)
SELECTOR_CKPT="${CHECKPOINT_DIR}/final_hhrl_choice_qwen2_ablation_vplgp_n100_t${SELECTOR_TID}.ckpt"
if [ ! -f "${SELECTOR_CKPT}" ]; then
    SELECTOR_CKPT="${CHECKPOINT_DIR}/hhrl_choice_qwen2_ablation_vplgp_n100_t${SELECTOR_TID}.ckpt"
    if [ ! -f "${SELECTOR_CKPT}" ]; then
        SELECTOR_CKPT="${CHECKPOINT_DIR}/40_hhrl_choice_qwen2_ablation_vplgp_n100_t${SELECTOR_TID}.ckpt"
    fi
fi

echo "Starting Ablation (Qwen 2): VPL + GP (no orthogonal loss) - RL Training, N=100"
echo "Task ID: ${tid}"
echo "Device: ${device}"
echo "Selector checkpoint: ${SELECTOR_CKPT} (tid=${SELECTOR_TID})"
echo "RLHF config: cfg/vpl-gp-no-ortho/hrl.yaml"
echo "RLHF checkpoint: ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_ablation_vplgp_n100_t${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo ""

# Check if selector checkpoint exists
if [ ! -f "${SELECTOR_CKPT}" ]; then
    echo "ERROR: Selector checkpoint not found: ${SELECTOR_CKPT}"
    echo "Please ensure hhst_vplgp_qwen_n100.sh (tid=${SELECTOR_TID}) has completed successfully."
    exit 1
fi
echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
echo ""

# Create temporary selector config with checkpoint path (for Qwen 2)
temp_selector_config="/tmp/selector_config_vplgp_qwen_n100_${tid}_$$.yaml"
cp cfg/vpl-gp-no-ortho/hhst.yaml "${temp_selector_config}"
# Update for Qwen 2
sed -i "s|save_to:.*|save_to: \"${SELECTOR_CKPT}\"|" "${temp_selector_config}"
sed -i "s|type:.*gemma.*|type: 'Qwen/Qwen2-0.5B@huggingface_llm'|" "${temp_selector_config}"
sed -i "s|client_num:.*|client_num: 100|" "${temp_selector_config}"

nohup python -u federatedscope/llm/rlhf/main.py \
    --selector-cfg-file "${temp_selector_config}" \
    --cfg cfg/vpl-gp-no-ortho/hrl.yaml \
    device ${device} \
    data.root ${DATA_ROOT} \
    model.type "Qwen/Qwen2-0.5B@huggingface_llm" \
    train.optimizer.lr 0.00001 \
    llm.grad_accum_step 32 \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_rlhf_qwen2_ablation_vplgp_n100_t${tid}.ckpt \
    expname "vplgp_no_ortho_qwen2_hrl_n100_t${tid}" \
    > outputs/${tid}.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}.log"

# Clean up temp config file after job starts
(sleep 30 && rm -f "${temp_selector_config}") &

