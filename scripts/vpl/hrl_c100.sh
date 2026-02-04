#!/bin/bash

# RL training script for VPL (Variational Preference Learning) with 100 clients
# GPU 3 reserved for this experiment

tid=20100
# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set PYTHONPATH to use the current directory's federatedscope
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Selector checkpoint from vpl/hhst_c100.sh (tid=10100)
SELECTOR_TID=10100
SELECTOR_CKPT="/hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vpl_c100_${SELECTOR_TID}.ckpt"

echo "Starting HRL VPL (task ID: ${tid}, 100 clients, GPU 3)"
echo "Selector checkpoint: ${SELECTOR_CKPT} (from vpl/hhst_c100.sh, tid=${SELECTOR_TID})"
echo "RLHF config: cfg/vpl/hrl.yaml"
echo "RLHF checkpoint: /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_vpl_choice_gemma_fedbiscuit_u3_c100_${tid}.ckpt"
echo "Log file: outputs/${tid}_c100.log"
echo ""

# Check if selector checkpoint exists
if [ ! -f "${SELECTOR_CKPT}" ]; then
    echo "ERROR: Selector checkpoint not found: ${SELECTOR_CKPT}"
    echo "Please run vpl/hhst_c100.sh first to generate the selector checkpoint."
    exit 1
fi
echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
echo ""

# Create temporary selector config with checkpoint path
temp_selector_config="/tmp/selector_config_${tid}_$$.yaml"
cp cfg/vpl/hhst.yaml "${temp_selector_config}"
sed -i "s|save_to:.*|save_to: \"${SELECTOR_CKPT}\"|" "${temp_selector_config}"

nohup python -u federatedscope/llm/rlhf/main.py \
    --selector-cfg-file "${temp_selector_config}" \
    --cfg cfg/vpl/hrl.yaml \
    device 3 \
    federate.client_num 100 \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_vpl_choice_gemma_fedbiscuit_u3_c100_${tid}.ckpt \
    expname "vpl_hrl_c100_t${tid}" \
    > outputs/${tid}_c100.log 2>&1 &

PID=$!
echo "Process started in background. PID: ${PID}"
echo "Monitor with: tail -f outputs/${tid}_c100.log"

# Clean up temp config file after job starts
(sleep 30 && rm -f "${temp_selector_config}") &
