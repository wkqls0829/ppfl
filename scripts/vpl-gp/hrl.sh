#!/bin/bash

# RL training script for VPL-GP (Variational Preference Learning with Gumbel Softmax Prior)
# This script runs RLHF training using a VPL-GP selector checkpoint

tid=50200
# GPU 5 reserved for VPL-GP algorithm (set in config file: device: 5)
# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set PYTHONPATH to use the current directory's federatedscope
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Selector checkpoint from vpl-gp/hhst.sh (tid=50100)
SELECTOR_TID=50100
SELECTOR_CKPT="/hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vplgp_${SELECTOR_TID}.ckpt"

echo "Starting HRL VPL-GP test (task ID: ${tid})"
echo "Selector checkpoint: ${SELECTOR_CKPT} (from vpl-gp/hhst.sh, tid=${SELECTOR_TID})"
echo "Selector config: cfg/vpl-gp/test_hrl_selector.yaml (points to hhst.sh checkpoint)"
echo "RLHF config: cfg/vpl-gp/hrl.yaml"
echo "RLHF checkpoint: /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_vplgp_${tid}.ckpt"
echo "Log file: outputs/${tid}.log"
echo ""

# Check if selector checkpoint exists
if [ ! -f "${SELECTOR_CKPT}" ]; then
    echo "ERROR: Selector checkpoint not found: ${SELECTOR_CKPT}"
    echo "Please run vpl-gp/hhst.sh first to generate the selector checkpoint."
    exit 1
fi
echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
echo ""

nohup python -u federatedscope/llm/rlhf/main.py \
    --selector-cfg-file cfg/vpl-gp/test_hrl_selector.yaml \
    --cfg cfg/vpl-gp/hrl.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_vplgp_${tid}.ckpt \
    expname "vplgp_test_hrl_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "Process started in background. PID: $!"
echo "Monitor with: tail -f outputs/${tid}.log"
