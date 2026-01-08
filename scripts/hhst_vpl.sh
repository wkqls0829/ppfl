#!/bin/bash

# Variational Preference Learning (VPL) training script
# Based on hhst.sh but uses VPL trainer

tid=10100
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg cfg/gemma_hhrl_vpl.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vpl_${tid}.ckpt \
    > outputs/${tid}.log 2>&1 &
