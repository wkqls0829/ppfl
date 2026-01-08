#!/bin/bash

# Test script for VPL (Variational Preference Learning) training with reduced rounds
# This is a quick test version of hhst_vpl.sh

tid=10101  # Different test ID for VPL
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run with VPL test configuration (5 rounds instead of 150)
nohup python -u federatedscope/main.py \
    --cfg cfg/gemma_hhrl_vpl_test.yaml \
    federate.save_to checkpoints/test_hhrl_choice_gemma_fedbiscuit_u3_vpl_${tid}.ckpt \
    > outputs/${tid}.log 2>&1 &
