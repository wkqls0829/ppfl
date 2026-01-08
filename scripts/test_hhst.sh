#!/bin/bash

# Test script for hh-rlhf training with reduced rounds
# This is a quick test version of hhst.sh

tid=10001  # Different test ID
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run with test configuration (5 rounds instead of 150)
nohup python -u federatedscope/main.py \
    --cfg cfg/gemma_hhrl_test.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/test_hhrl_choice_gemma_fedbiscuit_u3_${tid}.ckpt \
    > outputs/${tid}.log 2>&1 &
