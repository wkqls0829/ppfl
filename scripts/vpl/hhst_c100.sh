#!/bin/bash

# Variational Preference Learning (VPL) training script with 100 clients
# Binary selector training with VPL

tid=10100
# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl/hhst.yaml \
    device 3 \
    federate.client_num 100 \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vpl_c100_${tid}.ckpt \
    expname "vpl_hhst_c100_t${tid}" \
    > outputs/${tid}_c100.log 2>&1 &

echo "VPL HHST training started (task ID: ${tid}, 100 clients, GPU 3)"
echo "Log file: outputs/${tid}_c100.log"
echo "Monitor with: tail -f outputs/${tid}_c100.log"
