#!/bin/bash

# VPL-GP HHST training script (orthogonal loss disabled)
# Based on 50024 config but WITHOUT orthogonal loss
# Single GPU mode: GPU 2 (specified in config file)

tid=40024
# GPU is specified in config file (device: 2)
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-gp/hhst-ortho-40024.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_${tid}.ckpt \
    expname "vplgp_hhst_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "VPL-GP HHST training started (task ID: ${tid})"
echo "Config: cfg/vpl-gp/hhst-ortho-40024.yaml (VPL-GP WITHOUT orthogonal loss)"
echo "Hyperparameters: batch_size=8, lr=0.0001, grad_accum_step=4, local_update_steps=30"
echo "GPU: 2 (specified in config file: device: 2)"
echo "WandB project: fvpl-selector"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
