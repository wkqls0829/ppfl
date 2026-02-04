#!/bin/bash

# VPL-GP HHST training script (orthogonal loss enabled)
# Client 100 version of 50024
# Single GPU mode: GPU 4 (specified in config file)

tid=50224
# GPU is specified in config file (device: 4)
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-gp/hhst-ortho-50224.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_${tid}.ckpt \
    expname "vplgp_hhst_ortho_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "VPL-GP HHST training started (task ID: ${tid})"
echo "Config: cfg/vpl-gp/hhst-ortho-50224.yaml (VPL-GP with orthogonal loss, client 100 version of 50024)"
echo "Hyperparameters: batch_size=8, lr=0.0001, grad_accum_step=4, local_update_steps=30"
echo "GPU: 4 (specified in config file: device: 4)"
echo "WandB project: fvpl-selector"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
