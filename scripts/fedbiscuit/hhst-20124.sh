#!/bin/bash

# FedBiscuit baseline training script with 50 clients
# Based on hhst-20023.sh but scaled to 50 clients
# For comparison with VPL-GP experiment (50024 scaled to 50 clients)
# Single GPU mode: GPU 0 (specified in config file)

tid=20124
# GPU is specified in config file (device: 0)
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg cfg/fedbiscuit/hhst-20124.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_${tid}.ckpt \
    expname "fedbiscuit_hhst_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "FedBiscuit HHST baseline training started (task ID: ${tid}, 50 clients)"
echo "Config: cfg/fedbiscuit/hhst-20124.yaml (baseline: Multi-LoRA, 50 clients, no VPL)"
echo "Hyperparameters: batch_size=8, lr=0.0001, AdamW, local_update_steps=30, eval_freq=5"
echo "GPU: 0 (specified in config file: device: 0)"
echo "WandB project: fvpl-selector (same as 50024 for comparison)"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
