#!/bin/bash

# Federated Variational Preference Learning (FedVPL) training script
# Based on vpl-gp 50024 but WITHOUT GP prior and orthogonal loss
# Naive VPL implementation for federated learning (baseline)
# Single GPU mode: GPU 2 (specified in config file)

tid=30024
# GPU is specified in config file (device: 2)
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg cfg/fedvpl/hhst-30024.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_fedvpl_${tid}.ckpt \
    expname "fedvpl_hhst_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "FedVPL HHST training started (task ID: ${tid})"
echo "Config: cfg/fedvpl/hhst-30024.yaml (baseline: no GP prior, no orthogonal loss, same hyperparameters as 50024)"
echo "Hyperparameters: batch_size=16, lr=0.0001, KL_weight=0.1, variance_cap=-3.0"
echo "GPU: 2 (specified in config file: device: 2)"
echo "WandB project: fvpl-selector (same as 50024 for comparison)"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
