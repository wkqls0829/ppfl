#!/bin/bash

# Local RL Training Only (No Selector Checkpoint)
# This script runs RL training without using selector checkpoint
# Similar to FedDPO approach - direct RL training on preference data
# Based on scripts/feddpo/hrl-10000.sh

tid=56000  # Task ID for local RL only experiment

# GPU is specified in config file (device: 0)
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export TOKENIZERS_PARALLELISM=false

# Set PYTHONPATH to use the current directory's federatedscope
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Local RL only does NOT require a selector checkpoint (similar to FedDPO)
# DPO directly learns preferences without a separate selector model
# However, rlhf/main.py requires --selector-cfg-file argument
# We'll use the same config file for selector_cfg (since we don't use selector)

# Determine checkpoint directory (local server)
if [ -d "/hdd/hdd3/kjb" ]; then
    CHECKPOINT_DIR="/hdd/hdd3/kjb/checkpoints"
    DATA_ROOT="/hdd/hdd3/kjb"
else
    CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
    DATA_ROOT="$PROJECT_ROOT/data"
fi
mkdir -p "$CHECKPOINT_DIR" "$DATA_ROOT"

# Create config file for this experiment
CONFIG_FILE="cfg/hpsearch/rl-local-only/rl_local_${tid}.yaml"
mkdir -p $(dirname $CONFIG_FILE)

# Base config (FedDPO style)
CONFIG_BASE="cfg/feddpo/hrl-10000.yaml"

if [ ! -f "$CONFIG_BASE" ]; then
    echo "ERROR: Base config file not found: $CONFIG_BASE"
    exit 1
fi

# Copy base config and modify
cp $CONFIG_BASE $CONFIG_FILE

# Update config with experiment-specific settings
python3 << EOF
import yaml
import sys
import os

config_file = "$CONFIG_FILE"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Set device (local server)
config['use_gpu'] = True
config['device'] = 0

# Update checkpoint path
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_rlhf_gemma_choice_local_only_t${tid}.ckpt"

# Update data root
config['data']['root'] = "$DATA_ROOT"

# Update expname
config['expname'] = "rl_local_only_t${tid}"

# RL settings (no selector - direct preference learning)
config['llm']['rlhf_use_variational_selection'] = False
config['llm']['rlhf_use_variational_generation'] = False

# Save config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Config file created: {config_file}")
EOF

# Run experiment (nohup for background execution on local server)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg $CONFIG_FILE \
    --selector-cfg-file $CONFIG_FILE \
    > outputs/${tid}.log 2>&1 &

echo "Local RL only training started (task ID: ${tid})"
echo "Config: $CONFIG_FILE (pure DPO, no VPL, no selector required)"
echo "Hyperparameters: batch_size=1, lr=0.0001, grad_accum_step=4, reward_coeff=0.1"
echo "GPU: 0 (specified in config file: device: 0)"
echo "WandB project: fvpl-rl (same as other RL experiments)"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
echo ""
echo "Note: Local RL only does NOT require a separate selector checkpoint."
echo "      DPO directly learns preferences from pairwise comparisons."
echo "      This is a baseline experiment to compare with selector-based RL training."
