#!/bin/bash

# Local RL Training Only (No Selector Checkpoint)
# This script runs RL training without using selector checkpoint
# Similar to FedDPO approach - direct RL training on preference data
# Based on scripts/feddpo/hrl-10000.sh

# Parse arguments
TID=$1  # Task ID (e.g., 01000)
CLIENT_NUM=$2  # Number of clients (10, 50, 100)
GPU_ID=$3  # GPU ID (2, 3, 4)

if [ -z "$TID" ] || [ -z "$CLIENT_NUM" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: $0 <tid> <client_num> <gpu_id>"
    echo "  tid: Task ID (e.g., 01000)"
    echo "  client_num: Number of clients (10, 50, 100)"
    echo "  gpu_id: GPU ID (2, 3, 4)"
    exit 1
fi

# GPU is specified as argument
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export TOKENIZERS_PARALLELISM=false

# Set PYTHONPATH to use the current directory's federatedscope
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Determine checkpoint directory (local server)
if [ -d "/hdd/hdd3/kjb" ]; then
    CHECKPOINT_DIR="/hdd/hdd3/kjb/checkpoints"
    DATA_ROOT="/hdd/hdd3/kjb"
else
    CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
    DATA_ROOT="$PROJECT_ROOT/data"
fi
mkdir -p "$CHECKPOINT_DIR" "$DATA_ROOT"

# Determine sample_client_num based on client_num
if [ "$CLIENT_NUM" -eq 10 ]; then
    SAMPLE_CLIENT_NUM=5
elif [ "$CLIENT_NUM" -eq 50 ] || [ "$CLIENT_NUM" -eq 100 ]; then
    SAMPLE_CLIENT_NUM=10
else
    echo "ERROR: Invalid client_num: $CLIENT_NUM (must be 10, 50, or 100)"
    exit 1
fi

# Create config file for this experiment
CONFIG_FILE="cfg/local/rl_local_${TID}.yaml"
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
config['device'] = int("$GPU_ID")

# Update federate settings
config['federate']['client_num'] = int("$CLIENT_NUM")
config['federate']['sample_client_num'] = $SAMPLE_CLIENT_NUM

# Update checkpoint path
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_rlhf_gemma_choice_local_only_t${TID}.ckpt"

# Update data root
config['data']['root'] = "$DATA_ROOT"

# Update expname
config['expname'] = "rl_local_only_t${TID}_n${CLIENT_NUM}"

# RL settings (no selector - direct preference learning)
config['llm']['rlhf_use_variational_selection'] = False
config['llm']['rlhf_use_variational_generation'] = False

# Baseline comparison: use LoRA adapter so winrate can compare fine-tuned vs baseline (disable_adapter)
if 'adapter' not in config['llm']:
    config['llm']['adapter'] = {}
config['llm']['adapter']['use'] = True
config['llm']['adapter']['count'] = 3

# Eval: baseline comparison for winrate
if 'eval' not in config:
    config['eval'] = {}
config['eval']['use_baseline_model_for_winrate'] = True

# Save config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Config file created: {config_file}")
EOF

# Run experiment (nohup for background execution on local server)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg $CONFIG_FILE \
    --selector-cfg-file $CONFIG_FILE \
    > outputs/${TID}.log 2>&1 &

echo "Local RL only training started (task ID: ${TID})"
echo "Config: $CONFIG_FILE (pure DPO, no VPL, no selector required)"
echo "Client num: $CLIENT_NUM, Sample client num: $SAMPLE_CLIENT_NUM"
echo "Hyperparameters: batch_size=1, lr=0.0001, grad_accum_step=4, reward_coeff=0.1"
echo "GPU: $GPU_ID (specified in config file: device: $GPU_ID)"
echo "WandB project: fvpl-rl (same as other RL experiments)"
echo "Log file: outputs/${TID}.log"
echo "Monitor with: tail -f outputs/${TID}.log"
echo ""
echo "Note: Local RL only does NOT require a separate selector checkpoint."
echo "      DPO directly learns preferences from pairwise comparisons."
echo "      This is a baseline experiment to compare with selector-based RL training."
