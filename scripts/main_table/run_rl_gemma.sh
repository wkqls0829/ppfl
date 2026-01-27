#!/bin/bash

# Main Table RL Training Script for Gemma-2B
# SLURM cluster execution script
# TID range: 63100-63132 (Gemma-2B RL experiments)

#SBATCH -p A6000,RTX6000ADA  # Exclude RTX4090(24GB) and A5000(24GB) to avoid OOM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 3-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

# Parse arguments
MODEL="gemma-2b"
METHOD=$1  # feddpo, fedbiscuit, fedvpl, fedvpagp
CLIENT_COUNT=$2  # 10, 50, 100
RL_TID=$3  # RL Task ID (e.g., 63100)
SELECTOR_TID=$4  # Selector Task ID (e.g., 62100)

if [ -z "$METHOD" ] || [ -z "$CLIENT_COUNT" ] || [ -z "$RL_TID" ] || [ -z "$SELECTOR_TID" ]; then
    echo "Usage: $0 <method> <client_count> <rl_tid> <selector_tid>"
    echo "  method: feddpo, fedbiscuit, fedvpl, fedvpagp"
    echo "  client_count: 10, 50, 100"
    echo "  rl_tid: RL Task ID (e.g., 63100)"
    echo "  selector_tid: Selector Task ID (e.g., 62100)"
    exit 1
fi

# Set working directory
WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR

# Set PYTHONPATH
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# Set CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load environment variables from .env file if it exists
if [ -f "$WORK_DIR/.env" ]; then
    export $(cat $WORK_DIR/.env | grep -v '^#' | xargs)
    echo "Loaded environment variables from .env file"
fi

# Set Hugging Face cache directory (use local repo to avoid cache issues)
export HF_HOME="$WORK_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORK_DIR/.cache/huggingface/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Check if Hugging Face token is set (required for gated models like Gemma)
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "WARNING: HF_TOKEN or HUGGING_FACE_HUB_TOKEN not set."
    echo "Gemma-2B is a gated model and may require authentication."
    echo "Set HF_TOKEN in .env file or export it before running."
fi

# Set checkpoint path (local repo instead of /hdd/hdd3)
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p $CHECKPOINT_DIR

# Method-specific settings (check early to determine if selector is needed)
case $METHOD in
    feddpo)
        USE_SELECTOR=false
        ;;
    fedbiscuit)
        USE_SELECTOR=false
        ;;
    fedvpl)
        USE_SELECTOR=true
        ;;
    fedvpagp)
        USE_SELECTOR=true
        ;;
    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

# Check if selector checkpoint exists (only for methods that use selector)
SELECTOR_CKPT=""
if [ "$USE_SELECTOR" == "true" ]; then
    SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
    if [ ! -f "$SELECTOR_CKPT" ]; then
        # Try regular checkpoint
        SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
        if [ ! -f "$SELECTOR_CKPT" ]; then
            echo "ERROR: Selector checkpoint not found:"
            echo "  Tried: $CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
            echo "  Tried: $CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
            exit 1
        fi
    fi
    echo "Using selector checkpoint: $SELECTOR_CKPT"
else
    echo "Method $METHOD does not require selector checkpoint (USE_SELECTOR=false)"
fi

# Method-specific settings (USE_SELECTOR already determined above)
case $METHOD in
    feddpo)
        TRAINER="llmdporewardtrainer"
        CONFIG_BASE="cfg/feddpo/hrl-10000.yaml"
        ;;
    fedbiscuit)
        TRAINER="llmdporewardtrainer"
        CONFIG_BASE="cfg/fedbiscuit/hrl.yaml"
        ;;
    fedvpl)
        TRAINER="llmdporewardtrainer"
        CONFIG_BASE="cfg/vpl/hrl.yaml"
        ;;
    fedvpagp)
        TRAINER="llmdporewardtrainer"
        CONFIG_BASE="cfg/vpl-gp/hrl.yaml"
        ;;
    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

# Create config file for this experiment
CONFIG_FILE="cfg/main_table/${MODEL}/${METHOD}/hrl_n${CLIENT_COUNT}_${RL_TID}.yaml"
mkdir -p $(dirname $CONFIG_FILE)

# Check if base config exists
if [ ! -f "$CONFIG_BASE" ]; then
    echo "ERROR: Base config file not found: $CONFIG_BASE"
    echo "Please ensure the base config file exists."
    exit 1
fi

# Copy base config and modify
cp $CONFIG_BASE $CONFIG_FILE

# Verify config file was created
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Failed to create config file: $CONFIG_FILE"
    exit 1
fi

# Update config with experiment-specific settings
python3 << EOF
import yaml
import sys
import os

config_file = "$CONFIG_FILE"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Set Hugging Face cache directory in config
if 'llm' not in config:
    config['llm'] = {}
if 'cache' not in config['llm']:
    config['llm']['cache'] = {}
config['llm']['cache']['model'] = "$WORK_DIR/.cache/huggingface/transformers"

# Set device to 0 (SLURM sets CUDA_VISIBLE_DEVICES, so always use device 0)
config['use_gpu'] = True
config['device'] = 0

# Set num_workers to 0 to avoid "Too many open files" error in cluster environment
if 'dataloader' not in config:
    config['dataloader'] = {}
config['dataloader']['num_workers'] = 0

# Update federate settings
config['federate']['client_num'] = 1  # RL uses single client
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_rlhf_${MODEL}_choice_${METHOD}_t${RL_TID}.ckpt"

# Update data root (local repo)
config['data']['root'] = "$WORK_DIR/data"

# Update model type
config['model']['type'] = 'google/gemma-2b@huggingface_llm'

# Update trainer
config['trainer']['type'] = "$TRAINER"

# Update expname
config['expname'] = "${METHOD}_${MODEL}_n${CLIENT_COUNT}_rl_t${RL_TID}"

# For Gemma-2B, update learning rate
if "$MODEL" == "gemma-2b":
    config['train']['optimizer']['lr'] = 0.0001
    config['llm']['grad_accum_step'] = 4

# For VPL methods, add selector checkpoint and VPL settings
if "$USE_SELECTOR" == "true":
    config['llm']['rlhf_use_variational_selection'] = True
    config['llm']['rlhf_use_variational_generation'] = False
    config['llm']['rlhf_selector_checkpoint'] = "$SELECTOR_CKPT"
    
    # VPL settings
    config['llm']['vpl_latent_dim'] = 32
    config['llm']['vpl_feature_method'] = 'choice_logits'
    config['llm']['vpl_use_feature_difference'] = True
    config['llm']['vpl_use_difference_only'] = True
    config['llm']['vpl_gp_temperature'] = 1.0
    
    # For FedVPA-GP
    if "$METHOD" == "fedvpagp":
        config['llm']['vpl_use_gp_prior'] = True

# RL settings
config['llm']['reward_coeff'] = 0.1
config['llm']['max_prompts_for_generation'] = 50
config['llm']['generation_batch_size'] = 3

# Disable reward model evaluation
if 'eval' not in config:
    config['eval'] = {}
config['eval']['metrics'] = ['loss', 'acc']  # Reward model evaluation disabled
# Remove reward evaluation related settings
if 'max_samples_for_reward' in config['eval']:
    del config['eval']['max_samples_for_reward']
if 'use_baseline_model_for_winrate' in config['eval']:
    del config['eval']['use_baseline_model_for_winrate']
if 'use_gpt_api_for_winrate' in config['eval']:
    del config['eval']['use_gpt_api_for_winrate']
if 'openai_model' in config['eval']:
    del config['eval']['openai_model']

# Save config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Config file created: {config_file}")
EOF

# Find selector config file (for VPL methods)
SELECTOR_CFG=""
if [ "$USE_SELECTOR" == "true" ]; then
    SELECTOR_CFG="cfg/main_table/${MODEL}/${METHOD}/hhst_n${CLIENT_COUNT}_${SELECTOR_TID}.yaml"
    if [ ! -f "$SELECTOR_CFG" ]; then
        echo "WARNING: Selector config not found: $SELECTOR_CFG"
        SELECTOR_CFG=""
    fi
fi

# Run experiment
echo "Starting RL training: $METHOD, $MODEL, N=$CLIENT_COUNT, RL_TID=$RL_TID"
echo "Config: $CONFIG_FILE"
echo "Selector checkpoint: $SELECTOR_CKPT"

if [ -n "$SELECTOR_CFG" ]; then
    python -u federatedscope/llm/rlhf/main.py \
        --cfg $CONFIG_FILE \
        --selector-cfg-file $SELECTOR_CFG \
        > outputs/${RL_TID}.log 2>&1
else
    python -u federatedscope/llm/rlhf/main.py \
        --cfg $CONFIG_FILE \
        > outputs/${RL_TID}.log 2>&1
fi

echo "RL experiment completed: TID=$RL_TID"
