#!/bin/bash

# Hyperparameter Search RL Training Script
# SLURM cluster execution script
# TID range: 55000-55038 (RL experiments)

#SBATCH -p A6000,RTX6000ADA  # Exclude RTX4090(24GB) and A5000(24GB) to avoid OOM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 3-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

# Parse arguments
RL_TID=$1  # RL Task ID (e.g., 55000)
SELECTOR_TID=$2  # Selector Task ID (e.g., 54000)

if [ -z "$RL_TID" ] || [ -z "$SELECTOR_TID" ]; then
    echo "Usage: $0 <rl_tid> <selector_tid>"
    echo "  rl_tid: RL Task ID (e.g., 55000)"
    echo "  selector_tid: Selector Task ID (e.g., 54000)"
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

# Determine data root and checkpoint directory
if [ -d "/hdd/hdd3/kjb" ]; then
    DATA_ROOT="/hdd/hdd3/kjb"
    CHECKPOINT_BASE="/hdd/hdd3/kjb/checkpoints"
else
    DATA_ROOT="$WORK_DIR/data"
    CHECKPOINT_BASE="$WORK_DIR/checkpoints"
fi
mkdir -p "$DATA_ROOT" "$CHECKPOINT_BASE"

# Set Hugging Face cache directory
export HF_HOME="$WORK_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORK_DIR/.cache/huggingface/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

CHECKPOINT_DIR="$CHECKPOINT_BASE"
MODEL="gemma-2b"
METHOD="vplgp"

# Check if selector checkpoint exists
# Note: Selector saves as: hhrl_choice_gemma-2b_fedbiscuit_u3_vplgp_ortho_t${SELECTOR_TID}.ckpt
# Try with _ortho_ suffix first (hpsearch naming convention)
SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${SELECTOR_TID}.ckpt"
if [ ! -f "$SELECTOR_CKPT" ]; then
    SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${SELECTOR_TID}.ckpt"
    if [ ! -f "$SELECTOR_CKPT" ]; then
        SELECTOR_CKPT="$CHECKPOINT_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${SELECTOR_TID}.ckpt"
        if [ ! -f "$SELECTOR_CKPT" ]; then
            # Fallback: try without _ortho_ (for compatibility)
            SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
            if [ ! -f "$SELECTOR_CKPT" ]; then
                SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
                if [ ! -f "$SELECTOR_CKPT" ]; then
                    SELECTOR_CKPT="$CHECKPOINT_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
                    if [ ! -f "$SELECTOR_CKPT" ]; then
                        echo "ERROR: Selector checkpoint not found:"
                        echo "  Tried: $CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${SELECTOR_TID}.ckpt"
                        echo "  Tried: $CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${SELECTOR_TID}.ckpt"
                        echo "  Tried: $CHECKPOINT_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${SELECTOR_TID}.ckpt"
                        echo "  Tried: $CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
                        echo "  Tried: $CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
                        echo "  Tried: $CHECKPOINT_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
                        echo ""
                        echo "Available checkpoints in $CHECKPOINT_DIR:"
                        ls -lh "$CHECKPOINT_DIR"/*${SELECTOR_TID}* 2>/dev/null | head -10 || echo "  No checkpoints found for TID ${SELECTOR_TID}"
                        exit 1
                    fi
                fi
            fi
        fi
    fi
fi
echo "✓ Using selector checkpoint: $SELECTOR_CKPT"

# Base config
CONFIG_BASE="cfg/vpl-gp/hrl.yaml"
TRAINER="llmdporewardtrainer"

# Create config file for this experiment
CONFIG_FILE="cfg/hpsearch/vpl-gp-rl/hrl_${RL_TID}.yaml"
mkdir -p $(dirname $CONFIG_FILE)

# Check if base config exists
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

# Update data root
config['data']['root'] = "$DATA_ROOT"

# Update model type
config['model']['type'] = 'google/gemma-2b@huggingface_llm'

# Update trainer
config['trainer']['type'] = "$TRAINER"

# Update expname
config['expname'] = "vplgp_hrl_ortho_t${RL_TID}"

# Learning rate
config['train']['optimizer']['lr'] = 0.0001
config['llm']['grad_accum_step'] = 4

# VPL settings
config['llm']['rlhf_use_variational_selection'] = True
config['llm']['rlhf_use_variational_generation'] = False
config['llm']['rlhf_selector_checkpoint'] = "$SELECTOR_CKPT"
config['llm']['vpl_latent_dim'] = 32
config['llm']['vpl_feature_method'] = 'choice_logits'
config['llm']['vpl_use_feature_difference'] = True
config['llm']['vpl_use_difference_only'] = True
config['llm']['vpl_use_gp_prior'] = True
config['llm']['vpl_gp_temperature'] = 1.0

# RL settings
config['llm']['reward_coeff'] = 0.1
config['llm']['max_prompts_for_generation'] = 50
config['llm']['generation_batch_size'] = 3
config['llm']['max_samples_for_reward'] = 30
config['llm']['use_gpt_api_for_winrate'] = True
config['llm']['use_baseline_model_for_winrate'] = True
config['llm']['openai_model'] = 'gpt-4o-mini'

# Eval settings
if 'eval' not in config:
    config['eval'] = {}
config['eval']['freq'] = 10
config['eval']['use_gpt_api_for_winrate'] = True
config['eval']['use_baseline_model_for_winrate'] = True
config['eval']['metrics'] = ['loss', 'acc', 'helpfulness_winrate', 'harmlessness_winrate']

# Save config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Config file created: {config_file}")
EOF

# Find selector config file
SELECTOR_CFG="cfg/hpsearch/vpl-gp/phase_${SELECTOR_TID}.yaml"
if [ ! -f "$SELECTOR_CFG" ]; then
    echo "WARNING: Selector config not found: $SELECTOR_CFG"
    SELECTOR_CFG=""
fi

# Run experiment
echo "Starting RL training: RL_TID=$RL_TID, Selector_TID=$SELECTOR_TID"
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
