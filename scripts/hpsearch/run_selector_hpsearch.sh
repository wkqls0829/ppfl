#!/bin/bash

# Hyperparameter Search Selector Training Script
# SLURM cluster execution script
# TID range: 54000-54038 (selector experiments)

#SBATCH -p A6000,RTX6000ADA  # Exclude RTX4090(24GB) and A5000(24GB) to avoid OOM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

# Parse arguments
TID=$1  # Task ID (e.g., 54000)

if [ -z "$TID" ]; then
    echo "Usage: $0 <tid>"
    echo "  tid: Task ID (e.g., 54000)"
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

# Set checkpoint path (local repo)
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p $CHECKPOINT_DIR

# Base config
CONFIG_BASE="cfg/vpl-gp/hhst.yaml"
TRAINER="vplgprewardchoicetrainer"
MODEL="gemma-2b"

# Create config file for this experiment
CONFIG_FILE="cfg/hpsearch/vpl-gp/phase_${TID}.yaml"
mkdir -p $(dirname $CONFIG_FILE)

# Check if base config exists
if [ ! -f "$CONFIG_BASE" ]; then
    echo "ERROR: Base config file not found: $CONFIG_BASE"
    exit 1
fi

# Copy base config and modify
cp $CONFIG_BASE $CONFIG_FILE

# Determine hyperparameters based on TID
# Phase 1: 54000-54006
# Phase 2: 54007-54013
# Phase 3: 54014-54016
# Phase 4: 54017
# Phase 5: 54018-54038

# Default values (Phase 1-4 optimal)
ORTHOGONAL_WEIGHT=1.0
ORTHONORM_WEIGHT=0.1
PROTOTYPE_SCALE=5.0
KL_WEIGHT=0.1
GP_TEMPERATURE=1.0
LR=0.0001

# Phase 1: Orthogonal Loss Parameters (54000-54006)
if [ $TID -ge 54000 ] && [ $TID -le 54006 ]; then
    case $TID in
        54000) ORTHOGONAL_WEIGHT=0.2 ;;
        54001) ORTHOGONAL_WEIGHT=1.0 ;;
        54002) ORTHOGONAL_WEIGHT=5.0 ;;
        54003) ORTHONORM_WEIGHT=0.0 ;;
        54004) ORTHONORM_WEIGHT=0.5 ;;
        54005) PROTOTYPE_SCALE=2.0 ;;
        54006) PROTOTYPE_SCALE=10.0 ;;
    esac
fi

# Phase 2: VPL Core Parameters (54007-54013)
if [ $TID -ge 54007 ] && [ $TID -le 54013 ]; then
    case $TID in
        54007) KL_WEIGHT=0.02 ;;
        54008) KL_WEIGHT=0.05 ;;
        54009) KL_WEIGHT=0.1 ;;
        54010) KL_WEIGHT=0.2 ;;
        54011) GP_TEMPERATURE=0.5 ;;
        54012) GP_TEMPERATURE=2.0 ;;
        54013) GP_TEMPERATURE=5.0 ;;
    esac
fi

# Phase 3: Learning Rate (54014-54016)
if [ $TID -ge 54014 ] && [ $TID -le 54016 ]; then
    case $TID in
        54014) LR=0.00005 ;;
        54015) LR=0.0001 ;;
        54016) LR=0.0002 ;;
    esac
fi

# Phase 4: Combined Best Parameters (54017)
# Uses default values (already set)

# Phase 5: Fine-grained Search (54018-54038)
if [ $TID -ge 54018 ] && [ $TID -le 54038 ]; then
    # Sub-phase 5.1: Orthogonal Weight (54018-54024)
    if [ $TID -ge 54018 ] && [ $TID -le 54024 ]; then
        case $TID in
            54018) ORTHOGONAL_WEIGHT=0.1 ;;
            54019) ORTHOGONAL_WEIGHT=0.2 ;;
            54020) ORTHOGONAL_WEIGHT=0.5 ;;
            54021) ORTHOGONAL_WEIGHT=1.0 ;;
            54022) ORTHOGONAL_WEIGHT=2.0 ;;
            54023) ORTHOGONAL_WEIGHT=5.0 ;;
            54024) ORTHOGONAL_WEIGHT=10.0 ;;
        esac
    fi
    
    # Sub-phase 5.2: Orthonorm Weight (54025-54030)
    # Note: This should use optimal value from 5.1, but for now using default
    if [ $TID -ge 54025 ] && [ $TID -le 54030 ]; then
        case $TID in
            54025) ORTHONORM_WEIGHT=0.0 ;;
            54026) ORTHONORM_WEIGHT=0.05 ;;
            54027) ORTHONORM_WEIGHT=0.1 ;;
            54028) ORTHONORM_WEIGHT=0.2 ;;
            54029) ORTHONORM_WEIGHT=0.5 ;;
            54030) ORTHONORM_WEIGHT=1.0 ;;
        esac
    fi
    
    # Sub-phase 5.3: KL Weight (54031-54037)
    # Note: This should use optimal values from 5.1-5.2, but for now using default
    if [ $TID -ge 54031 ] && [ $TID -le 54037 ]; then
        case $TID in
            54031) KL_WEIGHT=0.01 ;;
            54032) KL_WEIGHT=0.02 ;;
            54033) KL_WEIGHT=0.05 ;;
            54034) KL_WEIGHT=0.1 ;;
            54035) KL_WEIGHT=0.2 ;;
            54036) KL_WEIGHT=0.5 ;;
            54037) KL_WEIGHT=1.0 ;;
        esac
    fi
    
    # Sub-phase 5.4: Optimal Combination (54038)
    # Uses default values (already set)
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
config['federate']['client_num'] = 10
config['federate']['sample_client_num'] = 5

# Update checkpoint path (local repo)
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_vplgp_ortho_t${TID}.ckpt"

# Update data root (local repo)
config['data']['root'] = "$WORK_DIR/data"

# Update model type
config['model']['type'] = 'google/gemma-2b@huggingface_llm'

# Update trainer
config['trainer']['type'] = "$TRAINER"

# Update expname
config['expname'] = "vplgp_hhst_ortho_t${TID}"

# Learning rate and batch size
config['train']['optimizer']['lr'] = float("$LR")
config['dataloader']['batch_size'] = 8
config['llm']['grad_accum_step'] = 4

# VPL-GP hyperparameters
config['llm']['vpl_use_gp_prior'] = True
config['llm']['vpl_latent_dim'] = 32
config['llm']['vpl_kl_weight'] = float("$KL_WEIGHT")
config['llm']['vpl_gp_temperature'] = float("$GP_TEMPERATURE")
config['llm']['vpl_feature_method'] = 'choice_logits'
config['llm']['vpl_use_feature_difference'] = True
config['llm']['vpl_use_difference_only'] = True
config['llm']['vpl_max_logvar'] = -3.0
config['llm']['vpl_orthogonal_weight'] = float("$ORTHOGONAL_WEIGHT")
config['llm']['vpl_orthogonal_orthonorm_weight'] = float("$ORTHONORM_WEIGHT")
config['llm']['vpl_use_manual_orthogonal_labels'] = True
config['llm']['vpl_num_prototypes'] = 2
config['llm']['vpl_prototype_scale'] = float("$PROTOTYPE_SCALE")
config['llm']['vpl_tsne_visualize_freq'] = 10

# Save config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Config file created: {config_file}")
EOF

# Print hyperparameters
echo "Hyperparameters:"
echo "  orthogonal_weight=$ORTHOGONAL_WEIGHT"
echo "  orthonorm_weight=$ORTHONORM_WEIGHT"
echo "  prototype_scale=$PROTOTYPE_SCALE"
echo "  kl_weight=$KL_WEIGHT"
echo "  gp_temperature=$GP_TEMPERATURE"
echo "  lr=$LR"

# Run experiment
echo "Starting selector training: TID=$TID"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_vplgp_ortho_t${TID}.ckpt"

python -u federatedscope/main.py \
    --cfg $CONFIG_FILE \
    > outputs/${TID}.log 2>&1

echo "Experiment completed: TID=$TID"
