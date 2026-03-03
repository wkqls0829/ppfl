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
# Phase 3: 54014-54022 (refinement around 54005, 54008, 54013)
# Phase 4: 54023-54025 (Learning Rate)
# Phase 5: 54026 (Combined Best)
# Phase 6: 54027-54047 (Fine-grained)

# Default values (Phase 1-2 optimal)
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

# Phase 3: Refinement around 54005 (prototype_scale), 54008 (kl), 54013 (gp_temperature)
if [ $TID -ge 54014 ] && [ $TID -le 54022 ]; then
    PROTOTYPE_SCALE=2.0
    KL_WEIGHT=0.1
    GP_TEMPERATURE=1.0
    case $TID in
        54014) PROTOTYPE_SCALE=1.0 ;;
        54015) PROTOTYPE_SCALE=2.0 ;;
        54016) PROTOTYPE_SCALE=3.0 ;;
        54017) KL_WEIGHT=0.03 ;;
        54018) KL_WEIGHT=0.05 ;;
        54019) KL_WEIGHT=0.08 ;;
        54020) GP_TEMPERATURE=3.0 ; KL_WEIGHT=0.05 ;;
        54021) GP_TEMPERATURE=5.0 ; KL_WEIGHT=0.05 ;;
        54022) GP_TEMPERATURE=7.0 ; KL_WEIGHT=0.05 ;;
    esac
fi

# Phase 4: Learning Rate (54023-54025)
if [ $TID -ge 54023 ] && [ $TID -le 54025 ]; then
    case $TID in
        54023) LR=0.00005 ;;
        54024) LR=0.0001 ;;
        54025) LR=0.0002 ;;
    esac
fi

# Phase 5: Combined Best Parameters (54026)
# Phase 1–4 최적/중간값 조합: orthogonal=1.0, orthonorm=0.1, prototype_scale=2.0 (54005/54015),
# kl_weight=0.05 (54008/54018), gp_temperature=5.0 (54013/54021), lr=0.0001 (54024)
if [ $TID -eq 54026 ]; then
    ORTHOGONAL_WEIGHT=1.0
    ORTHONORM_WEIGHT=0.1
    PROTOTYPE_SCALE=2.0
    KL_WEIGHT=0.05
    GP_TEMPERATURE=5.0
    LR=0.0001
fi

# Phase 6: Fine-grained Search (54027-54047)
if [ $TID -ge 54027 ] && [ $TID -le 54047 ]; then
    # Sub-phase 6.1: Orthogonal Weight (54027-54033)
    if [ $TID -ge 54027 ] && [ $TID -le 54033 ]; then
        case $TID in
            54027) ORTHOGONAL_WEIGHT=0.1 ;;
            54028) ORTHOGONAL_WEIGHT=0.2 ;;
            54029) ORTHOGONAL_WEIGHT=0.5 ;;
            54030) ORTHOGONAL_WEIGHT=1.0 ;;
            54031) ORTHOGONAL_WEIGHT=2.0 ;;
            54032) ORTHOGONAL_WEIGHT=5.0 ;;
            54033) ORTHOGONAL_WEIGHT=10.0 ;;
        esac
    fi
    
    # Sub-phase 6.2: Orthonorm Weight (54034-54039)
    if [ $TID -ge 54034 ] && [ $TID -le 54039 ]; then
        case $TID in
            54034) ORTHONORM_WEIGHT=0.0 ;;
            54035) ORTHONORM_WEIGHT=0.05 ;;
            54036) ORTHONORM_WEIGHT=0.1 ;;
            54037) ORTHONORM_WEIGHT=0.2 ;;
            54038) ORTHONORM_WEIGHT=0.5 ;;
            54039) ORTHONORM_WEIGHT=1.0 ;;
        esac
    fi
    
    # Sub-phase 6.3: KL Weight (54040-54046)
    if [ $TID -ge 54040 ] && [ $TID -le 54046 ]; then
        case $TID in
            54040) KL_WEIGHT=0.01 ;;
            54041) KL_WEIGHT=0.02 ;;
            54042) KL_WEIGHT=0.05 ;;
            54043) KL_WEIGHT=0.1 ;;
            54044) KL_WEIGHT=0.2 ;;
            54045) KL_WEIGHT=0.5 ;;
            54046) KL_WEIGHT=1.0 ;;
        esac
    fi
    
    # Sub-phase 6.4: Optimal Combination (54047)
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
