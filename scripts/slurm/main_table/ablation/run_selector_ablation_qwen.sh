#!/bin/bash

# Ablation Study Selector Training Script for Qwen 2
# SLURM cluster execution script
# TID range: 62400-62432 (Qwen 2 ablation selector experiments)

#SBATCH -p A6000,RTX6000ADA  # Exclude RTX4090(24GB) and A5000(24GB) to avoid OOM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

# Parse arguments
MODEL="qwen2"
METHOD=$1  # vplgp (VPL + GP), vplortho (VPL + Ortho)
CLIENT_COUNT=$2  # 10, 50, 100
TID=$3  # Task ID (e.g., 62400)

if [ -z "$METHOD" ] || [ -z "$CLIENT_COUNT" ] || [ -z "$TID" ]; then
    echo "Usage: $0 <method> <client_count> <tid>"
    echo "  method: vplgp (VPL + GP prior, no orthogonal loss), vplortho (VPL + orthogonal loss, no GP prior)"
    echo "  client_count: 10, 50, 100"
    echo "  tid: Task ID (e.g., 62400)"
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

# Determine sample_client_num (always 10 clients per round)
SAMPLE_CLIENT_NUM=10

# Set checkpoint path (local repo instead of /hdd/hdd3)
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p $CHECKPOINT_DIR

# Method-specific settings
case $METHOD in
    vplgp)
        TRAINER="vplgprewardchoicetrainer"
        CONFIG_BASE="cfg/vpl-gp-no-ortho/hhst.yaml"
        METHOD_NAME="vplgp"
        ;;
    vplortho)
        TRAINER="vplrewardchoicetrainer"
        CONFIG_BASE="cfg/vpl-ortho/hhst.yaml"
        METHOD_NAME="vplortho"
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Supported methods: vplgp, vplortho"
        exit 1
        ;;
esac

# Create config file for this experiment
CONFIG_FILE="cfg/main_table/ablation/${MODEL}/${METHOD_NAME}/hhst_n${CLIENT_COUNT}_${TID}.yaml"
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
config['federate']['client_num'] = int("$CLIENT_COUNT")
config['federate']['sample_client_num'] = $SAMPLE_CLIENT_NUM

# Update checkpoint path (local repo)
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_choice_${MODEL}_ablation_${METHOD_NAME}_t${TID}.ckpt"

# Update data root (local repo)
config['data']['root'] = "$WORK_DIR/data"

# Update model type for Qwen 2
config['model']['type'] = 'Qwen/Qwen2-0.5B@huggingface_llm'

# Update trainer
config['trainer']['type'] = "$TRAINER"

# Update expname
config['expname'] = "${METHOD_NAME}_${MODEL}_n${CLIENT_COUNT}_t${TID}"

# For Qwen 2, update learning rate and batch size
if "$MODEL" == "qwen2":
    config['train']['optimizer']['lr'] = 0.00001  # Qwen 2 uses lower LR
    config['dataloader']['batch_size'] = 16
    config['llm']['grad_accum_step'] = 1

# For VPL + GP (no orthogonal loss), ensure GP prior settings
if "$METHOD" == "vplgp":
    config['llm']['vpl_use_gp_prior'] = True
    config['llm']['vpl_latent_dim'] = 32
    config['llm']['vpl_kl_weight'] = 0.02
    config['llm']['vpl_gp_temperature'] = 1.0
    config['llm']['vpl_feature_method'] = 'choice_logits'
    config['llm']['vpl_use_feature_difference'] = True
    config['llm']['vpl_use_difference_only'] = True
    config['llm']['vpl_max_logvar'] = -3.0
    # Ensure orthogonal loss is disabled
    config['llm']['vpl_orthogonal_weight'] = 0.0
    config['llm']['vpl_orthogonal_orthonorm_weight'] = 0.0
    config['llm']['vpl_use_manual_orthogonal_labels'] = False
    config['llm']['vpl_num_prototypes'] = 0
    config['llm']['vpl_prototype_scale'] = 0.0
    config['llm']['vpl_tsne_visualize_freq'] = 10

# For VPL + Ortho (no GP prior), ensure orthogonal loss settings
if "$METHOD" == "vplortho":
    config['llm']['vpl_use_gp_prior'] = False
    config['llm']['vpl_latent_dim'] = 32
    config['llm']['vpl_kl_weight'] = 0.1
    config['llm']['vpl_feature_method'] = 'choice_logits'
    config['llm']['vpl_use_feature_difference'] = True
    config['llm']['vpl_use_difference_only'] = True
    config['llm']['vpl_max_logvar'] = -3.0
    # Ensure orthogonal loss is enabled
    config['llm']['vpl_orthogonal_weight'] = 1.0
    config['llm']['vpl_orthogonal_orthonorm_weight'] = 0.0
    config['llm']['vpl_use_manual_orthogonal_labels'] = True
    config['llm']['vpl_num_prototypes'] = 2
    config['llm']['vpl_prototype_scale'] = 5.0
    config['llm']['vpl_tsne_visualize_freq'] = 10

# Save config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Config file created: {config_file}")
EOF

# Run experiment
echo "Starting ablation selector training: $METHOD_NAME, $MODEL, N=$CLIENT_COUNT, TID=$TID"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_DIR/hhrl_choice_${MODEL}_ablation_${METHOD_NAME}_t${TID}.ckpt"

python -u federatedscope/main.py \
    --cfg $CONFIG_FILE \
    > outputs/${TID}.log 2>&1

echo "Ablation experiment completed: TID=$TID"
