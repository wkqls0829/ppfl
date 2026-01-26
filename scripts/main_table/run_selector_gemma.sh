#!/bin/bash

# Main Table Selector Training Script for Gemma-2B
# SLURM cluster execution script
# TID range: 62100-62132 (Gemma-2B selector experiments)

#SBATCH -p A6000,RTX4090,RTX6000ADA,A5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

# Parse arguments
MODEL="gemma-2b"
METHOD=$1  # feddpo, fedbiscuit, fedvpl, fedvpagp
CLIENT_COUNT=$2  # 10, 50, 100
TID=$3  # Task ID (e.g., 62100)

if [ -z "$METHOD" ] || [ -z "$CLIENT_COUNT" ] || [ -z "$TID" ]; then
    echo "Usage: $0 <method> <client_count> <tid>"
    echo "  method: feddpo, fedbiscuit, fedvpl, fedvpagp"
    echo "  client_count: 10, 50, 100"
    echo "  tid: Task ID (e.g., 62100)"
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

# Determine sample_client_num (always 10 clients per round)
SAMPLE_CLIENT_NUM=10

# Set checkpoint path (local repo instead of /hdd/hdd3)
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p $CHECKPOINT_DIR

# Method-specific settings
case $METHOD in
    feddpo)
        TRAINER="llmdporewardchoicetrainer"
        CONFIG_BASE="cfg/feddpo/hhst.yaml"
        ;;
    fedbiscuit)
        TRAINER="llmrewardchoicetrainer"
        CONFIG_BASE="cfg/fedbiscuit/hhst.yaml"
        ;;
    fedvpl)
        TRAINER="vplrewardchoicetrainer"
        CONFIG_BASE="cfg/vpl/hhst.yaml"
        ;;
    fedvpagp)
        TRAINER="vplgprewardchoicetrainer"
        CONFIG_BASE="cfg/vpl-gp/hhst.yaml"
        ;;
    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

# Create config file for this experiment
CONFIG_FILE="cfg/main_table/${MODEL}/${METHOD}/hhst_n${CLIENT_COUNT}_${TID}.yaml"
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

config_file = "$CONFIG_FILE"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Update federate settings
config['federate']['client_num'] = int("$CLIENT_COUNT")
config['federate']['sample_client_num'] = $SAMPLE_CLIENT_NUM

# Update checkpoint path (local repo)
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${TID}.ckpt"

# Update data root (local repo)
config['data']['root'] = "$WORK_DIR/data"

# Update model type
config['model']['type'] = 'google/gemma-2b@huggingface_llm'

# Update trainer
config['trainer']['type'] = "$TRAINER"

# Update expname
config['expname'] = "${METHOD}_${MODEL}_n${CLIENT_COUNT}_t${TID}"

# For Gemma-2B, update learning rate and batch size
if "$MODEL" == "gemma-2b":
    config['train']['optimizer']['lr'] = 0.0001
    config['dataloader']['batch_size'] = 8
    config['llm']['grad_accum_step'] = 4

# For FedVPA-GP, add hyperparameters from hyperparameter search results
if "$METHOD" == "fedvpagp":
    config['llm']['vpl_use_gp_prior'] = True
    config['llm']['vpl_latent_dim'] = 32
    config['llm']['vpl_kl_weight'] = 0.02  # Updated from hyperparameter search
    config['llm']['vpl_gp_temperature'] = 1.0
    config['llm']['vpl_feature_method'] = 'choice_logits'
    config['llm']['vpl_use_feature_difference'] = True
    config['llm']['vpl_use_difference_only'] = True
    config['llm']['vpl_max_logvar'] = -3.0
    config['llm']['vpl_orthogonal_weight'] = 1.0
    config['llm']['vpl_orthogonal_orthonorm_weight'] = 0.0  # Updated from hyperparameter search
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
echo "Starting selector training: $METHOD, $MODEL, N=$CLIENT_COUNT, TID=$TID"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${TID}.ckpt"

python -u federatedscope/main.py \
    --cfg $CONFIG_FILE \
    > outputs/${TID}.log 2>&1

echo "Experiment completed: TID=$TID"
