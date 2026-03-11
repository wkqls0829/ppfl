#!/bin/bash

# Main Table RL Training Script for Qwen 2
# SLURM cluster execution script
# TID range: 63200-63232 (Qwen 2 RL experiments)

#SBATCH -p A6000,RTX6000ADA  # Exclude RTX4090(24GB) and A5000(24GB) to avoid OOM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 3-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

# Always use biscuit conda env when available (required for torch etc.)
# On SLURM cluster, skip conda (job env may use module or different path)
if [ -z "$SLURM_JOB_ID" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate biscuit 2>/dev/null || true
fi

# Parse arguments
MODEL="qwen2"
METHOD=$1  # feddpo, fedbiscuit, fedvpl, fedvpagp
CLIENT_COUNT=$2  # 10, 50, 100
RL_TID=$3  # RL Task ID (e.g., 63200)
SELECTOR_TID=$4  # Selector Task ID (e.g., 62200)

if [ -z "$METHOD" ] || [ -z "$CLIENT_COUNT" ] || [ -z "$RL_TID" ] || [ -z "$SELECTOR_TID" ]; then
    echo "Usage: $0 <method> <client_count> <rl_tid> <selector_tid>"
    echo "  method: feddpo, fedbiscuit, fedvpl, fedvpagp"
    echo "  client_count: 10, 50, 100"
    echo "  rl_tid: RL Task ID (e.g., 63200)"
    echo "  selector_tid: Selector Task ID (e.g., 62200)"
    exit 1
fi

# Set working directory: SLURM cluster vs local server
# SLURM: job runs on cluster, use cluster home. Local: use repo from script path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$SLURM_JOB_ID" ]; then
    WORK_DIR="/home2/jbkoo/ppfl"
else
    WORK_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$WORK_DIR" || { echo "ERROR: cannot cd to $WORK_DIR"; exit 1; }

# Determine data root and checkpoint directory based on environment
# Local server: /hdd/hdd3/kjb exists → use /hdd/hdd3/kjb
# Cluster (SLURM): /hdd/hdd3/kjb doesn't exist → use WORK_DIR
if [ -d "/hdd/hdd3/kjb" ]; then
    # Local server environment
    DATA_ROOT="/hdd/hdd3/kjb"
    CHECKPOINT_BASE="/hdd/hdd3/kjb/checkpoints"
    echo "Local server environment detected. Using data root: $DATA_ROOT"
else
    # Cluster environment (SLURM job, /hdd/hdd3 doesn't exist)
    DATA_ROOT="$WORK_DIR/data"
    CHECKPOINT_BASE="$WORK_DIR/checkpoints"
    echo "Cluster environment detected. Using data root: $DATA_ROOT"
fi
mkdir -p "$DATA_ROOT" "$CHECKPOINT_BASE"

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

# Use environment-specific checkpoint directory (already set above)
CHECKPOINT_DIR="$CHECKPOINT_BASE"

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
    # Try CHECKPOINT_DIR first, then WORK_DIR/checkpoints (where DP/local runs save)
    for CHECK_DIR in "$CHECKPOINT_DIR" "$WORK_DIR/checkpoints"; do
        [ -d "$CHECK_DIR" ] || continue
        echo "Checking for selector checkpoint in: $CHECK_DIR"
        echo "Looking for: MODEL=${MODEL}, METHOD=${METHOD}, SELECTOR_TID=${SELECTOR_TID}"
        ls -lh "$CHECK_DIR"/*${METHOD}*${SELECTOR_TID}* 2>/dev/null | head -10 || true
        
        SELECTOR_CKPT="$CHECK_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
        [ -f "$SELECTOR_CKPT" ] && break
        SELECTOR_CKPT="$CHECK_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
        [ -f "$SELECTOR_CKPT" ] && break
        SELECTOR_CKPT="$CHECK_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
        [ -f "$SELECTOR_CKPT" ] && break
        SELECTOR_CKPT=""
    done
    
    if [ -z "$SELECTOR_CKPT" ] || [ ! -f "$SELECTOR_CKPT" ]; then
        echo "ERROR: Selector checkpoint not found:"
        echo "  Tried in $CHECKPOINT_DIR and $WORK_DIR/checkpoints:"
        echo "    final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
        echo "    hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
        echo "    40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
        echo ""
        echo "Available in $WORK_DIR/checkpoints:"
        ls -lh "$WORK_DIR/checkpoints"/*${METHOD}* 2>/dev/null | head -20 || echo "  (none)"
        exit 1
    fi
    echo "✓ Using selector checkpoint: $SELECTOR_CKPT"
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

# Update data root (environment-specific)
config['data']['root'] = "$DATA_ROOT"

# Update model type for Qwen 2
config['model']['type'] = 'Qwen/Qwen2-0.5B@huggingface_llm'

# Update trainer
config['trainer']['type'] = "$TRAINER"

# Update expname
config['expname'] = "${METHOD}_${MODEL}_n${CLIENT_COUNT}_rl_t${RL_TID}"

# For Qwen 2, update learning rate
if "$MODEL" == "qwen2":
    config['train']['optimizer']['lr'] = 0.00001  # Qwen 2 uses lower LR
    config['llm']['grad_accum_step'] = 32  # Qwen 2 uses higher grad_accum_step

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
    config['llm']['vpl_gp_temperature'] = 1.0  # Qwen Phase 4 (match selector)
    
    # For FedVPA-GP
    if "$METHOD" == "fedvpagp":
        config['llm']['vpl_use_gp_prior'] = True
        config['llm']['vpl_kl_weight'] = 0.05  # Match selector (Qwen Phase 4)

# RL settings
config['llm']['reward_coeff'] = 0.1
config['llm']['max_prompts_for_generation'] = 50
config['llm']['generation_batch_size'] = 3

# Baseline comparison: use LoRA adapter so winrate can compare fine-tuned vs baseline (disable_adapter)
if 'adapter' not in config['llm']:
    config['llm']['adapter'] = {}
config['llm']['adapter']['use'] = True
config['llm']['adapter']['count'] = 3

# New RL eval settings: eval every 10 rounds, GPT API winrate (baseline comparison)
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
