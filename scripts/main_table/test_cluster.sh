#!/bin/bash

# Main Table Cluster Test Script
# Quick test to verify cluster setup before running full experiments
# Tests both Selector and RL training with minimal rounds and data

#SBATCH -p A6000,RTX4090,RTX6000ADA,A5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-02:00:00
#SBATCH -o /home/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

# Check if running inside SLURM job
if [ -z "$SLURM_JOB_ID" ]; then
    echo "ERROR: This script must be run via SLURM (sbatch)"
    echo ""
    echo "Usage:"
    echo "  sbatch $0 <model> <method>"
    echo ""
    echo "Example:"
    echo "  sbatch $0 gemma-2b fedvpagp"
    echo ""
    echo "The cluster login nodes don't have GPUs. You must submit this as a SLURM job."
    exit 1
fi

# Check if GPU is available
echo "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU may not be available."
    exit 1
fi

# Verify GPU is accessible
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: Cannot access GPU. Make sure you're running in a SLURM job with GPU allocation."
    echo "Current SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
    exit 1
fi

# Display GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Parse arguments
MODEL=$1  # gemma-2b or qwen2
METHOD=$2  # feddpo, fedbiscuit, fedvpl, fedvpagp

if [ -z "$MODEL" ] || [ -z "$METHOD" ]; then
    echo "Usage: $0 <model> <method>"
    echo "  model: gemma-2b, qwen2"
    echo "  method: feddpo, fedbiscuit, fedvpl, fedvpagp"
    echo ""
    echo "Example:"
    echo "  sbatch $0 gemma-2b fedvpagp"
    exit 1
fi

# Set working directory
WORK_DIR="/home/jbkoo/ppfl"
cd $WORK_DIR

# Set PYTHONPATH
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# Set CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load environment variables from .env file if it exists
if [ -f "$WORK_DIR/.env" ]; then
    export $(cat $WORK_DIR/.env | grep -v '^#' | xargs)
    echo "✓ Loaded environment variables from .env file"
else
    echo "⚠ WARNING: .env file not found. API keys may not be set."
fi

# Test IDs (use 9xxxx range for tests)
SELECTOR_TEST_TID=90001
RL_TEST_TID=91001

# Set checkpoint path
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p $CHECKPOINT_DIR

echo "=========================================="
echo "Main Table Cluster Test"
echo "=========================================="
echo "Model: $MODEL"
echo "Method: $METHOD"
echo "Selector Test TID: $SELECTOR_TEST_TID"
echo "RL Test TID: $RL_TEST_TID"
echo "=========================================="
echo ""

# Method-specific settings
case $METHOD in
    feddpo)
        TRAINER_SELECTOR="llmdporewardchoicetrainer"
        TRAINER_RL="llmdporewardtrainer"
        CONFIG_BASE_SELECTOR="cfg/feddpo/hhst.yaml"
        CONFIG_BASE_RL="cfg/feddpo/hrl-10000.yaml"
        USE_SELECTOR=false
        ;;
    fedbiscuit)
        TRAINER_SELECTOR="llmrewardchoicetrainer"
        TRAINER_RL="llmdporewardtrainer"
        CONFIG_BASE_SELECTOR="cfg/fedbiscuit/hhst.yaml"
        CONFIG_BASE_RL="cfg/fedbiscuit/hrl.yaml"
        USE_SELECTOR=false
        ;;
    fedvpl)
        TRAINER_SELECTOR="vplrewardchoicetrainer"
        TRAINER_RL="llmdporewardtrainer"
        CONFIG_BASE_SELECTOR="cfg/vpl/hhst.yaml"
        CONFIG_BASE_RL="cfg/vpl/hrl.yaml"
        USE_SELECTOR=true
        ;;
    fedvpagp)
        TRAINER_SELECTOR="vplgprewardchoicetrainer"
        TRAINER_RL="llmdporewardtrainer"
        CONFIG_BASE_SELECTOR="cfg/vpl-gp/hhst.yaml"
        CONFIG_BASE_RL="cfg/vpl-gp/hrl.yaml"
        USE_SELECTOR=true
        ;;
    *)
        echo "ERROR: Unknown method: $METHOD"
        exit 1
        ;;
esac

# Model-specific settings
if [ "$MODEL" == "gemma-2b" ]; then
    MODEL_TYPE="google/gemma-2b@huggingface_llm"
    LR=0.0001
    BATCH_SIZE=8
    GRAD_ACCUM=4
elif [ "$MODEL" == "qwen2" ]; then
    MODEL_TYPE="Qwen/Qwen2-0.5B@huggingface_llm"
    LR=0.00001
    BATCH_SIZE=16
    GRAD_ACCUM=1
else
    echo "ERROR: Unknown model: $MODEL"
    exit 1
fi

# ==========================================
# Step 1: Test Selector Training
# ==========================================
echo ""
echo "=========================================="
echo "Step 1: Testing Selector Training"
echo "=========================================="

SELECTOR_CONFIG="cfg/main_table/test/${MODEL}/${METHOD}/hhst_test_${SELECTOR_TEST_TID}.yaml"
mkdir -p $(dirname $SELECTOR_CONFIG)

# Create selector test config
python3 << EOF
import yaml
import os

config_file = "$SELECTOR_CONFIG"
with open("$CONFIG_BASE_SELECTOR", 'r') as f:
    config = yaml.safe_load(f)

# Test settings: minimal rounds and data
config['federate']['client_num'] = 10
config['federate']['sample_client_num'] = 5
config['federate']['total_round_num'] = 2  # Only 2 rounds for quick test
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_test_${SELECTOR_TEST_TID}.ckpt"
config['federate']['save_freq'] = 2

# Data settings: limit samples for quick test
config['data']['root'] = "$WORK_DIR/data"
if 'max_train_samples' not in config.get('data', {}):
    config.setdefault('data', {})['max_train_samples'] = 100
if 'max_test_samples' not in config.get('data', {}):
    config.setdefault('data', {})['max_test_samples'] = 50

# Model settings
config['model']['type'] = "$MODEL_TYPE"
config['trainer']['type'] = "$TRAINER_SELECTOR"

# GPU settings: ensure GPU is used (SLURM allocates GPU)
config['use_gpu'] = True
# Use GPU 0 (SLURM allocates single GPU)
config['device'] = 0

# Training settings
config['train']['optimizer']['lr'] = $LR
config['dataloader']['batch_size'] = $BATCH_SIZE
config['llm']['grad_accum_step'] = $GRAD_ACCUM
config['train']['local_update_steps'] = 5  # Reduced for quick test

# Expname
config['expname'] = "${METHOD}_${MODEL}_test_selector_t${SELECTOR_TEST_TID}"

# For FedVPA-GP, add hyperparameters
if "$METHOD" == "fedvpagp":
    config['llm']['vpl_use_gp_prior'] = True
    config['llm']['vpl_latent_dim'] = 32
    config['llm']['vpl_kl_weight'] = 0.1
    config['llm']['vpl_gp_temperature'] = 1.0
    config['llm']['vpl_feature_method'] = 'choice_logits'
    config['llm']['vpl_use_feature_difference'] = True
    config['llm']['vpl_use_difference_only'] = True
    config['llm']['vpl_max_logvar'] = -3.0
    config['llm']['vpl_orthogonal_weight'] = 1.0
    config['llm']['vpl_orthogonal_orthonorm_weight'] = 0.1
    config['llm']['vpl_use_manual_orthogonal_labels'] = True
    config['llm']['vpl_num_prototypes'] = 2
    config['llm']['vpl_prototype_scale'] = 5.0

# WandB settings
config['wandb']['name_project'] = 'fvpl-selector-test'

# Save config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"✓ Selector test config created: {config_file}")
EOF

# Run selector test
echo "Running selector test..."
python -u federatedscope/main.py \
    --cfg $SELECTOR_CONFIG \
    > outputs/${SELECTOR_TEST_TID}_test.log 2>&1

SELECTOR_EXIT_CODE=$?

if [ $SELECTOR_EXIT_CODE -eq 0 ]; then
    echo "✓ Selector test PASSED"
    SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_test_${SELECTOR_TEST_TID}.ckpt"
    if [ -f "$SELECTOR_CKPT" ]; then
        echo "✓ Selector checkpoint created: $SELECTOR_CKPT"
    else
        echo "⚠ WARNING: Selector checkpoint not found (may be normal for test)"
    fi
else
    echo "✗ Selector test FAILED (exit code: $SELECTOR_EXIT_CODE)"
    echo "Check log: outputs/${SELECTOR_TEST_TID}_test.log"
    exit 1
fi

# ==========================================
# Step 2: Test RL Training (if selector passed)
# ==========================================
if [ "$USE_SELECTOR" == "true" ] && [ ! -f "$SELECTOR_CKPT" ]; then
    echo ""
    echo "⚠ WARNING: Selector checkpoint not found. Skipping RL test."
    echo "This is OK for methods that don't require selector (FedDPO, FedBiscuit)."
    exit 0
fi

echo ""
echo "=========================================="
echo "Step 2: Testing RL Training"
echo "=========================================="

RL_CONFIG="cfg/main_table/test/${MODEL}/${METHOD}/hrl_test_${RL_TEST_TID}.yaml"
mkdir -p $(dirname $RL_CONFIG)

# Create RL test config
python3 << EOF
import yaml
import os

config_file = "$RL_CONFIG"
with open("$CONFIG_BASE_RL", 'r') as f:
    config = yaml.safe_load(f)

# Test settings: minimal rounds and data
config['federate']['client_num'] = 1  # RL uses single client
config['federate']['total_round_num'] = 2  # Only 2 rounds for quick test
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_rlhf_${MODEL}_choice_${METHOD}_test_${RL_TEST_TID}.ckpt"
config['federate']['save_freq'] = 2

# Data settings: limit samples for quick test
config['data']['root'] = "$WORK_DIR/data"
if 'max_train_samples' not in config.get('data', {}):
    config.setdefault('data', {})['max_train_samples'] = 50
if 'max_test_samples' not in config.get('data', {}):
    config.setdefault('data', {})['max_test_samples'] = 20

# Model settings
config['model']['type'] = "$MODEL_TYPE"
config['trainer']['type'] = "$TRAINER_RL"

# GPU settings: ensure GPU is used (SLURM allocates GPU)
config['use_gpu'] = True
# Use GPU 0 (SLURM allocates single GPU)
config['device'] = 0

# Training settings
config['train']['optimizer']['lr'] = $LR
config['llm']['grad_accum_step'] = $GRAD_ACCUM
config['train']['local_update_steps'] = 5  # Reduced for quick test

# RL settings (minimal for test)
config['llm']['max_prompts_for_generation'] = 10  # Reduced from 50
config['llm']['generation_batch_size'] = 2
config['llm']['max_samples_for_reward'] = 10  # Reduced from 30
config['llm']['use_gpt_api_for_winrate'] = False  # Disable GPT API for test (faster)
config['llm']['use_baseline_model_for_winrate'] = False

# For VPL methods, add selector checkpoint
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

# OpenAI API key from environment variable
if 'OPENAI_API_KEY' in os.environ:
    if 'eval' not in config:
        config['eval'] = {}
    config['eval']['openai_api_key'] = os.environ['OPENAI_API_KEY']

# Expname
config['expname'] = "${METHOD}_${MODEL}_test_rl_t${RL_TEST_TID}"

# WandB settings
config['wandb']['name_project'] = 'fvpl-rl-test'

# Save config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"✓ RL test config created: {config_file}")
EOF

# Find selector config file (for VPL methods)
SELECTOR_CFG=""
if [ "$USE_SELECTOR" == "true" ]; then
    SELECTOR_CFG="$SELECTOR_CONFIG"
fi

# Run RL test
echo "Running RL test..."
if [ -n "$SELECTOR_CFG" ]; then
    python -u federatedscope/llm/rlhf/main.py \
        --cfg $RL_CONFIG \
        --selector-cfg-file $SELECTOR_CFG \
        > outputs/${RL_TEST_TID}_test.log 2>&1
else
    python -u federatedscope/llm/rlhf/main.py \
        --cfg $RL_CONFIG \
        > outputs/${RL_TEST_TID}_test.log 2>&1
fi

RL_EXIT_CODE=$?

if [ $RL_EXIT_CODE -eq 0 ]; then
    echo "✓ RL test PASSED"
else
    echo "✗ RL test FAILED (exit code: $RL_EXIT_CODE)"
    echo "Check log: outputs/${RL_TEST_TID}_test.log"
    exit 1
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Model: $MODEL"
echo "Method: $METHOD"
echo "Selector Test: $([ $SELECTOR_EXIT_CODE -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "RL Test: $([ $RL_EXIT_CODE -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo ""
echo "Log files:"
echo "  Selector: outputs/${SELECTOR_TEST_TID}_test.log"
echo "  RL: outputs/${RL_TEST_TID}_test.log"
echo ""
if [ $SELECTOR_EXIT_CODE -eq 0 ] && [ $RL_EXIT_CODE -eq 0 ]; then
    echo "✓ All tests PASSED! Ready for main table experiments."
else
    echo "✗ Some tests FAILED. Please check logs before running main experiments."
    exit 1
fi
