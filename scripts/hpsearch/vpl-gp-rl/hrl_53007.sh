#!/bin/bash

# Variational Preference Learning with Gumbel Softmax Prior (VPL-GP) RL training script
# HRL (hh-rlhf) version with reward model evaluation
# Using 52000 selector checkpoint
# Single GPU mode: GPU specified in config file

tid=53007
selector_tid=52007  # Selector checkpoint task ID (hpsearch 52007)

# GPU is specified in config file
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if selector checkpoint exists (try final checkpoint first, then regular)
SELECTOR_CKPT_FINAL="/hdd/hdd3/kjb/checkpoints/final_hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_${selector_tid}.ckpt"
SELECTOR_CKPT="/hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_${selector_tid}.ckpt"

if [ -f "${SELECTOR_CKPT_FINAL}" ]; then
    SELECTOR_CKPT="${SELECTOR_CKPT_FINAL}"
    echo "✓ Using final selector checkpoint: ${SELECTOR_CKPT}"
elif [ -f "${SELECTOR_CKPT}" ]; then
    echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
else
    echo "ERROR: Selector checkpoint not found:"
    echo "  Tried: ${SELECTOR_CKPT_FINAL}"
    echo "  Tried: ${SELECTOR_CKPT}"
    echo "Please run hpsearch/vpl-gp selector experiment (tid=${selector_tid}) first to generate the selector checkpoint."
    exit 1
fi
echo ""

# Find selector config file
SELECTOR_CFG=""
if [ -f "cfg/hpsearch/vpl-gp/phase1_orthogonal_${selector_tid}.yaml" ]; then
    SELECTOR_CFG="cfg/hpsearch/vpl-gp/phase1_orthogonal_${selector_tid}.yaml"
elif [ -f "cfg/hpsearch/vpl-gp/phase2_vpl_core_${selector_tid}.yaml" ]; then
    SELECTOR_CFG="cfg/hpsearch/vpl-gp/phase2_vpl_core_${selector_tid}.yaml"
elif [ -f "cfg/hpsearch/vpl-gp/phase3_lr_${selector_tid}.yaml" ]; then
    SELECTOR_CFG="cfg/hpsearch/vpl-gp/phase3_lr_${selector_tid}.yaml"
elif [ -f "cfg/hpsearch/vpl-gp/phase4_combined_${selector_tid}.yaml" ]; then
    SELECTOR_CFG="cfg/hpsearch/vpl-gp/phase4_combined_${selector_tid}.yaml"
else
    echo "ERROR: Selector config file not found for tid=${selector_tid}"
    exit 1
fi

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/hpsearch/vpl-gp-rl/hrl_${tid}.yaml \
    --selector-cfg-file ${SELECTOR_CFG} \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_vplgp_ortho_${tid}.ckpt \
    expname "vplgp_hrl_ortho_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "VPL-GP HRL training started (task ID: ${tid})"
echo "Config: cfg/hpsearch/vpl-gp-rl/hrl_${tid}.yaml"
echo "Selector checkpoint: ${selector_tid} (hpsearch ${selector_tid})"
echo "Selector config: ${SELECTOR_CFG}"
echo "Hyperparameters: batch_size=1, lr=0.0001, grad_accum_step=4, reward_coeff=0.1"
echo "GPU: specified in config file"
echo "WandB project: fvpl-rl (separate from selector experiments)"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
