#!/bin/bash

# FedBiscuit RL training script (no variational elements)
# HRL (hh-rlhf) version with reward model evaluation
# Test run: 1 round eval, 10 rounds only
# Single GPU mode: GPU 4 (specified in config file)

tid=21000
selector_tid=20023  # Selector checkpoint task ID (fedbiscuit 20023)

# GPU is specified in config file (device: 4)
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false  # Disable tokenizers parallelism to avoid fork warnings 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if selector checkpoint exists (try final checkpoint first, then regular)
SELECTOR_CKPT_FINAL="/hdd/hdd3/kjb/checkpoints/final_hhrl_choice_gemma_fedbiscuit_u3_${selector_tid}.ckpt"
SELECTOR_CKPT="/hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_${selector_tid}.ckpt"

if [ -f "${SELECTOR_CKPT_FINAL}" ]; then
    SELECTOR_CKPT="${SELECTOR_CKPT_FINAL}"
    echo "✓ Using final selector checkpoint: ${SELECTOR_CKPT}"
elif [ -f "${SELECTOR_CKPT}" ]; then
    echo "✓ Selector checkpoint found: ${SELECTOR_CKPT}"
else
    echo "ERROR: Selector checkpoint not found:"
    echo "  Tried: ${SELECTOR_CKPT_FINAL}"
    echo "  Tried: ${SELECTOR_CKPT}"
    echo "Please run fedbiscuit/hhst-${selector_tid}.sh first to generate the selector checkpoint (tid=${selector_tid})."
    exit 1
fi
echo ""

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/fedbiscuit/hrl-21000.yaml \
    --selector-cfg-file cfg/fedbiscuit/hhst-${selector_tid}.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_fedbiscuit_${tid}.ckpt \
    expname "fedbiscuit_hrl_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "FedBiscuit HRL training started (task ID: ${tid})"
echo "Config: cfg/fedbiscuit/hrl-21000.yaml (no variational elements, FedBiscuit algorithm, test run)"
echo "Selector checkpoint: ${selector_tid} (fedbiscuit 20023)"
echo "Hyperparameters: batch_size=1, lr=0.0001, grad_accum_step=4, reward_coeff=0.1"
echo "Total rounds: 10 (test run), Eval freq: 1 (every round)"
echo "GPU: 4 (specified in config file: device: 4)"
echo "WandB project: fvpl-rl (separate from selector experiments)"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
