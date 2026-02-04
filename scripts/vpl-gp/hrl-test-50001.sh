#!/bin/bash

# Variational Preference Learning with Gumbel-Softmax Prior (VPL-GP) RL test script
# Test run with gemma-2b model (tid: 50001 RL test)
# Using selector checkpoint from 50001 test experiment

tid=50001
selector_tid=50001  # Selector checkpoint task ID (test experiment)

# GPU is specified in config file (device: 0)
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

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
    echo "Please run selector test experiment (tid=${selector_tid}) first to generate the selector checkpoint."
    exit 1
fi
echo ""

# Find selector config file
SELECTOR_CFG="cfg/vpl-gp/hhst-ortho-50001-test.yaml"
if [ ! -f "${SELECTOR_CFG}" ]; then
    echo "WARNING: Selector config file not found: ${SELECTOR_CFG}"
    echo "Continuing without --selector-cfg-file option..."
    SELECTOR_CFG=""
fi

if [ -n "${SELECTOR_CFG}" ]; then
    nohup python -u federatedscope/llm/rlhf/main.py \
        --cfg cfg/vpl-gp/hrl-test-50001.yaml \
        --selector-cfg-file ${SELECTOR_CFG} \
        federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_vplgp_ortho_${tid}_test.ckpt \
        expname "vplgp_hrl_ortho_t${tid}_test" \
        > outputs/${tid}_rl_test.log 2>&1 &
else
    nohup python -u federatedscope/llm/rlhf/main.py \
        --cfg cfg/vpl-gp/hrl-test-50001.yaml \
        federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_vplgp_ortho_${tid}_test.ckpt \
        expname "vplgp_hrl_ortho_t${tid}_test" \
        > outputs/${tid}_rl_test.log 2>&1 &
fi

echo "VPL-GP HRL test training started (task ID: ${tid} RL test)"
echo "Config: cfg/vpl-gp/hrl-test-50001.yaml"
echo "Model: google/gemma-2b@huggingface_llm"
echo "Selector checkpoint: ${selector_tid} (test experiment)"
echo "Selector config: ${SELECTOR_CFG}"
echo "Test configuration: 5 rounds, 100 train samples, 50 test samples"
echo "Hyperparameters: batch_size=1, lr=0.0001, grad_accum_step=4, reward_coeff=0.1"
echo "GPU: 0 (specified in config file)"
echo "WandB project: fvpl-rl"
echo "Log file: outputs/${tid}_rl_test.log"
echo "Monitor with: tail -f outputs/${tid}_rl_test.log"
