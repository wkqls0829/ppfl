#!/bin/bash

# Federated Direct Preference Optimization (FedDPO) RL training script
# HRL (hh-rlhf) version with DPO training - 50 clients (scaled version)
# FedDPO is pure DPO for federated learning (baseline)
# WITHOUT VPL, GP prior, or orthogonal loss
# Direct Preference Optimization: https://arxiv.org/abs/2305.18290
# Single GPU mode: GPU 7 (specified in config file)

tid=11124
# GPU is specified in config file (device: 7)
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export TOKENIZERS_PARALLELISM=false

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# FedDPO does NOT require a selector checkpoint (unlike FedVPL/FedBiscuit)
# DPO directly learns preferences without a separate selector model
# However, rlhf/main.py requires --selector-cfg-file argument
# We'll use a dummy config file or the same config file for selector_cfg
# Since FedDPO doesn't use selector, we can pass the same config file

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/feddpo/hrl-11124.yaml \
    --selector-cfg-file cfg/feddpo/hrl-11124.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_feddpo_${tid}.ckpt \
    expname "feddpo_hrl_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "FedDPO HRL training started (task ID: ${tid})"
echo "Config: cfg/feddpo/hrl-11124.yaml (pure DPO, no VPL, no selector required)"
echo "Hyperparameters: batch_size=1, lr=0.0001, grad_accum_step=4, reward_coeff=0.1"
echo "GPU: 7 (specified in config file: device: 7)"
echo "Client num: 50 (scaled version, same as 20124)"
echo "WandB project: fvpl-rl (same as other RL experiments)"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
echo ""
echo "Note: FedDPO does NOT require a separate selector checkpoint."
echo "      DPO directly learns preferences from pairwise comparisons."
