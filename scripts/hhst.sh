#!/bin/bash

num_client=30
data_path=~/dplora/news/data/30/1
data_name=news
num_rounds=100
client_epochs=1
model=FacebookAI/roberta-base #google-bert/bert-base-cased
mode=ttlora
projection_type=global_mag #BA_mag
learning_rate=5e-4

tid=10000
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg cfg/gemma_hhrl.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_${tid}.ckpt \
    > outputs/${tid}.log 2>&1 &
