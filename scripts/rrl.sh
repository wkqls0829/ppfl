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

tid=20001
# export CUDA_LAUNCH_BLOCKING=1 
# export CUDA_VISIBLE_DEVICES=2 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
nohup python -u federatedscope/llm/rlhf/main.py \
    --selector-cfg-file cfg/reward_eval.yaml \
    --cfg cfg/reward_eval_rl.yaml \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_gemma_fedbiscuit_u3.ckpt \
    expname hhrl/rlhf_gemma/hhrl_choice_gemma_fedbiscuit_u3 \
    > outputs/${tid}.log 2>&1 &
