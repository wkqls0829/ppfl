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

tid=10001
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
nohup python -u federatedscope/main.py \
    --cfg cfg/reward_eval.yaml \
    > outputs/${tid}.log 2>&1 &
