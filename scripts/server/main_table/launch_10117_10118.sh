#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_no_kl_ortho_only_10117.yaml \
    > outputs/10117.log 2>&1 &
echo "10117 PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_kl_ortho_10118.yaml \
    > outputs/10118.log 2>&1 &
echo "10118 PID=$!"
