#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_add_highlr_10208.yaml \
    > outputs/10208.log 2>&1 &
echo "10208 (add + high LR) PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_add_10209.yaml \
    > outputs/10209.log 2>&1 &
echo "10209 (hybrid + add) PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_concat_10210.yaml \
    > outputs/10210.log 2>&1 &
echo "10210 (hybrid + concat) PID=$!"
