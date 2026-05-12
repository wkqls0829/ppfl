#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_concat_10206.yaml \
    > outputs/10206.log 2>&1 &
echo "10206 (z-concat prefix) PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_embedding_no_ortho_10207.yaml \
    > outputs/10207.log 2>&1 &
echo "10207 (z-add no ortho) PID=$!"
