#!/bin/bash
# 10204: Full dropout (logit_dropout=1.0) on GPU 0
# 10303: Pull loss only ablation on GPU 2
# 10304: Orthonorm only ablation on GPU 5

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

mkdir -p outputs

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_full_dropout_10204.yaml \
    > outputs/10204.log 2>&1 &
echo "10204 (full dropout) PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_pull_only_10303.yaml \
    > outputs/10303.log 2>&1 &
echo "10303 (pull only) PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_orthonorm_only_10304.yaml \
    > outputs/10304.log 2>&1 &
echo "10304 (orthonorm only) PID=$!"
