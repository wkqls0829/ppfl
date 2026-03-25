#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_comparison_fedvpl_11201.yaml \
    > outputs/11201.log 2>&1 &
echo "11201 PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_comparison_kl_ortho_11203.yaml \
    > outputs/11203.log 2>&1 &
echo "11203 PID=$!"
