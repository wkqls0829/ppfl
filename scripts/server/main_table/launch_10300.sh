#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_fixed_uniform_weights_10300.yaml \
    > outputs/10300.log 2>&1 &
echo "10300 PID=$!"
