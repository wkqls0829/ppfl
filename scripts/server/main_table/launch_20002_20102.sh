#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_ultrafeedback/fedvpagp_z_hybrid_adrop_20002.yaml \
    > outputs/20002.log 2>&1 &
echo "20002 (UltraFeedback hybrid) PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_3070_20102.yaml \
    > outputs/20102.log 2>&1 &
echo "20102 (HH-RLHF 30:70 hybrid) PID=$!"
