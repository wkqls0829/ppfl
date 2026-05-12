#!/bin/bash
# UltraFeedback selector experiments (4 categories, 40 clients, 40 rounds)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

mkdir -p outputs

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_ultrafeedback/fedbiscuit_30200.yaml \
    > outputs/30200.log 2>&1 &
echo "30200 (FedBiscuit UF) PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_ultrafeedback/fedvpagp_z_hybrid_adrop_30201.yaml \
    > outputs/30201.log 2>&1 &
echo "30201 (FedVPA-GP UF) PID=$!"
