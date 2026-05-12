#!/bin/bash
# UltraFeedback selector experiments (4 categories, 40 clients)
# 20000: FedBiscuit on GPU 0
# 20001: FedVPA-GP (KL+ortho, M=4) on GPU 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

mkdir -p outputs

echo "Launching UltraFeedback selector experiments..."

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_ultrafeedback/fedbiscuit_20000.yaml \
    > outputs/20000.log 2>&1 &
echo "20000 (FedBiscuit) PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_ultrafeedback/fedvpagp_kl_ortho_20001.yaml \
    > outputs/20001.log 2>&1 &
echo "20001 (FedVPA-GP) PID=$!"

echo "Monitor: tail -f outputs/2000{0,1}.log"
