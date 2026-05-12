#!/bin/bash
# 30:70 imbalanced split experiments (3 harmless, 7 helpful)
# 20100: FedBiscuit on GPU 2
# 20101: FedVPA-GP (KL+ortho) on GPU 3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

mkdir -p outputs

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedbiscuit_3070_20100.yaml \
    > outputs/20100.log 2>&1 &
echo "20100 (FedBiscuit 30:70) PID=$!"

nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_kl_ortho_3070_20101.yaml \
    > outputs/20101.log 2>&1 &
echo "20101 (FedVPA-GP 30:70) PID=$!"
