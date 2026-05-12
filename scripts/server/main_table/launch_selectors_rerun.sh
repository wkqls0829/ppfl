#!/bin/bash
# Re-run selectors with latent_projection/z_to_embedding excluded from FedAvg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

mkdir -p outputs

# 10211: hybrid + adapter dropout (main) on GPU 0
nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adapter_drop_10211.yaml \
    device 0 \
    > outputs/10211.log 2>&1 &
echo "10211 (adrop) PID=$!"

# 10212: hybrid + adrop + kmeans on GPU 1
nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    device 1 \
    > outputs/10212.log 2>&1 &
echo "10212 (kmeans) PID=$!"

# 10209: hybrid + add (no adrop, control) on GPU 2
nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_add_10209.yaml \
    device 2 \
    > outputs/10209.log 2>&1 &
echo "10209 (no adrop) PID=$!"

# 20002: UltraFeedback hybrid + adrop on GPU 3
nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_ultrafeedback/fedvpagp_z_hybrid_adrop_20002.yaml \
    device 3 \
    > outputs/20002.log 2>&1 &
echo "20002 (UF) PID=$!"

# 20102: 30:70 hybrid + adrop on GPU 4
nohup python3 -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_3070_20102.yaml \
    device 4 \
    > outputs/20102.log 2>&1 &
echo "20102 (30:70) PID=$!"

echo "5 selectors launched on GPUs 0-4. GPU 5,6 free."
