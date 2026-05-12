#!/bin/bash
# Imbalanced split experiments: 30:70 and 70:30
# FedBiscuit + FedVPA-GP on GPUs 1-4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

mkdir -p outputs

for tid_cfg in \
    "30000 cfg/main_table/qwen_hhrlhf/fedbiscuit_3070_30000.yaml" \
    "30001 cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_3070_30001.yaml" \
    "30002 cfg/main_table/qwen_hhrlhf/fedbiscuit_7030_30002.yaml" \
    "30003 cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_7030_30003.yaml"; do

    tid=$(echo $tid_cfg | awk '{print $1}')
    cfg=$(echo $tid_cfg | awk '{print $2}')

    nohup python3 -u federatedscope/main.py \
        --cfg "$cfg" \
        > outputs/${tid}.log 2>&1 &
    echo "Started TID ${tid} PID=$!"
done

echo "All 4 launched on GPUs 1-4."
