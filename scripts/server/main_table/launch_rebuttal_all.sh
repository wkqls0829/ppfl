#!/bin/bash
# Rebuttal experiments: imbalanced splits + orthogonal ablation
# All 40 rounds
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
    "30003 cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_7030_30003.yaml" \
    "30100 cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_pull_only_30100.yaml" \
    "30101 cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_orthonorm_only_30101.yaml"; do

    tid=$(echo $tid_cfg | awk '{print $1}')
    cfg=$(echo $tid_cfg | awk '{print $2}')

    nohup python3 -u federatedscope/main.py \
        --cfg "$cfg" \
        > outputs/${tid}.log 2>&1 &
    echo "Started TID ${tid} PID=$!"
done

echo "All 6 launched (GPUs 1-6). 11211 RL on GPU 0 unaffected."
