#!/bin/bash
# RL comparison: 5 experiments
# Adjust device in YAML configs before running, or override via CLI

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

echo "Launching 5 RL comparison experiments..."

for tid_cfg in \
    "11200 cfg/main_table/qwen_hhrlhf/hrl_comparison_fedbiscuit_11200.yaml" \
    "11201 cfg/main_table/qwen_hhrlhf/hrl_comparison_fedvpl_11201.yaml" \
    "11202 cfg/main_table/qwen_hhrlhf/hrl_comparison_kl_only_11202.yaml" \
    "11203 cfg/main_table/qwen_hhrlhf/hrl_comparison_kl_ortho_11203.yaml" \
    "11117 cfg/main_table/qwen_hhrlhf/hrl_comparison_ortho_only_11117.yaml"; do

    tid=$(echo $tid_cfg | awk '{print $1}')
    cfg=$(echo $tid_cfg | awk '{print $2}')

    nohup python3 -u federatedscope/main.py \
        --cfg "$cfg" \
        > outputs/${tid}.log 2>&1 &
    echo "Started TID ${tid} PID=$!"
done

echo "All 5 launched."
