#!/bin/bash
# Main comparison: FedBiscuit vs FedVPL vs FedVPA-GP(KL only) vs FedVPA-GP(KL+ortho)
# GPUs 1-4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

echo "Launching 4 comparison experiments..."

for tid_cfg in \
    "10200 cfg/main_table/qwen_hhrlhf/fedvpagp_comparison_fedbiscuit_10200.yaml" \
    "10201 cfg/main_table/qwen_hhrlhf/fedvpagp_comparison_fedvpl_10201.yaml" \
    "10202 cfg/main_table/qwen_hhrlhf/fedvpagp_comparison_kl_only_10202.yaml" \
    "10203 cfg/main_table/qwen_hhrlhf/fedvpagp_comparison_kl_ortho_10203.yaml"; do

    tid=$(echo $tid_cfg | awk '{print $1}')
    cfg=$(echo $tid_cfg | awk '{print $2}')

    nohup python3 -u federatedscope/main.py \
        --cfg "$cfg" \
        > outputs/${tid}.log 2>&1 &
    echo "Started TID ${tid} PID=$!"
done

echo "All 4 launched. Monitor: tail -f outputs/1020{0,1,2,3}.log"
