#!/bin/bash
# KL weight ablation: 6 experiments on GPUs 0-5
# Baseline: deep_proj + logit_dropout + max_logvar=-4.0 + manual_labels
# Varying: vpl_kl_weight = {0.1, 0.05, 0.01, 0.001, 0.0, 0.2}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

echo "Launching 6 KL ablation experiments..."

for tid_cfg in \
    "10110 cfg/main_table/qwen_hhrlhf/fedvpagp_kl_ablation_10110.yaml hhrl_choice_qwen2_fedvpagp_kl_ablation test_fedvpagp_kl_ablation_qwen" \
    "10111 cfg/main_table/qwen_hhrlhf/fedvpagp_kl_ablation_10111.yaml hhrl_choice_qwen2_fedvpagp_kl_ablation test_fedvpagp_kl_ablation_qwen" \
    "10112 cfg/main_table/qwen_hhrlhf/fedvpagp_kl_ablation_10112.yaml hhrl_choice_qwen2_fedvpagp_kl_ablation test_fedvpagp_kl_ablation_qwen" \
    "10113 cfg/main_table/qwen_hhrlhf/fedvpagp_kl_ablation_10113.yaml hhrl_choice_qwen2_fedvpagp_kl_ablation test_fedvpagp_kl_ablation_qwen" \
    "10114 cfg/main_table/qwen_hhrlhf/fedvpagp_kl_ablation_10114.yaml hhrl_choice_qwen2_fedvpagp_kl_ablation test_fedvpagp_kl_ablation_qwen" \
    "10115 cfg/main_table/qwen_hhrlhf/fedvpagp_kl_ablation_10115.yaml hhrl_choice_qwen2_fedvpagp_kl_ablation test_fedvpagp_kl_ablation_qwen"; do

    tid=$(echo $tid_cfg | awk '{print $1}')
    cfg=$(echo $tid_cfg | awk '{print $2}')
    ckpt_name=$(echo $tid_cfg | awk '{print $3}')
    expname=$(echo $tid_cfg | awk '{print $4}')

    nohup python3 -u federatedscope/main.py \
        --cfg "$cfg" \
        federate.save_to "/hdd/hdd3/kjb/checkpoints/${ckpt_name}_t${tid}.ckpt" \
        expname "${expname}_t${tid}" \
        > outputs/${tid}.log 2>&1 &
    echo "Started TID ${tid} PID=$!"
done

echo "All 6 launched. Monitor: tail -f outputs/1011{0,1,2,3,4,5}.log"
