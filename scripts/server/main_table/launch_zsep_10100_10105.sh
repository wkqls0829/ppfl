#!/bin/bash
# Z-separation experiments: 3 variants × 2 max_logvar settings
# 10100-10102: max_logvar=-4.0 on GPUs 0,1,2
# 10103-10105: max_logvar=-5.0 on GPUs 3,4,5

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

echo "Launching 6 z-separation experiments..."

for tid_cfg in \
    "10100 cfg/main_table/qwen_hhrlhf/fedvpagp_relaxed_kl_10100.yaml hhrl_choice_qwen2_fedvpagp_relaxed_kl test_fedvpagp_relaxed_kl_qwen" \
    "10101 cfg/main_table/qwen_hhrlhf/fedvpagp_deep_proj_dropout_10101.yaml hhrl_choice_qwen2_fedvpagp_deep_proj_dropout test_fedvpagp_deep_proj_dropout_qwen" \
    "10102 cfg/main_table/qwen_hhrlhf/fedvpagp_combined_maxkl_10102.yaml hhrl_choice_qwen2_fedvpagp_combined_maxkl test_fedvpagp_combined_maxkl_qwen" \
    "10103 cfg/main_table/qwen_hhrlhf/fedvpagp_relaxed_kl_10103.yaml hhrl_choice_qwen2_fedvpagp_relaxed_kl test_fedvpagp_relaxed_kl_qwen" \
    "10104 cfg/main_table/qwen_hhrlhf/fedvpagp_deep_proj_dropout_10104.yaml hhrl_choice_qwen2_fedvpagp_deep_proj_dropout test_fedvpagp_deep_proj_dropout_qwen" \
    "10105 cfg/main_table/qwen_hhrlhf/fedvpagp_combined_maxkl_10105.yaml hhrl_choice_qwen2_fedvpagp_combined_maxkl test_fedvpagp_combined_maxkl_qwen"; do

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

echo "All 6 launched. Monitor: tail -f outputs/1010{0,1,2,3,4,5}.log"
