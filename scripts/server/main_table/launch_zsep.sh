#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

for tid_cfg in \
    "10013 cfg/main_table/qwen_hhrlhf/fedvpagp_relaxed_kl_10013.yaml hhrl_choice_qwen2_fedvpagp_relaxed_kl test_fedvpagp_relaxed_kl_qwen" \
    "10014 cfg/main_table/qwen_hhrlhf/fedvpagp_deep_proj_dropout_10014.yaml hhrl_choice_qwen2_fedvpagp_deep_proj_dropout test_fedvpagp_deep_proj_dropout_qwen" \
    "10015 cfg/main_table/qwen_hhrlhf/fedvpagp_combined_10015.yaml hhrl_choice_qwen2_fedvpagp_combined test_fedvpagp_combined_qwen" \
    "10016 cfg/main_table/qwen_hhrlhf/fedvpagp_combined_maxkl_10016.yaml hhrl_choice_qwen2_fedvpagp_combined_maxkl test_fedvpagp_combined_maxkl_qwen"; do

    tid=$(echo $tid_cfg | awk '{print $1}')
    cfg=$(echo $tid_cfg | awk '{print $2}')
    ckpt_name=$(echo $tid_cfg | awk '{print $3}')
    expname=$(echo $tid_cfg | awk '{print $4}')

    nohup python3 -u federatedscope/main.py \
        --cfg "$cfg" \
        federate.save_to "/hdd/hdd3/kjb/checkpoints/${ckpt_name}_t${tid}.ckpt" \
        expname "${expname}_t${tid}" \
        > outputs/${tid}.log 2>&1 &
    echo "Started TID ${tid} on PID=$!"
done
