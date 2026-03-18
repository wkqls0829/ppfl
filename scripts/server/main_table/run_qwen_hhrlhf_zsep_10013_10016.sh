#!/bin/bash
# Run 4 z-separation fix experiments on Qwen2-0.5B + HH-RLHF (GPUs 3-6)
# TID 10013: Relaxed logvar + stronger KL (beta=0.5)
# TID 10014: Deep projection + 50% logit dropout
# TID 10015: Combined (relaxed + deep + dropout + beta=0.5)
# TID 10016: Combined + max KL (beta=1.0)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

echo "Project root: $PROJECT_ROOT"
echo "Starting 4 z-separation experiments (Qwen2-0.5B + HH-RLHF)..."

# TID 10013: Relaxed logvar + stronger KL (GPU 3)
tid=10013
CUDA_VISIBLE_DEVICES=3 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_relaxed_kl_10013.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_fedvpagp_relaxed_kl_t${tid}.ckpt" \
    expname "test_fedvpagp_relaxed_kl_qwen_t${tid}" \
    > outputs/${tid}.log 2>&1 &
echo "Started TID ${tid} (relaxed logvar + KL=0.5) on GPU 3, PID=$!"

# TID 10014: Deep projection + logit dropout (GPU 4)
tid=10014
CUDA_VISIBLE_DEVICES=4 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_deep_proj_dropout_10014.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_fedvpagp_deep_proj_dropout_t${tid}.ckpt" \
    expname "test_fedvpagp_deep_proj_dropout_qwen_t${tid}" \
    > outputs/${tid}.log 2>&1 &
echo "Started TID ${tid} (deep projection + 50% logit dropout) on GPU 4, PID=$!"

# TID 10015: Combined (GPU 5)
tid=10015
CUDA_VISIBLE_DEVICES=5 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_combined_10015.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_fedvpagp_combined_t${tid}.ckpt" \
    expname "test_fedvpagp_combined_qwen_t${tid}" \
    > outputs/${tid}.log 2>&1 &
echo "Started TID ${tid} (combined: relaxed+deep+dropout+KL=0.5) on GPU 5, PID=$!"

# TID 10016: Combined + max KL (GPU 6)
tid=10016
CUDA_VISIBLE_DEVICES=6 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_combined_maxkl_10016.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_fedvpagp_combined_maxkl_t${tid}.ckpt" \
    expname "test_fedvpagp_combined_maxkl_qwen_t${tid}" \
    > outputs/${tid}.log 2>&1 &
echo "Started TID ${tid} (combined + KL=1.0) on GPU 6, PID=$!"

echo ""
echo "All 4 experiments launched. Monitor with:"
echo "  tail -f outputs/10013.log  # Relaxed logvar + KL=0.5"
echo "  tail -f outputs/10014.log  # Deep projection + logit dropout"
echo "  tail -f outputs/10015.log  # Combined (KL=0.5)"
echo "  tail -f outputs/10016.log  # Combined + max KL=1.0"
