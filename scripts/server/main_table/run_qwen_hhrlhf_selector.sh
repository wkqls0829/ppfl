#!/bin/bash
# Run 4 selector experiments on Qwen2-0.5B + HH-RLHF (GPUs 3-6)
# TID 10000: FedBiscuit | TID 10001: FedVPL | TID 10002: VPL-GP (no ortho) | TID 10003: FedVPA-GP (full)
# Paper Table 3 hyperparameters

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

echo "Project root: $PROJECT_ROOT"
echo "Starting 4 selector experiments (Qwen2-0.5B + HH-RLHF)..."

# TID 10000: FedBiscuit (GPU 3)
tid=10000
CUDA_VISIBLE_DEVICES=3 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedbiscuit_10000.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_fedbiscuit_t${tid}.ckpt" \
    expname "test_fedbiscuit_qwen_t${tid}" \
    > outputs/${tid}.log 2>&1 &
echo "Started TID ${tid} (FedBiscuit) on GPU 3, PID=$!"

# TID 10001: FedVPL (GPU 4)
tid=10001
CUDA_VISIBLE_DEVICES=4 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpl_10001.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_fedvpl_t${tid}.ckpt" \
    expname "test_fedvpl_qwen_t${tid}" \
    > outputs/${tid}.log 2>&1 &
echo "Started TID ${tid} (FedVPL) on GPU 4, PID=$!"

# TID 10002: VPL-GP no ortho (GPU 5)
tid=10002
CUDA_VISIBLE_DEVICES=5 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/vplgp_no_ortho_10002.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_vplgp_no_ortho_t${tid}.ckpt" \
    expname "test_vplgp_no_ortho_qwen_t${tid}" \
    > outputs/${tid}.log 2>&1 &
echo "Started TID ${tid} (VPL-GP no ortho) on GPU 5, PID=$!"

# TID 10003: FedVPA-GP full (GPU 6)
tid=10003
CUDA_VISIBLE_DEVICES=6 nohup python -u federatedscope/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/fedvpagp_10003.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_choice_qwen2_fedvpagp_t${tid}.ckpt" \
    expname "test_fedvpagp_qwen_t${tid}" \
    > outputs/${tid}.log 2>&1 &
echo "Started TID ${tid} (FedVPA-GP full) on GPU 6, PID=$!"

echo ""
echo "All 4 experiments launched. Monitor with:"
echo "  tail -f outputs/10000.log  # FedBiscuit"
echo "  tail -f outputs/10001.log  # FedVPL"
echo "  tail -f outputs/10002.log  # VPL-GP (no ortho)"
echo "  tail -f outputs/10003.log  # FedVPA-GP (full)"
