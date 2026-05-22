#!/bin/bash
# N=10 Main Table — retrain the 3 VPL selectors with z-conditioning
# (commit 3b7e10e). The first attempt's selectors lacked
# vpl_use_z_embedding, so their variational encoder was never
# co-adapted to embedding-injection; RL collapsed. FedBiscuit selector
# (10200) is non-VPL and is NOT retrained.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

if [[ "${CONDA_DEFAULT_ENV}" != "biscuit" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate biscuit
fi

CFG_DIR=cfg/main_table/qwen_hhrlhf
CKPT_DIR=/hdd/hdd3/kjb/checkpoints

mkdir -p outputs

# --- FedVPL selector (10201, GPU 1) ---------------------------------------
nohup python -u federatedscope/main.py \
    --cfg "$CFG_DIR/fedvpagp_comparison_fedvpl_10201.yaml" \
    device 1 \
    federate.save_to "$CKPT_DIR/hhrl_choice_qwen2_mt_n10_fedvpl_t10201.ckpt" \
    expname "mt_n10_fedvpl_sel_v2_10201" \
    > outputs/mt_n10_10201.log 2>&1 &
echo "Launched 10201 (FedVPL selector v2) on GPU 1, PID=$!"

# --- FedVPA-GP kl_only selector (10202, GPU 2) ---------------------------
nohup python -u federatedscope/main.py \
    --cfg "$CFG_DIR/fedvpagp_comparison_kl_only_10202.yaml" \
    device 2 \
    federate.save_to "$CKPT_DIR/hhrl_choice_qwen2_mt_n10_kl_only_t10202.ckpt" \
    expname "mt_n10_kl_only_sel_v2_10202" \
    > outputs/mt_n10_10202.log 2>&1 &
echo "Launched 10202 (FedVPA-GP kl_only selector v2) on GPU 2, PID=$!"

# --- FedVPA-GP full (kl_ortho) selector (10203, GPU 3) -------------------
nohup python -u federatedscope/main.py \
    --cfg "$CFG_DIR/fedvpagp_comparison_kl_ortho_10203.yaml" \
    device 3 \
    federate.save_to "$CKPT_DIR/hhrl_choice_qwen2_mt_n10_kl_ortho_t10203.ckpt" \
    expname "mt_n10_kl_ortho_sel_v2_10203" \
    > outputs/mt_n10_10203.log 2>&1 &
echo "Launched 10203 (FedVPA-GP full kl_ortho selector v2) on GPU 3, PID=$!"

echo ""
echo "3 VPL selectors relaunched. Tail logs:"
echo "  tail -f outputs/mt_n10_1020{1,2,3}.log"
echo ""
echo "Wall-clock ~3-5h. Then launch_mt_n10_rl.sh for Stage 2."
