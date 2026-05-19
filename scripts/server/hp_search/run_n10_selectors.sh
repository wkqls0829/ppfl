#!/bin/bash
# N=10 HP search: launch 5 selectors in parallel on GPUs 0-4.
# See documents/HP_SEARCH_PLAN.md and documents/HANDOFF_OTHER_SERVER.md.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure conda env is active. If not invoked from `conda activate biscuit`,
# source it explicitly.
if [[ "${CONDA_DEFAULT_ENV}" != "biscuit" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate biscuit
fi

CFG=cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml
CKPT_DIR=/hdd/hdd3/kjb/checkpoints

mkdir -p outputs

# --- baseline (90200, GPU 0) ----------------------------------------------
nohup python -u federatedscope/main.py \
    --cfg "$CFG" \
    device 0 \
    federate.save_to "$CKPT_DIR/hhrl_choice_qwen2_hpsearch_n10_baseline_t90200.ckpt" \
    expname "hpsearch_n10_baseline_90200" \
    > outputs/90200.log 2>&1 &
echo "Launched 90200 (baseline) on GPU 0, PID=$!"

# --- kl_low (90201, GPU 1) ------------------------------------------------
nohup python -u federatedscope/main.py \
    --cfg "$CFG" \
    device 1 \
    llm.vpl_kl_weight 0.001 \
    federate.save_to "$CKPT_DIR/hhrl_choice_qwen2_hpsearch_n10_kl_low_t90201.ckpt" \
    expname "hpsearch_n10_kl_low_90201" \
    > outputs/90201.log 2>&1 &
echo "Launched 90201 (kl_low) on GPU 1, PID=$!"

# --- kl_high (90202, GPU 2) -----------------------------------------------
nohup python -u federatedscope/main.py \
    --cfg "$CFG" \
    device 2 \
    llm.vpl_kl_weight 0.1 \
    federate.save_to "$CKPT_DIR/hhrl_choice_qwen2_hpsearch_n10_kl_high_t90202.ckpt" \
    expname "hpsearch_n10_kl_high_90202" \
    > outputs/90202.log 2>&1 &
echo "Launched 90202 (kl_high) on GPU 2, PID=$!"

# --- ortho_high (90203, GPU 3) --------------------------------------------
nohup python -u federatedscope/main.py \
    --cfg "$CFG" \
    device 3 \
    llm.vpl_orthogonal_weight 5.0 \
    federate.save_to "$CKPT_DIR/hhrl_choice_qwen2_hpsearch_n10_ortho_high_t90203.ckpt" \
    expname "hpsearch_n10_ortho_high_90203" \
    > outputs/90203.log 2>&1 &
echo "Launched 90203 (ortho_high) on GPU 3, PID=$!"

# --- reward_high selector (90204, GPU 4) ----------------------------------
# reward_coeff is a Stage-2 knob; selector identical to baseline. Run a
# separate selector anyway so all five share the same launch flow, then
# 91204 will reuse THIS checkpoint at Stage 2 with reward_coeff=0.5.
nohup python -u federatedscope/main.py \
    --cfg "$CFG" \
    device 4 \
    federate.save_to "$CKPT_DIR/hhrl_choice_qwen2_hpsearch_n10_reward_high_t90204.ckpt" \
    expname "hpsearch_n10_reward_high_90204" \
    > outputs/90204.log 2>&1 &
echo "Launched 90204 (reward_high) on GPU 4, PID=$!"

echo ""
echo "All 5 selectors launched. Tail logs with:"
echo "  tail -f outputs/9020{0,1,2,3,4}.log"
echo ""
echo "Stage 1 ~5h. Final ckpts: final_hhrl_choice_qwen2_hpsearch_n10_*_t9020*.ckpt"
