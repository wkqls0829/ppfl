#!/bin/bash
# N=10 HP search Stage 2: 5 RL runs in parallel on GPUs 0-4.
# Uses federatedscope/llm/rlhf/main.py (NOT main.py) with --selector-cfg-file.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "${CONDA_DEFAULT_ENV}" != "biscuit" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate biscuit
fi

RL_CFG=cfg/main_table/qwen_hhrlhf/hrl_z_hybrid_adrop_kmeans_11212.yaml
SEL_CFG=cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml
CKPT_DIR=/hdd/hdd3/kjb/checkpoints

mkdir -p outputs

# --- baseline RL (91200, GPU 0) -------------------------------------------
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 0 \
    llm.rlhf_selector_checkpoint "$CKPT_DIR/final_hhrl_choice_qwen2_hpsearch_n10_baseline_t90200.ckpt" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_baseline_t91200.ckpt" \
    expname "hpsearch_n10_baseline_rl_91200" \
    > outputs/91200.log 2>&1 &
echo "Launched 91200 (baseline) on GPU 0, PID=$!"

# --- kl_low RL (91201, GPU 1) ---------------------------------------------
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 1 \
    llm.vpl_kl_weight 0.001 \
    llm.rlhf_selector_checkpoint "$CKPT_DIR/final_hhrl_choice_qwen2_hpsearch_n10_kl_low_t90201.ckpt" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_kl_low_t91201.ckpt" \
    expname "hpsearch_n10_kl_low_rl_91201" \
    > outputs/91201.log 2>&1 &
echo "Launched 91201 (kl_low) on GPU 1, PID=$!"

# --- kl_high RL (91202, GPU 2) --------------------------------------------
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 2 \
    llm.vpl_kl_weight 0.1 \
    llm.rlhf_selector_checkpoint "$CKPT_DIR/final_hhrl_choice_qwen2_hpsearch_n10_kl_high_t90202.ckpt" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_kl_high_t91202.ckpt" \
    expname "hpsearch_n10_kl_high_rl_91202" \
    > outputs/91202.log 2>&1 &
echo "Launched 91202 (kl_high) on GPU 2, PID=$!"

# --- ortho_high RL (91203, GPU 3) -----------------------------------------
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 3 \
    llm.vpl_orthogonal_weight 5.0 \
    llm.rlhf_selector_checkpoint "$CKPT_DIR/final_hhrl_choice_qwen2_hpsearch_n10_ortho_high_t90203.ckpt" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_ortho_high_t91203.ckpt" \
    expname "hpsearch_n10_ortho_high_rl_91203" \
    > outputs/91203.log 2>&1 &
echo "Launched 91203 (ortho_high) on GPU 3, PID=$!"

# --- reward_high RL (91204, GPU 4) ----------------------------------------
# Uses the dedicated reward_high selector ckpt (90204). reward_coeff
# applies to Stage 2 only.
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 4 \
    llm.reward_coeff 0.5 \
    llm.rlhf_selector_checkpoint "$CKPT_DIR/final_hhrl_choice_qwen2_hpsearch_n10_reward_high_t90204.ckpt" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_reward_high_t91204.ckpt" \
    expname "hpsearch_n10_reward_high_rl_91204" \
    > outputs/91204.log 2>&1 &
echo "Launched 91204 (reward_high) on GPU 4, PID=$!"

echo ""
echo "All 5 RL stages launched. Tail logs with:"
echo "  tail -f outputs/9120{0,1,2,3,4}.log"
echo ""
echo "Stage 2 ~6h. Final ckpts: 50_hhrl_rlhf_qwen2_hpsearch_n10_*_t9120*.ckpt"
