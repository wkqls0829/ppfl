#!/bin/bash
# Tier-2 N=10 HP search: 8 RL runs in parallel on GPUs 0-7.
# All reuse the reward_high tier-1 selector ckpt; only RL stage runs.
# Group A (92204-92206): 3 seeds of reward_high @ LR=5e-5.
# Group B (93000-93004): fairness checks, rc sweep, 100-round probes.
# See documents/HP_SEARCH_RESULTS_N10.md for the rationale.

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
SEL_CKPT="$CKPT_DIR/final_hhrl_choice_qwen2_hpsearch_n10_reward_high_t90204.ckpt"

mkdir -p outputs

if [[ ! -f "$SEL_CKPT" ]]; then
    echo "ERROR: tier-1 selector ckpt missing: $SEL_CKPT" >&2
    exit 1
fi

# ============================================================
# Group A — 3 seeds of reward_high @ LR=5e-5 (GPUs 0-2)
# ============================================================

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 0 \
    seed 1 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.5 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_tier2_reward_high_lr5e5_seed1_t92204.ckpt" \
    expname "hpsearch_n10_t2_reward_high_lr5e5_s1_92204" \
    > outputs/92204.log 2>&1 &
echo "Launched 92204 (reward_high LR=5e-5 seed=1) on GPU 0, PID=$!"

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 1 \
    seed 2 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.5 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_tier2_reward_high_lr5e5_seed2_t92205.ckpt" \
    expname "hpsearch_n10_t2_reward_high_lr5e5_s2_92205" \
    > outputs/92205.log 2>&1 &
echo "Launched 92205 (reward_high LR=5e-5 seed=2) on GPU 1, PID=$!"

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 2 \
    seed 3 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.5 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_tier2_reward_high_lr5e5_seed3_t92206.ckpt" \
    expname "hpsearch_n10_t2_reward_high_lr5e5_s3_92206" \
    > outputs/92206.log 2>&1 &
echo "Launched 92206 (reward_high LR=5e-5 seed=3) on GPU 2, PID=$!"

# ============================================================
# Group B — Fairness checks + rc sweep + 100-round probes (GPUs 3-7)
# ============================================================

# 93000 — baseline at LR=5e-5 (fairness check vs reward_high@LR=5e-5)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 3 \
    seed 0 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.1 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_tier2_baseline_lr5e5_t93000.ckpt" \
    expname "hpsearch_n10_t2_baseline_lr5e5_93000" \
    > outputs/93000.log 2>&1 &
echo "Launched 93000 (baseline LR=5e-5) on GPU 3, PID=$!"

# 93001 — reward_high with rc=1.0, LR=5e-5 (push the winning knob further)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 4 \
    seed 0 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 1.0 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_tier2_rc1.0_lr5e5_t93001.ckpt" \
    expname "hpsearch_n10_t2_rc1.0_lr5e5_93001" \
    > outputs/93001.log 2>&1 &
echo "Launched 93001 (rc=1.0 LR=5e-5) on GPU 4, PID=$!"

# 93002 — reward_high, LR=5e-5, 100 rounds (does the winner keep improving?)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 5 \
    seed 0 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.5 \
    federate.total_round_num 100 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_tier2_reward_high_lr5e5_r100_t93002.ckpt" \
    expname "hpsearch_n10_t2_reward_high_lr5e5_r100_93002" \
    > outputs/93002.log 2>&1 &
echo "Launched 93002 (reward_high LR=5e-5 100 rounds) on GPU 5, PID=$!"

# 93003 — baseline at default LR=1e-5, 100 rounds (does baseline catch up?)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 6 \
    seed 0 \
    train.optimizer.lr 1e-5 \
    llm.reward_coeff 0.1 \
    federate.total_round_num 100 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_tier2_baseline_r100_t93003.ckpt" \
    expname "hpsearch_n10_t2_baseline_r100_93003" \
    > outputs/93003.log 2>&1 &
echo "Launched 93003 (baseline 100 rounds) on GPU 6, PID=$!"

# 93004 — reward_high with rc=0.3, LR=5e-5 (middle of rc sweep)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device 7 \
    seed 0 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.3 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n10_tier2_rc0.3_lr5e5_t93004.ckpt" \
    expname "hpsearch_n10_t2_rc0.3_lr5e5_93004" \
    > outputs/93004.log 2>&1 &
echo "Launched 93004 (rc=0.3 LR=5e-5) on GPU 7, PID=$!"

echo ""
echo "All 8 tier-2 RL stages launched. Tail logs with:"
echo "  tail -f outputs/{92204,92205,92206,93000,93001,93002,93003,93004}.log"
echo ""
echo "Expected wall-clock: ~12-15h for 50-round runs, ~24h for the two"
echo "100-round runs (93002, 93003). All run in parallel."
