#!/bin/bash
# Tier-2 N=100 HP search: up to 8 RL runs in parallel.
# All reuse the reward_high tier-1 selector ckpt; only RL stage runs.
# Group A (92104-92106): 3 seeds of reward_high @ LR=5e-5.
# Group B (93100-93104): fairness checks, rc sweep, 100-round probes.
# Mirrors documents/HP_SEARCH_RESULTS_N10.md's tier-2 plan for N=100.
#
# Default GPU mapping uses 0..7. If GPUs 0/1 are in use by another
# tenant, set HP_TIER2_GPUS="2,3,4,5,6" (or any 8-element list)
# before invoking the script and only the first N indices will be
# assigned; runs whose GPU is unset are skipped.

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

RL_CFG=cfg/main_table/qwen_hhrlhf/hrl_z_hybrid_adrop_kmeans_11212.yaml
SEL_CFG=cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml
CKPT_DIR=/hdd/hdd3/kjb/checkpoints
SEL_CKPT="$CKPT_DIR/final_hhrl_choice_qwen2_hpsearch_n100_reward_high_t90104.ckpt"

mkdir -p outputs

if [[ ! -f "$SEL_CKPT" ]]; then
    echo "ERROR: tier-1 selector ckpt missing: $SEL_CKPT" >&2
    exit 1
fi

# GPU assignment — override HP_TIER2_GPUS to avoid shared GPUs.
IFS=',' read -ra GPUS <<< "${HP_TIER2_GPUS:-0,1,2,3,4,5,6,7}"

# ============================================================
# Group A — 3 seeds of reward_high @ LR=5e-5 (GPUs 0-2)
# ============================================================

# 92104 — reward_high, LR=5e-5, seed=1
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device "${GPUS[0]}" \
    federate.client_num 100 \
    seed 1 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.5 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n100_tier2_reward_high_lr5e5_seed1_t92104.ckpt" \
    expname "hpsearch_n100_t2_reward_high_lr5e5_s1_92104" \
    > outputs/92104.log 2>&1 &
echo "Launched 92104 (reward_high LR=5e-5 seed=1) on GPU ${GPUS[0]}, PID=$!"

# 92105 — reward_high, LR=5e-5, seed=2
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device "${GPUS[1]}" \
    federate.client_num 100 \
    seed 2 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.5 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n100_tier2_reward_high_lr5e5_seed2_t92105.ckpt" \
    expname "hpsearch_n100_t2_reward_high_lr5e5_s2_92105" \
    > outputs/92105.log 2>&1 &
echo "Launched 92105 (reward_high LR=5e-5 seed=2) on GPU ${GPUS[1]}, PID=$!"

# 92106 — reward_high, LR=5e-5, seed=3
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device "${GPUS[2]}" \
    federate.client_num 100 \
    seed 3 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.5 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n100_tier2_reward_high_lr5e5_seed3_t92106.ckpt" \
    expname "hpsearch_n100_t2_reward_high_lr5e5_s3_92106" \
    > outputs/92106.log 2>&1 &
echo "Launched 92106 (reward_high LR=5e-5 seed=3) on GPU ${GPUS[2]}, PID=$!"

# ============================================================
# Group B — Fairness checks + rc sweep + 100-round probes (GPUs 3-7)
# ============================================================

# 93100 — baseline at LR=5e-5 (fairness check vs reward_high@LR=5e-5)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device "${GPUS[3]}" \
    federate.client_num 100 \
    seed 0 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.1 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n100_tier2_baseline_lr5e5_t93100.ckpt" \
    expname "hpsearch_n100_t2_baseline_lr5e5_93100" \
    > outputs/93100.log 2>&1 &
echo "Launched 93100 (baseline LR=5e-5) on GPU ${GPUS[3]}, PID=$!"

# 93101 — reward_high with rc=1.0, LR=5e-5 (push the winning knob further)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device "${GPUS[4]}" \
    federate.client_num 100 \
    seed 0 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 1.0 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n100_tier2_rc1.0_lr5e5_t93101.ckpt" \
    expname "hpsearch_n100_t2_rc1.0_lr5e5_93101" \
    > outputs/93101.log 2>&1 &
echo "Launched 93101 (rc=1.0 LR=5e-5) on GPU ${GPUS[4]}, PID=$!"

# 93102 — reward_high, LR=5e-5, 100 rounds (does the winner keep improving?)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device "${GPUS[5]}" \
    federate.client_num 100 \
    seed 0 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.5 \
    federate.total_round_num 100 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n100_tier2_reward_high_lr5e5_r100_t93102.ckpt" \
    expname "hpsearch_n100_t2_reward_high_lr5e5_r100_93102" \
    > outputs/93102.log 2>&1 &
echo "Launched 93102 (reward_high LR=5e-5 100 rounds) on GPU ${GPUS[5]}, PID=$!"

# 93103 — baseline at default LR=1e-5, 100 rounds (does baseline catch up?)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device "${GPUS[6]}" \
    federate.client_num 100 \
    seed 0 \
    train.optimizer.lr 1e-5 \
    llm.reward_coeff 0.1 \
    federate.total_round_num 100 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n100_tier2_baseline_r100_t93103.ckpt" \
    expname "hpsearch_n100_t2_baseline_r100_93103" \
    > outputs/93103.log 2>&1 &
echo "Launched 93103 (baseline 100 rounds) on GPU ${GPUS[6]}, PID=$!"

# 93104 — reward_high with rc=0.3, LR=5e-5 (middle of rc sweep)
nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg "$RL_CFG" \
    --selector-cfg-file "$SEL_CFG" \
    device "${GPUS[7]}" \
    federate.client_num 100 \
    seed 0 \
    train.optimizer.lr 5e-5 \
    llm.reward_coeff 0.3 \
    llm.rlhf_selector_checkpoint "$SEL_CKPT" \
    federate.save_to "$CKPT_DIR/hhrl_rlhf_qwen2_hpsearch_n100_tier2_rc0.3_lr5e5_t93104.ckpt" \
    expname "hpsearch_n100_t2_rc0.3_lr5e5_93104" \
    > outputs/93104.log 2>&1 &
echo "Launched 93104 (rc=0.3 LR=5e-5) on GPU ${GPUS[7]}, PID=$!"

echo ""
echo "All 8 tier-2 RL stages launched. Tail logs with:"
echo "  tail -f outputs/{92104,92105,92106,93100,93101,93102,93103,93104}.log"
echo ""
echo "Expected wall-clock: ~5-7h for 50-round runs (post-eval-fix), ~10-14h"
echo "for the two 100-round runs (93102, 93103). All run in parallel."
