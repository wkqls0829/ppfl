#!/bin/bash
# Qwen2 RL (next step after DP selector) with biscuit env
# **Local server only** (not for SLURM). Use biscuit env, CUDA_VISIBLE_DEVICES via gpu_id.
# Usage: bash scripts/server/differential_privacy/run_rl_qwen_dp.sh <rl_tid> <selector_tid> <gpu_id> [client_count]
#   e.g. bash scripts/server/differential_privacy/run_rl_qwen_dp.sh 93230 82230 5
#        bash scripts/server/differential_privacy/run_rl_qwen_dp.sh 93231 82231 6 10

set -e

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate biscuit || true
fi

RL_TID=$1
SELECTOR_TID=$2
GPU_ID=$3
CLIENT_COUNT=${4:-10}

MODEL="qwen2"
METHOD="fedvpagp"

if [ -z "$RL_TID" ] || [ -z "$SELECTOR_TID" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: $0 <rl_tid> <selector_tid> <gpu_id> [client_count]"
    echo "  e.g. $0 93230 82230 5"
    echo "       $0 93231 82231 6 10"
    exit 1
fi

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WORK_DIR"

if [ -d "/hdd/hdd3/kjb" ]; then
    DATA_ROOT="/hdd/hdd3/kjb"
    CHECKPOINT_BASE="/hdd/hdd3/kjb/checkpoints"
else
    DATA_ROOT="$WORK_DIR/data"
    CHECKPOINT_BASE="$WORK_DIR/checkpoints"
fi
mkdir -p "$CHECKPOINT_BASE"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SELECTOR_CKPT="$CHECKPOINT_BASE/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
if [ ! -f "$SELECTOR_CKPT" ]; then
    SELECTOR_CKPT="$CHECKPOINT_BASE/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
fi
if [ ! -f "$SELECTOR_CKPT" ]; then
    echo "ERROR: Selector checkpoint not found for $SELECTOR_TID"
    echo "  Tried: $CHECKPOINT_BASE/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
    echo "  Tried: $CHECKPOINT_BASE/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
    exit 1
fi

SELECTOR_CFG="cfg/main_table/${MODEL}/${METHOD}/hhst_n${CLIENT_COUNT}_${SELECTOR_TID}.yaml"
if [ ! -f "$SELECTOR_CFG" ]; then
    echo "WARNING: Selector config not found: $SELECTOR_CFG (will run without --selector-cfg-file)"
    SELECTOR_CFG=""
fi

CONFIG_FILE="cfg/main_table/${MODEL}/${METHOD}/hrl_n${CLIENT_COUNT}_${RL_TID}.yaml"
mkdir -p "$(dirname "$CONFIG_FILE")"

cp cfg/vpl-gp/hrl.yaml "$CONFIG_FILE"

python3 << EOF
import os, yaml
cfg_path = "$CONFIG_FILE"
with open(cfg_path, "r") as f:
    c = yaml.safe_load(f)
c["use_gpu"] = True
c["device"] = 0
c.setdefault("data", {})["root"] = "$DATA_ROOT"
c.setdefault("federate", {})["save_to"] = "$CHECKPOINT_BASE/hhrl_rlhf_${MODEL}_choice_${METHOD}_t${RL_TID}.ckpt"
c.setdefault("model", {})["type"] = "Qwen/Qwen2-0.5B@huggingface_llm"
c.setdefault("trainer", {})["type"] = "llmdporewardtrainer"
c["expname"] = "fedvpagp_${MODEL}_n${CLIENT_COUNT}_rl_t${RL_TID}"
c.setdefault("train", {}).setdefault("optimizer", {})["lr"] = 1e-5
c.setdefault("llm", {})
c["llm"]["rlhf_use_variational_selection"] = True
c["llm"]["rlhf_use_variational_generation"] = False
c["llm"]["rlhf_selector_checkpoint"] = "$SELECTOR_CKPT"
c["llm"]["vpl_latent_dim"] = 32
c["llm"]["vpl_feature_method"] = "choice_logits"
c["llm"]["vpl_use_feature_difference"] = True
c["llm"]["vpl_use_difference_only"] = True
c["llm"]["vpl_gp_temperature"] = 1.0
c["llm"]["vpl_use_gp_prior"] = True
c["llm"]["vpl_kl_weight"] = 0.05
c["llm"]["reward_coeff"] = 0.1
c["llm"]["max_prompts_for_generation"] = 50
c["llm"]["generation_batch_size"] = 3
c["llm"]["grad_accum_step"] = 32
c.setdefault("dataloader", {})["num_workers"] = 0
c.setdefault("eval", {})["freq"] = 10
c["eval"]["use_gpt_api_for_winrate"] = True
c["eval"]["use_baseline_model_for_winrate"] = True
c["eval"]["metrics"] = ["loss", "acc", "helpfulness_winrate", "harmlessness_winrate"]
with open(cfg_path, "w") as f:
    yaml.dump(c, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
print("Wrote", cfg_path)
EOF

echo "Starting RL: RL_TID=${RL_TID}, selector=${SELECTOR_TID}, GPU=${GPU_ID}"
if [ -n "$SELECTOR_CFG" ]; then
    python -u federatedscope/llm/rlhf/main.py --cfg "$CONFIG_FILE" --selector-cfg-file "$SELECTOR_CFG" > "outputs/${RL_TID}.log" 2>&1
else
    python -u federatedscope/llm/rlhf/main.py --cfg "$CONFIG_FILE" > "outputs/${RL_TID}.log" 2>&1
fi
echo "RL completed: ${RL_TID}"
