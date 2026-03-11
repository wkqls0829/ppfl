#!/bin/bash

# Qwen2 Main Table Selector with Differential Privacy (NbAFL)
# **Local server only** (not for SLURM). Use biscuit env; GPU via config device (gpu_id arg).
# Usage:
#   bash scripts/server/differential_privacy/run_qwen_main_table_dp.sh <method> <tid> <gpu_id>
#     method: fedbiscuit, fedvpagp
#     tid   : e.g., 82210, 82230
#     gpu_id: local GPU index to use (e.g., 5 or 6)

set -e

# Always run inside biscuit conda env if available
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate biscuit || echo "WARNING: failed to activate biscuit env, continuing in current env"
fi

METHOD=$1      # fedbiscuit, fedvpagp
TID=$2         # 82210, 82230, ...
GPU_ID=$3      # 5, 6, ...

MODEL="qwen2"
CLIENT_COUNT=10   # 62210 / 62230 are N=10 main-table selectors

if [ -z "$METHOD" ] || [ -z "$TID" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: $0 <method> <tid> <gpu_id>"
    echo "  method : fedbiscuit, fedvpagp"
    echo "  tid    : e.g., 82210, 82230"
    echo "  gpu_id : local GPU index (e.g., 5, 6)"
    exit 1
fi

case "$METHOD" in
    fedbiscuit)
        TRAINER="llmrewardchoicetrainer"
        CONFIG_BASE="cfg/fedbiscuit/hhst.yaml"
        ;;
    fedvpagp)
        TRAINER="vplgprewardchoicetrainer"
        CONFIG_BASE="cfg/vpl-gp/hhst.yaml"
        ;;
    *)
        echo "Unknown method: $METHOD (expected: fedbiscuit or fedvpagp)"
        exit 1
        ;;
esac

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WORK_DIR"

export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

if [ ! -f "$CONFIG_BASE" ]; then
    echo "ERROR: Base config not found: $CONFIG_BASE"
    exit 1
fi

# We keep the same path pattern as main_table selectors but with new TID (8xxxx)
CONFIG_FILE="cfg/main_table/${MODEL}/${METHOD}/hhst_n${CLIENT_COUNT}_${TID}.yaml"
mkdir -p "$(dirname "$CONFIG_FILE")"

cp "$CONFIG_BASE" "$CONFIG_FILE"

python3 << EOF
import os
import yaml

config_file = "$CONFIG_FILE"
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

WORK_DIR = "$WORK_DIR"
MODEL = "$MODEL"
METHOD = "$METHOD"
CLIENT_COUNT = int("$CLIENT_COUNT")
TID = "$TID"
CHECKPOINT_DIR = "$CHECKPOINT_DIR"

# Ensure llm + cache block
config.setdefault("llm", {})
config["llm"].setdefault("cache", {})
config["llm"]["cache"]["model"] = os.path.join(WORK_DIR, ".cache/huggingface/transformers")

# Basic GPU / dataloader settings
config["use_gpu"] = True
config["device"] = int("$GPU_ID")  # use GPU directly in config (no CUDA_VISIBLE_DEVICES)

config.setdefault("dataloader", {})
config["dataloader"]["num_workers"] = 0

# Federate settings (N=10 main-table)
config.setdefault("federate", {})
config["federate"]["client_num"] = CLIENT_COUNT
config["federate"]["sample_client_num"] = 5
config["federate"]["save_to"] = f"{CHECKPOINT_DIR}/hhrl_choice_{MODEL}_fedbiscuit_u3_{METHOD}_t{TID}.ckpt"

# Data root (local)
config.setdefault("data", {})
config["data"]["root"] = os.path.join(WORK_DIR, "data")

# Model + trainer
config.setdefault("model", {})
config["model"]["type"] = "Qwen/Qwen2-0.5B@huggingface_llm"

config.setdefault("trainer", {})
config["trainer"]["type"] = "$TRAINER"

# Exp name
config["expname"] = f"{METHOD}_{MODEL}_n{CLIENT_COUNT}_t{TID}"

# Optimizer / batch size / grad_accum for Qwen main table
config.setdefault("train", {})
config["train"].setdefault("optimizer", {})
config["train"]["optimizer"]["lr"] = 1e-5
config["dataloader"]["batch_size"] = 16
config["llm"]["grad_accum_step"] = 1

# FedVPA-GP (METHOD = fedvpagp) Qwen Phase 4 hyperparameters
if METHOD == "fedvpagp":
    llm = config["llm"]
    llm["vpl_use_gp_prior"] = True
    llm["vpl_latent_dim"] = 32
    llm["vpl_kl_weight"] = 0.05        # Qwen Phase 4 combined best
    llm["vpl_gp_temperature"] = 1.0    # Qwen Phase 4
    llm["vpl_feature_method"] = "choice_logits"
    llm["vpl_use_feature_difference"] = True
    llm["vpl_use_difference_only"] = True
    llm["vpl_max_logvar"] = -3.0
    llm["vpl_orthogonal_weight"] = 1.0
    llm["vpl_orthogonal_orthonorm_weight"] = 0.1
    llm["vpl_use_manual_orthogonal_labels"] = False
    llm["vpl_num_prototypes"] = 2
    llm["vpl_prototype_scale"] = 2.0
    llm["vpl_tsne_visualize_freq"] = 10

# Enable differential privacy via NbAFL wrapper
config.setdefault("nbafl", {})
config["nbafl"]["use"] = True
# DP hyperparameters (initial setting; can be tuned later)
config["nbafl"]["mu"] = 0.0          # regularizer factor
config["nbafl"]["epsilon"] = 8.0     # privacy budget (smaller = stronger privacy)
config["nbafl"]["w_clip"] = 1.0      # weight clipping threshold
config["nbafl"]["constant"] = 30.0   # scaling constant (as in FS NbAFL implementation)

with open(config_file, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"DP config written: {config_file}")
EOF

echo "Starting DP selector training: METHOD=${METHOD}, MODEL=${MODEL}, N=${CLIENT_COUNT}, TID=${TID}, GPU=${GPU_ID}"
echo "Config: ${CONFIG_FILE}"

python -u federatedscope/main.py \
    --cfg "$CONFIG_FILE" \
    > "outputs/${TID}.log" 2>&1

echo "DP experiment completed: TID=${TID}"

