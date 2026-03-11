#!/bin/bash

# LLaMA 7B VPL-GP HHST selector experiment
# Based on main-table Qwen FedVPA-GP config (TID 62230), but using LLaMA-7B
# TID: 92230, GPU: 2

set -e

# Always run inside biscuit conda env if available
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate biscuit || echo "WARNING: failed to activate biscuit env, continuing in current env"
fi

WORK_DIR="/home/kjb/ppfl"
cd "$WORK_DIR"

# Use local GPU 2
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

BASE_CFG="cfg/vpl-gp/hhst.yaml"
if [ ! -f "$BASE_CFG" ]; then
  echo "ERROR: Base config not found: $BASE_CFG"
  exit 1
fi

CFG_DIR="cfg/llama-vplgp"
mkdir -p "$CFG_DIR"
CONFIG_FILE="${CFG_DIR}/hhst_llama_92230.yaml"

cp "$BASE_CFG" "$CONFIG_FILE"

python3 << EOF
import os
import yaml

work_dir = "$WORK_DIR"
config_file = "$CONFIG_FILE"

with open(config_file, "r") as f:
    cfg = yaml.safe_load(f)

# Basic paths
cfg["use_gpu"] = True
cfg["device"] = 0  # mapped by CUDA_VISIBLE_DEVICES

cfg.setdefault("data", {})
cfg["data"]["root"] = os.path.join(work_dir, "data")

cfg.setdefault("federate", {})
cfg["federate"]["save_to"] = os.path.join(
    work_dir,
    "checkpoints",
    "hhrl_choice_llama2-7b_vplgp_92230.ckpt",
)

# LLaMA 7B model
cfg.setdefault("model", {})
cfg["model"]["type"] = "meta-llama/Llama-2-7b-chat-hf@huggingface_llm"

# Trainer
cfg.setdefault("trainer", {})
cfg["trainer"]["type"] = "vplgprewardchoicetrainer"

# LLaMA-friendly training hyperparameters
cfg.setdefault("train", {})
opt = cfg["train"].setdefault("optimizer", {})
opt["type"] = opt.get("type", "AdamW")
opt["lr"] = 1e-5

cfg.setdefault("dataloader", {})
cfg["dataloader"]["batch_size"] = 2

cfg.setdefault("llm", {})
cfg["llm"]["grad_accum_step"] = 8  # effective batch 16

# Hugging Face cache
cfg["llm"].setdefault("cache", {})
cfg["llm"]["cache"]["model"] = os.path.join(
    work_dir, ".cache", "huggingface", "transformers"
)

# VPL-GP hyperparameters (Qwen Phase 4 style, from 62230)
llm = cfg["llm"]
llm["vpl_use_gp_prior"] = True
llm["vpl_latent_dim"] = 32
llm["vpl_kl_weight"] = 0.05
llm["vpl_gp_temperature"] = 1.0
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

# Exp name / logging
cfg["expname"] = "vplgp_hhst_llama_92230"

with open(config_file, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Wrote LLaMA VPL-GP config: {config_file}")
EOF

echo "Starting LLaMA 7B VPL-GP selector (TID 92230) on GPU 2"
echo "Config: $CONFIG_FILE"

python -u federatedscope/main.py \
  --cfg "$CONFIG_FILE" \
  > outputs/92230.log 2>&1

echo "LLaMA 7B VPL-GP experiment completed: TID 92230"

