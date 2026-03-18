#!/bin/bash
# Launch all 4 RL experiments for Qwen2-0.5B + HH-RLHF (Stage 2)
# FedBiscuit (GPU 2), FedVPL (GPU 3), VPL-GP no ortho (GPU 4), FedVPA-GP (GPU 6)

set -e

# Conda activation
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate biscuit || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Load .env for OPENAI_API_KEY
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    echo "Loaded .env (OPENAI_API_KEY for winrate eval)"
fi

PYTHON="/home/kjb/.conda/envs/biscuit/bin/python"
CFG_DIR="cfg/main_table/qwen_hhrlhf"
mkdir -p outputs

echo "=== Launching RL Stage (Stage 2) ==="

# 1. FedBiscuit RL (TID 11000) on GPU 2
#    Uses --selector-cfg-file to load multi-LoRA selector for reward scoring
echo "[11000] FedBiscuit RL on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_fedbiscuit_11000.yaml \
    --selector-cfg-file $CFG_DIR/fedbiscuit_10000.yaml \
    > outputs/11000.log 2>&1 &
echo "  PID: $!, Log: outputs/11000.log"

# 2. FedVPL RL (TID 11001) on GPU 3
#    Uses rlhf_selector_checkpoint for VPL variational selection
echo "[11001] FedVPL RL on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_fedvpl_11001.yaml \
    --selector-cfg-file $CFG_DIR/fedvpl_10001.yaml \
    > outputs/11001.log 2>&1 &
echo "  PID: $!, Log: outputs/11001.log"

# 3. VPL-GP no ortho RL (TID 11002) on GPU 4
echo "[11002] VPL-GP no ortho RL on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_vplgp_no_ortho_11002.yaml \
    --selector-cfg-file $CFG_DIR/vplgp_no_ortho_10002.yaml \
    > outputs/11002.log 2>&1 &
echo "  PID: $!, Log: outputs/11002.log"

# 4. FedVPA-GP full RL (TID 11003) on GPU 6
echo "[11003] FedVPA-GP RL on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_fedvpagp_11003.yaml \
    --selector-cfg-file $CFG_DIR/fedvpagp_10003.yaml \
    > outputs/11003.log 2>&1 &
echo "  PID: $!, Log: outputs/11003.log"

echo ""
echo "=== All 4 RL experiments launched ==="
echo "Monitor: tail -f outputs/1100{0,1,2,3}.log"
