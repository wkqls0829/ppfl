#!/bin/bash
# RL experiments for all hybrid z-embedding selectors
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

if [ -f .env ]; then
    set -a; source .env; set +a
    echo "Loaded .env (OPENAI_API_KEY=${OPENAI_API_KEY:0:10}...)"
fi

PYTHON="python3"
CFG_HH="cfg/main_table/qwen_hhrlhf"
CFG_UF="cfg/main_table/qwen_ultrafeedback"
mkdir -p outputs

echo "[11211] hybrid+adrop RL on GPU 0..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_HH/hrl_z_hybrid_adrop_11211.yaml \
    --selector-cfg-file $CFG_HH/fedvpagp_z_hybrid_adapter_drop_10211.yaml \
    > outputs/11211.log 2>&1 &
echo "  PID: $!"

echo "[11212] hybrid+adrop+kmeans RL on GPU 1..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_HH/hrl_z_hybrid_adrop_kmeans_11212.yaml \
    --selector-cfg-file $CFG_HH/fedvpagp_z_hybrid_adrop_kmeans_10212.yaml \
    > outputs/11212.log 2>&1 &
echo "  PID: $!"

echo "[21002] UltraFeedback hybrid+adrop RL on GPU 2..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_UF/hrl_z_hybrid_adrop_21002.yaml \
    --selector-cfg-file $CFG_UF/fedvpagp_z_hybrid_adrop_20002.yaml \
    > outputs/21002.log 2>&1 &
echo "  PID: $!"

echo "[21102] 30:70 hybrid+adrop RL on GPU 3..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_HH/hrl_z_hybrid_adrop_3070_21102.yaml \
    --selector-cfg-file $CFG_HH/fedvpagp_z_hybrid_adrop_3070_20102.yaml \
    > outputs/21102.log 2>&1 &
echo "  PID: $!"

echo "[11209] hybrid+add (no adrop) RL on GPU 4..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_HH/hrl_z_hybrid_add_11209.yaml \
    --selector-cfg-file $CFG_HH/fedvpagp_z_hybrid_add_10209.yaml \
    > outputs/11209.log 2>&1 &
echo "  PID: $!"

echo "[11210] hybrid+concat (no adrop) RL on GPU 5..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_HH/hrl_z_hybrid_concat_11210.yaml \
    --selector-cfg-file $CFG_HH/fedvpagp_z_hybrid_concat_10210.yaml \
    > outputs/11210.log 2>&1 &
echo "  PID: $!"

echo "All 6 RL experiments launched."
