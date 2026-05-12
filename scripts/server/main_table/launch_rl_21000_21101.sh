#!/bin/bash
# RL experiments for UltraFeedback + 30:70 split
# 21000: FedBiscuit UF on GPU 0
# 21001: FedVPA-GP UF on GPU 1
# 21100: FedBiscuit 30:70 on GPU 2
# 21101: FedVPA-GP 30:70 on GPU 3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env (OPENAI_API_KEY=${OPENAI_API_KEY:0:10}...)"
fi

mkdir -p outputs

# UltraFeedback
echo "[21000] FedBiscuit RL UltraFeedback on GPU 0..."
nohup python3 -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_ultrafeedback/hrl_fedbiscuit_21000.yaml \
    --selector-cfg-file cfg/main_table/qwen_ultrafeedback/fedbiscuit_20000.yaml \
    > outputs/21000.log 2>&1 &
echo "  PID: $!"

echo "[21001] FedVPA-GP RL UltraFeedback on GPU 1..."
nohup python3 -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_ultrafeedback/hrl_fedvpagp_kl_ortho_21001.yaml \
    --selector-cfg-file cfg/main_table/qwen_ultrafeedback/fedvpagp_kl_ortho_20001.yaml \
    > outputs/21001.log 2>&1 &
echo "  PID: $!"

# 30:70 split
echo "[21100] FedBiscuit RL 30:70 on GPU 2..."
nohup python3 -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_fedbiscuit_3070_21100.yaml \
    --selector-cfg-file cfg/main_table/qwen_hhrlhf/fedbiscuit_3070_20100.yaml \
    > outputs/21100.log 2>&1 &
echo "  PID: $!"

echo "[21101] FedVPA-GP RL 30:70 on GPU 3..."
nohup python3 -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_fedvpagp_kl_ortho_3070_21101.yaml \
    --selector-cfg-file cfg/main_table/qwen_hhrlhf/fedvpagp_kl_ortho_3070_20101.yaml \
    > outputs/21101.log 2>&1 &
echo "  PID: $!"

echo "All 4 launched."
