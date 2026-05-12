#!/bin/bash
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

nohup python3 -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/main_table/qwen_hhrlhf/hrl_z_hybrid_adrop_11211.yaml \
    --selector-cfg-file cfg/main_table/qwen_hhrlhf/fedvpagp_z_hybrid_adapter_drop_10211.yaml \
    > outputs/11211.log 2>&1 &
echo "11211 PID=$!"
