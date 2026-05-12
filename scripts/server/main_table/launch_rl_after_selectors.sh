#!/bin/bash
# Wait for selectors to finish, then launch RL experiments
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

# Wait for HH-RLHF selectors (10209, 10211, 10212, 20102)
echo "Waiting for HH-RLHF selectors to finish..."
while true; do
    running=0
    for tid in 10209 10211 10212 20102; do
        if ps aux | grep -v grep | grep "${tid}" | grep -q python; then
            running=$((running + 1))
        fi
    done
    if [ "$running" -eq 0 ]; then
        echo "All HH-RLHF selectors finished!"
        break
    fi
    echo "  $(date): $running HH-RLHF selectors still running..."
    sleep 120
done

# Delete any generation caches
rm -f /hdd/hdd3/kjb/hh-rlhf/generated_choose_*z_hybrid*.json 2>/dev/null

# Launch HH-RLHF RL experiments
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

echo "[11209] hybrid+add RL on GPU 2..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_HH/hrl_z_hybrid_add_11209.yaml \
    --selector-cfg-file $CFG_HH/fedvpagp_z_hybrid_add_10209.yaml \
    > outputs/11209.log 2>&1 &
echo "  PID: $!"

echo "[21102] 30:70 hybrid+adrop RL on GPU 4..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_HH/hrl_z_hybrid_adrop_3070_21102.yaml \
    --selector-cfg-file $CFG_HH/fedvpagp_z_hybrid_adrop_3070_20102.yaml \
    > outputs/21102.log 2>&1 &
echo "  PID: $!"

echo "4 HH-RLHF RL experiments launched."

# Wait for UltraFeedback selector (20002)
echo "Waiting for UltraFeedback selector to finish..."
while true; do
    if ! ps aux | grep -v grep | grep "20002" | grep -q python; then
        echo "UltraFeedback selector finished!"
        break
    fi
    echo "  $(date): UltraFeedback selector still running..."
    sleep 120
done

rm -f /hdd/hdd3/kjb/ultrafeedback/generated_choose_*z_hybrid*.json 2>/dev/null

echo "[21002] UltraFeedback hybrid+adrop RL on GPU 3..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_UF/hrl_z_hybrid_adrop_21002.yaml \
    --selector-cfg-file $CFG_UF/fedvpagp_z_hybrid_adrop_20002.yaml \
    > outputs/21002.log 2>&1 &
echo "  PID: $!"

echo "All 5 RL experiments launched."
