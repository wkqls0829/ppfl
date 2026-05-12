#!/bin/bash
# Wait for rebuttal selectors to finish, then launch RL
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

CFG="cfg/main_table/qwen_hhrlhf"
mkdir -p outputs

echo "Waiting for selectors 30000-30003, 30100-30101 to finish..."
while true; do
    running=0
    for tid in 30000 30001 30002 30003 30100 30101; do
        if ps aux | grep -v grep | grep "${tid}" | grep -q python; then
            running=$((running + 1))
        fi
    done
    if [ "$running" -eq 0 ]; then
        echo "All selectors finished!"
        break
    fi
    echo "  $(date): $running selectors still running..."
    sleep 60
done

# Delete any generation caches
rm -f /hdd/hdd3/kjb/hh-rlhf/generated_choose_*3100*.json /hdd/hdd3/kjb/hh-rlhf/generated_choose_*3000*.json 2>/dev/null

echo "Launching 6 RL experiments..."

python3 -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG/hrl_fedbiscuit_3070_31000.yaml \
    --selector-cfg-file $CFG/fedbiscuit_3070_30000.yaml \
    > outputs/31000.log 2>&1 &
echo "31000 PID=$!"

python3 -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG/hrl_fedvpagp_3070_31001.yaml \
    --selector-cfg-file $CFG/fedvpagp_z_hybrid_adrop_3070_30001.yaml \
    > outputs/31001.log 2>&1 &
echo "31001 PID=$!"

python3 -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG/hrl_fedbiscuit_7030_31002.yaml \
    --selector-cfg-file $CFG/fedbiscuit_7030_30002.yaml \
    > outputs/31002.log 2>&1 &
echo "31002 PID=$!"

python3 -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG/hrl_fedvpagp_7030_31003.yaml \
    --selector-cfg-file $CFG/fedvpagp_z_hybrid_adrop_7030_30003.yaml \
    > outputs/31003.log 2>&1 &
echo "31003 PID=$!"

python3 -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG/hrl_fedvpagp_pull_only_31100.yaml \
    --selector-cfg-file $CFG/fedvpagp_z_hybrid_adrop_pull_only_30100.yaml \
    > outputs/31100.log 2>&1 &
echo "31100 PID=$!"

python3 -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG/hrl_fedvpagp_orthonorm_only_31101.yaml \
    --selector-cfg-file $CFG/fedvpagp_z_hybrid_adrop_orthonorm_only_30101.yaml \
    > outputs/31101.log 2>&1 &
echo "31101 PID=$!"

echo "All 6 RL experiments launched."
