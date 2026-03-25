#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/kjb/ppfl:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
cd /home/kjb/ppfl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biscuit

# Load .env for OPENAI_API_KEY (winrate eval)
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "Loaded .env"
fi

PYTHON="python3"
CFG_DIR="cfg/main_table/qwen_hhrlhf"
mkdir -p outputs

# 11200: FedBiscuit RL on GPU 4
echo "[11200] FedBiscuit RL on GPU 4..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_comparison_fedbiscuit_11200.yaml \
    --selector-cfg-file $CFG_DIR/fedvpagp_comparison_fedbiscuit_10200.yaml \
    > outputs/11200.log 2>&1 &
echo "  PID: $!"

# 11201: FedVPL RL on GPU 5
echo "[11201] FedVPL RL on GPU 5..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_comparison_fedvpl_11201.yaml \
    --selector-cfg-file $CFG_DIR/fedvpagp_comparison_fedvpl_10201.yaml \
    > outputs/11201.log 2>&1 &
echo "  PID: $!"

# 11203: FedVPA-GP (KL+ortho) RL on GPU 6
echo "[11203] FedVPA-GP (KL+ortho) RL on GPU 6..."
nohup $PYTHON -u federatedscope/llm/rlhf/main.py \
    --cfg $CFG_DIR/hrl_comparison_kl_ortho_11203.yaml \
    --selector-cfg-file $CFG_DIR/fedvpagp_comparison_kl_ortho_10203.yaml \
    > outputs/11203.log 2>&1 &
echo "  PID: $!"

echo "All 3 launched."
