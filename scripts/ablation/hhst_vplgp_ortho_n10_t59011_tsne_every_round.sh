#!/bin/bash

# VPL-GP-Ortho - 10 rounds only, t-SNE every round (for z distribution evolution)
# Variant of 59001: tid=59011, GPU 6

tid=59011
device=6  # GPU 6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd $PROJECT_ROOT

[ -f "$PROJECT_ROOT/.env" ] && export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${PROJECT_ROOT}/.cache/huggingface/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

if [ -d "/hdd/hdd3/kjb" ]; then
    CHECKPOINT_DIR="/hdd/hdd3/kjb/checkpoints"
    DATA_ROOT="/hdd/hdd3/kjb"
else
    CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
    DATA_ROOT="$PROJECT_ROOT/data"
fi
mkdir -p $CHECKPOINT_DIR

echo "VPL-GP-Ortho 10-round t-SNE-every-round experiment (tid=${tid}, GPU ${device})"
echo "  total_round_num=10, vpl_tsne_visualize_freq=1"
echo "  Log: outputs/${tid}.log"
echo ""

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-gp/hhst.yaml \
    device ${device} \
    federate.client_num 10 \
    federate.total_round_num 10 \
    llm.vpl_tsne_visualize_freq 1 \
    llm.vpl_kl_weight 0.1 \
    llm.vpl_orthogonal_weight 1.0 \
    llm.vpl_orthogonal_orthonorm_weight 0.02 \
    llm.vpl_use_manual_orthogonal_labels True \
    llm.vpl_num_prototypes 2 \
    llm.vpl_prototype_scale 5.0 \
    data.root ${DATA_ROOT} \
    federate.save_to ${CHECKPOINT_DIR}/hhrl_choice_gemma_vplgp_ortho_n10_t${tid}.ckpt \
    expname "vplgp_ortho_hhst_n10_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "PID: $!"
echo "Monitor: tail -f outputs/${tid}.log"
