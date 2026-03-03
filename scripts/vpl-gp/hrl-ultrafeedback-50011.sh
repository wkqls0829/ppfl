#!/bin/bash
# VPL-GP RL (RLHF) - UltraFeedback, selector from 50010 (full 50-round)
# TID 50011, GPU 3. Selector cfg: hhst-ultrafeedback-50010.yaml

tid=50011
selector_tid=50010

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

SELECTOR_CKPT_FINAL="/hdd/hdd3/kjb/checkpoints/final_ultrafeedback_choice_gemma_fedbiscuit_u3_vplgp_ortho_${selector_tid}.ckpt"
SELECTOR_CKPT="/hdd/hdd3/kjb/checkpoints/ultrafeedback_choice_gemma_fedbiscuit_u3_vplgp_ortho_${selector_tid}.ckpt"

if [ -f "${SELECTOR_CKPT_FINAL}" ]; then
    SELECTOR_CKPT="${SELECTOR_CKPT_FINAL}"
    echo "Using final selector checkpoint: ${SELECTOR_CKPT}"
elif [ -f "${SELECTOR_CKPT}" ]; then
    echo "Using selector checkpoint: ${SELECTOR_CKPT}"
else
    echo "ERROR: UltraFeedback selector checkpoint not found:"
    echo "  Tried: ${SELECTOR_CKPT_FINAL}"
    echo "  Tried: ${SELECTOR_CKPT}"
    echo "Run hhst-ultrafeedback-50010.sh first (tid=${selector_tid})."
    exit 1
fi
echo ""

if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/vpl-gp/hrl-ultrafeedback-50011.yaml \
    --selector-cfg-file cfg/vpl-gp/hhst-ultrafeedback-50010.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_vplgp_ortho_uf_${tid}.ckpt" \
    expname "vplgp_hrl_ultrafeedback_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "VPL-GP UltraFeedback RL started (TID: ${tid}, GPU: 3)"
echo "Selector: 50010 (hhst-ultrafeedback-50010.yaml)"
echo "Log: outputs/${tid}.log — tail -f outputs/${tid}.log"
