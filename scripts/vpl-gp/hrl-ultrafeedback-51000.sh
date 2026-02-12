#!/bin/bash
# VPL-GP RL (RLHF) training - UltraFeedback version
# TID 51000: uses UltraFeedback selector checkpoint from 50000 (hhst-ultrafeedback-50000-test)
# Standard pattern: same as hrl-ortho-51000.sh but selector = UltraFeedback 50000

tid=51000
selector_tid=50000  # UltraFeedback selector (hhst-ultrafeedback-50000-test)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# UltraFeedback selector checkpoint names (try final_ first, then round-saved)
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
    echo "Run hhst-ultrafeedback-50000-test.sh first (tid=${selector_tid})."
    exit 1
fi
echo ""

nohup python -u federatedscope/llm/rlhf/main.py \
    --cfg cfg/vpl-gp/hrl-ultrafeedback-51000.yaml \
    --selector-cfg-file cfg/vpl-gp/hhst-ultrafeedback-50000-test.yaml \
    federate.save_to "/hdd/hdd3/kjb/checkpoints/hhrl_rlhf_gemma_choice_vplgp_ortho_uf_${tid}.ckpt" \
    expname "vplgp_hrl_ultrafeedback_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "VPL-GP UltraFeedback RL training started (task ID: ${tid})"
echo "Config: cfg/vpl-gp/hrl-ultrafeedback-51000.yaml"
echo "Selector: UltraFeedback 50000 (cfg: hhst-ultrafeedback-50000-test.yaml)"
echo "Checkpoint: hhrl_rlhf_gemma_choice_vplgp_ortho_uf_${tid}.ckpt"
echo "GPU: 4 (device in config)"
echo "Log: outputs/${tid}.log — monitor: tail -f outputs/${tid}.log"
