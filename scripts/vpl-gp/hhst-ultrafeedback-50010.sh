#!/bin/bash
# VPL-GP UltraFeedback Binary Selector - full training (50 rounds)
# TID 50010, GPU 3. Same round/settings as other binary selectors.

tid=50010

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-gp/hhst-ultrafeedback-50010.yaml \
    > outputs/${tid}.log 2>&1 &

echo "VPL-GP HHST UltraFeedback (full) started (TID: ${tid})"
echo "Config: cfg/vpl-gp/hhst-ultrafeedback-50010.yaml"
echo "Rounds: 50, GPU: 3, full dataset (no max_train_samples limit)"
echo "Log: outputs/${tid}.log — tail -f outputs/${tid}.log"
