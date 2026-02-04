#!/bin/bash

# Variational Preference Learning with Gumbel-Softmax Prior (VPL-GP) test script
# Test run with gemma-2b model (tid: 50001)
# Based on 50000 test configuration, adjusted for gemma-2b

tid=50001

# GPU is specified in config file (device: 0)
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg cfg/vpl-gp/hhst-ortho-50001-test.yaml \
    > outputs/${tid}.log 2>&1 &

echo "VPL-GP HHST test training started (task ID: ${tid})"
echo "Config: cfg/vpl-gp/hhst-ortho-50001-test.yaml"
echo "Model: google/gemma-2b@huggingface_llm"
echo "Test configuration: 5 rounds, 1000 train samples, 200 test samples"
echo "Hyperparameters: batch_size=8, lr=0.0001, grad_accum_step=4"
echo "GPU: 0 (specified in config file)"
echo "WandB project: fvpl-selector"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
