#!/bin/bash

# Generic script template for hyperparameter search experiments
# Usage: bash run_experiment.sh <experiment_id> <config_file>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_id> <config_file>"
    echo "Example: $0 50025 cfg/hpsearch/vpl-gp/phase1_orthogonal_50025.yaml"
    exit 1
fi

tid=$1
config_file=$2

# GPU is specified in config file
# Do NOT set CUDA_VISIBLE_DEVICES - let the config file handle GPU assignment

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Set PYTHONPATH to use the current directory's federatedscope instead of other installations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

nohup python -u federatedscope/main.py \
    --cfg ${config_file} \
    federate.save_to /hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_${tid}.ckpt \
    expname "vplgp_hhst_ortho_t${tid}" \
    > outputs/${tid}.log 2>&1 &

echo "VPL-GP HHST hyperparameter search experiment started (task ID: ${tid})"
echo "Config: ${config_file}"
echo "Log file: outputs/${tid}.log"
echo "Monitor with: tail -f outputs/${tid}.log"
