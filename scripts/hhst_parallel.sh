#!/bin/bash

# Multi-GPU parallel training script for HH-RLHF dataset
# Usage: ./scripts/hhst_parallel.sh [GPU_IDS] [PROCESS_NUM]
# Example: ./scripts/hhst_parallel.sh "0,1,2" 3
#          ./scripts/hhst_parallel.sh "4,5,6,7" 4

# Default values
DEFAULT_GPUS="0,1"
DEFAULT_PROCESS_NUM=2
TID=10001

# Parse arguments
if [ $# -ge 1 ]; then
    GPU_IDS="$1"
else
    GPU_IDS="$DEFAULT_GPUS"
fi

if [ $# -ge 2 ]; then
    PROCESS_NUM="$2"
else
    # Count number of GPUs from GPU_IDS
    PROCESS_NUM=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
fi

# Validate GPU IDs
if [ -z "$GPU_IDS" ]; then
    echo "Error: GPU IDs cannot be empty"
    exit 1
fi

# Check if GPUs are available
echo "Checking GPU availability..."
for gpu_id in $(echo "$GPU_IDS" | tr ',' ' '); do
    if ! nvidia-smi -i "$gpu_id" &>/dev/null; then
        echo "Warning: GPU $gpu_id may not be available"
    fi
done

# Set CUDA_VISIBLE_DEVICES to only use specified GPUs
# This remaps the GPUs so they appear as 0, 1, 2, ... to the program
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Set PyTorch CUDA allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set PYTHONPATH to use the current directory's federatedscope
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create output directory if it doesn't exist
mkdir -p outputs

# Log file name with GPU info (different from hhst.sh which uses outputs/10000.log)
LOG_FILE="outputs/${TID}_parallel_gpu${GPU_IDS//,/_}_p${PROCESS_NUM}.log"

echo "=========================================="
echo "Multi-GPU Parallel Training Configuration"
echo "=========================================="
echo "GPU IDs (physical): $GPU_IDS"
echo "Process number: $PROCESS_NUM"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Log file: $LOG_FILE"
echo "=========================================="

# Run the training
nohup python -u federatedscope/main.py \
    --cfg cfg/gemma_hhrl_parallel.yaml \
    federate.process_num "$PROCESS_NUM" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Or check GPU usage with: watch -n 1 nvidia-smi"
