#!/bin/bash

# Quick script to check GPU availability and usage
# Usage: ./scripts/check_gpu.sh

echo "=========================================="
echo "GPU Availability Check"
echo "=========================================="
echo ""

# Show all GPUs with their status
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "GPU %s: %s\n", $1, $2; printf "  Memory: %s / %s\n", $3, $4; printf "  Utilization: %s\n", $5; printf "  Temperature: %s°C\n\n", $6}'

echo "=========================================="
echo "Recommended GPU Selection:"
echo "=========================================="
echo "Look for GPUs with:"
echo "  - Low memory usage (< 20%)"
echo "  - Low utilization (< 10%)"
echo "  - Low temperature (< 50°C)"
echo ""
echo "Example usage:"
echo "  ./scripts/hhst_parallel.sh \"0,1\" 2    # Use GPU 0 and 1 with 2 processes"
echo "  ./scripts/hhst_parallel.sh \"2,3,4\" 3  # Use GPU 2, 3, 4 with 3 processes"
echo ""
