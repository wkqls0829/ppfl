#!/bin/bash

# Run all local RL training only experiments
# TID: 01000 (N=10, GPU=2), 01001 (N=50, GPU=3), 01002 (N=100, GPU=4)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Starting Local RL Training Only Experiments"
echo "=========================================="
echo ""

# Experiment 1: N=10, GPU=2
echo "Starting experiment 1: TID=01000, N=10, GPU=2"
bash $SCRIPT_DIR/run_rl_local.sh 01000 10 2
sleep 5

# Experiment 2: N=50, GPU=3
echo ""
echo "Starting experiment 2: TID=01001, N=50, GPU=3"
bash $SCRIPT_DIR/run_rl_local.sh 01001 50 3
sleep 5

# Experiment 3: N=100, GPU=4
echo ""
echo "Starting experiment 3: TID=01002, N=100, GPU=4"
bash $SCRIPT_DIR/run_rl_local.sh 01002 100 4

echo ""
echo "=========================================="
echo "All experiments started"
echo "=========================================="
echo ""
echo "Monitor experiments:"
echo "  tail -f outputs/01000.log  # N=10, GPU=2"
echo "  tail -f outputs/01001.log  # N=50, GPU=3"
echo "  tail -f outputs/01002.log  # N=100, GPU=4"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
