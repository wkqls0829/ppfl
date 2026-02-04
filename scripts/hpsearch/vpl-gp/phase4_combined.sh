#!/bin/bash

# Phase 4: Combined Best Parameters
# Experiment: 52017 (1 experiment)
# NOTE: Update phase4_combined_52017.yaml with best parameters from Phase 1-3 before running

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Phase 4: Combined Best Parameters"
echo "=========================================="
echo "Experiment: 52017"
echo ""
echo "WARNING: Make sure to update cfg/hpsearch/vpl-gp/phase4_combined_52017.yaml"
echo "         with the best parameters from Phase 1-3 before running!"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

tid=52017
config_file="cfg/hpsearch/vpl-gp/phase4_combined_52017.yaml"

echo "Starting experiment ${tid}..."
bash ${SCRIPT_DIR}/run_experiment.sh ${tid} ${config_file}

echo ""
echo "Phase 4 experiment started!"
echo "Monitor progress with: tail -f outputs/52017.log"
