#!/bin/bash

# Phase 3: Learning Rate Search
# Experiments: 52014-52016 (3 experiments)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Phase 3: Learning Rate Search"
echo "=========================================="
echo "Experiments: 52014-52016"
echo ""

# Array of experiment IDs and their config files
declare -a experiments=(
    "52014:cfg/hpsearch/vpl-gp/phase3_lr_52014.yaml"
    "52015:cfg/hpsearch/vpl-gp/phase3_lr_52015.yaml"
    "52016:cfg/hpsearch/vpl-gp/phase3_lr_52016.yaml"
)

for exp in "${experiments[@]}"; do
    IFS=':' read -r tid config_file <<< "$exp"
    echo "Starting experiment ${tid}..."
    bash ${SCRIPT_DIR}/run_experiment.sh ${tid} ${config_file}
    sleep 2  # Small delay between experiments
done

echo ""
echo "All Phase 3 experiments started!"
echo "Monitor progress with: tail -f outputs/5201*.log"
