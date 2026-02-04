#!/bin/bash

# Phase 1: Orthogonal Loss Weight Search
# Experiments: 52000-52006 (7 experiments)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Phase 1: Orthogonal Loss Weight Search"
echo "=========================================="
echo "Experiments: 52000-52006"
echo ""

# Array of experiment IDs and their config files
declare -a experiments=(
    "52000:cfg/hpsearch/vpl-gp/phase1_orthogonal_52000.yaml"
    "52001:cfg/hpsearch/vpl-gp/phase1_orthogonal_52001.yaml"
    "52002:cfg/hpsearch/vpl-gp/phase1_orthogonal_52002.yaml"
    "52003:cfg/hpsearch/vpl-gp/phase1_orthogonal_52003.yaml"
    "52004:cfg/hpsearch/vpl-gp/phase1_orthogonal_52004.yaml"
    "52005:cfg/hpsearch/vpl-gp/phase1_orthogonal_52005.yaml"
    "52006:cfg/hpsearch/vpl-gp/phase1_orthogonal_52006.yaml"
)

for exp in "${experiments[@]}"; do
    IFS=':' read -r tid config_file <<< "$exp"
    echo "Starting experiment ${tid}..."
    bash ${SCRIPT_DIR}/run_experiment.sh ${tid} ${config_file}
    sleep 2  # Small delay between experiments
done

echo ""
echo "All Phase 1 experiments started!"
echo "Monitor progress with: tail -f outputs/5200*.log"
