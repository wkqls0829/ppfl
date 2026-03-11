#!/bin/bash

# Phase 2: VPL Core Parameters Search
# Experiments: 52007-52013 (7 experiments)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Phase 2: VPL Core Parameters Search"
echo "=========================================="
echo "Experiments: 52007-52013"
echo ""

# Array of experiment IDs and their config files
declare -a experiments=(
    "52007:cfg/hpsearch/vpl-gp/phase2_vpl_core_52007.yaml"
    "52008:cfg/hpsearch/vpl-gp/phase2_vpl_core_52008.yaml"
    "52009:cfg/hpsearch/vpl-gp/phase2_vpl_core_52009.yaml"
    "52010:cfg/hpsearch/vpl-gp/phase2_vpl_core_52010.yaml"
    "52011:cfg/hpsearch/vpl-gp/phase2_vpl_core_52011.yaml"
    "52012:cfg/hpsearch/vpl-gp/phase2_vpl_core_52012.yaml"
    "52013:cfg/hpsearch/vpl-gp/phase2_vpl_core_52013.yaml"
)

for exp in "${experiments[@]}"; do
    IFS=':' read -r tid config_file <<< "$exp"
    echo "Starting experiment ${tid}..."
    bash ${SCRIPT_DIR}/run_experiment.sh ${tid} ${config_file}
    sleep 2  # Small delay between experiments
done

echo ""
echo "All Phase 2 experiments started!"
echo "Monitor progress with: tail -f outputs/5201*.log"
