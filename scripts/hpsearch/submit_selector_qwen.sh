#!/bin/bash

# Submit Hyperparameter Search Selector Experiments (Qwen 2)
# Usage: bash submit_selector_qwen.sh <phase>
#   phase: 1, 2, 3, 4, or 5
# TID range: 54100-54138

PHASE=$1
if [ -z "$PHASE" ]; then
    echo "Usage: $0 <phase>"
    echo "  phase: 1, 2, 3, 4, or 5"
    echo ""
    echo "Phase ranges (Qwen):"
    echo "  Phase 1: 54100-54106 (7 experiments)"
    echo "  Phase 2: 54107-54113 (7 experiments)"
    echo "  Phase 3: 54114-54116 (3 experiments)"
    echo "  Phase 4: 54117 (1 experiment)"
    echo "  Phase 5: 54118-54138 (21 experiments)"
    exit 1
fi

WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR

case $PHASE in
    1) START_TID=54100; END_TID=54106; PHASE_NAME="Phase 1: Orthogonal Loss (Qwen)" ;;
    2) START_TID=54107; END_TID=54113; PHASE_NAME="Phase 2: VPL Core (Qwen)" ;;
    3) START_TID=54114; END_TID=54116; PHASE_NAME="Phase 3: Learning Rate (Qwen)" ;;
    4) START_TID=54117; END_TID=54117; PHASE_NAME="Phase 4: Combined Best (Qwen)" ;;
    5) START_TID=54118; END_TID=54138; PHASE_NAME="Phase 5: Fine-grained (Qwen)" ;;
    *) echo "ERROR: Invalid phase: $PHASE"; exit 1 ;;
esac

echo "=========================================="
echo "$PHASE_NAME"
echo "TID range: $START_TID-$END_TID"
echo "=========================================="

for tid in $(seq $START_TID $END_TID); do
    echo "  Submitting: TID=$tid"
    sbatch --job-name="hp_sel_qwen_p${PHASE}_${tid}" scripts/hpsearch/run_selector_hpsearch_qwen.sh $tid
    sleep 1
done
echo "Phase $PHASE (Qwen) selector jobs submitted."
