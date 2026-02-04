#!/bin/bash

# Submit Hyperparameter Search Selector Experiments
# Usage: bash submit_selector.sh <phase>
#   phase: 1, 2, 3, 4, or 5

PHASE=$1

if [ -z "$PHASE" ]; then
    echo "Usage: $0 <phase>"
    echo "  phase: 1, 2, 3, 4, or 5"
    echo ""
    echo "Phase ranges:"
    echo "  Phase 1: 54000-54006 (7 experiments)"
    echo "  Phase 2: 54007-54013 (7 experiments)"
    echo "  Phase 3: 54014-54016 (3 experiments)"
    echo "  Phase 4: 54017 (1 experiment)"
    echo "  Phase 5: 54018-54038 (21 experiments)"
    exit 1
fi

WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR

# Determine TID range based on phase
case $PHASE in
    1)
        START_TID=54000
        END_TID=54006
        PHASE_NAME="Phase 1: Orthogonal Loss Parameters"
        ;;
    2)
        START_TID=54007
        END_TID=54013
        PHASE_NAME="Phase 2: VPL Core Parameters"
        ;;
    3)
        START_TID=54014
        END_TID=54016
        PHASE_NAME="Phase 3: Learning Rate"
        ;;
    4)
        START_TID=54017
        END_TID=54017
        PHASE_NAME="Phase 4: Combined Best Parameters"
        ;;
    5)
        START_TID=54018
        END_TID=54038
        PHASE_NAME="Phase 5: Fine-grained Hyperparameter Search"
        ;;
    *)
        echo "ERROR: Invalid phase: $PHASE"
        echo "Phase must be 1, 2, 3, 4, or 5"
        exit 1
        ;;
esac

echo "=========================================="
echo "Submitting $PHASE_NAME"
echo "TID range: $START_TID-$END_TID"
echo "=========================================="

for tid in $(seq $START_TID $END_TID); do
    echo "  Submitting: TID=$tid"
    sbatch --job-name="hp_sel_phase${PHASE}_${tid}" \
           scripts/hpsearch/run_selector_hpsearch.sh \
           $tid
    sleep 1
done

echo ""
echo "Phase $PHASE selector jobs submitted."
