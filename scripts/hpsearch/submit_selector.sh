#!/bin/bash

# Submit Hyperparameter Search Selector Experiments
# Usage: bash submit_selector.sh <phase>
#   phase: 1, 2, 3, 4, 5, or 6

PHASE=$1

if [ -z "$PHASE" ]; then
    echo "Usage: $0 <phase>"
    echo "  phase: 1, 2, 3, 4, 5, or 6"
    echo ""
    echo "Phase ranges:"
    echo "  Phase 1: 54000-54006 (7 experiments) - Orthogonal Loss"
    echo "  Phase 2: 54007-54013 (7 experiments) - VPL Core"
    echo "  Phase 3: 54014-54022 (9 experiments) - Refinement (54005/54008/54013 기반)"
    echo "  Phase 4: 54023-54025 (3 experiments) - Learning Rate"
    echo "  Phase 5: 54026 (1 experiment) - Combined Best"
    echo "  Phase 6: 54027-54047 (21 experiments) - Fine-grained"
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
        END_TID=54022
        PHASE_NAME="Phase 3: Refinement (prototype_scale / kl_weight / gp_temperature)"
        ;;
    4)
        START_TID=54023
        END_TID=54025
        PHASE_NAME="Phase 4: Learning Rate"
        ;;
    5)
        START_TID=54026
        END_TID=54026
        PHASE_NAME="Phase 5: Combined Best Parameters"
        ;;
    6)
        START_TID=54027
        END_TID=54047
        PHASE_NAME="Phase 6: Fine-grained Hyperparameter Search"
        ;;
    *)
        echo "ERROR: Invalid phase: $PHASE"
        echo "Phase must be 1, 2, 3, 4, 5, or 6"
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
