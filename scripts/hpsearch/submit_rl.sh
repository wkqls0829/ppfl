#!/bin/bash

# Submit Hyperparameter Search RL Experiments
# Usage: bash submit_rl.sh <phase>
#   phase: 1, 2, 3, 4, or 5
# Note: Run this after the corresponding selector phase completes

PHASE=$1

if [ -z "$PHASE" ]; then
    echo "Usage: $0 <phase>"
    echo "  phase: 1, 2, 3, 4, or 5"
    echo ""
    echo "Phase ranges:"
    echo "  Phase 1: RL 55000-55006, Selector 54000-54006 (7 experiments)"
    echo "  Phase 2: RL 55007-55013, Selector 54007-54013 (7 experiments)"
    echo "  Phase 3: RL 55014-55016, Selector 54014-54016 (3 experiments)"
    echo "  Phase 4: RL 55017, Selector 54017 (1 experiment)"
    echo "  Phase 5: RL 55018-55038, Selector 54018-54038 (21 experiments)"
    exit 1
fi

WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR

# Determine TID range based on phase
case $PHASE in
    1)
        SELECTOR_START=54000
        SELECTOR_END=54006
        RL_START=55000
        RL_END=55006
        PHASE_NAME="Phase 1: Orthogonal Loss Parameters"
        ;;
    2)
        SELECTOR_START=54007
        SELECTOR_END=54013
        RL_START=55007
        RL_END=55013
        PHASE_NAME="Phase 2: VPL Core Parameters"
        ;;
    3)
        SELECTOR_START=54014
        SELECTOR_END=54016
        RL_START=55014
        RL_END=55016
        PHASE_NAME="Phase 3: Learning Rate"
        ;;
    4)
        SELECTOR_START=54017
        SELECTOR_END=54017
        RL_START=55017
        RL_END=55017
        PHASE_NAME="Phase 4: Combined Best Parameters"
        ;;
    5)
        SELECTOR_START=54018
        SELECTOR_END=54038
        RL_START=55018
        RL_END=55038
        PHASE_NAME="Phase 5: Fine-grained Hyperparameter Search"
        ;;
    *)
        echo "ERROR: Invalid phase: $PHASE"
        echo "Phase must be 1, 2, 3, 4, or 5"
        exit 1
        ;;
esac

CHECKPOINT_DIR="$WORK_DIR/checkpoints"
MODEL="gemma-2b"
METHOD="vplgp"

echo "=========================================="
echo "Submitting $PHASE_NAME RL Experiments"
echo "RL TID range: $RL_START-$RL_END"
echo "Selector TID range: $SELECTOR_START-$SELECTOR_END"
echo "=========================================="

# Calculate number of experiments
NUM_EXP=$((SELECTOR_END - SELECTOR_START + 1))

for i in $(seq 0 $((NUM_EXP - 1))); do
    selector_tid=$((SELECTOR_START + i))
    rl_tid=$((RL_START + i))
    
    # Check if selector checkpoint exists
    SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${selector_tid}.ckpt"
    if [ ! -f "$SELECTOR_CKPT" ]; then
        SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${selector_tid}.ckpt"
        if [ ! -f "$SELECTOR_CKPT" ]; then
            SELECTOR_CKPT="$CHECKPOINT_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${selector_tid}.ckpt"
        fi
    fi
    
    if [ ! -f "$SELECTOR_CKPT" ]; then
        echo "  WARNING: Selector checkpoint not found for TID=$selector_tid"
        echo "  Skipping RL job..."
        continue
    fi
    
    echo "  Submitting: RL_TID=$rl_tid, Selector_TID=$selector_tid"
    sbatch --job-name="hp_rl_phase${PHASE}_${rl_tid}" \
           scripts/hpsearch/run_rl_hpsearch.sh \
           $rl_tid $selector_tid
    sleep 1
done

echo ""
echo "Phase $PHASE RL jobs submitted."
