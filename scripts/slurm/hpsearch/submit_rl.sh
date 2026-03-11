#!/bin/bash

# Submit Hyperparameter Search RL Experiments
# Usage: bash submit_rl.sh <phase>
#   phase: 1, 2, 3, 4, 5, or 6
# Note: Run this after the corresponding selector phase completes

PHASE=$1

if [ -z "$PHASE" ]; then
    echo "Usage: $0 <phase>"
    echo "  phase: 1, 2, 3, 4, 5, or 6"
    echo ""
    echo "Phase ranges:"
    echo "  Phase 1: RL 55000-55006, Selector 54000-54006 (7 experiments)"
    echo "  Phase 2: RL 55007-55013, Selector 54007-54013 (7 experiments)"
    echo "  Phase 3: RL 55014-55022, Selector 54014-54022 (9 experiments)"
    echo "  Phase 4: RL 55023-55025, Selector 54023-54025 (3 experiments)"
    echo "  Phase 5: RL 55026, Selector 54026 (1 experiment)"
    echo "  Phase 6: RL 55027-55047, Selector 54027-54047 (21 experiments)"
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
        SELECTOR_END=54022
        RL_START=55014
        RL_END=55022
        PHASE_NAME="Phase 3: Refinement (prototype_scale / kl_weight / gp_temperature)"
        ;;
    4)
        SELECTOR_START=54023
        SELECTOR_END=54025
        RL_START=55023
        RL_END=55025
        PHASE_NAME="Phase 4: Learning Rate"
        ;;
    5)
        SELECTOR_START=54026
        SELECTOR_END=54026
        RL_START=55026
        RL_END=55026
        PHASE_NAME="Phase 5: Combined Best Parameters"
        ;;
    6)
        SELECTOR_START=54027
        SELECTOR_END=54047
        RL_START=55027
        RL_END=55047
        PHASE_NAME="Phase 6: Fine-grained Hyperparameter Search"
        ;;
    *)
        echo "ERROR: Invalid phase: $PHASE"
        echo "Phase must be 1, 2, 3, 4, 5, or 6"
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
    # Note: Selector saves as: hhrl_choice_gemma-2b_fedbiscuit_u3_vplgp_ortho_t${selector_tid}.ckpt
    # Try with _ortho_ suffix first (hpsearch naming convention)
    SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${selector_tid}.ckpt"
    if [ ! -f "$SELECTOR_CKPT" ]; then
        SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${selector_tid}.ckpt"
        if [ ! -f "$SELECTOR_CKPT" ]; then
            SELECTOR_CKPT="$CHECKPOINT_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${selector_tid}.ckpt"
            if [ ! -f "$SELECTOR_CKPT" ]; then
                # Fallback: try without _ortho_ (for compatibility)
                SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${selector_tid}.ckpt"
                if [ ! -f "$SELECTOR_CKPT" ]; then
                    SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${selector_tid}.ckpt"
                    if [ ! -f "$SELECTOR_CKPT" ]; then
                        SELECTOR_CKPT="$CHECKPOINT_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${selector_tid}.ckpt"
                    fi
                fi
            fi
        fi
    fi
    
    if [ ! -f "$SELECTOR_CKPT" ]; then
        echo "  WARNING: Selector checkpoint not found for TID=$selector_tid"
        echo "  Skipping RL job..."
        continue
    fi
    
    echo "  Submitting: RL_TID=$rl_tid, Selector_TID=$selector_tid"
    sbatch --job-name="hp_rl_phase${PHASE}_${rl_tid}" \
           scripts/slurm/hpsearch/run_rl_hpsearch.sh \
           $rl_tid $selector_tid
    sleep 1
done

echo ""
echo "Phase $PHASE RL jobs submitted."
