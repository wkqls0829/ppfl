#!/bin/bash

# Submit Hyperparameter Search RL Experiments (Qwen 2)
# Usage: bash submit_rl_qwen.sh <phase>
#   phase: 1, 2, 3, 4, or 5
# RL TID: 55100-55138, Selector TID: 54100-54138

PHASE=$1
if [ -z "$PHASE" ]; then
    echo "Usage: $0 <phase>"
    echo "  phase: 1, 2, 3, 4, or 5"
    echo ""
    echo "Phase ranges (Qwen):"
    echo "  Phase 1: RL 55100-55106, Selector 54100-54106"
    echo "  Phase 2: RL 55107-55113, Selector 54107-54113"
    echo "  Phase 3: RL 55114-55116, Selector 54114-54116"
    echo "  Phase 4: RL 55117, Selector 54117"
    echo "  Phase 5: RL 55118-55138, Selector 54118-54138"
    exit 1
fi

WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR

case $PHASE in
    1) SELECTOR_START=54100; SELECTOR_END=54106; RL_START=55100; RL_END=55106; PHASE_NAME="Phase 1: Orthogonal Loss (Qwen)" ;;
    2) SELECTOR_START=54107; SELECTOR_END=54113; RL_START=55107; RL_END=55113; PHASE_NAME="Phase 2: VPL Core (Qwen)" ;;
    3) SELECTOR_START=54114; SELECTOR_END=54116; RL_START=55114; RL_END=55116; PHASE_NAME="Phase 3: Learning Rate (Qwen)" ;;
    4) SELECTOR_START=54117; SELECTOR_END=54117; RL_START=55117; RL_END=55117; PHASE_NAME="Phase 4: Combined Best (Qwen)" ;;
    5) SELECTOR_START=54118; SELECTOR_END=54138; RL_START=55118; RL_END=55138; PHASE_NAME="Phase 5: Fine-grained (Qwen)" ;;
    *) echo "ERROR: Invalid phase: $PHASE"; exit 1 ;;
esac

if [ -d "/hdd/hdd3/kjb" ]; then
    CHECKPOINT_DIR="/hdd/hdd3/kjb/checkpoints"
else
    CHECKPOINT_DIR="$WORK_DIR/checkpoints"
fi
MODEL="qwen2"
METHOD="vplgp"

echo "=========================================="
echo "$PHASE_NAME"
echo "RL TID: $RL_START-$RL_END, Selector TID: $SELECTOR_START-$SELECTOR_END"
echo "=========================================="

NUM_EXP=$((SELECTOR_END - SELECTOR_START + 1))
for i in $(seq 0 $((NUM_EXP - 1))); do
    selector_tid=$((SELECTOR_START + i))
    rl_tid=$((RL_START + i))
    SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${selector_tid}.ckpt"
    if [ ! -f "$SELECTOR_CKPT" ]; then
        SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${selector_tid}.ckpt"
    fi
    if [ ! -f "$SELECTOR_CKPT" ]; then
        SELECTOR_CKPT="$CHECKPOINT_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${selector_tid}.ckpt"
    fi
    if [ ! -f "$SELECTOR_CKPT" ]; then
        echo "  WARNING: Qwen selector checkpoint not found for TID=$selector_tid, skipping RL job"
        continue
    fi
    echo "  Submitting: RL_TID=$rl_tid, Selector_TID=$selector_tid"
    sbatch --job-name="hp_rl_qwen_p${PHASE}_${rl_tid}" scripts/hpsearch/run_rl_hpsearch_qwen.sh $rl_tid $selector_tid
    sleep 1
done
echo "Phase $PHASE (Qwen) RL jobs submitted."
