#!/bin/bash
# ============================================================================
# run_rl.sh — Universal RL (Stage 2) training script
# Works on both SLURM cluster and local server.
#
# Usage:
#   # Explicit TIDs:
#   bash scripts/run_rl.sh --method fedvpagp --model gemma --clients 10 \
#       --rl-tid 63130 --selector-tid 62130 [--gpu 3]
#
#   # Auto TIDs (computed from main table ranges):
#   bash scripts/run_rl.sh --method fedvpagp --model gemma --clients 10 [--gpu 3]
#
#   # Dry run:
#   bash scripts/run_rl.sh --method fedvpagp --model gemma --clients 10 --dry-run
# ============================================================================

#SBATCH -p A6000,RTX6000ADA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 3-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
METHOD="" MODEL="" CLIENT_COUNT="" RL_TID="" SELECTOR_TID="" GPU_ID="" DRY_RUN=false
EXTRA_PYTHON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)        METHOD="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --clients)       CLIENT_COUNT="$2"; shift 2 ;;
        --rl-tid)        RL_TID="$2"; shift 2 ;;
        --selector-tid)  SELECTOR_TID="$2"; shift 2 ;;
        --gpu)           GPU_ID="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        --extra-python)  EXTRA_PYTHON="$2"; shift 2 ;;
        *)
            # Positional backward compat: METHOD CLIENT_COUNT RL_TID SELECTOR_TID
            if [ -z "$METHOD" ]; then METHOD="$1"
            elif [ -z "$CLIENT_COUNT" ]; then CLIENT_COUNT="$1"
            elif [ -z "$RL_TID" ]; then RL_TID="$1"
            elif [ -z "$SELECTOR_TID" ]; then SELECTOR_TID="$1"
            else echo "Unknown argument: $1"; exit 1
            fi
            shift ;;
    esac
done

if [ -z "$METHOD" ] || [ -z "$MODEL" ] || [ -z "$CLIENT_COUNT" ]; then
    echo "Usage: $0 --method <method> --model <model> --clients <N> [--rl-tid <tid>] [--selector-tid <tid>] [--gpu <id>] [--dry-run]"
    echo ""
    echo "  Methods: feddpo, fedbiscuit, fedvpl, fedvpagp"
    echo "  Models:  gemma (gemma-2b), qwen (qwen2)"
    echo "  Clients: 10, 50, 100"
    exit 1
fi

# ── Source shared libraries ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/hyperparams.sh"

setup_environment "$SCRIPT_DIR"

# ── Resolve parameters ───────────────────────────────────────────────────────
MODEL_CANONICAL=$(get_model_canonical "$MODEL")

# Auto-compute TIDs if not provided
if [ -z "$RL_TID" ]; then
    RL_BASE=$(get_rl_base_tid "$MODEL")
    RL_TID=$(calc_tid "$RL_BASE" "$METHOD" "$CLIENT_COUNT")
    echo "Auto-computed RL_TID=$RL_TID"
fi
if [ -z "$SELECTOR_TID" ]; then
    SELECTOR_BASE=$(get_selector_base_tid "$MODEL")
    SELECTOR_TID=$(calc_tid "$SELECTOR_BASE" "$METHOD" "$CLIENT_COUNT")
    echo "Auto-computed SELECTOR_TID=$SELECTOR_TID"
fi

TRAINER=$(get_rl_trainer)
CONFIG_BASE=$(get_rl_config_base "$METHOD")
MODEL_HF=$(get_model_hf_id "$MODEL")
CKPT_PATH="$CHECKPOINT_DIR/hhrl_rlhf_${MODEL_CANONICAL}_choice_${METHOD}_t${RL_TID}.ckpt"

# ── Find selector checkpoint (if needed) ─────────────────────────────────────
USE_SELECTOR=false
SELECTOR_CKPT=""
if method_needs_selector "$METHOD"; then
    USE_SELECTOR=true
    if [ "$DRY_RUN" = false ]; then
        find_selector_checkpoint "$MODEL_CANONICAL" "$METHOD" "$SELECTOR_TID" || exit 1
    fi
fi

# ── Build Python config snippet ─────────────────────────────────────────────
PYTHON_SNIPPET="
config['federate']['client_num'] = 1
config['federate']['save_to'] = '$CKPT_PATH'
config['model']['type'] = '$MODEL_HF'
config['trainer']['type'] = '$TRAINER'
config['expname'] = '${METHOD}_${MODEL_CANONICAL}_n${CLIENT_COUNT}_rl_t${RL_TID}'
$(get_rl_model_params "$MODEL")
$(get_rl_common_params)
"

# Add VPL params if method needs selector
if [ "$USE_SELECTOR" = true ]; then
    PYTHON_SNIPPET="$PYTHON_SNIPPET
$(get_vpl_rl_params "$MODEL" "$METHOD" "$SELECTOR_CKPT")
"
fi

# Add any extra Python snippet
if [ -n "$EXTRA_PYTHON" ]; then
    PYTHON_SNIPPET="$PYTHON_SNIPPET
$EXTRA_PYTHON
"
fi

# ── Generate config ──────────────────────────────────────────────────────────
CONFIG_FILE="cfg/experiments/${MODEL_CANONICAL}/${METHOD}/hrl_n${CLIENT_COUNT}_t${RL_TID}.yaml"

# ── Dry run: print summary and exit ─────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "=== DRY RUN: RL Training ==="
    echo "  Method:       $METHOD"
    echo "  Model:        $MODEL_CANONICAL ($MODEL_HF)"
    echo "  Clients:      $CLIENT_COUNT"
    echo "  RL TID:       $RL_TID"
    echo "  Selector TID: $SELECTOR_TID"
    echo "  Use selector: $USE_SELECTOR"
    echo "  Trainer:      $TRAINER"
    echo "  Base config:  $CONFIG_BASE"
    echo "  Output cfg:   $CONFIG_FILE"
    echo "  Checkpoint:   $CKPT_PATH"
    echo "  Environment:  $ENV_TYPE"
    echo "  GPU:          ${GPU_ID:-auto}"
    echo ""
    echo "Would generate config and run:"
    echo "  python -u federatedscope/llm/rlhf/main.py --cfg $CONFIG_FILE"
    exit 0
fi

generate_config "$CONFIG_BASE" "$CONFIG_FILE" "$PYTHON_SNIPPET"

# ── Find selector config for --selector-cfg-file ────────────────────────────
SELECTOR_CFG_FLAG=""
if [ "$USE_SELECTOR" = true ]; then
    SELECTOR_CFG="cfg/experiments/${MODEL_CANONICAL}/${METHOD}/hhst_n${CLIENT_COUNT}_t${SELECTOR_TID}.yaml"
    if [ ! -f "$SELECTOR_CFG" ]; then
        # Try legacy path
        SELECTOR_CFG="cfg/main_table/${MODEL_CANONICAL}/${METHOD}/hhst_n${CLIENT_COUNT}_${SELECTOR_TID}.yaml"
    fi
    if [ -f "$SELECTOR_CFG" ]; then
        SELECTOR_CFG_FLAG="--selector-cfg-file $SELECTOR_CFG"
        echo "Using selector config: $SELECTOR_CFG"
    else
        echo "WARNING: Selector config not found, running without --selector-cfg-file"
    fi
fi

# ── Set GPU if specified (local server only) ─────────────────────────────────
if [ -n "$GPU_ID" ] && [ "$ENV_TYPE" = "local" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Using GPU $GPU_ID"
fi

# ── Run ──────────────────────────────────────────────────────────────────────
echo ""
echo "Starting RL training: $METHOD, $MODEL_CANONICAL, N=$CLIENT_COUNT, RL_TID=$RL_TID"
echo "Config: $CONFIG_FILE"
[ "$USE_SELECTOR" = true ] && echo "Selector checkpoint: $SELECTOR_CKPT"

if [ "$ENV_TYPE" = "local" ]; then
    nohup python -u federatedscope/llm/rlhf/main.py \
        --cfg "$CONFIG_FILE" \
        $SELECTOR_CFG_FLAG \
        > "outputs/${RL_TID}.log" 2>&1 &
    echo "Started in background (PID=$!). Log: outputs/${RL_TID}.log"
else
    python -u federatedscope/llm/rlhf/main.py \
        --cfg "$CONFIG_FILE" \
        $SELECTOR_CFG_FLAG \
        > "outputs/${RL_TID}.log" 2>&1
    echo "RL experiment completed: TID=$RL_TID"
fi
