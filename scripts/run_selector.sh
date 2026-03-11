#!/bin/bash
# ============================================================================
# run_selector.sh — Universal selector (Stage 1) training script
# Works on both SLURM cluster and local server.
#
# Usage:
#   # Explicit TID:
#   bash scripts/run_selector.sh --method fedvpagp --model gemma --clients 10 --tid 62130 [--gpu 3]
#
#   # Auto TID (computed from main table ranges):
#   bash scripts/run_selector.sh --method fedvpagp --model gemma --clients 10 [--gpu 3]
#
#   # SLURM submission:
#   sbatch scripts/run_selector.sh --method fedvpagp --model gemma --clients 10 --tid 62130
#
#   # Dry run (print config, don't execute):
#   bash scripts/run_selector.sh --method fedvpagp --model gemma --clients 10 --dry-run
# ============================================================================

#SBATCH -p A6000,RTX6000ADA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
METHOD="" MODEL="" CLIENT_COUNT="" TID="" GPU_ID="" DRY_RUN=false
EXTRA_PYTHON=""  # additional Python snippet for config modification

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)       METHOD="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --clients)      CLIENT_COUNT="$2"; shift 2 ;;
        --tid)          TID="$2"; shift 2 ;;
        --gpu)          GPU_ID="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --extra-python) EXTRA_PYTHON="$2"; shift 2 ;;
        *)
            # Support positional args for backward compat: METHOD CLIENT_COUNT TID
            if [ -z "$METHOD" ]; then METHOD="$1"
            elif [ -z "$CLIENT_COUNT" ]; then CLIENT_COUNT="$1"
            elif [ -z "$TID" ]; then TID="$1"
            else echo "Unknown argument: $1"; exit 1
            fi
            shift ;;
    esac
done

if [ -z "$METHOD" ] || [ -z "$MODEL" ] || [ -z "$CLIENT_COUNT" ]; then
    echo "Usage: $0 --method <method> --model <model> --clients <N> [--tid <tid>] [--gpu <id>] [--dry-run]"
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

# Auto-compute TID if not provided
if [ -z "$TID" ]; then
    SELECTOR_BASE=$(get_selector_base_tid "$MODEL")
    TID=$(calc_tid "$SELECTOR_BASE" "$METHOD" "$CLIENT_COUNT")
    echo "Auto-computed TID=$TID"
fi

SAMPLE_CLIENT_NUM=$(get_sample_client_num "$CLIENT_COUNT")
TRAINER=$(get_selector_trainer "$METHOD")
CONFIG_BASE=$(get_selector_config_base "$METHOD")
MODEL_HF=$(get_model_hf_id "$MODEL")
CKPT_PATH="$CHECKPOINT_DIR/hhrl_choice_${MODEL_CANONICAL}_fedbiscuit_u3_${METHOD}_t${TID}.ckpt"

# ── Build Python config snippet ─────────────────────────────────────────────
PYTHON_SNIPPET="
config['federate']['client_num'] = $CLIENT_COUNT
config['federate']['sample_client_num'] = $SAMPLE_CLIENT_NUM
config['federate']['save_to'] = '$CKPT_PATH'
config['model']['type'] = '$MODEL_HF'
config['trainer']['type'] = '$TRAINER'
config['expname'] = '${METHOD}_${MODEL_CANONICAL}_n${CLIENT_COUNT}_t${TID}'
$(get_selector_model_params "$MODEL")
"

# Add FedVPA-GP params if needed
if [ "$METHOD" = "fedvpagp" ]; then
    PYTHON_SNIPPET="$PYTHON_SNIPPET
$(get_fedvpagp_selector_params "$MODEL")
"
fi

# Add any extra Python snippet
if [ -n "$EXTRA_PYTHON" ]; then
    PYTHON_SNIPPET="$PYTHON_SNIPPET
$EXTRA_PYTHON
"
fi

# ── Generate config ──────────────────────────────────────────────────────────
CONFIG_FILE="cfg/experiments/${MODEL_CANONICAL}/${METHOD}/hhst_n${CLIENT_COUNT}_t${TID}.yaml"

# ── Dry run: print summary and exit ─────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "=== DRY RUN: Selector Training ==="
    echo "  Method:     $METHOD"
    echo "  Model:      $MODEL_CANONICAL ($MODEL_HF)"
    echo "  Clients:    $CLIENT_COUNT (sample=$SAMPLE_CLIENT_NUM)"
    echo "  TID:        $TID"
    echo "  Trainer:    $TRAINER"
    echo "  Base config:$CONFIG_BASE"
    echo "  Output cfg: $CONFIG_FILE"
    echo "  Checkpoint: $CKPT_PATH"
    echo "  Environment:$ENV_TYPE"
    echo "  GPU:        ${GPU_ID:-auto}"
    echo ""
    echo "Would generate config and run:"
    echo "  python -u federatedscope/main.py --cfg $CONFIG_FILE"
    exit 0
fi

generate_config "$CONFIG_BASE" "$CONFIG_FILE" "$PYTHON_SNIPPET"

# ── Set GPU if specified (local server only) ─────────────────────────────────
if [ -n "$GPU_ID" ] && [ "$ENV_TYPE" = "local" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Using GPU $GPU_ID"
fi

# ── Run ──────────────────────────────────────────────────────────────────────
echo ""
echo "Starting selector training: $METHOD, $MODEL_CANONICAL, N=$CLIENT_COUNT, TID=$TID"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CKPT_PATH"

if [ "$ENV_TYPE" = "local" ]; then
    # Local: run in background with nohup
    nohup python -u federatedscope/main.py \
        --cfg "$CONFIG_FILE" \
        > "outputs/${TID}.log" 2>&1 &
    echo "Started in background (PID=$!). Log: outputs/${TID}.log"
else
    # SLURM: run in foreground (SLURM manages the job)
    python -u federatedscope/main.py \
        --cfg "$CONFIG_FILE" \
        > "outputs/${TID}.log" 2>&1
    echo "Experiment completed: TID=$TID"
fi
