#!/bin/bash
# ============================================================================
# submit.sh — Unified batch orchestrator for experiment submission
# Supports SLURM (with --dependency chaining) and local server execution.
#
# Usage:
#   # Submit full pipeline (selector + RL with automatic dependency):
#   bash scripts/submit.sh --model gemma --phase all
#
#   # Submit only selector or RL:
#   bash scripts/submit.sh --model gemma --phase selector
#   bash scripts/submit.sh --model gemma --phase rl
#
#   # Single experiment:
#   bash scripts/submit.sh --model gemma --method fedvpagp --clients 10 --phase all
#
#   # Local server (sequential, specific GPU):
#   bash scripts/submit.sh --model gemma --method fedvpagp --clients 10 --phase all --gpu 3
#
#   # Dry run:
#   bash scripts/submit.sh --model gemma --phase all --dry-run
# ============================================================================

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
MODEL="" PHASE="all" GPU_ID="" DRY_RUN=false
FILTER_METHOD="" FILTER_CLIENTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2"; shift 2 ;;
        --phase)    PHASE="$2"; shift 2 ;;  # selector, rl, all
        --method)   FILTER_METHOD="$2"; shift 2 ;;
        --clients)  FILTER_CLIENTS="$2"; shift 2 ;;
        --gpu)      GPU_ID="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Usage: $0 --model <gemma|qwen> [--phase <selector|rl|all>] [--method <method>] [--clients <N>] [--gpu <id>] [--dry-run]"
    exit 1
fi

# ── Source shared libraries ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/hyperparams.sh"

setup_environment "$SCRIPT_DIR"

MODEL_CANONICAL=$(get_model_canonical "$MODEL")

# ── Build experiment matrix ──────────────────────────────────────────────────
# FedDPO only needs RL (no selector training)
SELECTOR_METHODS=("fedbiscuit" "fedvpl" "fedvpagp")
RL_METHODS=("feddpo" "fedbiscuit" "fedvpl" "fedvpagp")
CLIENT_COUNTS=("10" "50" "100")

# Apply filters
if [ -n "$FILTER_METHOD" ]; then
    SELECTOR_METHODS=("$FILTER_METHOD")
    RL_METHODS=("$FILTER_METHOD")
fi
if [ -n "$FILTER_CLIENTS" ]; then
    CLIENT_COUNTS=("$FILTER_CLIENTS")
fi

SELECTOR_BASE=$(get_selector_base_tid "$MODEL")
RL_BASE=$(get_rl_base_tid "$MODEL")

DRY_FLAG=""
[ "$DRY_RUN" = true ] && DRY_FLAG="--dry-run"

GPU_FLAG=""
[ -n "$GPU_ID" ] && GPU_FLAG="--gpu $GPU_ID"

echo "=========================================="
echo "Experiment Submission: $MODEL_CANONICAL"
echo "  Phase:   $PHASE"
echo "  Env:     $ENV_TYPE"
echo "  Dry run: $DRY_RUN"
echo "=========================================="

# Track SLURM job IDs for dependency chaining: SELECTOR_JOBS[method:clients]=job_id
declare -A SELECTOR_JOBS=()

# ── Submit Selector Jobs ─────────────────────────────────────────────────────
if [ "$PHASE" = "selector" ] || [ "$PHASE" = "all" ]; then
    echo ""
    echo "--- Selector Jobs ---"

    for method in "${SELECTOR_METHODS[@]}"; do
        # Skip feddpo for selector (it has no selector stage)
        [ "$method" = "feddpo" ] && continue

        for clients in "${CLIENT_COUNTS[@]}"; do
            tid=$(calc_tid "$SELECTOR_BASE" "$method" "$clients")

            echo "  [$method N=$clients] TID=$tid"

            if [ "$ENV_TYPE" = "slurm" ]; then
                if [ "$DRY_RUN" = true ]; then
                    echo "    Would run: sbatch --job-name=sel_${method}_n${clients}_${tid} scripts/run_selector.sh --method $method --model $MODEL --clients $clients --tid $tid"
                else
                    JOB_OUTPUT=$(sbatch --job-name="sel_${method}_n${clients}_${tid}" \
                        "$SCRIPT_DIR/run_selector.sh" \
                        --method "$method" --model "$MODEL" --clients "$clients" --tid "$tid")
                    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
                    SELECTOR_JOBS["${method}:${clients}"]="$JOB_ID"
                    echo "    Submitted SLURM job $JOB_ID"
                fi
            else
                # Local: run directly (will nohup in background)
                bash "$SCRIPT_DIR/run_selector.sh" \
                    --method "$method" --model "$MODEL" --clients "$clients" --tid "$tid" \
                    $GPU_FLAG $DRY_FLAG
            fi
        done
    done
fi

# ── Submit RL Jobs ───────────────────────────────────────────────────────────
if [ "$PHASE" = "rl" ] || [ "$PHASE" = "all" ]; then
    echo ""
    echo "--- RL Jobs ---"

    for method in "${RL_METHODS[@]}"; do
        for clients in "${CLIENT_COUNTS[@]}"; do
            rl_tid=$(calc_tid "$RL_BASE" "$method" "$clients")
            selector_tid=$(calc_tid "$SELECTOR_BASE" "$method" "$clients")

            echo "  [$method N=$clients] RL_TID=$rl_tid, SEL_TID=$selector_tid"

            # Build dependency flag for SLURM
            DEP_FLAG=""
            if [ "$ENV_TYPE" = "slurm" ] && method_needs_selector "$method"; then
                SELECTOR_JOB="${SELECTOR_JOBS["${method}:${clients}"]:-}"
                if [ -n "$SELECTOR_JOB" ]; then
                    DEP_FLAG="--dependency=afterok:${SELECTOR_JOB}"
                    echo "    Depends on SLURM job $SELECTOR_JOB (selector)"
                fi
            fi

            if [ "$ENV_TYPE" = "slurm" ]; then
                if [ "$DRY_RUN" = true ]; then
                    echo "    Would run: sbatch ${DEP_FLAG} --job-name=rl_${method}_n${clients}_${rl_tid} scripts/run_rl.sh --method $method --model $MODEL --clients $clients --rl-tid $rl_tid --selector-tid $selector_tid"
                else
                    sbatch $DEP_FLAG --job-name="rl_${method}_n${clients}_${rl_tid}" \
                        "$SCRIPT_DIR/run_rl.sh" \
                        --method "$method" --model "$MODEL" --clients "$clients" \
                        --rl-tid "$rl_tid" --selector-tid "$selector_tid"
                    echo "    Submitted"
                fi
            else
                if [ "$DRY_RUN" = true ]; then
                    bash "$SCRIPT_DIR/run_rl.sh" \
                        --method "$method" --model "$MODEL" --clients "$clients" \
                        --rl-tid "$rl_tid" --selector-tid "$selector_tid" \
                        $GPU_FLAG --dry-run
                else
                    echo "    WARNING: On local server, RL jobs should be started manually after selector completes."
                    echo "    Run: bash scripts/run_rl.sh --method $method --model $MODEL --clients $clients --rl-tid $rl_tid --selector-tid $selector_tid $GPU_FLAG"
                fi
            fi
        done
    done
fi

echo ""
echo "=========================================="
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN complete. No jobs were submitted."
elif [ "$ENV_TYPE" = "slurm" ] && [ "$PHASE" = "all" ]; then
    echo "All jobs submitted with dependency chaining."
    echo "RL jobs will start automatically after their selector completes."
    echo "Monitor with: squeue -u \$USER"
else
    echo "Submission complete."
fi
echo "=========================================="
