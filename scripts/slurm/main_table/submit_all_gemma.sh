#!/bin/bash

# Submit all Gemma-2B main table experiments to SLURM
# This script submits all selector and RL jobs

WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR

# TID ranges
# Selector: 62100-62132
# RL: 63100-63132

# Experiment matrix
# Note: FedDPO is only for RL, not for selector training
declare -a SELECTOR_METHODS=("fedbiscuit" "fedvpl" "fedvpagp")
declare -a RL_METHODS=("feddpo" "fedbiscuit" "fedvpl" "fedvpagp")
declare -a CLIENT_COUNTS=("10" "50" "100")

# Base TIDs
SELECTOR_BASE=62100
RL_BASE=63100

# Method offsets (selector: FedDPO 제외)
declare -A SELECTOR_METHOD_OFFSETS=(
    ["fedbiscuit"]=10
    ["fedvpl"]=20
    ["fedvpagp"]=30
)

# Method offsets (RL: FedDPO 포함)
declare -A RL_METHOD_OFFSETS=(
    ["feddpo"]=0
    ["fedbiscuit"]=10
    ["fedvpl"]=20
    ["fedvpagp"]=30
)

# Client count offsets
declare -A CLIENT_OFFSETS=(
    ["10"]=0
    ["50"]=1
    ["100"]=2
)

echo "=========================================="
echo "Submitting Gemma-2B Main Table Experiments"
echo "(Selector + RL with automatic SLURM dependency chaining)"
echo "=========================================="

# Track selector job IDs for dependency chaining
declare -A SELECTOR_JOB_IDS=()

# Submit selector jobs (FedDPO 제외)
echo ""
echo "Submitting Selector Jobs (FedDPO는 RL에서만 사용)..."
for method in "${SELECTOR_METHODS[@]}"; do
    for client_count in "${CLIENT_COUNTS[@]}"; do
        method_offset=${SELECTOR_METHOD_OFFSETS[$method]}
        client_offset=${CLIENT_OFFSETS[$client_count]}
        tid=$((SELECTOR_BASE + method_offset + client_offset))

        echo "  Submitting: $method, N=$client_count, TID=$tid"
        JOB_OUTPUT=$(sbatch --job-name="sel_${method}_n${client_count}_${tid}" \
               scripts/slurm/main_table/run_selector_gemma.sh \
               $method $client_count $tid)
        JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
        SELECTOR_JOB_IDS["${method}:${client_count}"]="$JOB_ID"
        echo "    -> SLURM job $JOB_ID"
        sleep 1
    done
done

# Submit RL jobs with --dependency=afterok on their selector
echo ""
echo "Submitting RL Jobs (with dependency on selector completion)..."
for method in "${RL_METHODS[@]}"; do
    for client_count in "${CLIENT_COUNTS[@]}"; do
        method_offset=${RL_METHOD_OFFSETS[$method]}
        client_offset=${CLIENT_OFFSETS[$client_count]}
        selector_tid=$((SELECTOR_BASE + ${SELECTOR_METHOD_OFFSETS[$method]:-0} + client_offset))
        rl_tid=$((RL_BASE + method_offset + client_offset))

        # Build dependency flag (FedDPO has no selector)
        DEP_FLAG=""
        SEL_JOB="${SELECTOR_JOB_IDS["${method}:${client_count}"]:-}"
        if [ -n "$SEL_JOB" ]; then
            DEP_FLAG="--dependency=afterok:${SEL_JOB}"
        fi

        # FedDPO doesn't use selector
        if [ "$method" == "feddpo" ]; then
            selector_tid="dummy"
        fi

        echo "  Submitting: $method, N=$client_count, RL_TID=$rl_tid ${DEP_FLAG:+(depends on job $SEL_JOB)}"
        sbatch $DEP_FLAG --job-name="rl_${method}_n${client_count}_${rl_tid}" \
               scripts/slurm/main_table/run_rl_gemma.sh \
               $method $client_count $rl_tid $selector_tid
        sleep 1
    done
done

echo ""
echo "All jobs submitted with dependency chaining."
echo "RL jobs will start automatically after their selector completes."
echo "Monitor with: squeue -u \$USER"
