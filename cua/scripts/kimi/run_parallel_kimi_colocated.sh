#!/bin/bash
# ===============================================================================
# Kimi-K2.5 Colocated: vLLM server + data collection on same 2 reserved GPU nodes
# ===============================================================================
# Uses 2 reserved GPU nodes for both:
#   - Kimi vLLM Ray cluster (TP=16 across both nodes, using GPUs)
#   - Data collection on both nodes (using CPU)
#
# Supports both singularity (local KVM) and nvcf (remote NVCF VMs) runtimes.
#
# Flow:
#   1. Submits run_kimi.sbatch (with reservation) for 2 GPU nodes
#   2. Waits for vLLM to become healthy
#   3. SSH+enroot execs into each node's container to run data collection
#   4. Waits for both collectors, then cancels the server job
#
# Required env vars (for nvcf runtime):
#   NGC_API_KEY       - NVCF API key
#   NGC_ORG           - NVCF organization
#
# Usage:
#   bash run_parallel_kimi_colocated.sh
#   MAX_PARALLEL=12 RUNTIME=nvcf GENERATION_MODE=zenodo bash run_parallel_kimi_colocated.sh
#   MAX_PARALLEL=16 RUNTIME=singularity GENERATION_MODE=zenodo bash run_parallel_kimi_colocated.sh
#   MAX_PARALLEL=16 RUNTIME=nvcf_singularity GENERATION_MODE=zenodo bash run_parallel_kimi_colocated.sh
# Log files:
#   logs/slurm-<jobid>-server.out
#   logs/slurm-<jobid>-collector-1.out
#   logs/slurm-<jobid>-collector-2.out
# ============================================================================

export LOG_DIR="${LOG_DIR:-./logs}"

# Load .env as defaults (won't override existing env vars)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../../.env"
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        if [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"
fi

# Configurable parameters
RUNTIME="${RUNTIME:-singularity}"
GENERATION_MODE="${GENERATION_MODE:-spreadsheetbench}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-10000}"
TRAJECTORY_SAVE_DIR="${TRAJECTORY_SAVE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2/cua/trajectories/kimi_$GENERATION_MODE}"
# NVCF_FUNCTION_NAME_PREFIX is set after KIMI_JOB_ID is known (see below)

# Validate NVCF credentials if nvcf runtime
if [ "$RUNTIME" = "nvcf" ]; then
    if [ -z "$NGC_API_KEY" ]; then
        echo "[colocated] ERROR: NGC_API_KEY not set. Required for NVCF runtime."
        exit 1
    fi
    if [ -z "$NGC_ORG" ]; then
        echo "[colocated] ERROR: NGC_ORG not set. Required for NVCF runtime."
        exit 1
    fi
fi

# Create logs directory
mkdir -p "$LOG_DIR"

KIMI_JOB_ID=""
COLLECTOR_PIDS=()
KIMI_PORT=8000


# TODO: change to your proejct root dir
# PROJECT_ROOT="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2"
PROJECT_ROOT="/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server"
PROJECT_DIR="$PROJECT_ROOT/cua"

echo "============================================"
echo "Kimi-K2.5 Colocated (vLLM + Collection)"
echo "============================================"
echo "RUNTIME:           $RUNTIME"
echo "GENERATION_MODE:   $GENERATION_MODE"
echo "MAX_PARALLEL:      $MAX_PARALLEL (per node)"
echo "MAX_TRAJECTORIES:  $MAX_TRAJECTORIES (per node)"
echo "TRAJECTORY_SAVE_DIR: $TRAJECTORY_SAVE_DIR"
echo ""


# --- Cleanup: cancel server on exit ---
_CLEANUP_DONE=false
cleanup() {
    if $_CLEANUP_DONE; then return; fi
    _CLEANUP_DONE=true

    echo ""
    echo "[colocated] Cleaning up..."

    # 1. Kill collector SSH sessions
    if [ ${#COLLECTOR_PIDS[@]} -gt 0 ]; then
        echo "[colocated] Killing ${#COLLECTOR_PIDS[@]} collector(s)..."
        for pid in "${COLLECTOR_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
        done
    fi

    # 2. Cancel Kimi vLLM server
    if [ -n "$KIMI_JOB_ID" ]; then
        echo "[colocated] Cancelling Kimi vLLM job $KIMI_JOB_ID"
        SCANCEL_OUTPUT=$(scancel "$KIMI_JOB_ID" 2>&1)
        SCANCEL_EXIT=$?
        if [ $SCANCEL_EXIT -ne 0 ]; then
            echo "[colocated] WARNING: scancel failed (exit $SCANCEL_EXIT): $SCANCEL_OUTPUT"
        else
            echo "[colocated] scancel succeeded for job $KIMI_JOB_ID"
        fi
    fi

    # 3. Cleanup NVCF functions for this job
    if [ "$RUNTIME" = "nvcf" ] && [ -n "$NVCF_FUNCTION_NAME_PREFIX" ]; then
        echo "[colocated] Cleaning up NVCF functions with prefix: $NVCF_FUNCTION_NAME_PREFIX"
        python "$SCRIPT_DIR/cleanup_nvcf_functions.py" --name-prefix "$NVCF_FUNCTION_NAME_PREFIX" 2>&1 || true
    fi

    # 4. Remove head node file
    rm -f "$LOG_DIR/head_node_${KIMI_JOB_ID}"
}
trap cleanup EXIT
trap 'cleanup; exit 130' SIGTERM SIGINT


# --- 1. Submit Kimi vLLM server ---
if [ "$RUNTIME" = "nvcf" ]; then
    echo "[colocated] Submitting Kimi vLLM sbatch job (NVCF runtime)..."
#    KIMI_JOB_ID=$(sbatch \
#        --account=nvr_lpr_agentic \
#        --partition=batch_block1 \
#        --time=04:00:00 \
#        --output="$LOG_DIR/slurm-%j-server.out" \
#        --error="$LOG_DIR/slurm-%j-server.out" \
#        --parsable \
#        "./run_kimi.sbatch")
    KIMI_JOB_ID=$(sbatch \
    --account=llmservice_fm_vision \
    --reservation=sla_res_osworld_agent_vlm \
    --partition=batch_block1 \
    --time=04:00:00 \
    --output="$LOG_DIR/slurm-%j-server.out" \
    --error="$LOG_DIR/slurm-%j-server.out" \
    --parsable \
    "./run_kimi.sbatch")
else
    echo "[colocated] Submitting Kimi vLLM sbatch job (KVM runtime, reserved nodes)..."
    KIMI_JOB_ID=$(sbatch \
        --account=nvr_lacr_llm \
        --partition=batch_short \
        --time=02:00:00 \
        --output="$LOG_DIR/slurm-%j-server.out" \
        --error="$LOG_DIR/slurm-%j-server.out" \
        --parsable \
        "./run_kimi.sbatch")
fi

if [ -z "$KIMI_JOB_ID" ]; then
    echo "[colocated] ERROR: Kimi sbatch submission failed."
    exit 1
fi
echo "[colocated] Kimi vLLM job submitted: $KIMI_JOB_ID"

# Set job-specific NVCF function name prefix for isolated cleanup
NVCF_FUNCTION_NAME_PREFIX="kimi-${KIMI_JOB_ID}"
export NVCF_FUNCTION_NAME_PREFIX
echo "[colocated] NVCF function prefix: $NVCF_FUNCTION_NAME_PREFIX"

# --- 2. Wait for job to start and discover nodes ---
HEAD_NODE_FILE="$LOG_DIR/head_node_${KIMI_JOB_ID}"
echo "[colocated] Waiting for head node file: $HEAD_NODE_FILE"
ELAPSED=0
MAX_WAIT=43200  # 12 hours

MODEL_NODE=""
ALL_NODES=""
while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if job is still alive
    JOB_STATE=$(squeue -j "$KIMI_JOB_ID" -h -o %T 2>/dev/null)
    if [ -z "$JOB_STATE" ]; then
        echo "[colocated] ERROR: Kimi job $KIMI_JOB_ID disappeared from queue!"
        exit 1
    fi

    # Check for head node file (written by run_kimi.sbatch once it starts)
    if [ -f "$HEAD_NODE_FILE" ]; then
        MODEL_NODE=$(cat "$HEAD_NODE_FILE")
        if [ -n "$MODEL_NODE" ]; then
            ALL_NODES=$(scontrol show hostnames "$(squeue -j "$KIMI_JOB_ID" -h -o %N)")
            echo "[colocated] Kimi vLLM job running on: $(echo $ALL_NODES | tr '\n' ' ')"
            break
        fi
    fi

    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "[colocated] Still waiting for Kimi job to start (${ELAPSED}s)..."
    fi
done

if [ -z "$MODEL_NODE" ]; then
    echo "[colocated] ERROR: Kimi vLLM did not start within ${MAX_WAIT}s."
    exit 1
fi

mapfile -t NODES_ARRAY <<< "$ALL_NODES"

echo "[colocated] Head node (vLLM API): $MODEL_NODE"
echo "[colocated] All nodes: ${NODES_ARRAY[*]}"

# --- 3. Wait for vLLM health ---
echo "[colocated] Waiting for vLLM health at $MODEL_NODE:$KIMI_PORT..."
ELAPSED=0
MAX_HEALTH_WAIT=3600  # 2 hours

while [ $ELAPSED -lt $MAX_HEALTH_WAIT ]; do
    if curl -sf "http://$MODEL_NODE:$KIMI_PORT/health" > /dev/null 2>&1; then
        echo "[colocated] vLLM is healthy!"
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "[colocated] Still waiting for vLLM health (${ELAPSED}s)..."
    fi
done

if [ $ELAPSED -ge $MAX_HEALTH_WAIT ]; then
    echo "[colocated] ERROR: vLLM did not become healthy within ${MAX_HEALTH_WAIT}s."
    exit 1
fi

# --- 4. Launch data collection on each node via SSH+enroot ---
echo "[colocated] Launching data collection on ${#NODES_ARRAY[@]} node(s)..."
COLLECTOR_PIDS=()

for i in "${!NODES_ARRAY[@]}"; do
    node=${NODES_ARRAY[$i]}
    COLLECTOR_IDX=$((i + 1))
    CURRENT_LOG="$LOG_DIR/slurm-${KIMI_JOB_ID}-collector-${COLLECTOR_IDX}.out"

    echo "[colocated] Finding container on $node..."
    CONTAINER_PID=""
    while [ -z "$CONTAINER_PID" ]; do
        sleep 2
        CONTAINER_PID=$(ssh -q -o StrictHostKeyChecking=no "$node" \
            "enroot list -f | grep 'pyxis' | head -n 1 | awk '{print \$2}'" 2>/dev/null)
    done
    echo "[colocated] Container on $node ready, PID: $CONTAINER_PID"

    # Build NVCF env exports if needed
    NVCF_EXPORTS=""
    RUNTIME_ARG=""
    if [ "$RUNTIME" = "nvcf" ]; then
        NVCF_EXPORTS="export NGC_API_KEY=$NGC_API_KEY; export NGC_ORG=$NGC_ORG; export NVCF_FUNCTION_NAME_PREFIX=$NVCF_FUNCTION_NAME_PREFIX; export OSWORLD_SETUP_CACHE_DIR=/tmp/osworld_cache;"
        RUNTIME_ARG="--runtime nvcf"
    elif [ "$RUNTIME" = "nvcf_singularity" ]; then
        RUNTIME_ARG="--runtime nvcf_singularity"
    fi

    ssh -t -q -o StrictHostKeyChecking=no "$node" \
        "enroot exec $CONTAINER_PID /bin/bash -c '
            set -e
            export PYTHONUNBUFFERED=1
            $NVCF_EXPORTS

            echo \"[Collector $COLLECTOR_IDX] Starting data collection on $node ($RUNTIME)...\"
            cd $PROJECT_DIR
            python parallel_collect_kimi.py \
                --model_node $MODEL_NODE \
                --project_dir $PROJECT_DIR \
                --generation_mode $GENERATION_MODE \
                $RUNTIME_ARG \
                --max_parallel $MAX_PARALLEL \
                --max_trajectories $MAX_TRAJECTORIES \
                --trajectory_save_dir $TRAJECTORY_SAVE_DIR

            COLLECT_EXIT=\$?
            echo \"[Collector $COLLECTOR_IDX] Done (exit code \$COLLECT_EXIT)\"
            exit \$COLLECT_EXIT
        '" &> "$CURRENT_LOG" &
    COLLECTOR_PIDS+=($!)
    echo "[colocated] Collector $COLLECTOR_IDX launched on $node (PID ${COLLECTOR_PIDS[-1]})"
    echo "            Log: $CURRENT_LOG"
done

# --- 5. Wait for collectors ---
echo ""
echo "[colocated] All collectors launched. Waiting for completion..."
echo ""

FAILED=0
for i in "${!COLLECTOR_PIDS[@]}"; do
    COLLECTOR_NUM=$((i + 1))
    wait "${COLLECTOR_PIDS[$i]}" 2>/dev/null
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[colocated] Collector $COLLECTOR_NUM finished successfully."
    else
        echo "[colocated] Collector $COLLECTOR_NUM failed (exit code $EXIT_CODE)."
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "[colocated] All collectors finished. $FAILED/${#NODES_ARRAY[@]} failed."
echo "============================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
