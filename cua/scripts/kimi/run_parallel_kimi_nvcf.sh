#!/bin/bash
# ============================================================================
# Kimi-K2.5 Data Collection with NVCF Backend - Multi-Collector Launcher
# ============================================================================
# Same architecture as run_parallel_kimi.sh but uses NVCF for VMs instead of
# local KVM. Collector nodes don't need /dev/kvm or GPU — they just orchestrate
# remote NVCF VMs.
#
#   1. Submits Kimi vLLM sbatch job (2 GPU nodes, Ray cluster)
#   2. Waits for Kimi server to become healthy
#   3. Launches NUM_COLLECTORS instances of run_collector_kimi_nvcf.sh in parallel
#   4. Each collector submits a CPU-only holder job (no /dev/kvm needed)
#   5. Waits for all collectors to finish, then cancels Kimi server
#
# Required env vars:
#   NGC_API_KEY       - NVCF API key
#   NGC_ORG           - NVCF organization
#
# Usage:
#   NGC_API_KEY=nvapi-xxx NGC_ORG=my-org NUM_COLLECTORS=2 bash run_parallel_kimi_nvcf.sh
#
# ============================================================================

export LOG_DIR="${LOG_DIR:-./logs}"

# Configurable parameters
NUM_COLLECTORS="${NUM_COLLECTORS:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-10000}"
# TRAJECTORY_SAVE_DIR="${TRAJECTORY_SAVE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/mingjiel/workspace/data/jaehun/cua/trajectories/kimi-nvcf/}"
TRAJECTORY_SAVE_DIR="${TRAJECTORY_SAVE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/bcui/ProRL-Agent-Server/cua/trajectories/kimi-nvcf/}"
NVCF_FUNCTION_NAME_PREFIX="${NVCF_FUNCTION_NAME_PREFIX:-data-collection}"

# Validate NVCF credentials
if [ -z "$NGC_API_KEY" ]; then
    echo "[ERROR] NGC_API_KEY not set. Required for NVCF backend."
    exit 1
fi
if [ -z "$NGC_ORG" ]; then
    echo "[ERROR] NGC_ORG not set. Required for NVCF backend."
    exit 1
fi

# Create logs directory
mkdir -p "$LOG_DIR"

KIMI_JOB_ID=""
COLLECTOR_PIDS=()
KIMI_PORT=8000

echo "============================================"
echo "Kimi-K2.5 Data Collection (NVCF Backend)"
echo "============================================"
echo "NUM_COLLECTORS:    $NUM_COLLECTORS"
echo "MAX_PARALLEL:      $MAX_PARALLEL (per collector)"
echo "MAX_TRAJECTORIES:  $MAX_TRAJECTORIES (per collector)"
echo "NGC_ORG:           $NGC_ORG"
echo "NVCF_PREFIX:       $NVCF_FUNCTION_NAME_PREFIX"
echo ""


# --- Cleanup: cancel Kimi server on exit ---
cleanup() {
    echo ""
    echo "[run_parallel_kimi_nvcf.sh] Cleaning up..."

    # 1. Kill Collector Launcher Scripts
    if [ ${#COLLECTOR_PIDS[@]} -gt 0 ]; then
        echo "[run_parallel_kimi_nvcf.sh] Killing ${#COLLECTOR_PIDS[@]} collector launcher scripts..."
        for pid in "${COLLECTOR_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
        done
    fi

    # 2. Cancel Kimi vLLM server
    if [ -n "$KIMI_JOB_ID" ]; then
        echo "[run_parallel_kimi_nvcf.sh] Cancelling Kimi vLLM job $KIMI_JOB_ID"
        scancel "$KIMI_JOB_ID" 2>/dev/null
    fi

    # 3. Remove head node file
    rm -f "$LOG_DIR/head_node_${KIMI_JOB_ID}"
}
trap cleanup EXIT


# --- 1. Submit Kimi vLLM server ---
echo "[run_parallel_kimi_nvcf.sh] Submitting Kimi vLLM sbatch job..."
KIMI_JOB_ID=$(sbatch \
    --account=nvr_lacr_llm \
    --partition=batch_short \
    --time=02:00:00 \
    --output="$LOG_DIR/slurm-%j-server.out" \
    --error="$LOG_DIR/slurm-%j-server.out" \
    --parsable \
    "./run_kimi.sbatch")

if [ -z "$KIMI_JOB_ID" ]; then
    echo "[run_parallel_kimi_nvcf.sh] ERROR: Kimi sbatch submission failed."
    exit 1
fi
echo "[run_parallel_kimi_nvcf.sh] Kimi vLLM job submitted: $KIMI_JOB_ID"

# Wait for the job to start and discover the head node
HEAD_NODE_FILE="$LOG_DIR/head_node_${KIMI_JOB_ID}"
echo "[run_parallel_kimi_nvcf.sh] Waiting for head node file: $HEAD_NODE_FILE"
MODEL_NODE=""
ELAPSED=0
MAX_WAIT=43200  # 12 hours

while [ $ELAPSED -lt $MAX_WAIT ]; do
    JOB_STATE=$(squeue -j "$KIMI_JOB_ID" -h -o %T 2>/dev/null)
    if [ -z "$JOB_STATE" ]; then
        echo "[run_parallel_kimi_nvcf.sh] ERROR: Kimi job $KIMI_JOB_ID disappeared from queue!"
        exit 1
    fi

    if [ -f "$HEAD_NODE_FILE" ]; then
        MODEL_NODE=$(cat "$HEAD_NODE_FILE")
        if [ -n "$MODEL_NODE" ]; then
            echo "[run_parallel_kimi_nvcf.sh] Kimi vLLM job running. Head node: $MODEL_NODE"
            break
        fi
    fi

    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "[run_parallel_kimi_nvcf.sh] Still waiting for Kimi job to start (${ELAPSED}s)..."
    fi
done

if [ -z "$MODEL_NODE" ]; then
    echo "[run_parallel_kimi_nvcf.sh] ERROR: Kimi vLLM did not start within ${MAX_WAIT}s."
    exit 1
fi

# --- 2. Wait for Kimi vLLM health ---
echo "[run_parallel_kimi_nvcf.sh] Waiting for Kimi vLLM health at $MODEL_NODE:$KIMI_PORT..."
ELAPSED=0
MAX_HEALTH_WAIT=7200

while [ $ELAPSED -lt $MAX_HEALTH_WAIT ]; do
    if curl -sf "http://$MODEL_NODE:$KIMI_PORT/health" > /dev/null 2>&1; then
        echo "[run_parallel_kimi_nvcf.sh] Kimi vLLM is healthy!"
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "[run_parallel_kimi_nvcf.sh] Still waiting for Kimi health (${ELAPSED}s)..."
    fi
done

if [ $ELAPSED -ge $MAX_HEALTH_WAIT ]; then
    echo "[run_parallel_kimi_nvcf.sh] ERROR: Kimi vLLM did not become healthy within ${MAX_HEALTH_WAIT}s."
    exit 1
fi

# --- 3. Launch N Collector Instances ---
echo "[run_parallel_kimi_nvcf.sh] Launching $NUM_COLLECTORS collector(s)..."
export MODEL_NODE
COLLECTOR_PIDS=()

for i in $(seq 1 "$NUM_COLLECTORS"); do
    echo "[run_parallel_kimi_nvcf.sh] Starting collector $i..."
    CURRENT_LOG="$LOG_DIR/slurm-${KIMI_JOB_ID}-collector-nvcf-${i}.out"

    MODEL_NODE="$MODEL_NODE" \
    NVCF_FUNCTION_NAME_PREFIX="$NVCF_FUNCTION_NAME_PREFIX" \
    NGC_API_KEY="$NGC_API_KEY" \
    NGC_ORG="$NGC_ORG" \
    MAX_PARALLEL="$MAX_PARALLEL" \
    MAX_TRAJECTORIES="$MAX_TRAJECTORIES" \
    TRAJECTORY_SAVE_DIR="$TRAJECTORY_SAVE_DIR" \
        bash "./run_collector_kimi_nvcf.sh" "$i" &> "$CURRENT_LOG" &

    COLLECTOR_PIDS+=($!)
    echo "[run_parallel_kimi_nvcf.sh] Collector $i launched (PID ${COLLECTOR_PIDS[-1]})"
    echo "                            Log: $CURRENT_LOG"
done

# --- 4. Wait for all collectors ---
echo ""
echo "[run_parallel_kimi_nvcf.sh] All collectors launched. Waiting for completion..."
echo ""

FAILED=0
for i in "${!COLLECTOR_PIDS[@]}"; do
    COLLECTOR_NUM=$((i + 1))
    wait "${COLLECTOR_PIDS[$i]}" 2>/dev/null
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[run_parallel_kimi_nvcf.sh] Collector $COLLECTOR_NUM finished successfully."
    else
        echo "[run_parallel_kimi_nvcf.sh] Collector $COLLECTOR_NUM failed (exit code $EXIT_CODE)."
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "[run_parallel_kimi_nvcf.sh] All collectors finished. $FAILED/$NUM_COLLECTORS failed."
echo "============================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
