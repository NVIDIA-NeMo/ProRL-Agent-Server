#!/bin/bash
# ============================================================================
# Kimi-K2.5 Data Collection with NVCF Backend (Colocated on GPU Nodes)
# ============================================================================
# Runs data collection directly on the GPU server nodes (colocated with vLLM).
# Since NVCF VMs are remote, no /dev/kvm or extra CPU nodes are needed —
# collectors just orchestrate remote NVCF VMs from inside the vLLM container.
#
#   1. Submits Kimi vLLM sbatch job (2 GPU nodes, Ray cluster)
#   2. Waits for Kimi server to become healthy
#   3. SSH+enroot execs into each GPU node's container to run data collection
#   4. Waits for all collectors to finish, then cancels Kimi server
#
# Required env vars:
#   NGC_API_KEY       - NVCF API key
#   NGC_ORG           - NVCF organization
#
# Usage:
#   NGC_API_KEY=nvapi-xxx NGC_ORG=my-org bash run_parallel_kimi_nvcf.sh
#
# ============================================================================

export LOG_DIR="${LOG_DIR:-./logs}"

# Configurable parameters
MAX_PARALLEL="${MAX_PARALLEL:-16}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-10000}"
TRAJECTORY_SAVE_DIR="${TRAJECTORY_SAVE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/bcui/ProRL-Agent-Server/cua/trajectories/kimi-nvcf/}"
NVCF_FUNCTION_NAME_PREFIX="${NVCF_FUNCTION_NAME_PREFIX:-data-collection}"

PROJECT_ROOT="/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server"
PROJECT_DIR="$PROJECT_ROOT/cua"

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
echo "Kimi-K2.5 Data Collection (NVCF Colocated)"
echo "============================================"
echo "MAX_PARALLEL:      $MAX_PARALLEL (per node)"
echo "MAX_TRAJECTORIES:  $MAX_TRAJECTORIES (per node)"
echo "NGC_ORG:           $NGC_ORG"
echo "NVCF_PREFIX:       $NVCF_FUNCTION_NAME_PREFIX"
echo ""


# --- Helper: run NVCF cleanup ---
nvcf_cleanup() {
    echo "[nvcf] Cleaning up NVCF functions with prefix '$NVCF_FUNCTION_NAME_PREFIX'..."
    NVCF_FUNCTION_NAME_PREFIX="$NVCF_FUNCTION_NAME_PREFIX" \
    NGC_API_KEY="$NGC_API_KEY" \
    NGC_ORG="$NGC_ORG" \
        python "$PROJECT_DIR/cleanup_nvcf.py" --cleanup 2>&1 || \
        echo "[nvcf] WARNING: NVCF cleanup failed (non-fatal)"
}

# --- Cleanup: cancel Kimi server + NVCF functions on exit ---
cleanup() {
    echo ""
    echo "[nvcf] Cleaning up..."

    # 1. Kill collector SSH sessions
    if [ ${#COLLECTOR_PIDS[@]} -gt 0 ]; then
        echo "[nvcf] Killing ${#COLLECTOR_PIDS[@]} collector(s)..."
        for pid in "${COLLECTOR_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
        done
    fi

    # 2. Cancel Kimi vLLM server
    if [ -n "$KIMI_JOB_ID" ]; then
        echo "[nvcf] Cancelling Kimi vLLM job $KIMI_JOB_ID"
        scancel "$KIMI_JOB_ID" 2>/dev/null
    fi

    # 3. Remove head node file
    rm -f "$LOG_DIR/head_node_${KIMI_JOB_ID}"

    # 4. Clean up any remaining NVCF functions
    nvcf_cleanup
}
trap cleanup EXIT


# --- 0. Clean up stale NVCF functions from previous runs ---
nvcf_cleanup

# --- 1. Submit Kimi vLLM server ---
echo "[nvcf] Submitting Kimi vLLM sbatch job..."
KIMI_JOB_ID=$(sbatch \
    --account=llmservice_fm_vision \
    --partition=batch_short \
    --time=02:00:00 \
    --output="$LOG_DIR/slurm-%j-server.out" \
    --error="$LOG_DIR/slurm-%j-server.out" \
    --parsable \
    "./run_kimi.sbatch")

if [ -z "$KIMI_JOB_ID" ]; then
    echo "[nvcf] ERROR: Kimi sbatch submission failed."
    exit 1
fi
echo "[nvcf] Kimi vLLM job submitted: $KIMI_JOB_ID"

# Wait for the job to start and discover nodes
HEAD_NODE_FILE="$LOG_DIR/head_node_${KIMI_JOB_ID}"
echo "[nvcf] Waiting for head node file: $HEAD_NODE_FILE"
MODEL_NODE=""
ALL_NODES=""
ELAPSED=0
MAX_WAIT=43200  # 12 hours

while [ $ELAPSED -lt $MAX_WAIT ]; do
    JOB_STATE=$(squeue -j "$KIMI_JOB_ID" -h -o %T 2>/dev/null)
    if [ -z "$JOB_STATE" ]; then
        echo "[nvcf] ERROR: Kimi job $KIMI_JOB_ID disappeared from queue!"
        exit 1
    fi

    if [ -f "$HEAD_NODE_FILE" ]; then
        MODEL_NODE=$(cat "$HEAD_NODE_FILE")
        if [ -n "$MODEL_NODE" ]; then
            ALL_NODES=$(scontrol show hostnames "$(squeue -j "$KIMI_JOB_ID" -h -o %N)")
            echo "[nvcf] Kimi vLLM job running on: $(echo $ALL_NODES | tr '\n' ' ')"
            break
        fi
    fi

    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "[nvcf] Still waiting for Kimi job to start (${ELAPSED}s)..."
    fi
done

if [ -z "$MODEL_NODE" ]; then
    echo "[nvcf] ERROR: Kimi vLLM did not start within ${MAX_WAIT}s."
    exit 1
fi

mapfile -t NODES_ARRAY <<< "$ALL_NODES"
echo "[nvcf] Head node (vLLM API): $MODEL_NODE"
echo "[nvcf] All nodes: ${NODES_ARRAY[*]}"

# --- 2. Wait for Kimi vLLM health ---
echo "[nvcf] Waiting for Kimi vLLM health at $MODEL_NODE:$KIMI_PORT..."
ELAPSED=0
MAX_HEALTH_WAIT=7200

while [ $ELAPSED -lt $MAX_HEALTH_WAIT ]; do
    if curl -sf "http://$MODEL_NODE:$KIMI_PORT/health" > /dev/null 2>&1; then
        echo "[nvcf] Kimi vLLM is healthy!"
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "[nvcf] Still waiting for Kimi health (${ELAPSED}s)..."
    fi
done

if [ $ELAPSED -ge $MAX_HEALTH_WAIT ]; then
    echo "[nvcf] ERROR: Kimi vLLM did not become healthy within ${MAX_HEALTH_WAIT}s."
    exit 1
fi

# --- 3. Launch data collection on each GPU node via SSH+enroot ---
echo "[nvcf] Launching data collection on ${#NODES_ARRAY[@]} node(s)..."
COLLECTOR_PIDS=()

for i in "${!NODES_ARRAY[@]}"; do
    node=${NODES_ARRAY[$i]}
    COLLECTOR_IDX=$((i + 1))
    CURRENT_LOG="$LOG_DIR/slurm-${KIMI_JOB_ID}-collector-nvcf-${COLLECTOR_IDX}.out"

    echo "[nvcf] Finding container on $node..."
    CONTAINER_PID=""
    while [ -z "$CONTAINER_PID" ]; do
        sleep 2
        CONTAINER_PID=$(ssh -q -o StrictHostKeyChecking=no "$node" \
            "enroot list -f | grep 'pyxis' | head -n 1 | awk '{print \$2}'" 2>/dev/null)
    done
    echo "[nvcf] Container on $node ready, PID: $CONTAINER_PID"

    ssh -t -q -o StrictHostKeyChecking=no "$node" \
        "enroot exec $CONTAINER_PID /bin/bash -c '
            set -e
            export PYTHONUNBUFFERED=1
            export NGC_API_KEY=$NGC_API_KEY
            export NGC_ORG=$NGC_ORG
            export NVCF_FUNCTION_NAME_PREFIX=$NVCF_FUNCTION_NAME_PREFIX
            export OSWORLD_SETUP_CACHE_DIR=/tmp/osworld_cache

            # Activate Python venv with required dependencies
            source /lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server/cua/cua_env_reqs/bin/activate

            # Fix TLS certs: copy venv cacert.pem to container path if missing
            VENV_CACERT=/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server/cua/cua_env_reqs/lib/python3.12/site-packages/certifi/cacert.pem
            SYS_CACERT=/usr/local/lib/python3.12/dist-packages/certifi/cacert.pem
            if [ ! -f \$SYS_CACERT ] && [ -f \$VENV_CACERT ]; then
                mkdir -p /usr/local/lib/python3.12/dist-packages/certifi
                cp \$VENV_CACERT \$SYS_CACERT 2>/dev/null || true
            fi
            export REQUESTS_CA_BUNDLE=\$VENV_CACERT
            export SSL_CERT_FILE=\$VENV_CACERT
            export CURL_CA_BUNDLE=\$VENV_CACERT

            echo \"[Collector $COLLECTOR_IDX] Starting data collection on $node (NVCF backend)...\"
            cd $PROJECT_DIR
            python parallel_collect_kimi.py \
                --model_node $MODEL_NODE \
                --runtime nvcf \
                --max_parallel $MAX_PARALLEL \
                --max_trajectories $MAX_TRAJECTORIES \
                --trajectory_save_dir $TRAJECTORY_SAVE_DIR

            COLLECT_EXIT=\$?
            echo \"[Collector $COLLECTOR_IDX] Done (exit code \$COLLECT_EXIT)\"
            exit \$COLLECT_EXIT
        '" &> "$CURRENT_LOG" &
    COLLECTOR_PIDS+=($!)
    echo "[nvcf] Collector $COLLECTOR_IDX launched on $node (PID ${COLLECTOR_PIDS[-1]})"
    echo "        Log: $CURRENT_LOG"
done

# --- 4. Wait for all collectors ---
echo ""
echo "[nvcf] All collectors launched. Waiting for completion..."
echo ""

FAILED=0
for i in "${!COLLECTOR_PIDS[@]}"; do
    COLLECTOR_NUM=$((i + 1))
    wait "${COLLECTOR_PIDS[$i]}" 2>/dev/null
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[nvcf] Collector $COLLECTOR_NUM finished successfully."
    else
        echo "[nvcf] Collector $COLLECTOR_NUM failed (exit code $EXIT_CODE)."
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "[nvcf] All collectors finished. $FAILED/${#NODES_ARRAY[@]} failed."
echo "============================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
