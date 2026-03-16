#!/bin/bash
# ============================================================================
# Kimi Collector Launcher (SSH+Enroot Pattern)
# ============================================================================
# Submits a holder "sleep infinity" job on a reserved node, waits for
# the container to be ready, then SSH+enroot execs into it to run:
#   - Data collection via parallel_collect_kimi.py (no local vLLM needed)
#
# Required env vars:
#   MODEL_NODE        - hostname of the Kimi vLLM server head node
#
# Optional env vars:
#   MAX_PARALLEL      - parallel VMs per collector (default: 16)
#   MAX_TRAJECTORIES  - trajectories to collect (default: 10000)
#
# Optional arg:
#   $1 = collector index (for log naming, default: 0)
#
# Usage:
#   MODEL_NODE=pool0-03161 bash run_collector_kimi.sh 1
# ============================================================================

COLLECTOR_IDX="${1:-0}"
LOG_DIR="${LOG_DIR:-./logs}"

# Configs
# PROJECT_ROOT="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server"
PROJECT_ROOT="/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server"
PROJECT_DIR="$PROJECT_ROOT/cua"
COLLECTOR_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/cua_cpu.sqsh"

MAX_PARALLEL=${MAX_PARALLEL:-10}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-10000}

KIMI_PORT=8000

# Validate
if [ -z "$MODEL_NODE" ]; then
    echo "[Collector $COLLECTOR_IDX] ERROR: MODEL_NODE not set."
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "[Collector $COLLECTOR_IDX] MODEL_NODE=$MODEL_NODE"
echo "[Collector $COLLECTOR_IDX] MAX_PARALLEL=$MAX_PARALLEL MAX_TRAJECTORIES=$MAX_TRAJECTORIES"

# --- 1. Submit holder job on reserved node ---
echo "[Collector $COLLECTOR_IDX] Submitting holder job..."
COLLECTOR_JOB_ID=$(sbatch --parsable \
    --job-name="kimi_collector_${COLLECTOR_IDX}" \
    --account=nvr_lpr_agentic \
    --partition=cpu_short \
    --reservation=sla_res_osworld_agent_vlm_cpu_only \
    --mem=0 \
    --time=01:30:00 \
    --exclusive \
    --output="/dev/null" \
    --error="/dev/null" \
    --wrap="srun --container-image=$COLLECTOR_IMAGE --container-mounts=/lustre:/lustre sleep infinity")

if [ -z "$COLLECTOR_JOB_ID" ]; then
    echo "[Collector $COLLECTOR_IDX] ERROR: Job submission failed."
    exit 1
fi
echo "[Collector $COLLECTOR_IDX] Holder job submitted: $COLLECTOR_JOB_ID"

# --- 2. Cleanup trap ---
cleanup() {
    echo ""
    echo "[Collector $COLLECTOR_IDX] Cleaning up... Cancelling holder job $COLLECTOR_JOB_ID"
    scancel "$COLLECTOR_JOB_ID" 2>/dev/null
}
trap cleanup EXIT

# --- 3. Wait for job to start ---
echo "[Collector $COLLECTOR_IDX] Waiting for holder job to start..."
COLLECTOR_NODE=""
while [ -z "$COLLECTOR_NODE" ]; do
    JOB_STATE=$(squeue -j "$COLLECTOR_JOB_ID" -h -o %T 2>/dev/null)

    if [ "$JOB_STATE" == "RUNNING" ]; then
        COLLECTOR_NODE=$(squeue -j "$COLLECTOR_JOB_ID" -h -o %N)
    elif [ -z "$JOB_STATE" ]; then
        echo "[Collector $COLLECTOR_IDX] ERROR: Job $COLLECTOR_JOB_ID disappeared from queue!"
        exit 1
    fi
    sleep 2
done
echo "[Collector $COLLECTOR_IDX] Job RUNNING on node: $COLLECTOR_NODE"

# --- 4. Wait for container readiness ---
echo "[Collector $COLLECTOR_IDX] Polling for container readiness on $COLLECTOR_NODE..."
CONTAINER_PID=""
while [ -z "$CONTAINER_PID" ]; do
    sleep 2
    CONTAINER_PID=$(ssh -q -o StrictHostKeyChecking=no "$COLLECTOR_NODE" \
        "enroot list -f | grep 'pyxis' | grep 'sleep' | awk '{print \$2}' | head -n 1" 2>/dev/null)

    if [ -z "$CONTAINER_PID" ]; then
        printf "."
    fi
done
echo ""
echo "[Collector $COLLECTOR_IDX] Container ready, PID: $CONTAINER_PID"

# --- 5. SSH+enroot exec: run data collection ---
ssh -t -q -o StrictHostKeyChecking=no "$COLLECTOR_NODE" \
    "enroot exec $CONTAINER_PID /bin/bash -c '
        set -e
        export PYTHONUNBUFFERED=1

        # Wait for Kimi vLLM to be healthy
        echo \"[Collector $COLLECTOR_IDX] Waiting for Kimi vLLM at $MODEL_NODE:$KIMI_PORT...\"
        ELAPSED=0
        MAX_WAIT=7200
        while [ \$ELAPSED -lt \$MAX_WAIT ]; do
            if curl -sf http://$MODEL_NODE:$KIMI_PORT/health > /dev/null 2>&1; then
                echo \"[Collector $COLLECTOR_IDX] Kimi vLLM is healthy!\"
                break
            fi
            sleep 10
            ELAPSED=\$((ELAPSED + 10))
            if [ \$((ELAPSED % 60)) -eq 0 ]; then
                echo \"[Collector $COLLECTOR_IDX] Still waiting for Kimi vLLM (\${ELAPSED}s)...\"
            fi
        done

        if [ \$ELAPSED -ge \$MAX_WAIT ]; then
            echo \"[Collector $COLLECTOR_IDX] ERROR: Kimi vLLM did not become healthy within \${MAX_WAIT}s\"
            exit 1
        fi

        # Run data collection
        echo \"[Collector $COLLECTOR_IDX] Starting parallel data collection...\"
        cd $PROJECT_DIR
        python parallel_collect_kimi.py \
            --model_node $MODEL_NODE \
            --max_parallel $MAX_PARALLEL \
            --max_trajectories $MAX_TRAJECTORIES \
            --trajectory_save_dir $TRAJECTORY_SAVE_DIR

        COLLECT_EXIT=\$?
        echo \"[Collector $COLLECTOR_IDX] Data collection finished with exit code \$COLLECT_EXIT\"
        exit \$COLLECT_EXIT
    '"
