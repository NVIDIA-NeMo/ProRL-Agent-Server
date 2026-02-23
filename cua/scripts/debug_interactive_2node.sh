#!/bin/bash
# ============================================================================
# 2-Node Interactive GPU Debug Script
# ============================================================================
# Node 1 (Planner): Runs Qwen3-VL-235B vLLM server (tp=8, all 8 GPUs)
# Node 2 (Actor):   Runs UI-TARS-1.5-7B vLLM server (tp=4) + KVM for VMs
#
# Both vLLM servers auto-start, then you get an interactive shell on the
# Actor node for manual debugging / data collection.
# ============================================================================

IMAGE="/lustre/fsw/portfolios/nvr/users/bcui/images/cua-vllm-0.13.0.sqsh"

PROJECT_ROOT="/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server"
PROJECT_DIR="$PROJECT_ROOT/cua"

PLANNER_MODEL="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking"
# ACTOR_MODEL="ByteDance-Seed/UI-TARS-1.5-7B"
ACTOR_MODEL="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/bcui/huggingface_models/UI-TARS-1.5-7B"

PLANNER_PORT=8000
ACTOR_PORT=8000

# --- 1. Submit Holder Job (2 GPU nodes) ---
echo "[Local] Submitting 2-node GPU job..."

JOB_ID=$(sbatch --parsable \
    --job-name=debug_2node \
    --account=llmservice_fm_vision \
    --partition=interactive \
    --reservation=sla_res_osworld_agent_vlm \
    --gpus-per-node=8 \
    --nodes=2 \
    --ntasks-per-node=1 \
    --mem=0 \
    --time=04:00:00 \
    --exclusive \
    --output=$PROJECT_DIR/scripts/logs/debug_2node_holder_%j.out \
    --error=$PROJECT_DIR/scripts/logs/debug_2node_holder_%j.err \
    --wrap="srun --container-image=$IMAGE --container-mounts=/lustre:/lustre --container-writable sleep infinity")

if [ -z "$JOB_ID" ]; then
    echo "Error: Job submission failed."
    exit 1
fi

echo "[Local] Job submitted. ID: $JOB_ID"

# --- 2. Cleanup Trap ---
cleanup() {
    echo ""
    echo "[Local] Cleaning up... Cancelling Job $JOB_ID"
    scancel "$JOB_ID"
}
trap cleanup EXIT

# --- 3. Wait for Job to Start ---
echo "[Local] Waiting for job to start..."
NODES=""
while [ -z "$NODES" ]; do
    JOB_STATE=$(squeue -j "$JOB_ID" -h -o %T)
    if [ "$JOB_STATE" == "RUNNING" ]; then
        NODES=$(squeue -j "$JOB_ID" -h -o %N)
    elif [ -z "$JOB_STATE" ]; then
        echo "Error: Job disappeared from queue!"
        exit 1
    fi
    sleep 2
done

# Expand nodelist to individual hostnames
ALL_NODES=$(scontrol show hostnames "$NODES")
PLANNER_NODE=$(echo "$ALL_NODES" | head -n 1)
ACTOR_NODE=$(echo "$ALL_NODES" | tail -n 1)

echo "[Local] Job is RUNNING"
echo "[Local] Planner Node: $PLANNER_NODE"
echo "[Local] Actor Node:   $ACTOR_NODE"

# --- 4. Wait for Containers on Both Nodes ---
wait_for_container() {
    local node=$1
    local name=$2
    local pid=""

    echo "[Local] Waiting for container on $name ($node)..." >&2
    while [ -z "$pid" ]; do
        sleep 2
        pid=$(ssh -q -o StrictHostKeyChecking=no "$node" \
            "enroot list -f | grep 'pyxis' | grep 'sleep' | awk '{print \$2}' | head -n 1")
        if [ -z "$pid" ]; then
            printf "." >&2
        fi
    done
    echo "" >&2
    echo "[Local] $name container ready (PID: $pid)" >&2
    echo "$pid"
}

PLANNER_PID=$(wait_for_container "$PLANNER_NODE" "Planner")
ACTOR_PID=$(wait_for_container "$ACTOR_NODE" "Actor")

# --- 5. Launch Planner vLLM Server (background) ---
echo "[Local] Starting Planner vLLM server on $PLANNER_NODE..."
ssh -q -o StrictHostKeyChecking=no "$PLANNER_NODE" \
    "enroot exec $PLANNER_PID bash -c '
      mkdir -p $PROJECT_DIR/scripts/logs
      nohup vllm serve $PLANNER_MODEL \
        --api-key gen \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --limit-mm-per-prompt.video 0 \
        --limit-mm-per-prompt.image 3 \
        --async-scheduling \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.9 \
        > $PROJECT_DIR/scripts/logs/planner_debug.log 2>&1 &
      echo \"[Planner] vLLM server launched (PID: \$!)\"
    '" &

# --- 6. Launch Actor vLLM Server (background) ---
echo "[Local] Starting Actor vLLM server on $ACTOR_NODE..."
ssh -q -o StrictHostKeyChecking=no "$ACTOR_NODE" \
    "enroot exec $ACTOR_PID bash -c '
      mkdir -p $PROJECT_DIR/scripts/logs
      nohup vllm serve $ACTOR_MODEL \
        --served-model-name ByteDance-Seed/UI-TARS-1.5-7B \
        --api-key gen \
        --tensor-parallel-size 4 \
        --limit-mm-per-prompt.image 5 \
        --limit-mm-per-prompt.video 0 \
        --max-model-len 65536 \
        --disable-log-requests \
        --disable-log-stats \
        > $PROJECT_DIR/scripts/logs/actor_debug.log 2>&1 &
      echo \"[Actor] vLLM server launched (PID: \$!)\"
    '" &

wait  # wait for both SSH commands to return

# --- 7. Wait for Both Servers to Be Healthy ---
echo "[Local] Waiting for vLLM servers to become healthy..."

wait_for_server() {
    local node=$1
    local container_pid=$2
    local port=$3
    local name=$4
    local max_wait=600
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        if ssh -q -o StrictHostKeyChecking=no "$node" \
            "enroot exec $container_pid curl -sf http://localhost:$port/health" > /dev/null 2>&1; then
            echo "[Local] $name server healthy on $node:$port"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        if [ $((elapsed % 60)) -eq 0 ]; then
            echo "[Local] Still waiting for $name (${elapsed}s)..."
        fi
    done

    echo "[Local] ERROR: $name server did not start within ${max_wait}s"
    return 1
}

wait_for_server "$PLANNER_NODE" "$PLANNER_PID" "$PLANNER_PORT" "Planner" &
WAIT_PLANNER_PID=$!

wait_for_server "$ACTOR_NODE" "$ACTOR_PID" "$ACTOR_PORT" "Actor" &
WAIT_ACTOR_PID=$!

wait $WAIT_PLANNER_PID
PLANNER_OK=$?

wait $WAIT_ACTOR_PID
ACTOR_OK=$?

if [ $PLANNER_OK -ne 0 ] || [ $ACTOR_OK -ne 0 ]; then
    echo "[Local] ERROR: One or both servers failed to start."
    echo "[Local] Check logs:"
    echo "  Planner: $PROJECT_DIR/scripts/logs/planner_debug.log"
    echo "  Actor:   $PROJECT_DIR/scripts/logs/actor_debug.log"
    exit 1
fi

# --- 8. Launch Interactive Shell on Actor Node ---
echo ""
echo "=========================================================="
echo "  2-Node Interactive Debug Session"
echo "=========================================================="
echo "  Planner: $PLANNER_NODE (Qwen3-VL-235B, port $PLANNER_PORT)"
echo "  Actor:   $ACTOR_NODE (UI-TARS-1.5-7B, port $ACTOR_PORT)"
echo "  KVM:     available via reservation"
echo ""
echo "  Planner API: http://$PLANNER_NODE:$PLANNER_PORT"
echo "  Actor API:   http://localhost:$ACTOR_PORT"
echo ""
echo "  Logs:"
echo "    tail -f $PROJECT_DIR/scripts/logs/planner_debug.log"
echo "    tail -f $PROJECT_DIR/scripts/logs/actor_debug.log"
echo "=========================================================="

ssh -t -q -o StrictHostKeyChecking=no "$ACTOR_NODE" \
    "enroot exec $ACTOR_PID bash -c '
      cd $PROJECT_DIR
      source cua_env_reqs/bin/activate
      export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH
      export PLANNER_NODE=$PLANNER_NODE
      export PLANNER_PORT=$PLANNER_PORT
      export ACTOR_PORT=$ACTOR_PORT
      exec /bin/bash -l
    '"

# --- 9. End ---
echo "[Local] Session ended."
