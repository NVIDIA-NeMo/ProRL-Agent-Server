#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────────
# Polar SLURM Serve-Only Job Script
#
# Starts: vLLM server → Rollout service → Gateway node(s), then waits.
# Tasks are submitted separately via 'polar cluster submit-task'.
#
# All SBATCH directives are passed via sbatch CLI flags from 'polar cluster serve'.
# No hardcoded account, partition, or resource values.
#
# Required env vars (pass via --export):
#   POLAR_CODE        - Path to ProRL-Agent-Server on cluster
#   POLAR_WORKSPACE   - Base workspace path on cluster
#
# Optional env vars:
#   MODEL_NAME        - Model to serve (default from env.sh)
#   MODEL_PATH        - Model path/name for vLLM (default: $MODEL_NAME)
#   TENSOR_PARALLEL_SIZE - TP size for vLLM (default: 8)
# ────────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ -z "${POLAR_CODE:-}" ]; then
    echo "FATAL: POLAR_CODE not set. Use 'polar cluster serve -c cluster.yaml'." >&2
    exit 1
fi
if [ -z "${POLAR_WORKSPACE:-}" ]; then
    echo "FATAL: POLAR_WORKSPACE not set. Use 'polar cluster serve -c cluster.yaml'." >&2
    exit 1
fi

# Source environment setup from the cluster templates
TEMPLATE_DIR="${POLAR_CODE}/src/polar/cluster/templates"
source "${TEMPLATE_DIR}/env.sh"

# ── Job directory ──────────────────────────────────────────────────────────────
JOB_DIR="${POLAR_RESULTS}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}/logs"
echo "[polar_serve] Job directory: ${JOB_DIR}"
echo "[polar_serve] Job ID: ${SLURM_JOB_ID}"
echo "[polar_serve] Nodes: ${SLURM_JOB_NODELIST}"
echo "[polar_serve] GPUs per node: ${SLURM_GPUS_ON_NODE:-unknown}"

# ── Discover hostnames ─────────────────────────────────────────────────────────
HOSTNAMES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
NODE_0="${HOSTNAMES[0]}"
NUM_NODES="${#HOSTNAMES[@]}"
echo "[polar_serve] Allocated ${NUM_NODES} node(s): ${HOSTNAMES[*]}"

# ── Save results directory ─────────────────────────────────────────────────────
export SAVE_DIR="${JOB_DIR}/rollout_results"
mkdir -p "${SAVE_DIR}"

# ── Generate topology.yaml ─────────────────────────────────────────────────────
TOPOLOGY_PATH="${JOB_DIR}/topology.yaml"
python -m polar.cluster.topology \
    --output "${TOPOLOGY_PATH}" \
    --save-dir "${SAVE_DIR}"

export POLAR_TOPOLOGY="${TOPOLOGY_PATH}"
echo "[polar_serve] Generated topology: ${TOPOLOGY_PATH}"
cat "${TOPOLOGY_PATH}"

# ── Kill stale processes on ports from previous runs ─────────────────────────
echo "[polar_serve] Cleaning stale processes on ports..."
for port in "${VLLM_PORT}" "${ROLLOUT_PORT}"; do
    stale_pid=$(lsof -ti ":${port}" 2>/dev/null || true)
    if [ -n "${stale_pid}" ]; then
        echo "[polar_serve] Killing stale PID ${stale_pid} on port ${port}"
        kill -9 ${stale_pid} 2>/dev/null || true
        sleep 1
    fi
done
for ((i=0; i<NUM_NODES; i++)); do
    gw_port=$((GATEWAY_BASE_PORT + i))
    stale_pid=$(lsof -ti ":${gw_port}" 2>/dev/null || true)
    if [ -n "${stale_pid}" ]; then
        echo "[polar_serve] Killing stale PID ${stale_pid} on port ${gw_port}"
        kill -9 ${stale_pid} 2>/dev/null || true
        sleep 1
    fi
done

# ── PID tracking for cleanup ───────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo "[polar_serve] Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[polar_serve] Killing PID ${pid}"
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "[polar_serve] Cleanup complete."
}
trap cleanup EXIT INT TERM

# ── Source readiness poller ────────────────────────────────────────────────────
source "${TEMPLATE_DIR}/wait_for_service.sh"

# ── Launch vLLM ────────────────────────────────────────────────────────────────
echo "[polar_serve] Starting vLLM server on ${NODE_0}:${VLLM_PORT}..."
echo "[polar_serve]   Model: ${MODEL_PATH}"
echo "[polar_serve]   TP size: ${TENSOR_PARALLEL_SIZE}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-num-seqs "${MAX_NUM_SEQS:-64}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER:-hermes}" \
    --gdn-prefill-backend triton \
    --trust-remote-code \
    --host 0.0.0.0 \
    > "${JOB_DIR}/logs/vllm.log" 2>&1 &
PIDS+=($!)
echo "[polar_serve] vLLM PID: ${PIDS[-1]}"

wait_for_service "http://${NODE_0}:${VLLM_PORT}/health" "vLLM" 240 5 || exit 1

# ── Launch Rollout Service ─────────────────────────────────────────────────────
echo "[polar_serve] Starting rollout service on ${NODE_0}:${ROLLOUT_PORT}..."

python -m polar.cli serve_rollout -c "${TOPOLOGY_PATH}" \
    > "${JOB_DIR}/logs/rollout.log" 2>&1 &
PIDS+=($!)
echo "[polar_serve] Rollout PID: ${PIDS[-1]}"

wait_for_service "http://${NODE_0}:${ROLLOUT_PORT}/health" "Rollout" 30 2 || exit 1

# ── Launch Gateway Node(s) ─────────────────────────────────────────────────────
for ((i=0; i<NUM_NODES; i++)); do
    NODE_ID="node-$(printf '%02d' ${i})"
    GW_PORT=$((GATEWAY_BASE_PORT + i))
    GW_HOST="${HOSTNAMES[$i]}"

    echo "[polar_serve] Starting gateway ${NODE_ID} on ${GW_HOST}:${GW_PORT}..."

    if [ "${i}" -eq 0 ]; then
        python -m polar.cli serve_gateway \
            -c "${TOPOLOGY_PATH}" \
            --node-id "${NODE_ID}" \
            > "${JOB_DIR}/logs/gateway_${NODE_ID}.log" 2>&1 &
        PIDS+=($!)
    else
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${GW_HOST}" \
            --output="${JOB_DIR}/logs/gateway_${NODE_ID}.log" \
            --error="${JOB_DIR}/logs/gateway_${NODE_ID}.err" \
            bash -c "
                source '${TEMPLATE_DIR}/env.sh'
                export POLAR_TOPOLOGY='${TOPOLOGY_PATH}'
                export POLAR_GATEWAY_NODE_ID='${NODE_ID}'
                python -m polar.gateway.server
            " &
        PIDS+=($!)
    fi
    echo "[polar_serve] Gateway ${NODE_ID} PID: ${PIDS[-1]}"
done

# Wait for all gateways
for ((i=0; i<NUM_NODES; i++)); do
    GW_PORT=$((GATEWAY_BASE_PORT + i))
    GW_HOST="${HOSTNAMES[$i]}"
    wait_for_service "http://${GW_HOST}:${GW_PORT}/health" "Gateway node-$(printf '%02d' ${i})" 30 2 || exit 1
done

# ── All services ready ───────────────────────────────────────────────────────
echo "[polar_serve] All services running. Checking status..."
python -m polar.cli status -c "${TOPOLOGY_PATH}" --json > "${JOB_DIR}/logs/status.json" 2>&1 || true
python -m polar.cli status -c "${TOPOLOGY_PATH}" || true

# Write sentinel file so 'polar cluster serve' knows services are ready
cat > "${JOB_DIR}/.services_ready" <<EOF
TOPOLOGY=${TOPOLOGY_PATH}
ROLLOUT_URL=http://${NODE_0}:${ROLLOUT_PORT}
JOB_DIR=${JOB_DIR}
SIF_DIR=${POLAR_SIFS}
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo ""
echo "[polar_serve] ════════════════════════════════════════════════════════"
echo "[polar_serve] Services ready. Waiting for task submissions..."
echo "[polar_serve] Job ID: ${SLURM_JOB_ID}"
echo "[polar_serve] Topology: ${TOPOLOGY_PATH}"
echo "[polar_serve]"
echo "[polar_serve] Submit tasks with:"
echo "[polar_serve]   polar cluster submit-task -c <config> --job-id ${SLURM_JOB_ID} --example calculator --harness opencode"
echo "[polar_serve] ════════════════════════════════════════════════════════"

# Wait indefinitely — services stay alive until SLURM time limit or scancel
while true; do
    sleep 60
    # Periodic health check
    if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
        echo "[polar_serve] vLLM process died. Exiting."
        exit 1
    fi
done
