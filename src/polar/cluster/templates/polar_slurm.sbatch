#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────────
# Polar SLURM Job Script
#
# Orchestrates: vLLM server → Rollout service → Gateway node(s) → Task submission
#
# All SBATCH directives are passed via sbatch CLI flags from 'polar cluster launch'.
# No hardcoded account, partition, or resource values.
#
# Required env vars (pass via --export):
#   POLAR_CODE        - Path to ProRL-Agent-Server on cluster
#   POLAR_WORKSPACE   - Base workspace path on cluster
#   EXAMPLE           - "calculator" or "swegym" (default: calculator)
#   HARNESS           - Agent harness name (default: opencode)
#
# Optional env vars:
#   MODEL_NAME        - Model to serve (default from env.sh)
#   MODEL_PATH        - Model path/name for vLLM (default: $MODEL_NAME)
#   TENSOR_PARALLEL_SIZE - TP size for vLLM (default: 8)
#   NUM_ROLLOUTS      - Number of rollouts per task (default: 4)
#   TIMEOUT_SECONDS   - Per-task timeout (default: 900)
#   DEFAULT_SIF_IMAGE - Override SIF path for agent containers
#   TASK_FILE         - Submit a specific task JSON instead of using the example
# ────────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# SLURM copies the script to a local spool directory, so BASH_SOURCE[0] won't
# point back to the original source tree.  POLAR_CODE must be passed via --export.
if [ -z "${POLAR_CODE:-}" ]; then
    echo "FATAL: POLAR_CODE not set. Use 'polar cluster launch -c cluster.yaml'." >&2
    exit 1
fi
if [ -z "${POLAR_WORKSPACE:-}" ]; then
    echo "FATAL: POLAR_WORKSPACE not set. Use 'polar cluster launch -c cluster.yaml'." >&2
    exit 1
fi

# Source environment setup from the cluster templates
TEMPLATE_DIR="${POLAR_CODE}/src/polar/cluster/templates"
source "${TEMPLATE_DIR}/env.sh"

# ── Defaults ───────────────────────────────────────────────────────────────────
EXAMPLE="${EXAMPLE:-calculator}"
HARNESS="${HARNESS:-opencode}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"

# ── Job directory ──────────────────────────────────────────────────────────────
JOB_DIR="${POLAR_RESULTS}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}/logs"
echo "[polar_slurm] Job directory: ${JOB_DIR}"
echo "[polar_slurm] Job ID: ${SLURM_JOB_ID}"
echo "[polar_slurm] Nodes: ${SLURM_JOB_NODELIST}"
echo "[polar_slurm] GPUs per node: ${SLURM_GPUS_ON_NODE:-unknown}"

# ── Discover hostnames ─────────────────────────────────────────────────────────
HOSTNAMES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
NODE_0="${HOSTNAMES[0]}"
NUM_NODES="${#HOSTNAMES[@]}"
echo "[polar_slurm] Allocated ${NUM_NODES} node(s): ${HOSTNAMES[*]}"

# ── Resolve SIF image ─────────────────────────────────────────────────────────
if [ -z "${DEFAULT_SIF_IMAGE:-}" ]; then
    if [ "${EXAMPLE}" = "calculator" ]; then
        DEFAULT_SIF_IMAGE="${POLAR_SIFS}/calculator-${HARNESS}.sif"
    else
        # For swegym, each task has its own SIF — submit script handles this
        DEFAULT_SIF_IMAGE=""
    fi
fi

if [ -n "${DEFAULT_SIF_IMAGE}" ] && [ ! -f "${DEFAULT_SIF_IMAGE}" ]; then
    echo "[polar_slurm] WARNING: SIF not found: ${DEFAULT_SIF_IMAGE}"
    echo "[polar_slurm] Build it first with: polar cluster build-sif --example ${EXAMPLE} --harness ${HARNESS}"
fi

# ── Save results directory ─────────────────────────────────────────────────────
export SAVE_DIR="${JOB_DIR}/rollout_results"
mkdir -p "${SAVE_DIR}"

# ── Generate topology.yaml ─────────────────────────────────────────────────────
TOPOLOGY_PATH="${JOB_DIR}/topology.yaml"
python -m polar.cluster.topology \
    --output "${TOPOLOGY_PATH}" \
    --save-dir "${SAVE_DIR}" \
    ${DEFAULT_SIF_IMAGE:+--default-sif "${DEFAULT_SIF_IMAGE}"}

export POLAR_TOPOLOGY="${TOPOLOGY_PATH}"
echo "[polar_slurm] Generated topology: ${TOPOLOGY_PATH}"
cat "${TOPOLOGY_PATH}"

# ── Kill stale processes on ports from previous runs ─────────────────────────
echo "[polar_slurm] Cleaning stale processes on ports..."
for port in "${VLLM_PORT}" "${ROLLOUT_PORT}"; do
    stale_pid=$(lsof -ti ":${port}" 2>/dev/null || true)
    if [ -n "${stale_pid}" ]; then
        echo "[polar_slurm] Killing stale PID ${stale_pid} on port ${port}"
        kill -9 ${stale_pid} 2>/dev/null || true
        sleep 1
    fi
done
for ((i=0; i<NUM_NODES; i++)); do
    gw_port=$((GATEWAY_BASE_PORT + i))
    stale_pid=$(lsof -ti ":${gw_port}" 2>/dev/null || true)
    if [ -n "${stale_pid}" ]; then
        echo "[polar_slurm] Killing stale PID ${stale_pid} on port ${gw_port}"
        kill -9 ${stale_pid} 2>/dev/null || true
        sleep 1
    fi
done

# ── PID tracking for cleanup ───────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo "[polar_slurm] Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[polar_slurm] Killing PID ${pid}"
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "[polar_slurm] Cleanup complete."
}
trap cleanup EXIT INT TERM

# ── Source readiness poller ────────────────────────────────────────────────────
source "${TEMPLATE_DIR}/wait_for_service.sh"

# ── Launch vLLM ────────────────────────────────────────────────────────────────
echo "[polar_slurm] Starting vLLM server on ${NODE_0}:${VLLM_PORT}..."
echo "[polar_slurm]   Model: ${MODEL_PATH}"
echo "[polar_slurm]   TP size: ${TENSOR_PARALLEL_SIZE}"

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
echo "[polar_slurm] vLLM PID: ${PIDS[-1]}"

wait_for_service "http://${NODE_0}:${VLLM_PORT}/health" "vLLM" 240 5 || exit 1

# ── Launch Rollout Service ─────────────────────────────────────────────────────
echo "[polar_slurm] Starting rollout service on ${NODE_0}:${ROLLOUT_PORT}..."

python -m polar.cli serve_rollout -c "${TOPOLOGY_PATH}" \
    > "${JOB_DIR}/logs/rollout.log" 2>&1 &
PIDS+=($!)
echo "[polar_slurm] Rollout PID: ${PIDS[-1]}"

wait_for_service "http://${NODE_0}:${ROLLOUT_PORT}/health" "Rollout" 30 2 || exit 1

# ── Launch Gateway Node(s) ─────────────────────────────────────────────────────
for ((i=0; i<NUM_NODES; i++)); do
    NODE_ID="node-$(printf '%02d' ${i})"
    GW_PORT=$((GATEWAY_BASE_PORT + i))
    GW_HOST="${HOSTNAMES[$i]}"

    echo "[polar_slurm] Starting gateway ${NODE_ID} on ${GW_HOST}:${GW_PORT}..."

    if [ "${i}" -eq 0 ]; then
        # Gateway on node 0 runs locally (same as rollout + vLLM)
        python -m polar.cli serve_gateway \
            -c "${TOPOLOGY_PATH}" \
            --node-id "${NODE_ID}" \
            > "${JOB_DIR}/logs/gateway_${NODE_ID}.log" 2>&1 &
        PIDS+=($!)
    else
        # Gateway on remote node via srun
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
    echo "[polar_slurm] Gateway ${NODE_ID} PID: ${PIDS[-1]}"
done

# Wait for all gateways
for ((i=0; i<NUM_NODES; i++)); do
    GW_PORT=$((GATEWAY_BASE_PORT + i))
    GW_HOST="${HOSTNAMES[$i]}"
    wait_for_service "http://${GW_HOST}:${GW_PORT}/health" "Gateway node-$(printf '%02d' ${i})" 30 2 || exit 1
done

# ── Check overall status ───────────────────────────────────────────────────────
echo "[polar_slurm] All services running. Checking status..."
python -m polar.cli status -c "${TOPOLOGY_PATH}" --json > "${JOB_DIR}/logs/status.json" 2>&1 || true
python -m polar.cli status -c "${TOPOLOGY_PATH}" || true

# ── Submit Tasks ───────────────────────────────────────────────────────────────
echo ""
echo "[polar_slurm] ════════════════════════════════════════════════════════"
echo "[polar_slurm] Submitting tasks: EXAMPLE=${EXAMPLE}, HARNESS=${HARNESS}"
echo "[polar_slurm] ════════════════════════════════════════════════════════"

if [ -n "${TASK_FILE:-}" ]; then
    # Direct task file submission
    echo "[polar_slurm] Submitting task file: ${TASK_FILE}"
    python -m polar.cli submit "${TASK_FILE}" \
        -c "${TOPOLOGY_PATH}" \
        --json | tee "${JOB_DIR}/response.json"
else
    # Use the cluster task builder module
    INSTANCE_ID_ARGS=""
    if [ -n "${INSTANCE_IDS:-}" ]; then
        IFS=',' read -ra _IDS <<< "${INSTANCE_IDS}"
        for _id in "${_IDS[@]}"; do
            INSTANCE_ID_ARGS="${INSTANCE_ID_ARGS} --instance-id ${_id}"
        done
    fi
    python -m polar.cluster.tasks \
        --example "${EXAMPLE}" \
        --harness "${HARNESS}" \
        --topology "${TOPOLOGY_PATH}" \
        --sif-dir "${POLAR_SIFS}" \
        --output-dir "${JOB_DIR}/tasks" \
        --num-rollouts "${NUM_ROLLOUTS}" \
        --timeout-seconds "${TIMEOUT_SECONDS}" \
        ${INSTANCE_ID_ARGS}
fi

SUBMIT_EXIT=$?

# ── Results ────────────────────────────────────────────────────────────────────
echo ""
echo "[polar_slurm] ════════════════════════════════════════════════════════"
echo "[polar_slurm] Task submission finished (exit code: ${SUBMIT_EXIT})"
echo "[polar_slurm] Results directory: ${JOB_DIR}"
echo "[polar_slurm] Rollout results: ${SAVE_DIR}"
echo "[polar_slurm] ════════════════════════════════════════════════════════"

# Final status check
python -m polar.cli status -c "${TOPOLOGY_PATH}" || true

echo "[polar_slurm] Job complete."
exit ${SUBMIT_EXIT}
