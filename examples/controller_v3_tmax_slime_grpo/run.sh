#!/usr/bin/env bash
# Reserve GPUs 2-7 on the final rank for frozen Qwen workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SHARED_RUN="${SCRIPT_DIR}/../tmax_slime_grpo/run.sh"
: "${SLURM_NODEID:?}"
: "${RUN_DIR:?}"
: "${HF_CHECKPOINT:?}"
: "${POLR_TRAIN_VENV:?}"
: "${SGLANG_DIR:?}"

SMALL_NODE_RANK="$((NUM_NODES - 1))"
SMALL_NODE_FILE="${RUN_DIR}/startup/controller-v3-small-node"
mkdir -p "${RUN_DIR}/startup"
if [ "${SLURM_NODEID}" = "${SMALL_NODE_RANK}" ]; then
    SMALL_NODE="$(hostname -s)"
    printf '%s\n' "${SMALL_NODE}" >"${SMALL_NODE_FILE}.tmp"
    mv -f "${SMALL_NODE_FILE}.tmp" "${SMALL_NODE_FILE}"
else
    for _ in $(seq 1 600); do
        [ -s "${SMALL_NODE_FILE}" ] && break
        sleep 1
    done
    if [ ! -s "${SMALL_NODE_FILE}" ]; then
        echo "ERROR: timed out waiting for frozen-worker hostname" >&2
        exit 1
    fi
    read -r SMALL_NODE <"${SMALL_NODE_FILE}"
fi
if ! [[ "${SMALL_NODE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid frozen-worker hostname: ${SMALL_NODE}" >&2
    exit 1
fi
SMALL_ROUTER_PORT="${CONTROLLER_V3_SMALL_ROUTER_PORT:-19090}"
export CONTROLLER_V3_SMALL_ROUTER_BASE_URL="http://${SMALL_NODE}:${SMALL_ROUTER_PORT}/v1"
READY_FILE="${RUN_DIR}/startup/controller-v3-small-ready"
PYTHON_BIN="${POLR_TRAIN_VENV}/bin/python3"
LOCAL_NO_PROXY="0.0.0.0,127.0.0.1,localhost"
PIDS=()

cleanup_small_workers() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup_small_workers EXIT TERM INT

wait_http() {
    local url="$1" label="$2" pid="$3"
    for _ in $(seq 1 450); do
        if curl --noproxy '*' -fsS --max-time 2 "${url}" >/dev/null; then
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${label} exited during startup" >&2
            return 1
        fi
        sleep 2
    done
    echo "ERROR: timed out waiting for ${label}" >&2
    return 1
}

start_group() {
    local log="$1"
    shift
    setsid "$@" >"${log}" 2>&1 &
    LAST_PID=$!
    PIDS+=("${LAST_PID}")
}

if [ "${SLURM_NODEID}" = "${SMALL_NODE_RANK}" ]; then
    export RAY_NUM_GPUS_PER_NODE=2
    mkdir -p "${RUN_DIR}/startup" "${RUN_DIR}/controller-v3-small"
    rm -f "${READY_FILE}"

    server_args=(
        -m sglang.launch_server
        --model-path "${HF_CHECKPOINT}"
        --served-model-name nvidia/qwen/qwen3.6-35b-a3b
        --host 0.0.0.0
        --tp-size 2
        --ep-size 2
        --mem-fraction-static 0.77
        --max-running-requests 16
        --chunked-prefill-size 8192
        --max-prefill-tokens 16384
        --reasoning-parser qwen3
        --tool-call-parser qwen3_coder
        --linear-attn-backend triton
        --mamba-scheduler-strategy no_buffer
        --mamba-track-interval 256
        --disable-custom-all-reduce
    )

    for replica in 0 1 2; do
        first_gpu=$((2 + replica * 2))
        second_gpu=$((first_gpu + 1))
        port=$((SMALL_ROUTER_PORT + 1 + replica))
        start_group "${RUN_DIR}/controller-v3-small/replica-${replica}.log"             env CUDA_VISIBLE_DEVICES="${first_gpu},${second_gpu}"             NO_PROXY="${LOCAL_NO_PROXY}" no_proxy="${LOCAL_NO_PROXY}"             PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"             "${PYTHON_BIN}" "${server_args[@]}" --port "${port}"
        eval "replica_${replica}_pid=${LAST_PID}"
    done

    wait_http "http://127.0.0.1:$((SMALL_ROUTER_PORT + 1))/health_generate"         "small worker 0" "${replica_0_pid}"
    wait_http "http://127.0.0.1:$((SMALL_ROUTER_PORT + 2))/health_generate"         "small worker 1" "${replica_1_pid}"
    wait_http "http://127.0.0.1:$((SMALL_ROUTER_PORT + 3))/health_generate"         "small worker 2" "${replica_2_pid}"

    start_group "${RUN_DIR}/controller-v3-small/router.log"         env NO_PROXY="${LOCAL_NO_PROXY}" no_proxy="${LOCAL_NO_PROXY}"         PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"         "${PYTHON_BIN}" -m sglang_router.launch_router         --host 0.0.0.0 --port "${SMALL_ROUTER_PORT}"         --worker-urls         "http://127.0.0.1:$((SMALL_ROUTER_PORT + 1))"         "http://127.0.0.1:$((SMALL_ROUTER_PORT + 2))"         "http://127.0.0.1:$((SMALL_ROUTER_PORT + 3))"         --policy power_of_two         --max-concurrent-requests 48         --queue-size 512         --queue-timeout-secs 600
    router_pid="${LAST_PID}"
    wait_http "http://127.0.0.1:${SMALL_ROUTER_PORT}/health"         "small-worker router" "${router_pid}"
    printf 'ready\n' >"${READY_FILE}.tmp"
    mv -f "${READY_FILE}.tmp" "${READY_FILE}"
else
    export RAY_NUM_GPUS_PER_NODE=8
    for _ in $(seq 1 1800); do
        [ -s "${READY_FILE}" ] && break
        sleep 1
    done
    if [ ! -s "${READY_FILE}" ]; then
        echo "ERROR: timed out waiting for frozen Qwen workers" >&2
        exit 1
    fi
fi

bash "${SHARED_RUN}"
