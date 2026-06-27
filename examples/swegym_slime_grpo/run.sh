#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Async GRPO training on SWE-Gym via Polar + Slime (Qwen3.5-4B).
#
# Qwen3.5-4B is a VLM checkpoint (Qwen3_5ForConditionalGeneration) with
# hybrid attention (1 full + 3 GatedDeltaNet linear per 4 layers). Text-only
# RL requires the SGLang VLM input_ids patch (see MEMORY.md).
#
#
# Port layout:
#   9000        – SGLang router (slime-managed, load-balances engines)
#   8080        – Polar rollout server (task coordinator)
#   8100        – Polar gateway node (dispatches agent sessions)
#   8265        – Ray dashboard
#
# Weight sync: native GPU-to-GPU via NCCL every training step.
# Slime manages SGLang engines; Polar gateway proxies LLM calls to them.
# Dynamic-history: every trace in each agent session becomes one training
# sample, so gradients learn from *every* turn (not just the last one).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/swegym_slime_grpo}"
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

ensure_swegym_harness_ready() {
    "${PYTHON_BIN}" - <<'PY'
from swegym.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swegym.harness.grading import get_eval_report  # noqa: F401
from swegym.harness.test_spec import make_test_spec  # noqa: F401

needed = {"dask/dask", "python/mypy", "pandas-dev/pandas"}
missing = needed - set(MAP_REPO_VERSION_TO_SPECS)
if missing:
    raise SystemExit(f"SWE-Gym harness missing repo specs: {sorted(missing)}")
PY
}

is_path_like() {
    case "$1" in
        /*|./*|../*|~*) return 0 ;;
        *) return 1 ;;
    esac
}

detect_host_ip() {
    "${PYTHON_BIN}" - <<'PY'
import socket

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
    sock.close()
except Exception:
    try:
        print(socket.gethostbyname(socket.gethostname()))
    except Exception:
        print("127.0.0.1")
PY
}

resolve_host_ip() {
    local host="$1"
    if command -v getent >/dev/null 2>&1; then
        local ip
        ip="$(getent ahostsv4 "$host" | awk 'NR == 1 {print $1}')"
        if [ -n "$ip" ]; then
            echo "$ip"
            return
        fi
    fi
    echo "$host"
}

slurm_head_host() {
    if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1
    elif [ -n "${SLURM_JOB_NODELIST:-}" ]; then
        "${PYTHON_BIN}" - <<'PY'
import os
import re

nodelist = os.environ["SLURM_JOB_NODELIST"]
match = re.match(r"^(?P<prefix>[^\[]*)\[(?P<body>[^\]]+)\](?P<suffix>.*)$", nodelist)
if not match:
    print(nodelist.split(",")[0])
else:
    first = match.group("body").split(",", 1)[0].split("-", 1)[0]
    print(f"{match.group('prefix')}{first}{match.group('suffix')}")
PY
    else
        hostname
    fi
}

slurm_allocation_proxy_bypass_hosts() {
    if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
        return
    fi

    local host ip first=1
    while IFS= read -r host; do
        [ -n "$host" ] || continue
        ip="$(resolve_host_ip "$host")"
        if [ "$first" -eq 1 ]; then
            first=0
        else
            printf ','
        fi
        printf '%s,%s' "$host" "$ip"
    done < <(
        if command -v scontrol >/dev/null 2>&1; then
            scontrol show hostnames "$SLURM_JOB_NODELIST"
        else
            "${PYTHON_BIN}" - <<'PY'
import os
import re


def split_top_level(text):
    out, buf, depth = [], [], 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def expand_one(spec):
    match = re.match(r"^(?P<prefix>[^\[]*)\[(?P<body>[^\]]+)\](?P<suffix>.*)$", spec)
    if not match:
        yield spec
        return
    prefix, body, suffix = match.group("prefix"), match.group("body"), match.group("suffix")
    for item in body.split(","):
        if "-" not in item:
            yield f"{prefix}{item}{suffix}"
            continue
        start, end = item.split("-", 1)
        width = max(len(start), len(end))
        for value in range(int(start), int(end) + 1):
            yield f"{prefix}{value:0{width}d}{suffix}"


for part in split_top_level(os.environ["SLURM_JOB_NODELIST"]):
    for host in expand_one(part):
        print(host)
PY
        fi
    )
}

# ── External deps ──────────────────────────────────────────────────
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
if [ ! -f "${SLIME_DIR}/train_async.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}"
    echo "  git clone git@github.com:THUDM/slime.git ${SLIME_DIR}"
    exit 1
fi

MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
if [ ! -d "${MEGATRON_DIR}/megatron" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_DIR}"
    echo "  git clone https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_DIR}"
    exit 1
fi
if [ ! -f "${MEGATRON_DIR}/megatron/training/tokenizer/tokenizer.py" ]; then
    echo "ERROR: Megatron-LM at ${MEGATRON_DIR} is incompatible with this slime checkout" >&2
    echo "  megatron.training.tokenizer is required; use the 26.04-alpha-compatible checkout." >&2
    exit 1
fi

# ── Model ──────────────────────────────────────────────────────────
# Qwen3.5-4B: VLM checkpoint; we train text-only.  HF weights are loaded
# through slime_plugins.mbridge.qwen3_5 (text_config-aware) at convert-time.
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
RUN_ID="${RUN_ID:-swegym-slime-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/tmp/ckpt/swegym_slime_grpo_qwen35_4b}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"
mkdir -p "$SAVE_DIR"
if is_path_like "$HF_CHECKPOINT" && [ ! -e "$HF_CHECKPOINT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CHECKPOINT"
    echo "  hf download Qwen/Qwen3.5-4B"
    exit 1
fi
if [ ! -d "$REF_LOAD" ] || [ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: Megatron torch_dist checkpoint not found at $REF_LOAD"
    echo "  Run bash examples/swegym_slime_grpo/convert_weights.sh first."
    exit 1
fi

if [ "${REQUIRE_SWEGYM_HARNESS:-1}" = "1" ]; then
    if ! ensure_swegym_harness_ready; then
        echo "ERROR: SWE-Gym harness is not installed or is missing SWE-Gym repo specs." >&2
        echo "  Install it with: ${PYTHON_BIN} -m pip install 'swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7'" >&2
        exit 1
    fi
    echo "SWE-Gym harness ready."
fi

# shellcheck source=./model_args.sh
source "${SCRIPT_DIR}/model_args.sh"

# First run has an empty SAVE_DIR — slime's load_checkpoint asserts on empty.
# Pick REF_LOAD (torch_dist) until the first save lands.
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="$SAVE_DIR"
else
    LOAD_DIR="$REF_LOAD"
fi

# ── Data ───────────────────────────────────────────────────────────
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/swegym_train_293.jsonl}"
if [ ! -f "$PROMPT_DATA" ]; then
    echo "Preparing train data..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py"
fi

# ── Slurm/Ray node identity ─────────────────────────────────────────
RAY_NODE_RANK="${RAY_NODE_RANK:-${SLURM_PROCID:-0}}"
RAY_NUM_NODES="${RAY_NUM_NODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}"
RAY_HEAD_HOST="${RAY_HEAD_HOST:-$(slurm_head_host)}"
RAY_HEAD_IP="${RAY_HEAD_IP:-$(resolve_host_ip "$RAY_HEAD_HOST")}"
RAY_NODE_IP="${RAY_NODE_IP:-$(detect_host_ip)}"
RUN_DONE_FILE="${RUN_DONE_FILE:-${RUN_DIR}/ray_done}"

# The cluster image uses a container-cache proxy, and port 8080 can already be
# occupied. Keep Ray/Polar/SGLang control-plane traffic off the proxy.
CLUSTER_PROXY_BYPASS_HOSTS="$(slurm_allocation_proxy_bypass_hosts)"
PROXY_BYPASS_HOSTS="127.0.0.1,localhost,${RAY_HEAD_HOST},${RAY_HEAD_IP},${RAY_NODE_IP}"
if [ -n "$CLUSTER_PROXY_BYPASS_HOSTS" ]; then
    PROXY_BYPASS_HOSTS="${PROXY_BYPASS_HOSTS},${CLUSTER_PROXY_BYPASS_HOSTS}"
fi
export no_proxy="${no_proxy:+${no_proxy},}${PROXY_BYPASS_HOSTS}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${PROXY_BYPASS_HOSTS}"

# ── Runtime configs ─────────────────────────────────────────────────
export AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${PROJECT_ROOT}/tmp/swegym_apptainer_images}"
# Prefer apptainer in PATH (HPC modules etc.); fall back to /usr/bin for Ubuntu defaults.
export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-$(command -v apptainer || echo /usr/bin/apptainer)}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-${RAY_HEAD_IP}}"
export SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"
export POLAR_ROLLOUT_HOST="${POLAR_ROLLOUT_HOST:-0.0.0.0}"
export POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-18080}"
export POLAR_GATEWAY_HOST="${POLAR_GATEWAY_HOST:-0.0.0.0}"
export POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-18100}"
POLAR_PUBLIC_HOST="${POLAR_PUBLIC_HOST:-${RAY_HEAD_IP}}"
export POLAR_ROLLOUT_URL="${POLAR_ROLLOUT_URL:-http://${POLAR_PUBLIC_HOST}:${POLAR_ROLLOUT_PORT}}"
export POLAR_GATEWAY_URL="${POLAR_GATEWAY_URL:-http://${POLAR_PUBLIC_HOST}:${POLAR_GATEWAY_PORT}}"
export POLAR_CALLBACK_HOST="${POLAR_CALLBACK_HOST:-127.0.0.1}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-2}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-2400}"
export POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS:-2400}"
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}"
export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-32}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-32}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-32}"
POLAR_ROLLOUT_LOCAL_URL="${POLAR_ROLLOUT_LOCAL_URL:-http://127.0.0.1:${POLAR_ROLLOUT_PORT}}"
CONTROL_PLANE_BYPASS_HOSTS="${SGLANG_ROUTER_HOST},${POLAR_PUBLIC_HOST}"
export no_proxy="${no_proxy},${CONTROL_PLANE_BYPASS_HOSTS}"
export NO_PROXY="${NO_PROXY},${CONTROL_PLANE_BYPASS_HOSTS}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.yaml}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-${RUN_DIR}/topology.yaml}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config.yaml}"
COMPILER_CACHE_ROOT="${COMPILER_CACHE_ROOT:-${RUN_DIR}/compiler_cache}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${COMPILER_CACHE_ROOT}/torchinductor}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${COMPILER_CACHE_ROOT}/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Render YAML templates: only the listed ${VARS} are expanded, so literal
# $HOME / $... inside polar_config.yaml are left untouched.
mkdir -p "$(dirname "$TOPOLOGY_PATH")" "$(dirname "$CUSTOM_CONFIG_PATH")"
render_template() {
    local src="$1"
    local dst="$2"
    "${PYTHON_BIN}" - "$src" "$dst" <<'PY'
from pathlib import Path
import os
import sys

src, dst = map(Path, sys.argv[1:3])
text = src.read_text()
for name in (
    "SGLANG_ROUTER_BASE_URL",
    "AGENT_CLI_DIR",
    "APPTAINER_IMAGE_DIR",
    "POLAR_ROLLOUT_HOST",
    "POLAR_ROLLOUT_PORT",
    "POLAR_ROLLOUT_URL",
    "POLAR_GATEWAY_HOST",
    "POLAR_GATEWAY_PORT",
    "POLAR_GATEWAY_URL",
    "POLAR_CALLBACK_HOST",
    "POLAR_MAX_ASYNC_LEVEL",
    "POLAR_FULLY_ASYNC",
    "POLAR_REQUEST_TIMEOUT",
    "POLAR_TASK_TIMEOUT_SECONDS",
    "POLAR_MIN_COMPLETE_ACCEPT_FRACTION",
    "POLAR_MAX_INIT_WORKERS",
    "POLAR_MAX_RUN_WORKERS",
    "POLAR_MAX_POSTRUN_WORKERS",
    "POLAR_AGENT_HARNESS",
    "POLAR_AGENT_MODEL_NAME",
    "POLAR_AGENT_PATH",
    "POLAR_AGENT_RUNTIME_VOLUME",
    "POLAR_AGENT_STEP_LIMIT",
    "POLAR_AGENT_COST_LIMIT",
):
    text = text.replace("${" + name + "}", os.environ.get(name, ""))
dst.write_text(text)
PY
}
if [ "${RAY_NODE_RANK}" = "0" ]; then
    render_template "$TOPOLOGY_TEMPLATE" "$TOPOLOGY_PATH"
    render_template "$POLAR_CONFIG_TEMPLATE" "$CUSTOM_CONFIG_PATH"
else
    for _ in $(seq 1 120); do
        if [ -s "$TOPOLOGY_PATH" ] && [ -s "$CUSTOM_CONFIG_PATH" ]; then
            break
        fi
        sleep 1
    done
    if [ ! -s "$TOPOLOGY_PATH" ] || [ ! -s "$CUSTOM_CONFIG_PATH" ]; then
        echo "ERROR: rank ${RAY_NODE_RANK} timed out waiting for rendered configs" >&2
        exit 1
    fi
fi

echo "Using topology: ${TOPOLOGY_PATH}"
echo "Using Polar config: ${CUSTOM_CONFIG_PATH}"
echo "Using Apptainer image dir: ${APPTAINER_IMAGE_DIR}"
echo "Using run id: ${RUN_ID}"
echo "Using save dir: ${SAVE_DIR}"
echo "Using SGLang router URL for Polar gateway: ${SGLANG_ROUTER_BASE_URL}"
echo "Using Polar rollout URL: ${POLAR_ROLLOUT_URL}"
echo "Using Polar gateway URL: ${POLAR_GATEWAY_URL}"
echo "Using no_proxy: ${no_proxy}"

# ── Cleanup on exit ────────────────────────────────────────────────
PIDS=()
cleanup() {
    local status=$?
    echo "Shutting down..."
    if [ "${RAY_NODE_RANK}" = "0" ]; then
        touch "$RUN_DONE_FILE" 2>/dev/null || true
    fi
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
    return "$status"
}
trap cleanup EXIT

start_gpu_monitor() {
    if [ "${GPU_MONITOR_ENABLED:-1}" != "1" ]; then
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU monitor disabled: nvidia-smi not found"
        return
    fi
    if [ ! -f "${PROJECT_ROOT}/scripts/monitor_wandb_gpu.py" ]; then
        echo "GPU monitor disabled: scripts/monitor_wandb_gpu.py not found"
        return
    fi

    local monitor_dir metric_prefix csv_path node_role train_gpus rollout_gpus
    monitor_dir="${RUN_DIR}/gpu_monitor"
    mkdir -p "$monitor_dir"
    metric_prefix="${GPU_MONITOR_PREFIX:-polar_system}"
    node_role="${GPU_MONITOR_NODE_ROLE:-mixed}"
    train_gpus="${GPU_MONITOR_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
    rollout_gpus="${GPU_MONITOR_ROLLOUT_GPUS:-}"
    if [ "${RAY_NUM_NODES}" -gt 1 ]; then
        if [ -z "${GPU_MONITOR_NODE_ROLE:-}" ]; then
            if [ "${RAY_NODE_RANK}" -lt "${ACTOR_NUM_NODES}" ]; then
                node_role="actor"
            else
                node_role="rollout"
            fi
        fi
        if [ -z "${GPU_MONITOR_TRAIN_GPUS+x}" ]; then
            if [ "$node_role" = "actor" ]; then
                train_gpus="0,1,2,3,4,5,6,7"
            else
                train_gpus=""
            fi
        fi
        if [ -z "${GPU_MONITOR_ROLLOUT_GPUS+x}" ]; then
            if [ "$node_role" = "rollout" ]; then
                rollout_gpus="0,1,2,3,4,5,6,7"
            else
                rollout_gpus=""
            fi
        fi
        if [ "$node_role" = "rank" ]; then
            train_gpus=""
            rollout_gpus=""
            metric_prefix="${metric_prefix}/node_${RAY_NODE_RANK}"
        else
            metric_prefix="${metric_prefix}/${node_role}_node_${RAY_NODE_RANK}"
        fi
    fi
    csv_path="${monitor_dir}/node_${RAY_NODE_RANK}.csv"

    local wandb_args=("--no-wandb")
    if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-offline}" != "disabled" ]; then
        wandb_args=(
            "--wandb-run-id" "$RUN_ID"
            "--wandb-project" "${WANDB_PROJECT:-polar-swegym-grpo}"
            "--wandb-mode" "${GPU_MONITOR_WANDB_MODE:-shared}"
        )
        if [ -n "${WANDB_ENTITY:-}" ]; then
            wandb_args+=("--wandb-entity" "$WANDB_ENTITY")
        fi
    fi

    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/monitor_wandb_gpu.py" \
        --interval-s "${GPU_MONITOR_INTERVAL_S:-10}" \
        --out-csv "$csv_path" \
        --metric-prefix "$metric_prefix" \
        --train-gpus "$train_gpus" \
        --rollout-gpus "$rollout_gpus" \
        "${wandb_args[@]}" &
    PIDS+=($!)
}

wait_http_ok() {
    local name="$1"
    local url="$2"
    local attempts="${3:-60}"
    local i
    for i in $(seq 1 "$attempts"); do
        if curl --noproxy '*' -fsS --max-time 5 "$url" >/dev/null 2>&1; then
            echo "${name} healthy: ${url}"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: ${name} not healthy after ${attempts} attempts: ${url}" >&2
    return 1
}

wait_ray_dashboard() {
    local url="http://${RAY_HEAD_IP}:8265/api/version"
    local i
    for i in $(seq 1 90); do
        if curl --noproxy '*' -fsS --max-time 5 "$url" 2>/dev/null | "${PYTHON_BIN}" -c 'import json, sys; data=json.load(sys.stdin); assert data.get("ray_version")' >/dev/null 2>&1; then
            echo "Ray dashboard healthy: ${url}"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Ray dashboard not healthy after 90 attempts: ${url}" >&2
    return 1
}

# ── Step 1: Ray cluster ────────────────────────────────────────────
if [ "${RAY_NUM_NODES}" -gt 1 ]; then
    ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-$((RAY_NUM_NODES - 1))}"
else
    ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
fi
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-9}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-30000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32000}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-0}"
if [ $(((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT) % NUM_STEPS_PER_ROLLOUT)) -ne 0 ]; then
    echo "ERROR: rollout batch product must be divisible by NUM_STEPS_PER_ROLLOUT" >&2
    exit 1
fi
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT) / NUM_STEPS_PER_ROLLOUT))}"
EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
echo "Using rollout/global batch: ${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT}/${NUM_STEPS_PER_ROLLOUT}=${GLOBAL_BATCH_SIZE}, eval=${EVAL_GLOBAL_BATCH_SIZE}"
SEQUENCE_PARALLEL_ARGS=()
if [ "${SEQUENCE_PARALLEL}" = "1" ] || [ "${SEQUENCE_PARALLEL}" = "true" ]; then
    SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
fi
echo "Using sequence parallel: ${SEQUENCE_PARALLEL}"
WANDB_STEP_ARGS=()
if [ "${WANDB_ALWAYS_USE_TRAIN_STEP:-0}" = "1" ] || \
   [ "${WANDB_ALWAYS_USE_TRAIN_STEP:-0}" = "true" ]; then
    WANDB_STEP_ARGS=(--wandb-always-use-train-step)
    echo "Using W&B metric axis: train/step"
else
    echo "Using W&B metric axis: rollout/step"
fi
RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-}"
if [ -z "$RAY_NUM_GPUS_PER_NODE" ]; then
    if [ "${RAY_NUM_NODES}" -gt 1 ]; then
        RAY_NUM_GPUS_PER_NODE=8
    else
        RAY_NUM_GPUS_PER_NODE=$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))
    fi
fi

echo "=== Ray node rank ${RAY_NODE_RANK}/${RAY_NUM_NODES} head=${RAY_HEAD_IP} local=${RAY_NODE_IP} gpus/node=${RAY_NUM_GPUS_PER_NODE} ==="
ray stop --force 2>/dev/null || true
sleep 1
if [ "${RAY_NODE_RANK}" = "0" ]; then
    rm -f "$RUN_DONE_FILE"
    ray start --head \
        --node-ip-address "$RAY_HEAD_IP" \
        --num-gpus "$RAY_NUM_GPUS_PER_NODE" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265
else
    joined_ray=0
    for _ in $(seq 1 60); do
        if ray start \
            --address="${RAY_HEAD_IP}:6379" \
            --node-ip-address "$RAY_NODE_IP" \
            --num-gpus "$RAY_NUM_GPUS_PER_NODE" \
            --disable-usage-stats; then
            joined_ray=1
            break
        fi
        sleep 5
    done
    if [ "$joined_ray" != "1" ]; then
        echo "ERROR: Ray worker rank ${RAY_NODE_RANK} could not join ${RAY_HEAD_IP}:6379" >&2
        exit 1
    fi
    start_gpu_monitor
    while [ ! -f "$RUN_DONE_FILE" ]; do
        sleep 30
    done
    exit 0
fi

start_gpu_monitor
if [ "${RAY_NUM_NODES}" -gt 1 ]; then
    echo "Waiting for Ray workers..."
    sleep 45
fi
ray status || true
wait_ray_dashboard

# ── Step 2: Polar services (rank 0, CPU only) ──────────────────────
echo "=== Starting Polar rollout server (${POLAR_ROLLOUT_URL}) ==="
polar serve_rollout -c "${TOPOLOGY_PATH}" &
PIDS+=($!)
wait_http_ok "Polar rollout server" "${POLAR_ROLLOUT_LOCAL_URL}/health" 60

echo "=== Starting Polar gateway (${POLAR_GATEWAY_URL}) ==="
polar serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 &
PIDS+=($!)
sleep 2

# ── Step 3: Slime (manages SGLang engines + training) ──────────────

# cuDNN lib path — probe the active venv instead of hardcoding python3.13.
if [ -z "${CUDNN_LIB:-}" ]; then
    CUDNN_LIB="$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
fi
RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "${CUDNN_LIB}" ] && [ -d "$CUDNN_LIB" ]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
fi
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MASTER_ADDR\": \"${RAY_HEAD_IP}\",
    \"no_proxy\": \"${no_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-offline}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT:-polar-swegym-grpo}\",
    \"WANDB_GROUP\": \"${WANDB_GROUP:-swegym-qwen35-4b-async-grpo}\",
    \"WANDB_RUN_ID\": \"${RUN_ID}\",
    \"WANDB_RESUME\": \"${WANDB_RESUME:-allow}\",
    \"HF_TOKEN\": \"${HF_TOKEN:-}\",
    \"HUGGINGFACE_HUB_TOKEN\": \"${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD\": \"${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-}\",
    \"SLIME_ROLLOUT_BASE_PORT\": \"${SLIME_ROLLOUT_BASE_PORT:-15000}\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"2\"
  }
}"

# Rollout sizing: default 9 prompts × 16 trajectories = 144 trajectories/rollout.
# With the default 3 actor nodes, TP=2 gives DP=12, so 144 is divisible by
# Megatron's micro_batch_size(1) * data_parallel_size(12).
# With --dynamic-history each trajectory explodes into one sample per trace,
# so sample count per rollout is variable.
# The custom data source rounds epoch length up to 37 rollout batches, so all
# 293 train prompts are consumed once; the final fixed-size batch wraps 3 prompts.
RAY_JOB_ADDRESS="http://${RAY_HEAD_IP}:8265"
RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-polar-${SLURM_JOB_ID:-$$}}"
TRAINING_LIFECYCLE_ARGS=()
if [ -n "${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME:-}" ]; then
    TRAINING_LIFECYCLE_ARGS+=(--graceful-exit-at-unix-time "${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME}")
fi
if [ -n "${TRAINING_COMPLETE_MARKER:-}" ]; then
    TRAINING_LIFECYCLE_ARGS+=(--training-complete-marker "${TRAINING_COMPLETE_MARKER}")
fi
echo "=== Launching train_async.py (Ray submission ${RAY_JOB_SUBMISSION_ID}) ==="
ray job submit --address="${RAY_JOB_ADDRESS}" \
    --submission-id "${RAY_JOB_SUBMISSION_ID}" \
    --no-wait \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes "$ACTOR_NUM_NODES" \
    --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
    --rollout-num-gpus "$ROLLOUT_NUM_GPUS" \
    --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --ref-load "$REF_LOAD" \
    --load "$LOAD_DIR" \
    --dist-ckpt-strictness "${DIST_CKPT_STRICTNESS:-assume_ok_unexpected}" \
    --save "$SAVE_DIR" \
    --save-interval "${SAVE_INTERVAL:-10}" \
    "${TRAINING_LIFECYCLE_ARGS[@]}" \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rm-path slime_bridge.reward.reward_func \
    --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
    --custom-config-path "${CUSTOM_CONFIG_PATH}" \
    --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer \
    --prompt-data "$PROMPT_DATA" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --rollout-shuffle \
    --reward-key score \
    --num-epoch "${NUM_EPOCH:-1}" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --eval-global-batch-size "$EVAL_GLOBAL_BATCH_SIZE" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" \
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" \
    --dynamic-history \
    --num-steps-per-rollout "$NUM_STEPS_PER_ROLLOUT" \
    --seq-length "$SEQ_LENGTH" \
    --tensor-model-parallel-size 2 \
    "${SEQUENCE_PARALLEL_ARGS[@]}" \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
    --log-probs-chunk-size 256 \
    --distributed-timeout-minutes 30 \
    --advantage-estimator grpo \
    --normalize-advantages \
    --use-tis \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend "${ATTENTION_BACKEND:-auto}" \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static 0.8 \
    --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
    --use-wandb \
    "${WANDB_STEP_ARGS[@]}" \
    --wandb-mode "${WANDB_MODE:-offline}" \
    --wandb-run-id "$RUN_ID" \
    --wandb-project "${WANDB_PROJECT:-polar-swegym-grpo}" \
    --wandb-group "${WANDB_GROUP:-swegym-qwen35-4b-async-grpo}" \
    --disable-wandb-random-suffix \
    --sglang-router-port "$SGLANG_ROUTER_PORT"

# Ray's CLI normally owns the log WebSocket. A transient close code 1006 would
# otherwise make this shell exit and tear down a healthy multi-node job.
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/wait_ray_job.py" \
    --address "${RAY_JOB_ADDRESS}" \
    --submission-id "${RAY_JOB_SUBMISSION_ID}" \
    --retry-interval "${RAY_JOB_STATUS_RETRY_INTERVAL:-5}" \
    --max-status-errors "${RAY_JOB_MAX_STATUS_ERRORS:-120}"
