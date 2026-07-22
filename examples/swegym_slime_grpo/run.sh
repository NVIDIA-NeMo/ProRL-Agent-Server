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
#   8680        – SGLang router (slime-managed, load-balances engines)
#   18080       – Polar rollout server (task coordinator)
#   18100       – Polar gateway node (dispatches agent sessions)
#   8265        – Ray dashboard
#
# Weight sync: native GPU-to-GPU via NCCL every training step.
# Slime manages SGLang engines; Polar gateway proxies LLM calls to them.
# Dynamic-history: every trace in each agent session becomes one training
# sample, so gradients learn from *every* turn (not just the last one).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail
export SLIME_JOB_SCRIPT_START_UNIX_NS="${SLIME_JOB_SCRIPT_START_UNIX_NS:-$(date +%s%N)}"

SCRIPT_DIR="${POLAR_SHARED_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)}"
PROJECT_ROOT="${POLAR_TRAIN_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
# shellcheck source=./launcher_utils.sh
source "${SCRIPT_DIR}/launcher_utils.sh"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/swegym_slime_grpo}"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
export POLAR_ROLLOUT_SAVE_DIR="${POLAR_ROLLOUT_SAVE_DIR:-${RUN_DIR}/rollout_results}"
export POLAR_ROLLOUT_EXAMPLES_DIR="${POLAR_ROLLOUT_EXAMPLES_DIR:-${RUN_DIR}/trajectory_examples}"
export POLAR_ROLLOUT_EXAMPLE_INTERVAL="${POLAR_ROLLOUT_EXAMPLE_INTERVAL:-10}"
export POLAR_ROLLOUT_EXAMPLE_COUNT="${POLAR_ROLLOUT_EXAMPLE_COUNT:-2}"
export POLAR_ROLLOUT_EXAMPLES_WANDB="${POLAR_ROLLOUT_EXAMPLES_WANDB:-1}"
case "${POLAR_ROLLOUT_SAVE_DIR}" in
    /*) ;;
    *)
        echo "ERROR: POLAR_ROLLOUT_SAVE_DIR must be absolute: ${POLAR_ROLLOUT_SAVE_DIR}" >&2
        exit 1
        ;;
esac
case "${WANDB_DIR}" in
    /*) ;;
    *)
        echo "ERROR: WANDB_DIR must be absolute: ${WANDB_DIR}" >&2
        exit 1
        ;;
esac
case "${POLAR_ROLLOUT_EXAMPLES_DIR}" in
    /*) ;;
    *)
        echo "WARNING: trajectory examples disabled because POLAR_ROLLOUT_EXAMPLES_DIR is not absolute: ${POLAR_ROLLOUT_EXAMPLES_DIR}" >&2
        ;;
esac
if ! [[ "${POLAR_ROLLOUT_EXAMPLE_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARNING: invalid POLAR_ROLLOUT_EXAMPLE_INTERVAL=${POLAR_ROLLOUT_EXAMPLE_INTERVAL}; using 10" >&2
    export POLAR_ROLLOUT_EXAMPLE_INTERVAL=10
fi
if ! [[ "${POLAR_ROLLOUT_EXAMPLE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARNING: invalid POLAR_ROLLOUT_EXAMPLE_COUNT=${POLAR_ROLLOUT_EXAMPLE_COUNT}; using 2" >&2
    export POLAR_ROLLOUT_EXAMPLE_COUNT=2
fi
case "${POLAR_ROLLOUT_EXAMPLES_WANDB}" in
    1|true|yes|on|0|false|no|off) ;;
    *)
        echo "WARNING: invalid POLAR_ROLLOUT_EXAMPLES_WANDB=${POLAR_ROLLOUT_EXAMPLES_WANDB}; using 1" >&2
        export POLAR_ROLLOUT_EXAMPLES_WANDB=1
        ;;
esac
mkdir -p "${RUN_DIR}" "${WANDB_DIR}"
if [[ "${POLAR_ROLLOUT_EXAMPLES_DIR}" = /* ]] && \
   ! mkdir -p "${POLAR_ROLLOUT_EXAMPLES_DIR}"; then
    echo "WARNING: could not create trajectory-example directory; training will continue" >&2
fi
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

proxy_tcp_target_from_url() {
    local proxy_url="$1"
    "${PYTHON_BIN}" - "$proxy_url" <<'PY'
import sys
from urllib.parse import urlsplit

raw = sys.argv[1].strip()
if not raw:
    raise SystemExit("proxy URL must not be empty")
parsed = urlsplit(raw if "://" in raw else f"http://{raw}")
if not parsed.hostname:
    raise SystemExit("proxy URL has no hostname")
try:
    port = parsed.port
except ValueError as exc:
    raise SystemExit(f"invalid proxy port: {exc}") from exc
if port is None:
    port = 443 if parsed.scheme.lower() == "https" else 80
host = parsed.hostname
if ":" in host:
    host = f"[{host}]"
print(f"{host}:{port}")
PY
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

slurm_allocation_hosts() {
    local nodelist="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}" expanded
    if [ -z "${nodelist}" ]; then
        echo "ERROR: neither SLURM_JOB_NODELIST nor SLURM_NODELIST is set" >&2
        return 1
    fi
    if command -v scontrol >/dev/null 2>&1 && \
       expanded="$(scontrol show hostnames "${nodelist}" 2>/dev/null)" && \
       [ -n "${expanded}" ]; then
        printf '%s\n' "${expanded}"
        return
    fi

    # Training sqsh images intentionally do not contain the Slurm client. The
    # allocation environment is still propagated by srun/Pyxis, so expand its
    # hostlist locally instead of making topology rendering depend on scontrol.
    POLAR_SLURM_NODELIST="${nodelist}" "${PYTHON_BIN}" - <<'PY'
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
        range_match = re.fullmatch(r"(?P<start>\d+)-(?P<end>\d+)(?::(?P<step>\d+))?", item)
        if range_match is None:
            yield from expand_one(f"{prefix}{item}{suffix}")
            continue
        start = range_match.group("start")
        end = range_match.group("end")
        step = int(range_match.group("step") or "1")
        if step < 1:
            raise SystemExit(f"invalid zero hostlist step in {item!r}")
        width = max(len(start), len(end))
        start_value, end_value = int(start), int(end)
        direction = 1 if end_value >= start_value else -1
        stop = end_value + direction
        for value in range(start_value, stop, direction * step):
            yield from expand_one(f"{prefix}{value:0{width}d}{suffix}")


for part in split_top_level(os.environ["POLAR_SLURM_NODELIST"]):
    for host in expand_one(part):
        print(host)
PY
}

slurm_head_host() {
    local hosts
    if hosts="$(slurm_allocation_hosts 2>/dev/null)" && [ -n "${hosts}" ]; then
        printf '%s\n' "${hosts%%$'\n'*}"
    else
        hostname
    fi
}

slurm_allocation_proxy_bypass_hosts() {
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
    done < <(slurm_allocation_hosts 2>/dev/null || true)
}

# ── External deps ──────────────────────────────────────────────────
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
TMAX_TRAIN_MODE="${TMAX_TRAIN_MODE:-fully_async}"
TMAX_PROFILE_DISABLE_CHECKPOINT="${TMAX_PROFILE_DISABLE_CHECKPOINT:-0}"
TRAIN_MODE_ARGS=()
case "${TMAX_TRAIN_MODE}" in
    fully_async)
        SLIME_TRAIN_ENTRYPOINT="${SLIME_DIR}/train_async.py"
        ;;
    colocate)
        SLIME_TRAIN_ENTRYPOINT="${SLIME_DIR}/train.py"
        TRAIN_MODE_ARGS=(--colocate)
        if [ "${POLAR_FULLY_ASYNC:-false}" != "false" ]; then
            echo "ERROR: TMAX_TRAIN_MODE=colocate requires POLAR_FULLY_ASYNC=false" >&2
            exit 1
        fi
        if [ -n "${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME:-}" ]; then
            echo "ERROR: TMAX_TRAIN_MODE=colocate does not support graceful lifecycle arguments" >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: TMAX_TRAIN_MODE must be fully_async or colocate, got ${TMAX_TRAIN_MODE}" >&2
        exit 1
        ;;
esac

# TorchMemorySaver is required by Slime's colocated train/rollout path and is
# currently incompatible with CUDA allocator expandable segments. Keep the
# existing fully-async allocator behavior, but select a compatible allocator
# before constructing Ray's runtime environment for colocate. An explicit
# TMAX_PYTORCH_ALLOC_CONF remains available for controlled experiments.
TMAX_PYTORCH_ALLOC_CONF="$(polar_select_pytorch_allocator_config \
    "${TMAX_TRAIN_MODE}" "${TMAX_PYTORCH_ALLOC_CONF:-}")" || exit 1
echo "Using PyTorch allocator config: ${TMAX_PYTORCH_ALLOC_CONF}"
case "${TMAX_PROFILE_DISABLE_CHECKPOINT}" in
    0|1) ;;
    *)
        echo "ERROR: TMAX_PROFILE_DISABLE_CHECKPOINT must be 0 or 1" >&2
        exit 1
        ;;
esac
if [ ! -f "${SLIME_TRAIN_ENTRYPOINT}" ]; then
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
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_ID}}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/tmp/ckpt/swegym_slime_grpo_qwen35_4b}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"
mkdir -p "$SAVE_DIR"
export SLIME_TRAIN_PROGRESS_FILE="${SLIME_TRAIN_PROGRESS_FILE:-${SAVE_DIR}/train_progress.step}"
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

MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SCRIPT_DIR}/model_args.sh}"
if [ ! -r "${MODEL_ARGS_FILE}" ]; then
    echo "ERROR: Megatron model args file is not readable: ${MODEL_ARGS_FILE}" >&2
    exit 1
fi
# The TMax wrapper selects its Qwen3.5-9B file explicitly; standalone
# SWE-Gym retains the historical Qwen3.5-4B default above.
# shellcheck source=/dev/null
source "${MODEL_ARGS_FILE}"
if ! declare -p MODEL_ARGS >/dev/null 2>&1; then
    echo "ERROR: ${MODEL_ARGS_FILE} did not define the MODEL_ARGS array" >&2
    exit 1
fi
polar_validate_model_args "${MODEL_ARGS[@]}"
echo "Using model args: ${MODEL_ARGS_FILE}"

configure_resumed_checkpoint_eval_args() {
    RESUMED_CHECKPOINT_EVAL_ARGS=()
    case "${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN:-0}" in
        0|false)
            return 0
            ;;
        1|true) ;;
        *)
            echo "ERROR: TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN must be 0/1/false/true" >&2
            return 1
            ;;
    esac
    case "${TMAX_CONCURRENT_PRETRAIN_EVAL:-1}" in
        0|false) ;;
        *)
            echo "ERROR: resumed-checkpoint eval requires TMAX_CONCURRENT_PRETRAIN_EVAL=0" >&2
            return 1
            ;;
    esac
    case "${TMAX_TRAINING_EVAL_ENABLED:-${TMAX_EVAL_ENABLED:-0}}" in
        1|true) ;;
        *)
            echo "ERROR: resumed-checkpoint eval requires training-time eval (TMAX_TRAINING_EVAL_ENABLED=1)" >&2
            return 1
            ;;
    esac

    # Enable this exactly once for a logical run: when polar_select_load_dir
    # chose the caller's external numeric seed. A later allocation selects its
    # own SAVE_DIR and must not manufacture an extra resume-baseline eval.
    if [ -z "${REQUESTED_LOAD_DIR}" ] || \
       [ "${LOAD_DIR}" != "${REQUESTED_LOAD_DIR}" ] || \
       [ "${LOAD_DIR}" = "${SAVE_DIR}" ] || \
       polar_checkpoint_is_release_seed "${LOAD_DIR}"; then
        return 0
    fi
    local seed_iteration
    seed_iteration="$(tr -d '[:space:]' <"${LOAD_DIR}/latest_checkpointed_iteration.txt")"
    if ! [[ "${seed_iteration}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "ERROR: resumed-checkpoint eval requires a canonical numeric checkpoint tracker" >&2
        return 1
    fi
    RESUMED_CHECKPOINT_EVAL_ARGS=(--eval-resumed-checkpoint-before-train)
    echo "Using synchronous fixed eval for external seed checkpoint ${seed_iteration} before rollout $((seed_iteration + 1))"
}

# LOAD_DIR can seed a new RUN_ID/SAVE_DIR from an existing full checkpoint.
# After the first save lands, subsequent allocations resume SAVE_DIR instead
# of repeatedly going back to the seed checkpoint.
REQUESTED_LOAD_DIR="${LOAD_DIR:-}"
LOAD_DIR="$(polar_select_load_dir "$SAVE_DIR" "$REF_LOAD" "$REQUESTED_LOAD_DIR")"
if [ ! -d "$LOAD_DIR" ] || [ ! -s "$LOAD_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: load checkpoint not found or incomplete at $LOAD_DIR" >&2
    echo "  Expected a directory containing latest_checkpointed_iteration.txt." >&2
    exit 1
fi
echo "Using load dir: ${LOAD_DIR}"
LOAD_CHECKPOINT_ARGS=()
if polar_checkpoint_is_release_seed "$LOAD_DIR"; then
    # A release checkpoint contains initial model weights only. Megatron
    # reports it as iteration zero, so pin the RL cursor to the true first
    # rollout instead of trying to restore nonexistent rollout state 0.
    LOAD_CHECKPOINT_ARGS=(--start-rollout-id 0)
    echo "Using release checkpoint as model seed: start rollout 0"
fi
configure_resumed_checkpoint_eval_args || exit 1

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
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-${_POLAR_SGLANG_ROUTER_PORT_DEFAULT}}"
polar_validate_sglang_router_port "${SGLANG_ROUTER_PORT}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-${RAY_HEAD_IP}}"
export SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"
export POLAR_ROLLOUT_HOST="${POLAR_ROLLOUT_HOST:-0.0.0.0}"
export POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-18080}"
export POLAR_GATEWAY_HOST="${POLAR_GATEWAY_HOST:-0.0.0.0}"
export POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-18100}"
POLAR_PUBLIC_HOST="${POLAR_PUBLIC_HOST:-${RAY_HEAD_IP}}"
export POLAR_ROLLOUT_URL="${POLAR_ROLLOUT_URL:-http://${POLAR_PUBLIC_HOST}:${POLAR_ROLLOUT_PORT}}"
export POLAR_GATEWAY_URL="${POLAR_GATEWAY_URL:-http://${POLAR_PUBLIC_HOST}:${POLAR_GATEWAY_PORT}}"
export POLAR_SANDBOX_NETWORK="${POLAR_SANDBOX_NETWORK:-host}"
# AF_UNIX paths are limited to roughly 108 bytes on Linux. Keep the host-side
# tunnel under job-local /tmp rather than the much longer Lustre RUN_DIR.
export POLAR_UDS_DIR="${POLAR_UDS_DIR:-${POLAR_JOB_CACHE_ROOT:-/tmp/polar-${SLURM_JOB_ID:-$$}}/uds}"
export POLAR_GATEWAY_UDS_DIR="${POLAR_GATEWAY_UDS_DIR:-${POLAR_UDS_DIR}/gateway}"
export POLAR_PROXY_UDS_DIR="${POLAR_PROXY_UDS_DIR:-${POLAR_UDS_DIR}/proxy}"
export POLAR_GATEWAY_UDS_SOCKET="${POLAR_GATEWAY_UDS_SOCKET:-${POLAR_GATEWAY_UDS_DIR}/gateway.sock}"
export POLAR_PROXY_UDS_SOCKET="${POLAR_PROXY_UDS_SOCKET:-${POLAR_PROXY_UDS_DIR}/proxy.sock}"
export POLAR_UDS_TUNNEL_READY_FILE="${POLAR_UDS_TUNNEL_READY_FILE:-${POLAR_UDS_DIR}/ready}"
export POLAR_UDS_TUNNEL_BACKLOG="${POLAR_UDS_TUNNEL_BACKLOG:-4096}"
export POLAR_SANDBOX_HTTP_PROXY_PORT="${POLAR_SANDBOX_HTTP_PROXY_PORT:-28100}"
POLAR_PROXY_TCP_TARGET=""
export POLAR_SANDBOX_GATEWAY_UDS="${POLAR_SANDBOX_GATEWAY_UDS:-/polar/gateway/gateway.sock}"
POLAR_EFFECTIVE_HTTP_PROXY="${http_proxy:-}"
if [ -n "${POLAR_EFFECTIVE_HTTP_PROXY}" ]; then
    POLAR_PROXY_TCP_TARGET="$(proxy_tcp_target_from_url "${POLAR_EFFECTIVE_HTTP_PROXY}")"
    export POLAR_PROXY_UDS_ENABLED=1
    export POLAR_SANDBOX_HTTP_PROXY_UDS="${POLAR_SANDBOX_HTTP_PROXY_UDS:-/polar/proxy/proxy.sock}"
    printf -v POLAR_INTERNET_RUNTIME_VOLUME \
        '    internet_volumes:\n      - "%s:/polar/proxy:ro"' \
        "${POLAR_PROXY_UDS_DIR}"
else
    export POLAR_PROXY_UDS_ENABLED=0
    POLAR_INTERNET_RUNTIME_VOLUME=""
    export POLAR_SANDBOX_HTTP_PROXY_UDS=""
fi
export POLAR_INTERNET_RUNTIME_VOLUME
# TMax's task template always declares the job-private directory as an explicit
# bind. The gateway tunnel is useful in both modes: network=none requires it,
# while host-network A/B tasks marked allow_internet=false are still forced into
# network=none by the runtime and must retain model access. The proxy lives in a
# separate, policy-gated bind so an offline sandbox cannot open its UDS directly.
mkdir -p "${POLAR_UDS_DIR}" "${POLAR_GATEWAY_UDS_DIR}" "${POLAR_PROXY_UDS_DIR}"
chmod 700 "${POLAR_UDS_DIR}" "${POLAR_GATEWAY_UDS_DIR}" "${POLAR_PROXY_UDS_DIR}"
export POLAR_CALLBACK_HOST="${POLAR_CALLBACK_HOST:-127.0.0.1}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-2}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-2400}"
export POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS:-2400}"
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}"
export POLAR_EARLY_STOP_GRACE_SESSIONS="${POLAR_EARLY_STOP_GRACE_SESSIONS:-0}"
export POLAR_CANDIDATE_POOL_HEALTH_GATE_ENABLED="${POLAR_CANDIDATE_POOL_HEALTH_GATE_ENABLED:-false}"
export POLAR_CANDIDATE_POOL_HEALTH_MIN_OBSERVED_SESSIONS="${POLAR_CANDIDATE_POOL_HEALTH_MIN_OBSERVED_SESSIONS:-16}"
export POLAR_CANDIDATE_POOL_HEALTH_MIN_COMPLETION_FRACTION="${POLAR_CANDIDATE_POOL_HEALTH_MIN_COMPLETION_FRACTION:-0.1}"
export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-32}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-32}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-32}"
export POLAR_COMPLETION_QUEUE_SIZE="${POLAR_COMPLETION_QUEUE_SIZE:-16384}"
export POLAR_COMPLETION_WRITE_WORKERS="${POLAR_COMPLETION_WRITE_WORKERS:-8}"
export POLAR_COMPLETION_BATCH_SIZE="${POLAR_COMPLETION_BATCH_SIZE:-16}"
export POLAR_COMPLETION_WRITE_MAX_ATTEMPTS="${POLAR_COMPLETION_WRITE_MAX_ATTEMPTS:-3}"
export POLAR_COMPLETION_RETRY_BACKOFF_SECONDS="${POLAR_COMPLETION_RETRY_BACKOFF_SECONDS:-0.1}"
case "${POLAR_MULTI_GATEWAY:-0}" in
    1|true) export POLAR_MULTI_GATEWAY=1 ;;
    0|false) export POLAR_MULTI_GATEWAY=0 ;;
    *)
        echo "ERROR: POLAR_MULTI_GATEWAY must be 0/1/false/true" >&2
        exit 1
        ;;
esac
if [ "${POLAR_MULTI_GATEWAY}" = "1" ]; then
    export POLAR_GATEWAY_COUNT="${POLAR_GATEWAY_COUNT_OVERRIDE:-${RAY_NUM_NODES}}"
else
    export POLAR_GATEWAY_COUNT="${POLAR_GATEWAY_COUNT_OVERRIDE:-1}"
fi
if ! [[ "${POLAR_GATEWAY_COUNT}" =~ ^[1-9][0-9]*$ ]] || \
   [ "${POLAR_GATEWAY_COUNT}" -gt "${RAY_NUM_NODES}" ]; then
    echo "ERROR: gateway count must be in [1, RAY_NUM_NODES=${RAY_NUM_NODES}], got ${POLAR_GATEWAY_COUNT}" >&2
    exit 1
fi
if [ "${POLAR_MULTI_GATEWAY}" != "1" ] && [ "${POLAR_GATEWAY_COUNT}" -ne 1 ]; then
    echo "ERROR: POLAR_GATEWAY_COUNT_OVERRIDE>1 requires POLAR_MULTI_GATEWAY=1" >&2
    exit 1
fi

# The historical knobs describe the aggregate one-gateway capacity. Preserve
# that total when a gateway runs on every Slurm rank: multiplying these values
# per node recreates the SIF mount storm that multi-gateway is meant to remove.
polar_capacity_per_gateway() {
    local name="$1" total="$2" count="$3" value
    if ! [[ "${total}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer, got ${total}" >&2
        return 1
    fi
    if [ "${total}" -lt "${count}" ]; then
        echo "ERROR: ${name}=${total} cannot be split across ${count} gateways" >&2
        return 1
    fi
    value=$((total / count))
    printf '%s\n' "${value}"
}
export POLAR_GATEWAY_MAX_INIT_WORKERS="${POLAR_GATEWAY_MAX_INIT_WORKERS:-$(
    polar_capacity_per_gateway POLAR_MAX_INIT_WORKERS "${POLAR_MAX_INIT_WORKERS}" "${POLAR_GATEWAY_COUNT}"
)}"
export POLAR_GATEWAY_MAX_RUN_WORKERS="${POLAR_GATEWAY_MAX_RUN_WORKERS:-$(
    polar_capacity_per_gateway POLAR_MAX_RUN_WORKERS "${POLAR_MAX_RUN_WORKERS}" "${POLAR_GATEWAY_COUNT}"
)}"
export POLAR_GATEWAY_MAX_POSTRUN_WORKERS="${POLAR_GATEWAY_MAX_POSTRUN_WORKERS:-$(
    polar_capacity_per_gateway POLAR_MAX_POSTRUN_WORKERS "${POLAR_MAX_POSTRUN_WORKERS}" "${POLAR_GATEWAY_COUNT}"
)}"
export POLAR_GATEWAY_COMPLETION_QUEUE_SIZE="${POLAR_GATEWAY_COMPLETION_QUEUE_SIZE:-$(
    polar_capacity_per_gateway POLAR_COMPLETION_QUEUE_SIZE "${POLAR_COMPLETION_QUEUE_SIZE}" "${POLAR_GATEWAY_COUNT}"
)}"
export POLAR_GATEWAY_COMPLETION_WRITE_WORKERS="${POLAR_GATEWAY_COMPLETION_WRITE_WORKERS:-$(
    polar_capacity_per_gateway POLAR_COMPLETION_WRITE_WORKERS "${POLAR_COMPLETION_WRITE_WORKERS}" "${POLAR_GATEWAY_COUNT}"
)}"
unset -f polar_capacity_per_gateway
for _polar_capacity_name in \
    POLAR_GATEWAY_MAX_INIT_WORKERS \
    POLAR_GATEWAY_MAX_RUN_WORKERS \
    POLAR_GATEWAY_MAX_POSTRUN_WORKERS \
    POLAR_GATEWAY_COMPLETION_QUEUE_SIZE \
    POLAR_GATEWAY_COMPLETION_WRITE_WORKERS; do
    if ! [[ "${!_polar_capacity_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${_polar_capacity_name} must be a positive integer" >&2
        exit 1
    fi
done
unset _polar_capacity_name
for _polar_capacity_pair in \
    POLAR_MAX_INIT_WORKERS:POLAR_GATEWAY_MAX_INIT_WORKERS \
    POLAR_MAX_RUN_WORKERS:POLAR_GATEWAY_MAX_RUN_WORKERS \
    POLAR_MAX_POSTRUN_WORKERS:POLAR_GATEWAY_MAX_POSTRUN_WORKERS \
    POLAR_COMPLETION_QUEUE_SIZE:POLAR_GATEWAY_COMPLETION_QUEUE_SIZE \
    POLAR_COMPLETION_WRITE_WORKERS:POLAR_GATEWAY_COMPLETION_WRITE_WORKERS; do
    _polar_total_name="${_polar_capacity_pair%%:*}"
    _polar_per_gateway_name="${_polar_capacity_pair#*:}"
    _polar_effective_total=$((${!_polar_per_gateway_name} * POLAR_GATEWAY_COUNT))
    if [ "${_polar_effective_total}" -gt "${!_polar_total_name}" ]; then
        echo "ERROR: ${_polar_per_gateway_name} across ${POLAR_GATEWAY_COUNT} gateways exceeds aggregate ${_polar_total_name}=${!_polar_total_name}" >&2
        exit 1
    fi
done
unset _polar_capacity_pair _polar_total_name _polar_per_gateway_name \
    _polar_effective_total
if [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ]; then
    if [ "${SPILOT_EPISODE_ADMISSION_ENABLED:-false}" = "true" ]; then
        if [ "${SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT:-0}" -ne "${POLAR_GATEWAY_COUNT}" ]; then
            echo "ERROR: persisted SPilot admission gateway count ${SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT:-unset} does not match runtime gateway count ${POLAR_GATEWAY_COUNT}" >&2
            exit 1
        fi
        if [ $((SPILOT_QWEN_MAX_ACTIVE_EPISODES % POLAR_GATEWAY_COUNT)) -ne 0 ] || \
           [ $((SPILOT_GPT_MAX_ACTIVE_EPISODES % POLAR_GATEWAY_COUNT)) -ne 0 ]; then
            echo "ERROR: SPilot aggregate episode caps must be divisible by all ${POLAR_GATEWAY_COUNT} gateways" >&2
            exit 1
        fi
        _spilot_qwen_runtime_local="$((SPILOT_QWEN_MAX_ACTIVE_EPISODES / POLAR_GATEWAY_COUNT))"
        _spilot_gpt_runtime_local="$((SPILOT_GPT_MAX_ACTIVE_EPISODES / POLAR_GATEWAY_COUNT))"
        if [ "${SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES}" -ne "${_spilot_qwen_runtime_local}" ] || \
           [ "${SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES}" -ne "${_spilot_gpt_runtime_local}" ] || \
           [ "${SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY}" -ne "${_spilot_qwen_runtime_local}" ] || \
           [ "${SPILOT_GPT_GATEWAY_MAX_CONCURRENCY}" -ne "${_spilot_gpt_runtime_local}" ]; then
            echo "ERROR: rendered SPilot HTTP/episode caps do not match the runtime gateway split" >&2
            exit 1
        fi
        echo "Using SPilot episode admission: wait_budget=${SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS}s aggregate(qwen/gpt)=${SPILOT_QWEN_MAX_ACTIVE_EPISODES}/${SPILOT_GPT_MAX_ACTIVE_EPISODES} effective=${SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES}/${SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES} per_gateway=${SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES}/${SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES}"
        unset _spilot_qwen_runtime_local _spilot_gpt_runtime_local
    fi
fi
echo "Using Polar gateway fleet: count=${POLAR_GATEWAY_COUNT} per_gateway(init/run/post)=${POLAR_GATEWAY_MAX_INIT_WORKERS}/${POLAR_GATEWAY_MAX_RUN_WORKERS}/${POLAR_GATEWAY_MAX_POSTRUN_WORKERS} completion(queue/writers)=${POLAR_GATEWAY_COMPLETION_QUEUE_SIZE}/${POLAR_GATEWAY_COMPLETION_WRITE_WORKERS}"
POLAR_ROLLOUT_LOCAL_URL="${POLAR_ROLLOUT_LOCAL_URL:-http://127.0.0.1:${POLAR_ROLLOUT_PORT}}"
POLAR_GATEWAY_LOCAL_URL="${POLAR_GATEWAY_LOCAL_URL:-http://127.0.0.1:${POLAR_GATEWAY_PORT}}"
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
import re
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
    "POLAR_ROLLOUT_SAVE_DIR",
    "POLAR_GATEWAY_HOST",
    "POLAR_GATEWAY_PORT",
    "POLAR_GATEWAY_URL",
    "POLAR_MODEL_POOL_BASE_URL",
    "POLAR_SANDBOX_NETWORK",
    "POLAR_UDS_DIR",
    "POLAR_GATEWAY_UDS_DIR",
    "POLAR_PROXY_UDS_DIR",
    "POLAR_GATEWAY_UDS_SOCKET",
    "POLAR_PROXY_UDS_SOCKET",
    "POLAR_PROXY_UDS_ENABLED",
    "POLAR_UDS_TUNNEL_READY_FILE",
    "POLAR_UDS_TUNNEL_BACKLOG",
    "POLAR_SANDBOX_GATEWAY_UDS",
    "POLAR_SANDBOX_HTTP_PROXY_UDS",
    "POLAR_SANDBOX_HTTP_PROXY_PORT",
    "POLAR_APT_HTTP_SOURCE_POLICY",
    "POLAR_INTERNET_RUNTIME_VOLUME",
    "POLAR_CALLBACK_HOST",
    "POLAR_MAX_ASYNC_LEVEL",
    "POLAR_FULLY_ASYNC",
    "POLAR_REQUEST_TIMEOUT",
    "POLAR_TASK_TIMEOUT_SECONDS",
    "POLAR_TASK_TIMEOUT_FLOOR_SECONDS",
    "TMAX_TRAIN_AGENT_TIMEOUT_SECONDS",
    "POLAR_MIN_COMPLETE_ACCEPT_FRACTION",
    "POLAR_EARLY_STOP_GRACE_SESSIONS",
    "POLAR_CANDIDATE_POOL_HEALTH_GATE_ENABLED",
    "POLAR_CANDIDATE_POOL_HEALTH_MIN_OBSERVED_SESSIONS",
    "POLAR_CANDIDATE_POOL_HEALTH_MIN_COMPLETION_FRACTION",
    "TMAX_MAX_TOTAL_RESPONSE_LEN",
    "TMAX_TRAIN_PACK_LENGTH",
    "TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP",
    "SPILOT_QWEN_COST_WEIGHT",
    "SPILOT_GPT_COST_WEIGHT",
    "SPILOT_COST_PENALTY_LAMBDA",
    "SPILOT_COST_NORMALIZER",
    "SPILOT_LATENCY_PENALTY_LAMBDA",
    "SPILOT_LATENCY_NORMALIZER",
    "SPILOT_SLOT_LABEL_MODE",
    "SPILOT_ROUTING_MODE",
    "SPILOT_MAX_POOL_CALLS",
    "SPILOT_ROUTER_OBS_MAX_CHARS",
    "POLAR_MAX_INIT_WORKERS",
    "POLAR_MAX_RUN_WORKERS",
    "POLAR_MAX_POSTRUN_WORKERS",
    "POLAR_GATEWAY_MAX_INIT_WORKERS",
    "POLAR_GATEWAY_MAX_RUN_WORKERS",
    "POLAR_GATEWAY_MAX_POSTRUN_WORKERS",
    "POLAR_COMPLETION_QUEUE_SIZE",
    "POLAR_COMPLETION_WRITE_WORKERS",
    "POLAR_GATEWAY_COMPLETION_QUEUE_SIZE",
    "POLAR_GATEWAY_COMPLETION_WRITE_WORKERS",
    "SPILOT_EPISODE_ADMISSION_ENABLED",
    "SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS",
    "SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES",
    "SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES",
    "SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY",
    "SPILOT_GPT_GATEWAY_MAX_CONCURRENCY",
    "POLAR_COMPLETION_BATCH_SIZE",
    "POLAR_COMPLETION_WRITE_MAX_ATTEMPTS",
    "POLAR_COMPLETION_RETRY_BACKOFF_SECONDS",
    "POLAR_AGENT_HARNESS",
    "POLAR_AGENT_MODEL_NAME",
    "POLAR_AGENT_PATH",
    "POLAR_AGENT_RUNTIME_VOLUME",
    "POLAR_AGENT_STEP_LIMIT",
    "POLAR_AGENT_COST_LIMIT",
    "POLAR_AGENT_TEMPERATURE",
    "POLAR_AGENT_TOP_P",
    "POLAR_AGENT_MAX_TOKENS",
    "POLAR_AGENT_ENABLE_THINKING",
    "TMAX_EVAL_DATASET_NAME",
    "TMAX_EXTERNAL_EVAL_DATASET_NAME",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
):
    text = text.replace("${" + name + "}", os.environ.get(name, ""))
unresolved = sorted(set(re.findall(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", text)))
if unresolved:
    raise SystemExit(
        f"unresolved template variable(s) in {src}: {', '.join(unresolved)}"
    )
temporary = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
try:
    temporary.write_text(text)
    os.replace(temporary, dst)
finally:
    temporary.unlink(missing_ok=True)
PY
}
if [ "${RAY_NODE_RANK}" = "0" ]; then
    _polar_topology_prototype="${TOPOLOGY_PATH}.prototype.$$"
    render_template "$TOPOLOGY_TEMPLATE" "${_polar_topology_prototype}"
    _polar_topology_args=(
        --input "${_polar_topology_prototype}"
        --output "${TOPOLOGY_PATH}"
    )
    if [ "${POLAR_MULTI_GATEWAY}" = "1" ]; then
        if [ -z "${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}" ]; then
            echo "ERROR: multi-gateway topology requires SLURM_JOB_NODELIST or SLURM_NODELIST" >&2
            exit 1
        fi
        if ! _polar_gateway_host_output="$(slurm_allocation_hosts)" || \
           [ -z "${_polar_gateway_host_output}" ]; then
            echo "ERROR: failed to expand the Slurm allocation hostlist for multi-gateway topology" >&2
            exit 1
        fi
        mapfile -t _polar_gateway_hosts <<<"${_polar_gateway_host_output}"
        if [ "${#_polar_gateway_hosts[@]}" -lt "${POLAR_GATEWAY_COUNT}" ]; then
            echo "ERROR: Slurm hostlist expanded to ${#_polar_gateway_hosts[@]} hosts, fewer than ${POLAR_GATEWAY_COUNT} gateways" >&2
            exit 1
        fi
        _polar_gateway_hosts=("${_polar_gateway_hosts[@]:0:${POLAR_GATEWAY_COUNT}}")
        for _polar_gateway_host in "${_polar_gateway_hosts[@]}"; do
            _polar_topology_args+=(--gateway-host "${_polar_gateway_host}")
        done
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/render_gateway_topology.py" "${_polar_topology_args[@]}"
    rm -f "${_polar_topology_prototype}"
    unset _polar_topology_prototype _polar_topology_args \
        _polar_gateway_hosts _polar_gateway_host _polar_gateway_host_output
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
if [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ] && \
   [ "${SPILOT_EPISODE_ADMISSION_ENABLED:-false}" = "true" ]; then
    echo "[spilot admission run metric] fatal_job_count_total=${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT:-0}"
fi
echo "Using SGLang router URL for Polar gateway: ${SGLANG_ROUTER_BASE_URL}"
echo "Using Polar rollout URL: ${POLAR_ROLLOUT_URL}"
echo "Using Polar gateway URL: ${POLAR_GATEWAY_URL}"
echo "Using sandbox network: ${POLAR_SANDBOX_NETWORK}"
echo "Using sandbox gateway UDS: ${POLAR_GATEWAY_UDS_SOCKET}"
if [ "${POLAR_PROXY_UDS_ENABLED}" = "1" ]; then
    echo "Using sandbox proxy UDS: ${POLAR_PROXY_UDS_SOCKET} -> ${POLAR_PROXY_TCP_TARGET}"
fi
echo "Using no_proxy: ${no_proxy}"

# ── Cleanup on exit ────────────────────────────────────────────────
PIDS=()
PROCESS_GROUPS=()
POLAR_ROLLOUT_PID=""
POLAR_GATEWAY_PID=""
POLAR_UDS_TUNNEL_PID=""
POLAR_DISPATCHER_SHUTDOWN_TIMEOUT_SECONDS="${POLAR_DISPATCHER_SHUTDOWN_TIMEOUT_SECONDS:-120}"
if [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ] && \
   ! [[ "${POLAR_DISPATCHER_SHUTDOWN_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: formal SPilot requires POLAR_DISPATCHER_SHUTDOWN_TIMEOUT_SECONDS to be a positive integer" >&2
    exit 1
fi
export POLAR_DISPATCHER_SHUTDOWN_TIMEOUT_SECONDS
_POLAR_GATEWAY_MIN_SHUTDOWN_GRACE_SECONDS=210
if [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ]; then
    # The outer process bound must not kill the gateway while its fail-closed
    # runtime-containment proof is still within budget.
    _POLAR_GATEWAY_MIN_SHUTDOWN_GRACE_SECONDS=$((60 + POLAR_DISPATCHER_SHUTDOWN_TIMEOUT_SECONDS + 30))
fi
POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS="${POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS:-${_POLAR_GATEWAY_MIN_SHUTDOWN_GRACE_SECONDS}}"
if ! [[ "${POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS must be a non-negative integer" >&2
    exit 1
fi
if [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ] && \
   [ "${POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS}" -lt "${_POLAR_GATEWAY_MIN_SHUTDOWN_GRACE_SECONDS}" ]; then
    echo "ERROR: formal SPilot requires POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS>=${_POLAR_GATEWAY_MIN_SHUTDOWN_GRACE_SECONDS} (60s HTTP drain + ${POLAR_DISPATCHER_SHUTDOWN_TIMEOUT_SECONDS}s runtime proof + 30s margin)" >&2
    exit 1
fi
unset _POLAR_GATEWAY_MIN_SHUTDOWN_GRACE_SECONDS

polar_pid_is_active() {
    local pid="$1" proc_stat remainder state
    kill -0 "$pid" 2>/dev/null || return 1
    [ -r "/proc/${pid}/stat" ] || return 1
    proc_stat="$(<"/proc/${pid}/stat")"
    remainder="${proc_stat##*) }"
    state="${remainder%% *}"
    [ "$state" != "Z" ] && [ "$state" != "X" ]
}

polar_wait_for_pids_bounded() {
    local timeout_seconds="$1" deadline pid active
    shift
    [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || timeout_seconds=15
    deadline=$((SECONDS + timeout_seconds))
    while [ "$SECONDS" -lt "$deadline" ]; do
        active=0
        for pid in "$@"; do
            if polar_pid_is_active "$pid"; then
                active=1
                break
            fi
        done
        [ "$active" -eq 0 ] && return 0
        sleep 0.2
    done
    for pid in "$@"; do
        polar_pid_is_active "$pid" && return 1
    done
    return 0
}

polar_reap_stopped_pids() {
    local pid
    for pid in "$@"; do
        if ! polar_pid_is_active "$pid"; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}

polar_append_descendant_pids() {
    local parent_pid="$1" output_name="$2" child children
    local -n output_pids="$output_name"
    [ -r "/proc/${parent_pid}/task/${parent_pid}/children" ] || return 0
    children="$(<"/proc/${parent_pid}/task/${parent_pid}/children")" 2>/dev/null || return 0
    for child in $children; do
        output_pids+=("$child")
        polar_append_descendant_pids "$child" "$output_name"
    done
}

polar_terminate_pids_bounded() {
    local grace_seconds="$1" kill_grace_seconds pid
    shift
    [ "$#" -gt 0 ] || return 0
    [[ "$grace_seconds" =~ ^[0-9]+$ ]] || grace_seconds=15
    kill_grace_seconds="${POLAR_BACKGROUND_KILL_GRACE_SECONDS:-2}"
    [[ "$kill_grace_seconds" =~ ^[0-9]+$ ]] || kill_grace_seconds=2

    for pid in "$@"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    if ! polar_wait_for_pids_bounded "$grace_seconds" "$@"; then
        echo "Background shutdown exceeded ${grace_seconds}s; sending SIGKILL" >&2
        for pid in "$@"; do
            if polar_pid_is_active "$pid"; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        polar_wait_for_pids_bounded "$kill_grace_seconds" "$@" || true
    fi
    polar_reap_stopped_pids "$@"
    for pid in "$@"; do
        if polar_pid_is_active "$pid"; then
            echo "WARNING: background pid ${pid} remained alive after SIGKILL" >&2
        fi
    done
}

polar_process_group_is_active() {
    kill -0 -- "-$1" 2>/dev/null
}

polar_wait_for_process_groups_bounded() {
    local timeout_seconds="$1" deadline pgid active
    shift
    [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || timeout_seconds=15
    deadline=$((SECONDS + timeout_seconds))
    while [ "$SECONDS" -lt "$deadline" ]; do
        active=0
        for pgid in "$@"; do
            if polar_process_group_is_active "$pgid"; then
                active=1
                break
            fi
        done
        [ "$active" -eq 0 ] && return 0
        sleep 0.2
    done
    for pgid in "$@"; do
        polar_process_group_is_active "$pgid" && return 1
    done
    return 0
}

polar_terminate_process_groups_bounded() {
    local grace_seconds="$1" kill_grace_seconds pgid
    shift
    [ "$#" -gt 0 ] || return 0
    [[ "$grace_seconds" =~ ^[0-9]+$ ]] || grace_seconds=15
    kill_grace_seconds="${POLAR_BACKGROUND_KILL_GRACE_SECONDS:-2}"
    [[ "$kill_grace_seconds" =~ ^[0-9]+$ ]] || kill_grace_seconds=2

    for pgid in "$@"; do
        kill -TERM -- "-$pgid" 2>/dev/null || true
    done
    if ! polar_wait_for_process_groups_bounded "$grace_seconds" "$@"; then
        echo "Process-group shutdown exceeded ${grace_seconds}s; sending SIGKILL" >&2
        for pgid in "$@"; do
            if polar_process_group_is_active "$pgid"; then
                kill -KILL -- "-$pgid" 2>/dev/null || true
            fi
        done
        polar_wait_for_process_groups_bounded "$kill_grace_seconds" "$@" || true
    fi
    for pgid in "$@"; do
        if polar_process_group_is_active "$pgid"; then
            echo "WARNING: process group ${pgid} remained alive after SIGKILL" >&2
        fi
    done
}

polar_stop_ray_bounded() {
    local timeout_seconds ray_stop_pid
    timeout_seconds="${POLAR_RAY_STOP_TIMEOUT_SECONDS:-30}"
    [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || timeout_seconds=30
    ray stop --force 2>/dev/null &
    ray_stop_pid=$!
    if polar_wait_for_pids_bounded "$timeout_seconds" "$ray_stop_pid"; then
        polar_reap_stopped_pids "$ray_stop_pid"
        return
    fi
    echo "Ray shutdown exceeded ${timeout_seconds}s; terminating ray stop" >&2
    polar_terminate_pids_bounded 2 "$ray_stop_pid"
}

polar_shutdown_gateway_bounded() {
    local pid="$1" grace_seconds="$2" term_already_sent="${3:-0}"
    local kill_grace_seconds rc timed_out=0
    [ -n "${pid}" ] || return 0
    [[ "${grace_seconds}" =~ ^[0-9]+$ ]] || grace_seconds=210
    kill_grace_seconds="${POLAR_BACKGROUND_KILL_GRACE_SECONDS:-2}"
    [[ "${kill_grace_seconds}" =~ ^[0-9]+$ ]] || kill_grace_seconds=2

    # cleanup() starts gateway shutdown before the potentially-slow Ray stop
    # so both drains overlap. Do not deliver a second SIGTERM afterwards: it
    # can interrupt Uvicorn's already-running graceful shutdown and turn a
    # successful exit into status 143.
    if [ "${term_already_sent}" != 1 ]; then
        kill -TERM "${pid}" 2>/dev/null || true
    fi
    if ! polar_wait_for_pids_bounded "${grace_seconds}" "${pid}"; then
        timed_out=1
        echo "ERROR: Polar gateway shutdown exceeded ${grace_seconds}s; sending SIGKILL" >&2
        kill -KILL "${pid}" 2>/dev/null || true
        polar_wait_for_pids_bounded "${kill_grace_seconds}" "${pid}" || true
    fi
    if polar_pid_is_active "${pid}"; then
        echo "ERROR: Polar gateway pid ${pid} survived shutdown escalation" >&2
        return 1
    fi
    if wait "${pid}"; then
        rc=0
    else
        rc=$?
    fi
    if [ "${timed_out}" -ne 0 ]; then
        return 1
    fi
    if [ "${rc}" -ne 0 ]; then
        echo "ERROR: Polar gateway exited with status ${rc}; runtime teardown may be unproven" >&2
        return 1
    fi
    return 0
}

cleanup() {
    local status=$? final_status gateway_shutdown_failed=0
    local pid background_grace_seconds gateway_term_sent=0
    local -a background_pids=()
    local -a process_groups=("${PROCESS_GROUPS[@]}")
    trap - EXIT
    echo "Shutting down..."
    if [ "${RAY_NODE_RANK}" = "0" ]; then
        touch "$RUN_DONE_FILE" 2>/dev/null || true
    fi
    # W&B intentionally starts wandb-core in a new session, so it can escape
    # even the monitor's process group. Snapshot descendants before TERM can
    # reparent them, then include every recorded PID in the bounded teardown.
    for pid in "${PIDS[@]}"; do
        if [ -n "${POLAR_GATEWAY_PID}" ] && [ "${pid}" = "${POLAR_GATEWAY_PID}" ]; then
            continue
        fi
        background_pids+=("${pid}")
        polar_append_descendant_pids "$pid" background_pids
    done
    # Ask sidecars and Polar services to stop before Ray cleanup. This lets the
    # GPU monitor's bounded W&B finish run in parallel with ray stop.
    for pid in "${background_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${process_groups[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    if [ -n "${POLAR_GATEWAY_PID}" ]; then
        if kill -TERM "${POLAR_GATEWAY_PID}" 2>/dev/null; then
            gateway_term_sent=1
        fi
    fi
    polar_stop_ray_bounded
    if ! polar_shutdown_gateway_bounded \
        "${POLAR_GATEWAY_PID}" "${POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS}" \
        "${gateway_term_sent}"; then
        gateway_shutdown_failed=1
    fi
    background_grace_seconds="${POLAR_BACKGROUND_SHUTDOWN_GRACE_SECONDS:-20}"
    polar_terminate_pids_bounded "$background_grace_seconds" "${background_pids[@]}"
    # Groups received TERM before Ray shutdown and the per-PID grace above.
    # Escalate any remaining group member immediately instead of adding a
    # second full grace period to allocation teardown.
    polar_terminate_process_groups_bounded 0 "${process_groups[@]}"
    PIDS=()
    PROCESS_GROUPS=()
    final_status="${status}"
    if [ "${gateway_shutdown_failed}" -ne 0 ] && [ "${final_status}" -eq 0 ]; then
        # Only SPilot escalates an unprovable gateway teardown to a failed
        # allocation: router runs must not risk unaccounted provider spend
        # from leaked pool runtimes. Direct training keeps its outcome — a
        # completed run is not retroactively failed by teardown noise; the
        # ERROR lines above remain for operators.
        if [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ]; then
            final_status=70
        else
            echo "WARNING: gateway teardown was unprovable; keeping allocation exit status ${final_status} (non-SPilot harness)" >&2
        fi
    fi
    exit "${final_status}"
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

    local monitor_dir metric_prefix csv_path node_role train_gpus rollout_gpus monitor_pid
    local -a static_metric_args=()
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
            "--wandb-run-id" "$WANDB_RUN_ID"
            "--wandb-project" "${WANDB_PROJECT:-polar-swegym-grpo}"
            "--wandb-group" "${WANDB_GROUP:-swegym-qwen35-4b-async-grpo}"
            "--wandb-mode" "${GPU_MONITOR_WANDB_MODE:-shared}"
        )
        if [ -n "${WANDB_ENTITY:-}" ]; then
            wandb_args+=("--wandb-entity" "$WANDB_ENTITY")
        fi
    fi
    if [ "${RAY_NODE_RANK}" = "0" ] && \
       [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ] && \
       [ "${SPILOT_EPISODE_ADMISSION_ENABLED:-false}" = "true" ] && \
       [[ "${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT:-0}" =~ ^[0-9]+$ ]]; then
        static_metric_args=(
            --static-metric
            "polar/spilot_router/admission_fatal_job_count_total=${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT:-0}"
        )
    fi

    local -a monitor_command=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/monitor_wandb_gpu.py"
        --interval-s "${GPU_MONITOR_INTERVAL_S:-10}"
        --out-csv "$csv_path"
        --metric-prefix "$metric_prefix"
        --train-progress-file "$SLIME_TRAIN_PROGRESS_FILE"
        --wandb-finish-timeout-s "${GPU_MONITOR_WANDB_FINISH_TIMEOUT_S:-15}"
        --train-gpus "$train_gpus"
        --rollout-gpus "$rollout_gpus"
        "${static_metric_args[@]}"
        "${wandb_args[@]}"
    )
    if command -v setsid >/dev/null 2>&1; then
        setsid "${monitor_command[@]}" &
        monitor_pid=$!
        PROCESS_GROUPS+=("$monitor_pid")
    else
        "${monitor_command[@]}" &
        monitor_pid=$!
    fi
    PIDS+=("$monitor_pid")
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

wait_for_shared_marker() {
    local name="$1" path="$2" attempts="${3:-180}" i
    for i in $(seq 1 "${attempts}"); do
        if [ -f "${path}" ]; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: timed out waiting for ${name}: ${path}" >&2
    return 1
}

wait_uds_tunnel_ready() {
    local tunnel_pid="$1"
    local attempts="${2:-100}"
    local i rc
    for i in $(seq 1 "$attempts"); do
        if ! kill -0 "$tunnel_pid" 2>/dev/null; then
            if wait "$tunnel_pid"; then
                rc=0
            else
                rc=$?
            fi
            echo "ERROR: sandbox UDS tunnel exited before becoming ready (rc=${rc})" >&2
            return 1
        fi
        if [ -s "${POLAR_UDS_TUNNEL_READY_FILE}" ] && \
           [ -S "${POLAR_GATEWAY_UDS_SOCKET}" ] && \
           { [ "${POLAR_PROXY_UDS_ENABLED}" != "1" ] || [ -S "${POLAR_PROXY_UDS_SOCKET}" ]; }; then
            echo "Sandbox UDS tunnel ready: ${POLAR_UDS_TUNNEL_READY_FILE}"
            return 0
        fi
        sleep 0.1
    done
    echo "ERROR: sandbox UDS tunnel not ready after ${attempts} attempts" >&2
    return 1
}

start_sandbox_uds_tunnel() {
    mkdir -p "${POLAR_UDS_DIR}" "${POLAR_GATEWAY_UDS_DIR}" "${POLAR_PROXY_UDS_DIR}"
    chmod 700 "${POLAR_UDS_DIR}" "${POLAR_GATEWAY_UDS_DIR}" "${POLAR_PROXY_UDS_DIR}"
    rm -f "${POLAR_UDS_TUNNEL_READY_FILE}"

    local mappings=(
        "${POLAR_GATEWAY_UDS_SOCKET}=127.0.0.1:${POLAR_GATEWAY_PORT}"
    )
    if [ "${POLAR_PROXY_UDS_ENABLED}" = "1" ]; then
        mappings+=("${POLAR_PROXY_UDS_SOCKET}=${POLAR_PROXY_TCP_TARGET}")
    fi

    echo "=== Starting sandbox UDS tunnel (${#mappings[@]} endpoint(s)) ==="
    "${PYTHON_BIN}" -m polar.runtime.uds_tunnel \
        --ready-file "${POLAR_UDS_TUNNEL_READY_FILE}" \
        --socket-mode 0600 \
        --backlog "${POLAR_UDS_TUNNEL_BACKLOG}" \
        "${mappings[@]}" &
    POLAR_UDS_TUNNEL_PID=$!
    PIDS+=("${POLAR_UDS_TUNNEL_PID}")
    wait_uds_tunnel_ready "${POLAR_UDS_TUNNEL_PID}"
}

start_polar_gateway() {
    local node_id node_index
    if [ "${POLAR_MULTI_GATEWAY}" = "1" ]; then
        # SLURM_NODEID is the node's index in `scontrol show hostnames`, while
        # PROCID is a task index whose distribution policy can differ. There is
        # one task per node here, but use the authoritative node index anyway.
        node_index="${SLURM_NODEID:-${RAY_NODE_RANK}}"
        if ! [[ "${node_index}" =~ ^[0-9]+$ ]] || \
           [ "${node_index}" -ge "${POLAR_GATEWAY_COUNT}" ]; then
            echo "ERROR: invalid Slurm gateway node index ${node_index}" >&2
            return 1
        fi
        node_id="slurm-rank-${node_index}"
    else
        node_id="localhost-node-01"
    fi
    echo "=== Starting Polar gateway node_id=${node_id} host=$(hostname) ip=${RAY_NODE_IP} local=${POLAR_GATEWAY_LOCAL_URL} quotas=${POLAR_GATEWAY_MAX_INIT_WORKERS}/${POLAR_GATEWAY_MAX_RUN_WORKERS}/${POLAR_GATEWAY_MAX_POSTRUN_WORKERS} ==="
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" -m polar.cli serve_gateway \
        -c "${TOPOLOGY_PATH}" --node-id "${node_id}" &
    POLAR_GATEWAY_PID=$!
    PIDS+=("${POLAR_GATEWAY_PID}")
    wait_http_ok "Polar gateway ${node_id}" "${POLAR_GATEWAY_LOCAL_URL}/health" 60
    start_sandbox_uds_tunnel
}

wait_gateway_fleet_ready() {
    local expected="$1" ready_count=0 payload i
    echo "Waiting for ${expected} Polar gateway/UDS rank(s)..."
    for i in $(seq 1 180); do
        ready_count="$(
            find "${RAY_READY_DIR}" -maxdepth 1 -type f -name 'gateway_ready_rank_*' \
                | wc -l
        )"
        if [ "${ready_count}" -ge "${expected}" ]; then
            payload="$(curl --noproxy '*' -fsS --max-time 5 "${POLAR_ROLLOUT_LOCAL_URL}/nodes" 2>/dev/null || true)"
            if [ -n "${payload}" ] && "${PYTHON_BIN}" -c '
import json, sys
nodes = json.loads(sys.argv[1])
expected = int(sys.argv[2])
multi_gateway = sys.argv[3] == "1"
assert len(nodes) == expected
assert all(node.get("healthy") for node in nodes)
expected_ids = (
    {f"slurm-rank-{index}" for index in range(expected)}
    if multi_gateway
    else {"localhost-node-01"}
)
actual_ids = {str(node.get("node_id")) for node in nodes}
assert actual_ids == expected_ids
' "${payload}" "${expected}" "${POLAR_MULTI_GATEWAY}" >/dev/null 2>&1; then
                echo "Polar gateway fleet healthy: ${expected}/${expected}"
                return 0
            fi
        fi
        sleep 1
    done
    echo "ERROR: Polar gateway fleet not ready: markers=${ready_count}/${expected}" >&2
    curl --noproxy '*' -fsS --max-time 5 "${POLAR_ROLLOUT_LOCAL_URL}/nodes" >&2 || true
    return 1
}

wait_for_run_done_with_sidecars() {
    local pid
    while [ ! -f "${RUN_DONE_FILE}" ]; do
        if ! polar_check_spilot_admission_health; then
            return 1
        fi
        for pid in "${POLAR_GATEWAY_PID}" "${POLAR_UDS_TUNNEL_PID}"; do
            [ -n "${pid}" ] || continue
            if ! polar_pid_is_active "${pid}"; then
                wait "${pid}" 2>/dev/null || true
                echo "ERROR: rank ${RAY_NODE_RANK} Polar sidecar pid ${pid} exited early" >&2
                return 1
            fi
        done
        sleep 1
    done
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
ACTOR_TENSOR_MODEL_PARALLEL_SIZE="${ACTOR_TENSOR_MODEL_PARALLEL_SIZE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-9}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-30000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32000}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-0}"
if [ $(((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT) % NUM_STEPS_PER_ROLLOUT)) -ne 0 ]; then
    echo "ERROR: rollout batch product must be divisible by NUM_STEPS_PER_ROLLOUT" >&2
    exit 1
fi
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT) / NUM_STEPS_PER_ROLLOUT))}"
EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
CUSTOM_ROLLOUT_LOG_FUNCTION_PATH="${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH:-slime_bridge.rollout.log_rollout_trajectory_examples}"
echo "Using rollout/global batch: ${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT}/${NUM_STEPS_PER_ROLLOUT}=${GLOBAL_BATCH_SIZE}, eval=${EVAL_GLOBAL_BATCH_SIZE}"
TRAIN_LENGTH_ARGS=(--num-epoch "${NUM_EPOCH:-1}")
if [ -n "${TMAX_NUM_ROLLOUT:-}" ]; then
    if ! [[ "${TMAX_NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: TMAX_NUM_ROLLOUT must be a positive integer" >&2
        exit 1
    fi
    TRAIN_LENGTH_ARGS=(--num-rollout "${TMAX_NUM_ROLLOUT}")
    echo "Using explicit rollout-loop boundary: ${TMAX_NUM_ROLLOUT} (final rollout $((TMAX_NUM_ROLLOUT - 1)))"
fi
OPT_PARAM_SCHEDULER_ARGS=()
case "${TMAX_OVERRIDE_OPT_PARAM_SCHEDULER:-0}" in
    0|false) ;;
    1|true)
        OPT_PARAM_SCHEDULER_ARGS=(--override-opt-param-scheduler)
        echo "Using runtime optimizer scheduler configuration while preserving checkpoint progress"
        ;;
    *)
        echo "ERROR: TMAX_OVERRIDE_OPT_PARAM_SCHEDULER must be 0/1/false/true" >&2
        exit 1
        ;;
esac
echo "Using post-train rollout log hook: ${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH}"
TRAIN_DYNAMIC_SAMPLING_ARGS=()
if [ -n "${TMAX_DYNAMIC_SAMPLING_FILTER_PATH:-}" ]; then
    # This option is consumed only by the custom training rollout path. Eval
    # enters the bridge's one-shot evaluation branch before the filter is loaded.
    TRAIN_DYNAMIC_SAMPLING_ARGS=(
        --dynamic-sampling-filter-path
        "${TMAX_DYNAMIC_SAMPLING_FILTER_PATH}"
    )
    echo "Using training dynamic sampling filter: ${TMAX_DYNAMIC_SAMPLING_FILTER_PATH}"
fi
SEQUENCE_PARALLEL_ARGS=()
if [ "${SEQUENCE_PARALLEL}" = "1" ] || [ "${SEQUENCE_PARALLEL}" = "true" ]; then
    SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
fi
echo "Using sequence parallel: ${SEQUENCE_PARALLEL}"
WANDB_STEP_ARGS=()
if [ "${WANDB_ALWAYS_USE_TRAIN_STEP:-0}" = "1" ] || \
    [ "${WANDB_ALWAYS_USE_TRAIN_STEP:-0}" = "true" ]; then
    WANDB_STEP_ARGS=(--wandb-always-use-train-step)
    echo "Using W&B metric axes: business=train/step eval=eval/train_step GPU=per-node/train_step"
else
    echo "Using W&B metric axis: rollout/step"
fi
LOGPROB_GUARD_ARGS=()
if [ -n "${MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF:-}" ]; then
    if ! "${PYTHON_BIN}" -c \
        'import math,sys; value=float(sys.argv[1]); assert math.isfinite(value) and value > 0' \
        "${MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF}"; then
        echo "ERROR: MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF must be a finite number greater than zero" >&2
        exit 1
    fi
    LOGPROB_GUARD_ARGS=(
        --max-train-rollout-logprob-abs-diff
        "${MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF}"
    )
    echo "Using trainer/rollout log-probability fail-fast threshold: ${MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF}"
fi
POLICY_LOSS_TYPE="${POLICY_LOSS_TYPE:-ppo}"
POLICY_LOSS_ARGS=(--policy-loss-type "${POLICY_LOSS_TYPE}")
case "${POLICY_LOSS_TYPE}" in
    ppo)
        if [ "${USE_TIS:-1}" = "1" ] || [ "${USE_TIS:-1}" = "true" ]; then
            POLICY_LOSS_ARGS+=(--use-tis)
        fi
        ;;
    dppo)
        POLICY_LOSS_ARGS+=(
            --use-rollout-logprobs
            --dppo-divergence-type "${DPPO_DIVERGENCE_TYPE:-tv}"
            --dppo-divergence-threshold "${DPPO_DIVERGENCE_THRESHOLD:-0.1}"
        )
        ;;
    *)
        echo "ERROR: POLICY_LOSS_TYPE must be ppo or dppo, got ${POLICY_LOSS_TYPE}" >&2
        exit 1
        ;;
esac
GRPO_NORMALIZATION_ARGS=()
case "${GRPO_STD_NORMALIZATION:-1}" in
    0|false)
        GRPO_NORMALIZATION_ARGS=(--disable-grpo-std-normalization)
        ;;
    1|true)
        ;;
    *)
        echo "ERROR: GRPO_STD_NORMALIZATION must be 0/1/false/true, got ${GRPO_STD_NORMALIZATION}" >&2
        exit 1
        ;;
esac
OPTIMIZER_MEMORY_ARGS=()
case "${TMAX_OPTIMIZER_CPU_OFFLOAD:-0}" in
    0) ;;
    1)
        OPTIMIZER_MEMORY_ARGS=(
            --optimizer-cpu-offload
            --optimizer-offload-fraction 1.0
            --overlap-cpu-optimizer-d2h-h2d
            --use-precision-aware-optimizer
        )
        echo "Using full-precision CPU-offloaded optimizer state"
        ;;
    *)
        echo "ERROR: TMAX_OPTIMIZER_CPU_OFFLOAD must be 0 or 1" >&2
        exit 1
        ;;
esac
KL_LOSS_ARGS=()
if ! "${PYTHON_BIN}" -c \
    'import math,sys; value=float(sys.argv[1]); assert math.isfinite(value) and value >= 0' \
    "${KL_LOSS_COEF:-0.001}"; then
    echo "ERROR: KL_LOSS_COEF must be a finite non-negative number" >&2
    exit 1
fi
if "${PYTHON_BIN}" -c 'import sys; raise SystemExit(float(sys.argv[1]) == 0)' \
    "${KL_LOSS_COEF:-0.001}"; then
    KL_LOSS_ARGS=(
        --use-kl-loss
        --kl-loss-coef "${KL_LOSS_COEF:-0.001}"
        --kl-loss-type low_var_kl
    )
fi
SGLANG_REASONING_ARGS=()
if [ -n "${SGLANG_REASONING_PARSER:-}" ]; then
    SGLANG_REASONING_ARGS=(--sglang-reasoning-parser "${SGLANG_REASONING_PARSER}")
fi
SGLANG_FP32_LM_HEAD_ARGS=()
case "${SGLANG_ENABLE_FP32_LM_HEAD:-0}" in
    1|true)
        SGLANG_FP32_LM_HEAD_ARGS=(--sglang-enable-fp32-lm-head)
        ;;
    0|false)
        ;;
    *)
        echo "ERROR: SGLANG_ENABLE_FP32_LM_HEAD must be 0/1/false/true, got ${SGLANG_ENABLE_FP32_LM_HEAD}" >&2
        exit 1
        ;;
esac
configure_sglang_inference_args() {
    SGLANG_DETERMINISTIC_ARGS=()
    SGLANG_ATTENTION_BACKEND_ARGS=()
    case "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-0}" in
        1|true)
            SGLANG_DETERMINISTIC_ARGS=(--sglang-enable-deterministic-inference)
            echo "Using SGLang deterministic inference"
            ;;
        0|false) ;;
        *)
            echo "ERROR: SGLANG_ENABLE_DETERMINISTIC_INFERENCE must be 0/1/false/true, got ${SGLANG_ENABLE_DETERMINISTIC_INFERENCE}" >&2
            return 1
            ;;
    esac
    if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then
        SGLANG_ATTENTION_BACKEND_ARGS=(
            --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
        )
        echo "Using explicit SGLang attention backend: ${SGLANG_ATTENTION_BACKEND}"
    fi
}
configure_sglang_inference_args || exit 1
TRAINER_FP32_LM_HEAD_ARGS=()
case "${TMAX_ENABLE_FP32_LM_HEAD:-0}" in
    1|true)
        TRAINER_FP32_LM_HEAD_ARGS=(--enable-fp32-lm-head)
        echo "Using true FP32 actor LM head in the Megatron trainer"
        ;;
    0|false) ;;
    *)
        echo "ERROR: TMAX_ENABLE_FP32_LM_HEAD must be 0/1/false/true" >&2
        exit 1
        ;;
esac
PER_TOKEN_LOSS_ARGS=()
case "${CALCULATE_PER_TOKEN_LOSS:-0}" in
    1|true)
        PER_TOKEN_LOSS_ARGS=(--calculate-per-token-loss)
        ;;
    0|false) ;;
    *)
        echo "ERROR: CALCULATE_PER_TOKEN_LOSS must be 0/1/false/true" >&2
        exit 1
        ;;
esac
PRETRAIN_EVAL_ARGS=()
case "${TMAX_CONCURRENT_PRETRAIN_EVAL:-1}" in
    1|true)
        if [ "${TMAX_TRAIN_MODE}" = "colocate" ]; then
            echo "ERROR: TMAX_TRAIN_MODE=colocate requires TMAX_CONCURRENT_PRETRAIN_EVAL=0" >&2
            exit 1
        fi
        PRETRAIN_EVAL_ARGS=(--concurrent-pretrain-eval)
        PRETRAIN_EVAL_MODE=concurrent-with-rollout-0
        ;;
    0|false)
        PRETRAIN_EVAL_MODE=synchronous-before-rollout-0
        ;;
    *)
        echo "ERROR: TMAX_CONCURRENT_PRETRAIN_EVAL must be 0/1/false/true" >&2
        exit 1
        ;;
esac
if [ "${#RESUMED_CHECKPOINT_EVAL_ARGS[@]}" -gt 0 ]; then
    PRETRAIN_EVAL_ARGS+=("${RESUMED_CHECKPOINT_EVAL_ARGS[@]}")
    PRETRAIN_EVAL_MODE=synchronous-resumed-checkpoint-before-first-rollout
fi
EVAL_ARGS=()
if [ "${TMAX_TRAINING_EVAL_ENABLED:-${TMAX_EVAL_ENABLED:-0}}" = "1" ]; then
    if [ ! -s "${TMAX_EVAL_DATA:-}" ]; then
        echo "ERROR: fixed TMax eval data is missing or empty: ${TMAX_EVAL_DATA:-<unset>}" >&2
        exit 1
    fi
    CONFIGURED_EVAL_DATA_SHA256="${SLIME_EVAL_DATA_SHA256:-${TMAX_EVAL_DATA_SHA256:-}}"
    PRIMARY_EVAL_DATA_SHA256="$(
        "${PYTHON_BIN}" -c \
            'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
            "${TMAX_EVAL_DATA}"
    )"
    if [ -n "${CONFIGURED_EVAL_DATA_SHA256}" ] && \
       [ "${CONFIGURED_EVAL_DATA_SHA256}" != "${PRIMARY_EVAL_DATA_SHA256}" ]; then
        echo "ERROR: configured fixed eval SHA-256 does not match ${TMAX_EVAL_DATA}" >&2
        exit 1
    fi
    if ! [[ "${PRIMARY_EVAL_DATA_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: fixed eval SHA-256 is invalid: ${PRIMARY_EVAL_DATA_SHA256:-<unset>}" >&2
        exit 1
    fi
    FINAL_EVAL_DATA_SHA256="${PRIMARY_EVAL_DATA_SHA256}"
    if [ "${TMAX_EXTERNAL_EVAL_ENABLED:-0}" = "1" ]; then
        if [ ! -s "${TMAX_EXTERNAL_EVAL_DATA:-}" ]; then
            echo "ERROR: fixed external eval data is missing or empty: ${TMAX_EXTERNAL_EVAL_DATA:-<unset>}" >&2
            exit 1
        fi
        EXTERNAL_EVAL_DATA_SHA256="$(
            "${PYTHON_BIN}" -c \
                'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
                "${TMAX_EXTERNAL_EVAL_DATA}"
        )"
        if [ -n "${TMAX_EXTERNAL_EVAL_DATA_SHA256:-}" ] && \
           [ "${TMAX_EXTERNAL_EVAL_DATA_SHA256}" != "${EXTERNAL_EVAL_DATA_SHA256}" ]; then
            echo "ERROR: configured external eval SHA-256 does not match ${TMAX_EXTERNAL_EVAL_DATA}" >&2
            exit 1
        fi
        FINAL_EVAL_DATA_SHA256="$(
            "${PYTHON_BIN}" - \
                "${TMAX_EVAL_DATASET_NAME}" "${TMAX_EVAL_DATA}" \
                "${TMAX_EXTERNAL_EVAL_DATASET_NAME}" "${TMAX_EXTERNAL_EVAL_DATA}" <<'PY'
import hashlib
import json
import pathlib
import sys

records = []
for name, path in zip(sys.argv[1::2], sys.argv[2::2], strict=True):
    records.append({"name": name, "sha256": hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()})
payload = json.dumps(sorted(records, key=lambda item: item["name"]), sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(payload).hexdigest())
PY
        )"
        if [ -n "${TMAX_EVAL_BUNDLE_SHA256:-}" ] && \
           [ "${TMAX_EVAL_BUNDLE_SHA256}" != "${FINAL_EVAL_DATA_SHA256}" ]; then
            echo "ERROR: configured eval bundle SHA-256 does not match current eval files" >&2
            exit 1
        fi
        TMAX_EVAL_CONFIG_PATH="${TMAX_EVAL_CONFIG_PATH:-${RUN_DIR}/tmax-eval-config.json}"
        if [ "${RAY_NODE_RANK}" = "0" ]; then
            "${PYTHON_BIN}" \
                "${POLAR_TRAIN_PROJECT_ROOT:-${PROJECT_ROOT}}/examples/tmax_slime_grpo/build_eval_config.py" \
                --output "${TMAX_EVAL_CONFIG_PATH}" \
                --primary-name "${TMAX_EVAL_DATASET_NAME}" \
                --primary-path "${TMAX_EVAL_DATA}" \
                --primary-samples "${TMAX_EVAL_SAMPLES_PER_PROMPT}" \
                --primary-minimum "${TMAX_EVAL_MIN_VALID_SAMPLES}" \
                --primary-temperature "${TMAX_EVAL_TEMPERATURE:-0.2}" \
                --primary-top-p "${TMAX_EVAL_TOP_P:-1.0}" \
                --primary-max-response-len "${TMAX_EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}" \
                --external-name "${TMAX_EXTERNAL_EVAL_DATASET_NAME}" \
                --external-path "${TMAX_EXTERNAL_EVAL_DATA}" \
                --external-samples "${TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT}" \
                --external-minimum "${TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES}" \
                --external-temperature "${TMAX_EXTERNAL_EVAL_TEMPERATURE:-0.7}" \
                --external-top-p "${TMAX_EXTERNAL_EVAL_TOP_P:-0.95}" \
                --external-max-response-len "${TMAX_EXTERNAL_EVAL_MAX_RESPONSE_LEN:-16384}" \
                --external-agent-step-limit "${TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT:-64}" \
                --eval-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}" \
                --sglang-context-length "${SGLANG_CONTEXT_LENGTH}" \
                --model-max-context-length "${TMAX_MODEL_MAX_CONTEXT_LENGTH:-262144}"
        else
            for _ in $(seq 1 120); do
                [ -s "${TMAX_EVAL_CONFIG_PATH}" ] && break
                sleep 1
            done
            if [ ! -s "${TMAX_EVAL_CONFIG_PATH}" ]; then
                echo "ERROR: timed out waiting for multi-eval config ${TMAX_EVAL_CONFIG_PATH}" >&2
                exit 1
            fi
        fi
        EVAL_ARGS=(
            --eval-interval "${TMAX_EVAL_INTERVAL}"
            "${PRETRAIN_EVAL_ARGS[@]}"
            --eval-config "${TMAX_EVAL_CONFIG_PATH}"
            --eval-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
            --custom-eval-rollout-log-function-path slime_bridge.eval_logging.add_weighted_eval_metric
        )
        echo "Using fixed evals: ${TMAX_EVAL_DATASET_NAME} (${TMAX_EVAL_DATA}) + ${TMAX_EXTERNAL_EVAL_DATASET_NAME} (${TMAX_EXTERNAL_EVAL_DATA}), baseline + every ${TMAX_EVAL_INTERVAL} rollout(s) + final"
    else
        EVAL_ARGS=(
            --eval-interval "${TMAX_EVAL_INTERVAL}"
            "${PRETRAIN_EVAL_ARGS[@]}"
            --eval-prompt-data "${TMAX_EVAL_DATASET_NAME}" "${TMAX_EVAL_DATA}"
            --n-samples-per-eval-prompt "${TMAX_EVAL_SAMPLES_PER_PROMPT}"
            --min-eval-samples "${TMAX_EVAL_MIN_VALID_SAMPLES}"
            --eval-temperature "${TMAX_EVAL_TEMPERATURE:-0.2}"
            --eval-top-p "${TMAX_EVAL_TOP_P:-1.0}"
            --eval-max-response-len "${TMAX_EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}"
            --eval-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
        )
        echo "Using fixed eval: ${TMAX_EVAL_DATASET_NAME} (${TMAX_EVAL_DATA}, baseline + every ${TMAX_EVAL_INTERVAL} rollout(s) + final)"
    fi
    echo "Pretrain eval scheduling: ${PRETRAIN_EVAL_MODE}"
    if [ -n "${FINAL_EVAL_COMPLETE_MARKER:-}" ] && \
       [ "${TMAX_TRAIN_MODE}" = "fully_async" ]; then
        EVAL_ARGS+=(
            --final-eval-complete-marker "${FINAL_EVAL_COMPLETE_MARKER}"
            --final-eval-data-sha256 "${FINAL_EVAL_DATA_SHA256}"
        )
    elif [ -n "${FINAL_EVAL_COMPLETE_MARKER:-}" ]; then
        echo "Sync colocate mode will run eval without an async final-eval marker."
    fi
else
    echo "Training-time eval disabled; holdout remains available for data-split and integrity checks."
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
RAY_READY_DIR="${RUN_DIR}/startup/ray-${SLURM_JOB_ID:-manual}"
if [ "${RAY_NODE_RANK}" = "0" ]; then
    rm -rf "${RAY_READY_DIR}"
    mkdir -p "${RAY_READY_DIR}"
    touch "${RAY_READY_DIR}/initialized"
else
    for _ in $(seq 1 300); do
        [ -f "${RAY_READY_DIR}/initialized" ] && break
        sleep 0.1
    done
    if [ ! -f "${RAY_READY_DIR}/initialized" ]; then
        echo "ERROR: timed out waiting for Ray readiness directory" >&2
        exit 1
    fi
fi
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
    # Workers must not join until the newly-created head is listening. Without
    # this generation barrier a restarted allocation can briefly join a stale
    # head, publish rank_N, and then disappear when rank 0 kills that head.
    touch "${RAY_READY_DIR}/head_ready"
    touch "${RAY_READY_DIR}/rank_${RAY_NODE_RANK}"
else
    for _ in $(seq 1 300); do
        [ -f "${RAY_READY_DIR}/head_ready" ] && break
        sleep 0.1
    done
    if [ ! -f "${RAY_READY_DIR}/head_ready" ]; then
        echo "ERROR: timed out waiting for the new Ray head" >&2
        exit 1
    fi
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
    touch "${RAY_READY_DIR}/rank_${RAY_NODE_RANK}"
fi

start_gpu_monitor
if [ "${RAY_NODE_RANK}" = "0" ]; then
    if [ "${RAY_NUM_NODES}" -gt 1 ]; then
        echo "Waiting for ${RAY_NUM_NODES} Ray ranks to report ready..."
        ray_ready_count=0
        for _ in $(seq 1 180); do
            ray_ready_count="$(find "${RAY_READY_DIR}" -maxdepth 1 -type f -name 'rank_*' | wc -l)"
            [ "${ray_ready_count}" -ge "${RAY_NUM_NODES}" ] && break
            sleep 1
        done
        if [ "${ray_ready_count}" -lt "${RAY_NUM_NODES}" ]; then
            echo "ERROR: only ${ray_ready_count}/${RAY_NUM_NODES} Ray ranks became ready" >&2
            exit 1
        fi
    fi
    ray status || true
    wait_ray_dashboard
    "${PYTHON_BIN}" - "${RAY_NUM_NODES}" "$((RAY_NUM_NODES * RAY_NUM_GPUS_PER_NODE))" <<'PY'
import sys
import time

import ray

expected_nodes = int(sys.argv[1])
expected_gpus = int(sys.argv[2])
ray.init(address="auto", logging_level="ERROR")
deadline = time.monotonic() + 180
while True:
    live_nodes = sum(node.get("Alive", False) for node in ray.nodes())
    registered_gpus = int(ray.cluster_resources().get("GPU", 0))
    if live_nodes >= expected_nodes and registered_gpus >= expected_gpus:
        break
    if time.monotonic() >= deadline:
        raise SystemExit(
            "Ray cluster resource barrier timed out: "
            f"nodes={live_nodes}/{expected_nodes}, "
            f"GPUs={registered_gpus}/{expected_gpus}"
        )
    time.sleep(1)
ray.shutdown()
PY
    export SLIME_RAY_READY_UNIX_NS="$(date +%s%N)"

    # Keep one authoritative rollout server. Gateway processes on every rank
    # register with it and retain their node-local sandbox/UDS lifecycle.
    export SLIME_ROLLOUT_SERVICE_START_UNIX_NS="$(date +%s%N)"
    echo "=== Starting Polar rollout server (${POLAR_ROLLOUT_URL}) ==="
    polar serve_rollout -c "${TOPOLOGY_PATH}" &
    POLAR_ROLLOUT_PID=$!
    PIDS+=("${POLAR_ROLLOUT_PID}")
    wait_http_ok "Polar rollout server" "${POLAR_ROLLOUT_LOCAL_URL}/health" 60
    export SLIME_ROLLOUT_SERVICE_READY_UNIX_NS="$(date +%s%N)"
    touch "${RAY_READY_DIR}/polar_rollout_ready"
else
    wait_for_shared_marker \
        "rank 0 Polar rollout readiness" \
        "${RAY_READY_DIR}/polar_rollout_ready" \
        180
    wait_http_ok "remote Polar rollout server" "${POLAR_ROLLOUT_URL}/health" 60
fi

# ── Step 2: Polar gateway fleet (node-local CPU + UDS) ─────────────
_polar_gateway_rank=0
if [ "${RAY_NODE_RANK}" = "0" ] || { \
   [ "${POLAR_MULTI_GATEWAY}" = "1" ] && \
   [ "${RAY_NODE_RANK}" -lt "${POLAR_GATEWAY_COUNT}" ];
}; then
    _polar_gateway_rank=1
    if [ "${RAY_NODE_RANK}" = "0" ]; then
        export SLIME_GATEWAY_START_UNIX_NS="$(date +%s%N)"
        export SLIME_UDS_TUNNEL_START_UNIX_NS="${SLIME_GATEWAY_START_UNIX_NS}"
    fi
    start_polar_gateway
    touch "${RAY_READY_DIR}/gateway_ready_rank_${RAY_NODE_RANK}"
fi

if [ "${RAY_NODE_RANK}" = "0" ]; then
    wait_gateway_fleet_ready "${POLAR_GATEWAY_COUNT}"
    export SLIME_GATEWAY_READY_UNIX_NS="$(date +%s%N)"
    export SLIME_UDS_TUNNEL_READY_UNIX_NS="${SLIME_GATEWAY_READY_UNIX_NS}"
    export SLIME_SERVICES_READY_UNIX_NS="${SLIME_GATEWAY_READY_UNIX_NS}"
else
    if [ "${POLAR_MULTI_GATEWAY}" = "1" ] && \
       [ "${_polar_gateway_rank}" = "1" ]; then
        wait_for_run_done_with_sidecars
    else
        while [ ! -f "${RUN_DONE_FILE}" ]; do
            sleep 30
        done
    fi
    exit 0
fi
unset _polar_gateway_rank

# ── Step 3: Slime (manages SGLang engines + training) ──────────────

# cuDNN lib path — probe the active venv instead of hardcoding python3.13.
if [ -z "${CUDNN_LIB:-}" ]; then
    CUDNN_LIB="$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
fi
RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "${CUDNN_LIB}" ] && [ -d "$CUDNN_LIB" ]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
fi
export SLIME_RAY_JOB_SUBMIT_UNIX_NS="$(date +%s%N)"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MASTER_ADDR\": \"${RAY_HEAD_IP}\",
    \"SLIME_SUBMIT_UNIX_NS\": \"${SLIME_SUBMIT_UNIX_NS:-}\",
    \"SLIME_SLURM_BATCH_START_UNIX_NS\": \"${SLIME_SLURM_BATCH_START_UNIX_NS:-}\",
    \"SLIME_CONTAINER_ENTRY_UNIX_NS\": \"${SLIME_CONTAINER_ENTRY_UNIX_NS:-}\",
    \"SLIME_JOB_SCRIPT_START_UNIX_NS\": \"${SLIME_JOB_SCRIPT_START_UNIX_NS}\",
    \"SLIME_RAY_READY_UNIX_NS\": \"${SLIME_RAY_READY_UNIX_NS}\",
    \"SLIME_ROLLOUT_SERVICE_START_UNIX_NS\": \"${SLIME_ROLLOUT_SERVICE_START_UNIX_NS}\",
    \"SLIME_ROLLOUT_SERVICE_READY_UNIX_NS\": \"${SLIME_ROLLOUT_SERVICE_READY_UNIX_NS}\",
    \"SLIME_GATEWAY_START_UNIX_NS\": \"${SLIME_GATEWAY_START_UNIX_NS}\",
    \"SLIME_GATEWAY_READY_UNIX_NS\": \"${SLIME_GATEWAY_READY_UNIX_NS}\",
    \"SLIME_UDS_TUNNEL_START_UNIX_NS\": \"${SLIME_UDS_TUNNEL_START_UNIX_NS}\",
    \"SLIME_UDS_TUNNEL_READY_UNIX_NS\": \"${SLIME_UDS_TUNNEL_READY_UNIX_NS}\",
    \"SLIME_SERVICES_READY_UNIX_NS\": \"${SLIME_SERVICES_READY_UNIX_NS}\",
    \"SLIME_RAY_JOB_SUBMIT_UNIX_NS\": \"${SLIME_RAY_JOB_SUBMIT_UNIX_NS}\",
    \"SLIME_TRAIN_PROGRESS_FILE\": \"${SLIME_TRAIN_PROGRESS_FILE}\",
    \"GPU_MONITOR_PREFIX\": \"${GPU_MONITOR_PREFIX:-polar_system}\",
    \"GPU_MONITOR_NODE_ROLE\": \"${GPU_MONITOR_NODE_ROLE:-}\",
    \"SLURM_NNODES\": \"${RAY_NUM_NODES}\",
    \"POLAR_EVAL_DATA_INTEGRITY_B64\": \"${POLAR_EVAL_DATA_INTEGRITY_B64:-}\",
    \"POLAR_ROLLOUT_SAVE_DIR\": \"${POLAR_ROLLOUT_SAVE_DIR}\",
    \"POLAR_ROLLOUT_EXAMPLES_DIR\": \"${POLAR_ROLLOUT_EXAMPLES_DIR}\",
    \"POLAR_ROLLOUT_EXAMPLE_INTERVAL\": \"${POLAR_ROLLOUT_EXAMPLE_INTERVAL}\",
    \"POLAR_ROLLOUT_EXAMPLE_COUNT\": \"${POLAR_ROLLOUT_EXAMPLE_COUNT}\",
    \"POLAR_ROLLOUT_EXAMPLES_WANDB\": \"${POLAR_ROLLOUT_EXAMPLES_WANDB}\",
    \"no_proxy\": \"${no_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-offline}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT:-polar-swegym-grpo}\",
    \"WANDB_GROUP\": \"${WANDB_GROUP:-swegym-qwen35-4b-async-grpo}\",
    \"WANDB_RUN_ID\": \"${WANDB_RUN_ID}\",
    \"WANDB_RESUME\": \"${WANDB_RESUME:-allow}\",
    \"HF_TOKEN\": \"${HF_TOKEN:-}\",
    \"HUGGINGFACE_HUB_TOKEN\": \"${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}\",
    \"WANDB_DIR\": \"${WANDB_DIR}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"PYTORCH_ALLOC_CONF\": \"${TMAX_PYTORCH_ALLOC_CONF}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${TMAX_PYTORCH_ALLOC_CONF}\",
    \"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD\": \"${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-}\",
    \"SLIME_PROFILE_CUDA_PHASES\": \"${SLIME_PROFILE_CUDA_PHASES:-0}\",
    \"SLIME_ROLLOUT_BASE_PORT\": \"${SLIME_ROLLOUT_BASE_PORT:-2048}\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"2\"
  }
}"

# Each experiment supplies rollout groups, samples/group, and optimizer steps.
# The TMax wrapper uses 8 × 32 trajectories which form one GBS=256 step;
# standalone SWE-Gym keeps the defaults above. With --dynamic-history each
# trajectory can expand into multiple training samples, so --balance-data below
# distributes packed microbatches by estimated work across actor DP ranks.
RAY_JOB_ADDRESS="http://${RAY_HEAD_IP}:8265"
RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-polar-${SLURM_JOB_ID:-$$}}"
TRAIN_PROGRESS_ENV_JSON="$("${PYTHON_BIN}" -c 'import json, os; print(json.dumps({"SLIME_TRAIN_PROGRESS_FILE": os.environ["SLIME_TRAIN_PROGRESS_FILE"]}))')"
TRAINING_LIFECYCLE_ARGS=()
if [ "${TMAX_TRAIN_MODE}" = "fully_async" ] && \
   [ "${TMAX_PROFILE_DISABLE_CHECKPOINT}" = "0" ]; then
    if [ -n "${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME:-}" ]; then
        TRAINING_LIFECYCLE_ARGS+=(--graceful-exit-at-unix-time "${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME}")
    fi
    if [ -n "${TRAINING_COMPLETE_MARKER:-}" ]; then
        TRAINING_LIFECYCLE_ARGS+=(--training-complete-marker "${TRAINING_COMPLETE_MARKER}")
    fi
elif [ "${TMAX_PROFILE_DISABLE_CHECKPOINT}" = "1" ]; then
    echo "WARNING: profiling model checkpoint arguments and async lifecycle markers are disabled"
else
    echo "Sync colocate mode does not emit async lifecycle markers; do not use the checkpoint watcher for this run."
fi
SAVE_RETENTION_ARGS=()
if [ -n "${SAVE_RETAIN_INTERVAL:-}" ]; then
    SAVE_RETENTION_ARGS+=(--save-retain-interval "${SAVE_RETAIN_INTERVAL}")
fi
# Checkpoint formats. The HuggingFace safetensors export is written at every
# save so evaluation never needs a separate torch_dist->HF conversion job; the
# Megatron torch_dist checkpoint additionally carries fp32 optimizer state
# (roughly 10x the HF export) and is only required to resume training exactly.
# The raw Megatron-to-HF exporter copies tokenizer/config assets from
# HF_CHECKPOINT, so it needs a local snapshot directory rather than a hub id.
SAVE_FORMAT_ARGS=()
if [ -n "${SAVE_HF_ENABLED:-}" ]; then
    SAVE_HF_EFFECTIVE="${SAVE_HF_ENABLED}"
elif [ -d "${HF_CHECKPOINT}" ]; then
    SAVE_HF_EFFECTIVE=1
else
    SAVE_HF_EFFECTIVE=0
    echo "WARNING: HF safetensors export disabled: HF_CHECKPOINT=${HF_CHECKPOINT} is not a local directory" >&2
fi
case "${SAVE_HF_EFFECTIVE}" in
    1|true)
        if [ ! -d "${HF_CHECKPOINT}" ]; then
            echo "ERROR: SAVE_HF_ENABLED=1 requires HF_CHECKPOINT to be a local directory, got ${HF_CHECKPOINT}" >&2
            exit 1
        fi
        # A literal closing brace inside a ${VAR:-default} word terminates the
        # expansion early, so assign the default template separately.
        if [ -z "${SAVE_HF_TEMPLATE:-}" ]; then
            SAVE_HF_TEMPLATE="${SAVE_DIR}/hf/iter_{rollout_id:07d}"
        fi
        # Qwen3.5 checkpoints are VLM containers trained text-only: the frozen
        # vision/mtp tensors live only in the origin snapshot and must be
        # copied in for the export to be loadable standalone.
        SAVE_FORMAT_ARGS+=(--save-hf "${SAVE_HF_TEMPLATE}" --save-hf-add-missing-from-origin)
        ;;
    0|false) ;;
    *)
        echo "ERROR: SAVE_HF_ENABLED must be 0/1/false/true, got ${SAVE_HF_ENABLED}" >&2
        exit 1
        ;;
esac
case "${SAVE_MEGATRON:-1}" in
    1|true) ;;
    0|false)
        case "${SAVE_HF_EFFECTIVE}" in
            1|true) ;;
            *)
                echo "ERROR: SAVE_MEGATRON=0 requires the HF safetensors export to stay enabled, or nothing is saved" >&2
                exit 1
                ;;
        esac
        SAVE_FORMAT_ARGS+=(--no-save-megatron)
        echo "WARNING: Megatron torch_dist checkpointing disabled (SAVE_MEGATRON=0); this run cannot resume exactly - a resubmitted allocation restarts from the seed checkpoint" >&2
        ;;
    *)
        echo "ERROR: SAVE_MEGATRON must be 0/1/false/true, got ${SAVE_MEGATRON}" >&2
        exit 1
        ;;
esac
SAVE_PATH_ARGS=(--save "${SAVE_DIR}")
SAVE_INTERVAL_ARGS=()
if [ "${TMAX_PROFILE_DISABLE_CHECKPOINT}" = "0" ]; then
    SAVE_INTERVAL_ARGS=(--save-interval "${SAVE_INTERVAL:-10}")
else
    # Megatron requires --save-interval whenever --save is present, and Slime
    # forces a final save even when the interval exceeds the short profile.
    # Omit every model-checkpoint argument; rollout telemetry has independent
    # paths and fully-async metrics fall back to the rollout manager commit.
    SAVE_PATH_ARGS=()
    SAVE_RETENTION_ARGS=()
    SAVE_FORMAT_ARGS=()
fi
echo "=== Launching $(basename "${SLIME_TRAIN_ENTRYPOINT}") mode=${TMAX_TRAIN_MODE} (Ray submission ${RAY_JOB_SUBMISSION_ID}) ==="
# The custom reward post-processor already computes prompt-local GRPO
# advantages.  Slime's --normalize-advantages whitens them again across every
# response token in the global batch; with dynamic-history traces that leaks
# signal into otherwise all-zero prompt groups and introduces a length bias.
ray job submit --address="${RAY_JOB_ADDRESS}" \
    --submission-id "${RAY_JOB_SUBMISSION_ID}" \
    --no-wait \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" "${SLIME_TRAIN_ENTRYPOINT}" \
    --actor-num-nodes "$ACTOR_NUM_NODES" \
    --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
    "${TRAIN_MODE_ARGS[@]}" \
    --train-env-vars "$TRAIN_PROGRESS_ENV_JSON" \
    --rollout-num-gpus "$ROLLOUT_NUM_GPUS" \
    --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE" \
    --num-gpus-per-node "$RAY_NUM_GPUS_PER_NODE" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --ref-load "$REF_LOAD" \
    --load "$LOAD_DIR" \
    "${LOAD_CHECKPOINT_ARGS[@]}" \
    --dist-ckpt-strictness "${DIST_CKPT_STRICTNESS:-assume_ok_unexpected}" \
    "${OPT_PARAM_SCHEDULER_ARGS[@]}" \
    "${SAVE_PATH_ARGS[@]}" \
    "${SAVE_INTERVAL_ARGS[@]}" \
    "${SAVE_RETENTION_ARGS[@]}" \
    "${SAVE_FORMAT_ARGS[@]}" \
    "${TRAINING_LIFECYCLE_ARGS[@]}" \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rollout-log-function-path "${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH}" \
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
    "${TRAIN_LENGTH_ARGS[@]}" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    "${TRAIN_DYNAMIC_SAMPLING_ARGS[@]}" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --eval-global-batch-size "$EVAL_GLOBAL_BATCH_SIZE" \
    "${EVAL_ARGS[@]}" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" \
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" \
    --dynamic-history \
    --num-steps-per-rollout "$NUM_STEPS_PER_ROLLOUT" \
    --seq-length "$SEQ_LENGTH" \
    --tensor-model-parallel-size "${ACTOR_TENSOR_MODEL_PARALLEL_SIZE:-2}" \
    "${SEQUENCE_PARALLEL_ARGS[@]}" \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size "$CONTEXT_PARALLEL_SIZE" \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    "${PER_TOKEN_LOSS_ARGS[@]}" \
    --balance-data \
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-256}" \
    --distributed-timeout-minutes 30 \
    --advantage-estimator grpo \
    "${GRPO_NORMALIZATION_ARGS[@]}" \
    "${TRAINER_FP32_LM_HEAD_ARGS[@]}" \
    "${LOGPROB_GUARD_ARGS[@]}" \
    "${POLICY_LOSS_ARGS[@]}" \
    "${KL_LOSS_ARGS[@]}" \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr "${TRAIN_LR:-1e-6}" \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    "${OPTIMIZER_MEMORY_ARGS[@]}" \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend "${ATTENTION_BACKEND:-auto}" \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.8}" \
    --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
    --sglang-tool-call-parser qwen3_coder \
    "${SGLANG_REASONING_ARGS[@]}" \
    "${SGLANG_FP32_LM_HEAD_ARGS[@]}" \
    "${SGLANG_DETERMINISTIC_ARGS[@]}" \
    "${SGLANG_ATTENTION_BACKEND_ARGS[@]}" \
    --sglang-log-level-http "${SGLANG_LOG_LEVEL_HTTP:-warning}" \
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
    --use-wandb \
    "${WANDB_STEP_ARGS[@]}" \
    --wandb-mode "${WANDB_MODE:-offline}" \
    --wandb-run-id "$WANDB_RUN_ID" \
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
    --max-status-errors "${RAY_JOB_MAX_STATUS_ERRORS:-120}" &
RAY_JOB_WAITER_PID=$!
PIDS+=("${RAY_JOB_WAITER_PID}")
while polar_pid_is_active "${RAY_JOB_WAITER_PID}"; do
    if ! polar_check_spilot_admission_health; then
        kill -TERM "${RAY_JOB_WAITER_PID}" 2>/dev/null || true
        wait "${RAY_JOB_WAITER_PID}" 2>/dev/null || true
        exit 1
    fi
    for _polar_control_pid in \
        "${POLAR_ROLLOUT_PID}" \
        "${POLAR_GATEWAY_PID}" \
        "${POLAR_UDS_TUNNEL_PID}"; do
        [ -n "${_polar_control_pid}" ] || continue
        if ! polar_pid_is_active "${_polar_control_pid}"; then
            echo "ERROR: rank 0 Polar service pid ${_polar_control_pid} exited while the Ray job was running" >&2
            kill -TERM "${RAY_JOB_WAITER_PID}" 2>/dev/null || true
            wait "${RAY_JOB_WAITER_PID}" 2>/dev/null || true
            exit 1
        fi
    done
    sleep 1
done
unset _polar_control_pid
ray_job_wait_status=0
wait "${RAY_JOB_WAITER_PID}" || ray_job_wait_status=$?
if [ "${ray_job_wait_status}" -ne 0 ]; then
    echo "ERROR: Ray job waiter exited with status ${ray_job_wait_status}" >&2
    exit "${ray_job_wait_status}"
fi
