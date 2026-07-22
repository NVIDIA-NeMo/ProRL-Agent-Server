#!/usr/bin/env bash
# cw-dfw defaults for TMax Slime-GRPO on four 8xH100 nodes.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
USER_ROOT="$(dirname "${SPILOT_ROOT}")"
# shellcheck source=./lifecycle.sh
source "${SCRIPT_DIR}/lifecycle.sh"

export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
export TMAX_DATASET_DIR="${TMAX_DATASET_DIR:-${POLAR_DATA_ROOT}/tmax-15k}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${POLAR_DATA_ROOT}/tmax-15k-sif}"
export AGENT_CLI_DIR="${AGENT_CLI_DIR:-${POLAR_DATA_ROOT}/agent_cli/opt_node}"
export TMAX_SIF_PYTHON_BIN="${TMAX_SIF_PYTHON_BIN:-${USER_ROOT}/.python/polar/bin/python}"
export MINI_SWE_AGENT_RUNTIME_DIR="${MINI_SWE_AGENT_RUNTIME_DIR:-${POLAR_DATA_ROOT}/mini_swe_agent_runtime}"
export MINI_SWE_AGENT_CONTAINER_DIR="${MINI_SWE_AGENT_CONTAINER_DIR:-/opt/polar-mini-swe-agent}"
export MINI_SWE_AGENT_PYTHON_ROOT="${MINI_SWE_AGENT_PYTHON_ROOT:-${USER_ROOT}/tb_runs/pyportable/cpython-3.12.13-linux-x86_64-gnu}"
export MINI_SWE_AGENT_SPEC="${MINI_SWE_AGENT_SPEC:-mini-swe-agent==2.4.2}"
export MINI_SWE_AGENT_BIN="${MINI_SWE_AGENT_BIN:-${MINI_SWE_AGENT_RUNTIME_DIR}/bin/mini-swe-agent}"

export ACCOUNT="${ACCOUNT:-nvr_lpr_llm}"
export SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-H100}"
export SLURM_GPUS="${SLURM_GPUS:-8}"
export NUM_NODES="${NUM_NODES:-4}"
export WALL_TIME="${WALL_TIME:-4:00:00}"
export TMAX_MIN_WALL_TIME="${TMAX_MIN_WALL_TIME:-4:00:00}"
if ! _tmax_wall_seconds="$(tmax_slurm_duration_seconds "${WALL_TIME}")" || \
   ! _tmax_min_wall_seconds="$(tmax_slurm_duration_seconds "${TMAX_MIN_WALL_TIME}")"; then
    echo "ERROR: unsupported WALL_TIME=${WALL_TIME} or TMAX_MIN_WALL_TIME=${TMAX_MIN_WALL_TIME}" >&2
    return 1 2>/dev/null || exit 1
fi
if [ "${_tmax_wall_seconds}" -lt "${_tmax_min_wall_seconds}" ]; then
    echo "[tmax env] raising wall time ${WALL_TIME} -> ${TMAX_MIN_WALL_TIME} to amortize model startup and graceful checkpoint drain" >&2
    export WALL_TIME="${TMAX_MIN_WALL_TIME}"
    _tmax_wall_seconds="${_tmax_min_wall_seconds}"
fi

if ! [[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${SLURM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_NODES and SLURM_GPUS must be positive integers, got NUM_NODES=${NUM_NODES} SLURM_GPUS=${SLURM_GPUS}" >&2
    return 1 2>/dev/null || exit 1
fi
_tmax_multi_gateway_default=0
case "${POLAR_MULTI_GATEWAY:-${_tmax_multi_gateway_default}}" in
    1|true) export POLAR_MULTI_GATEWAY=1 ;;
    0|false) export POLAR_MULTI_GATEWAY=0 ;;
    *)
        echo "ERROR: POLAR_MULTI_GATEWAY must be 0/1/false/true" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
unset _tmax_multi_gateway_default
export POLAR_GATEWAY_COUNT_OVERRIDE="${POLAR_GATEWAY_COUNT_OVERRIDE:-}"
if [ -n "${POLAR_GATEWAY_COUNT_OVERRIDE}" ]; then
    if ! [[ "${POLAR_GATEWAY_COUNT_OVERRIDE}" =~ ^[1-9][0-9]*$ ]] || \
       [ "${POLAR_GATEWAY_COUNT_OVERRIDE}" -gt "${NUM_NODES}" ]; then
        echo "ERROR: POLAR_GATEWAY_COUNT_OVERRIDE must be in [1, NUM_NODES=${NUM_NODES}]" >&2
        return 1 2>/dev/null || exit 1
    fi
    if [ "${POLAR_MULTI_GATEWAY}" != "1" ] && \
       [ "${POLAR_GATEWAY_COUNT_OVERRIDE}" -ne 1 ]; then
        echo "ERROR: POLAR_GATEWAY_COUNT_OVERRIDE>1 requires POLAR_MULTI_GATEWAY=1" >&2
        return 1 2>/dev/null || exit 1
    fi
fi
_tmax_total_gpus="$((NUM_NODES * SLURM_GPUS))"
_tmax_short_limit_seconds="$((2 * 60 * 60))"

# Let Slurm choose whichever account-compatible H100 partition can start first.
# ``interactive`` accepts up to two nodes / sixteen GPUs and can run for four
# hours. ``batch_short`` accepts up to four nodes for at most two hours. Slurm
# applies a partition's QoS cap to the entire comma-separated request rather
# than skipping only that partition, so include only partitions which can
# admit the requested resources and wall time.
if [ -z "${PARTITION:-}" ]; then
    if [ "${NUM_NODES}" -le 2 ] && \
       [ "${_tmax_total_gpus}" -le 16 ]; then
        if [ "${_tmax_wall_seconds}" -le "${_tmax_short_limit_seconds}" ]; then
            export PARTITION="${TMAX_SMALL_SHORT_JOB_PARTITIONS:-interactive,batch_short,backfill,batch}"
        else
            export PARTITION="${TMAX_SMALL_JOB_PARTITIONS:-interactive,backfill,batch}"
        fi
    elif [ "${NUM_NODES}" -le 4 ] && \
         [ "${_tmax_wall_seconds}" -le "${_tmax_short_limit_seconds}" ]; then
        # Keep the low-capacity batch_short QoS separate. On cw-dfw, mixing it
        # with normal-QoS partitions lowers the effective priority and a live
        # QOSGrpNodeLimit can block the whole comma-separated request instead
        # of falling through to batch/backfill.
        export PARTITION="${TMAX_SHORT_JOB_PARTITIONS:-batch_short}"
    else
        # batch_large/batch_long reject the nvr_lpr_llm account.
        export PARTITION="${TMAX_LARGE_JOB_PARTITIONS:-backfill,batch}"
    fi
else
    export PARTITION
fi

_tmax_partition_list_contains() {
    local partition_name="$1"
    case ",${PARTITION}," in
        *",${partition_name},"*) return 0 ;;
        *) return 1 ;;
    esac
}
if _tmax_partition_list_contains interactive && \
   { [ "${NUM_NODES}" -gt 2 ] || [ "${_tmax_total_gpus}" -gt 16 ]; }; then
    echo "ERROR: PARTITION=${PARTITION} includes interactive, which is limited to at most 2 nodes and 16 total GPUs; requested ${NUM_NODES} nodes / ${_tmax_total_gpus} GPUs" >&2
    return 1 2>/dev/null || exit 1
fi
if _tmax_partition_list_contains batch_short && \
   { [ "${NUM_NODES}" -gt 4 ] || [ "${_tmax_wall_seconds}" -gt "${_tmax_short_limit_seconds}" ]; }; then
    echo "ERROR: PARTITION=${PARTITION} includes batch_short, which is limited to at most 4 nodes and 2:00:00; requested ${NUM_NODES} nodes / WALL_TIME=${WALL_TIME}" >&2
    return 1 2>/dev/null || exit 1
fi
unset -f _tmax_partition_list_contains
unset _tmax_wall_seconds _tmax_min_wall_seconds _tmax_total_gpus _tmax_short_limit_seconds
export CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
export SLURM_STEP_CPUS_PER_TASK="${SLURM_STEP_CPUS_PER_TASK:-120}"
export POLAR_SLURM_MEM_PER_NODE="${POLAR_SLURM_MEM_PER_NODE:-0}"
if ! [[ "${POLAR_SLURM_MEM_PER_NODE}" =~ ^(0|[1-9][0-9]*[KMGTP]?)$ ]]; then
    echo "ERROR: POLAR_SLURM_MEM_PER_NODE must be 0 or a positive Slurm memory value (for example 250G)" >&2
    return 1 2>/dev/null || exit 1
fi
export SUBMIT_BACKEND="${SUBMIT_BACKEND:-sbatch}"

DEFAULT_TRAIN_SQSH="${POLAR_DATA_ROOT}/container/flappydora-ubuntu22.04-cuda13.3.sqsh"
if [ ! -f "${DEFAULT_TRAIN_SQSH}" ] && [ -f "${USER_ROOT}/spilot-router/container/polar_train.sqsh" ]; then
    DEFAULT_TRAIN_SQSH="${USER_ROOT}/spilot-router/container/polar_train.sqsh"
fi
export POLR_TRAIN_SQSH="${POLR_TRAIN_SQSH:-${DEFAULT_TRAIN_SQSH}}"
export POLR_TRAIN_VENV="${POLR_TRAIN_VENV:-${USER_ROOT}/.python/polar}"
export TRAIN_CONTAINER_MOUNTS="${TRAIN_CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw}"
export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-/usr/bin/apptainer}"
export POLAR_APPTAINER_NO_INSTANCE="${POLAR_APPTAINER_NO_INSTANCE:-1}"
# TMax currently uses the proven fresh-exec path while the persistent broker's
# full-scale throughput is evaluated. The runtime itself keeps broker mode as
# its default, so other launchers are unchanged and can opt in here with 1.
export POLAR_APPTAINER_PERSISTENT_BROKER="${POLAR_APPTAINER_PERSISTENT_BROKER:-0}"
export POLAR_APPTAINER_NO_MOUNT_HOSTFS="${POLAR_APPTAINER_NO_MOUNT_HOSTFS:-1}"
export POLAR_APPTAINER_NO_MOUNT_TMP="${POLAR_APPTAINER_NO_MOUNT_TMP:-1}"
export POLAR_APPTAINER_ISOLATE_PID="${POLAR_APPTAINER_ISOLATE_PID:-1}"
export POLAR_APPTAINER_ISOLATE_IPC="${POLAR_APPTAINER_ISOLATE_IPC:-1}"
export POLAR_APPTAINER_CLEANENV="${POLAR_APPTAINER_CLEANENV:-1}"
# Popen admission alone is insufficient: Apptainer performs the expensive SIF
# and overlay mounts after Popen returns. Keep at most 32 complete broker
# startups (through socket readiness or failed-process reap) active per gateway
# process. A 48-SIF nested-Pyxis sweep performed while the full SIF rebuild was
# active completed with 0 failures at gates 8, 16, and 32. Gate 32 reduced p95
# startup to 1.62s and total wall time to 1.68s (versus 6.00s/6.02s at gate 8).
# The gateway has 48 INIT workers and the Slurm step reserves 120 CPUs, so 32
# leaves headroom for prepare/evaluator work while removing the multi-wave
# startup tail that starved SGLang. 120s still covers a cold Lustre mount
# without creating synchronized retry waves.
export POLAR_APPTAINER_BROKER_START_CONCURRENCY="${POLAR_APPTAINER_BROKER_START_CONCURRENCY:-32}"
# A live four-node sweep admitted eight copies of one SIF on a gateway: two
# task_000158 sessions reached the 120s socket timeout while unrelated images
# continued to start, and both copies succeeded on retry. Admit same-image
# starts two at a time before the aggregate gate to avoid per-SIF FUSE/mount
# stampedes without reducing concurrency across different task images.
export POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY="${POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY:-2}"
export POLAR_APPTAINER_BROKER_START_TIMEOUT_SEC="${POLAR_APPTAINER_BROKER_START_TIMEOUT_SEC:-120}"
export POLAR_APPTAINER_DIRECT_EXEC_RETRIES="${POLAR_APPTAINER_DIRECT_EXEC_RETRIES:-3}"
export POLAR_APPTAINER_BROKER_RETRY_BACKOFF_SEC="${POLAR_APPTAINER_BROKER_RETRY_BACKOFF_SEC:-5}"
export POLAR_APPTAINER_BROKER_RETRY_BACKOFF_MAX_SEC="${POLAR_APPTAINER_BROKER_RETRY_BACKOFF_MAX_SEC:-30}"
# Rootless Apptainer cannot create a bridge network on this cluster.  Its
# supported ``none`` network gives each sandbox session one persistent private
# loopback namespace; the shared launcher restores gateway/proxy access through
# bind-mounted Unix sockets, preventing fixed localhost services in concurrent
# TMax sessions from colliding.
export POLAR_SANDBOX_NETWORK="${POLAR_SANDBOX_NETWORK:-none}"
# The site container-cache endpoint correctly tunnels HTTPS CONNECT requests
# but returns its 59-byte registry health body for ordinary HTTP proxy GETs.
# TMax images commonly retain Ubuntu/Debian http:// apt sources, so upgrade
# those sources inside each disposable session overlay (never inside the SIF).
export POLAR_APT_HTTP_SOURCE_POLICY="${POLAR_APT_HTTP_SOURCE_POLICY:-https}"
case "${POLAR_SANDBOX_NETWORK}" in
    host|none) ;;
    *)
        echo "ERROR: POLAR_SANDBOX_NETWORK must be host or none, got ${POLAR_SANDBOX_NETWORK}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

export HF_HOME="${HF_HOME:-${USER_ROOT}/.cache/huggingface}"
# FlashInfer ignores XDG_CACHE_HOME and otherwise compiles into the job-local
# HOME that run_in_container.sh deliberately removes. Qwen3.5 TP>1 enables its
# TensorRT-LLM all-reduce fusion automatically; compiling that operator took
# about 117s independently on every rollout node in the first 9B run. Keep the
# namespace ABI-specific because FlashInfer's own cache key contains only its
# version and GPU architecture, not the Torch/CUDA/compiler ABI.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${POLAR_DATA_ROOT}/kernel-cache/cuda13.3-py312-torch2.11-sglang0.5.13-fi0.6.12-d768c14e-gcc13}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-9B}"
DEFAULT_REF_LOAD="${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-9B_torch_dist"
export REF_LOAD="${REF_LOAD:-${DEFAULT_REF_LOAD}}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-${REF_LOAD}}"
export SLIME_DIR="${SLIME_DIR:-${SPILOT_ROOT}/src/slime}"
export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SCRIPT_DIR}/model_args.sh}"
DEFAULT_MEGATRON_DIR="${POLAR_DATA_ROOT}/Megatron-LM-slime-v0.3.0"
ROUTER_MEGATRON_DIR="${USER_ROOT}/spilot-router/data/Megatron-LM-slime-v0.3.0"
WORKSPACE_MEGATRON_DIR="${SPILOT_ROOT}/src/Megatron-LM"
if [ ! -f "${DEFAULT_MEGATRON_DIR}/megatron/training/tokenizer/tokenizer.py" ] && \
   [ -f "${ROUTER_MEGATRON_DIR}/megatron/training/tokenizer/tokenizer.py" ]; then
    DEFAULT_MEGATRON_DIR="${ROUTER_MEGATRON_DIR}"
fi
if [ ! -f "${DEFAULT_MEGATRON_DIR}/megatron/training/tokenizer/tokenizer.py" ] && \
   [ -f "${WORKSPACE_MEGATRON_DIR}/megatron/training/tokenizer/tokenizer.py" ]; then
    DEFAULT_MEGATRON_DIR="${WORKSPACE_MEGATRON_DIR}"
fi
export MEGATRON_DIR="${MEGATRON_DIR:-${DEFAULT_MEGATRON_DIR}}"

# Keep the historical single-gateway/fresh-Apptainer execution path by
# default.  Gateway fan-out is an independent infrastructure experiment and
# must be opted into explicitly rather than being coupled to NUM_NODES.

# The four-node TMax jobs use one complete eight-GPU learner node. TP4 plus
# parallelism is the safe mapping for the pinned Qwen3.5 hybrid implementation:
# its GatedDeltaNet path does not implement mathematically correct CP>1 state
# propagation. One trainer node gives TP4 x DP2; the other 24 GPUs provide
# independent TP1 rollout engines. One optimizer step still consumes all
# 8*32=256 trajectories.
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ACTOR_TENSOR_MODEL_PARALLEL_SIZE="${ACTOR_TENSOR_MODEL_PARALLEL_SIZE:-4}"
# CP>1 would split the recurrent GatedDeltaNet state incorrectly in the pinned
# Megatron checkout, so long-sequence capacity comes from TP4 sequence
# parallelism and a one-sample dynamic microbatch rather than CP.
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
export TMAX_REQUIRE_FULL_GPU_ALLOCATION="${TMAX_REQUIRE_FULL_GPU_ALLOCATION:-1}"
export TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL="${TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL:-0}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-2048}"
# ``ROLLOUT_MAX_RESPONSE_LEN`` is the cap for one model turn. TMax separately
# caps the accumulated multi-turn response (model tokens plus tool
# observations) at 65,536. The trainer pack must preserve the 2,048-token
# prompt in addition to that complete response.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
export TMAX_MAX_TOTAL_RESPONSE_LEN="${TMAX_MAX_TOTAL_RESPONSE_LEN:-65536}"
for _tmax_token_budget_name in ROLLOUT_MAX_PROMPT_LEN ROLLOUT_MAX_RESPONSE_LEN TMAX_MAX_TOTAL_RESPONSE_LEN; do
    _tmax_token_budget_value="${!_tmax_token_budget_name}"
    if ! [[ "${_tmax_token_budget_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${_tmax_token_budget_name} must be a positive integer, got ${_tmax_token_budget_value}" >&2
        return 1 2>/dev/null || exit 1
    fi
done
unset _tmax_token_budget_name _tmax_token_budget_value
export TMAX_TRAIN_PACK_LENGTH="$((ROLLOUT_MAX_PROMPT_LEN + TMAX_MAX_TOTAL_RESPONSE_LEN))"
export SEQ_LENGTH="${SEQ_LENGTH:-${TMAX_TRAIN_PACK_LENGTH}}"
# Slime's dynamic scheduler multiplies this cap by CP, not TP. With CP=1 it
# must therefore admit one complete 67,584-token pack. TP4 sequence
# parallelism and full recomputation provide the per-GPU memory reduction.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${TMAX_TRAIN_PACK_LENGTH}}"
# The dynamic scheduler safely places an individual sample that exceeds its
# token cap in a microbatch by itself.  TMax keeps this opt-in disabled because
# a full 65k trajectory can be expensive; short-action harnesses such as
# SPilot may opt in to use the cap strictly as an aggregate microbatch bound.
export TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP="${TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP:-0}"
# Qwen3.5-9B declares a native 262,144-token context, and the observed TP=2
# SGLang engine KV pool holds about 1.568M tokens. A native-length request is
# therefore admissible without lowering the current static-memory fraction.
# The old 50k cap was smaller than the paper's 65,536 total-response budget and
# deterministically terminated otherwise healthy long Terminal-Bench runs;
# 131,072 could still truncate a 64-step trajectory, so use the native ceiling.
export TMAX_MODEL_MAX_CONTEXT_LENGTH="${TMAX_MODEL_MAX_CONTEXT_LENGTH:-262144}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-262144}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
# SGLang 0.5.13 implements this inside the LM-head matmul by promoting both
# hidden states and the stored head weight to FP32.  This aligns rollout
# logits/log-probabilities with the released recipe's FP32 LM-head projection.
export SGLANG_ENABLE_FP32_LM_HEAD="${SGLANG_ENABLE_FP32_LM_HEAD:-1}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export DIST_CKPT_STRICTNESS="${DIST_CKPT_STRICTNESS:-log_all}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-}"
if ! [[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SAVE_INTERVAL must be a positive integer, got ${SAVE_INTERVAL}" >&2
    return 1 2>/dev/null || exit 1
fi
if [ -n "${SAVE_RETAIN_INTERVAL}" ]; then
    if ! [[ "${SAVE_RETAIN_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: SAVE_RETAIN_INTERVAL must be a positive integer, got ${SAVE_RETAIN_INTERVAL}" >&2
        return 1 2>/dev/null || exit 1
    fi
    if [ "$((SAVE_RETAIN_INTERVAL % SAVE_INTERVAL))" -ne 0 ]; then
        echo "ERROR: SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL} must be divisible by SAVE_INTERVAL=${SAVE_INTERVAL}" >&2
        return 1 2>/dev/null || exit 1
    fi
fi
# Checkpoint formats. SAVE_HF_ENABLED=1 (default) writes an HF safetensors
# export to ${SAVE_DIR}/hf/iter_XXXXXXX at every save interval, so evaluation
# no longer needs export_hf_checkpoint.sh conversion jobs. SAVE_MEGATRON=0
# drops the Megatron torch_dist checkpoint (fp32 optimizer state, ~10x the HF
# export size); the run then cannot resume exactly after preemption or a
# graceful-deadline exit, and the lifecycle watchers never see a checkpoint
# tracker advance, so only disable it for runs that are disposable.
# Default the HF export on only when the exporter can actually run: it copies
# tokenizer/config assets from HF_CHECKPOINT, so a hub id (legacy run states)
# must degrade to the pre-HF-export behaviour instead of failing the run.
if [ -z "${SAVE_HF_ENABLED:-}" ]; then
    if [ -d "${HF_CHECKPOINT}" ]; then
        SAVE_HF_ENABLED=1
    else
        SAVE_HF_ENABLED=0
        echo "[tmax env] WARNING: HF safetensors export disabled: HF_CHECKPOINT=${HF_CHECKPOINT} is not a local directory" >&2
    fi
fi
export SAVE_HF_ENABLED
export SAVE_MEGATRON="${SAVE_MEGATRON:-1}"
case "${SAVE_HF_ENABLED}" in
    0|1|true|false) ;;
    *)
        echo "ERROR: SAVE_HF_ENABLED must be 0/1/false/true, got ${SAVE_HF_ENABLED}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
case "${SAVE_MEGATRON}" in
    0|1|true|false) ;;
    *)
        echo "ERROR: SAVE_MEGATRON must be 0/1/false/true, got ${SAVE_MEGATRON}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
case "${SAVE_MEGATRON}:${SAVE_HF_ENABLED}" in
    0:0|0:false|false:0|false:false)
        echo "ERROR: SAVE_MEGATRON=0 requires SAVE_HF_ENABLED=1, or nothing is saved at save time" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
export NUM_EPOCH="${NUM_EPOCH:-1}"
# Optional absolute rollout-loop boundary. Slime treats --num-rollout as an
# exclusive upper bound, so the matching final checkpoint/watcher target is
# always TMAX_NUM_ROLLOUT - 1. Keep the derived target explicit in run state so
# the trainer and watcher cannot silently follow different stopping contracts.
if [ -n "${TMAX_TARGET_ITER:-}" ] && \
   ! [[ "${TMAX_TARGET_ITER}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: TMAX_TARGET_ITER must be a non-negative integer" >&2
    return 1 2>/dev/null || exit 1
fi
if [ -n "${TMAX_NUM_ROLLOUT:-}" ]; then
    if ! [[ "${TMAX_NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: TMAX_NUM_ROLLOUT must be a positive integer" >&2
        return 1 2>/dev/null || exit 1
    fi
    _tmax_explicit_target="$((TMAX_NUM_ROLLOUT - 1))"
    if [ -n "${TMAX_TARGET_ITER:-}" ] && \
       [ "${TMAX_TARGET_ITER}" -ne "${_tmax_explicit_target}" ]; then
        echo "ERROR: TMAX_TARGET_ITER=${TMAX_TARGET_ITER} must equal TMAX_NUM_ROLLOUT-1=${_tmax_explicit_target}" >&2
        return 1 2>/dev/null || exit 1
    fi
    export TMAX_NUM_ROLLOUT
    export TMAX_TARGET_ITER="${_tmax_explicit_target}"
    unset _tmax_explicit_target
fi
# Match the released TMax recipe: behavior-anchored DPPO with binary-TV 0.1,
# no reference-policy KL term, and a constant 1e-6 learning rate. The shared
# SWE-Gym launcher retains PPO+TIS unless these TMax overrides are present.
export POLICY_LOSS_TYPE="${POLICY_LOSS_TYPE:-dppo}"
export DPPO_DIVERGENCE_TYPE="${DPPO_DIVERGENCE_TYPE:-tv}"
export DPPO_DIVERGENCE_THRESHOLD="${DPPO_DIVERGENCE_THRESHOLD:-0.1}"
export USE_TIS="${USE_TIS:-0}"
export TRAIN_LR="${TRAIN_LR:-1e-6}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0}"
export TMAX_ENABLE_FP32_LM_HEAD="${TMAX_ENABLE_FP32_LM_HEAD:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-1}"
if ! [[ "${TMAX_ENABLE_FP32_LM_HEAD}" =~ ^[01]$ ]] || \
   ! [[ "${CALCULATE_PER_TOKEN_LOSS}" =~ ^[01]$ ]]; then
    echo "ERROR: TMAX_ENABLE_FP32_LM_HEAD and CALCULATE_PER_TOKEN_LOSS must be 0 or 1" >&2
    return 1 2>/dev/null || exit 1
fi
# The released recipe uses centered (mean-subtracted) advantages, not GRPO's
# additional group-standard-deviation scaling.  The custom trajectory-level
# LOO post-processor reads this Slime argument directly.
export GRPO_STD_NORMALIZATION="${GRPO_STD_NORMALIZATION:-0}"
# A healthy Qwen3.5-4B baseline was ~0.005 while the incompatible 9B launch
# was ~10.  A threshold of 1.0 leaves ample room for genuine fully-async
# policy lag, but stops batches whose average token probability ratio is
# already far outside the useful TIS range before any gradient is applied.
export MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF="${MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF:-1.0}"
export TMAX_OPTIMIZER_CPU_OFFLOAD="${TMAX_OPTIMIZER_CPU_OFFLOAD:-0}"
if ! [[ "${TMAX_OPTIMIZER_CPU_OFFLOAD}" =~ ^[01]$ ]]; then
    echo "ERROR: TMAX_OPTIMIZER_CPU_OFFLOAD must be 0 or 1" >&2
    return 1 2>/dev/null || exit 1
fi

# ``fully_async`` is the production default.  ``colocate`` is an explicit
# synchronous profiling mode: Slime's asynchronous driver intentionally does
# not support sharing actor and rollout GPUs, while train.py does.
export TMAX_TRAIN_MODE="${TMAX_TRAIN_MODE:-fully_async}"
case "${TMAX_TRAIN_MODE}" in
    fully_async|colocate) ;;
    *)
        echo "ERROR: TMAX_TRAIN_MODE must be fully_async or colocate, got ${TMAX_TRAIN_MODE}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

_tmax_validate_resource_topology() {
    local name value
    local -a positive_names=(
        NUM_NODES SLURM_GPUS RAY_NUM_GPUS_PER_NODE
        ACTOR_NUM_NODES ACTOR_NUM_GPUS_PER_NODE
        ACTOR_TENSOR_MODEL_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE
        ROLLOUT_NUM_GPUS ROLLOUT_NUM_GPUS_PER_ENGINE
        ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT NUM_STEPS_PER_ROLLOUT
        SEQ_LENGTH MAX_TOKENS_PER_GPU TMAX_TRAIN_PACK_LENGTH
    )
    for name in "${positive_names[@]}"; do
        value="${!name}"
        if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: ${name} must be a positive integer, got ${value}" >&2
            return 1
        fi
    done

    if ! [[ "$TMAX_REQUIRE_FULL_GPU_ALLOCATION" =~ ^[01]$ ]] || \
       ! [[ "$TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL" =~ ^[01]$ ]] || \
       ! [[ "$TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP" =~ ^[01]$ ]]; then
        echo "ERROR: TMAX_REQUIRE_FULL_GPU_ALLOCATION, TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL, and TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP must be 0 or 1" >&2
        return 1
    fi

    local actor_gpus actor_parallel_size capacity allocated_gpus rollout_product expected_global_batch
    local required_gpus
    local global_batch actor_dp train_rollouts_per_dp min_train_rollouts_per_dp
    actor_gpus="$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))"
    capacity="$((NUM_NODES * RAY_NUM_GPUS_PER_NODE))"
    allocated_gpus="$((NUM_NODES * SLURM_GPUS))"
    rollout_product="$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"

    if [ "$RAY_NUM_GPUS_PER_NODE" -gt "$SLURM_GPUS" ]; then
        echo "ERROR: RAY_NUM_GPUS_PER_NODE=${RAY_NUM_GPUS_PER_NODE} exceeds allocated SLURM_GPUS=${SLURM_GPUS}" >&2
        return 1
    fi
    if [ "$ACTOR_NUM_NODES" -gt "$NUM_NODES" ] || \
       [ "$ACTOR_NUM_GPUS_PER_NODE" -gt "$RAY_NUM_GPUS_PER_NODE" ]; then
        echo "ERROR: actor topology does not fit NUM_NODES=${NUM_NODES} x RAY_NUM_GPUS_PER_NODE=${RAY_NUM_GPUS_PER_NODE}" >&2
        return 1
    fi
    if [ "${TMAX_TRAIN_MODE}" = "colocate" ]; then
        if [ "${actor_gpus}" -gt "${ROLLOUT_NUM_GPUS}" ]; then
            required_gpus="${actor_gpus}"
        else
            required_gpus="${ROLLOUT_NUM_GPUS}"
        fi
    else
        required_gpus="$((actor_gpus + ROLLOUT_NUM_GPUS))"
    fi
    if [ "${required_gpus}" -gt "$capacity" ]; then
        echo "ERROR: ${TMAX_TRAIN_MODE} actor (${actor_gpus}) / rollout (${ROLLOUT_NUM_GPUS}) require ${required_gpus} GPUs, exceeding Ray capacity ${capacity}" >&2
        echo "  For one node, explicitly use e.g. ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=4." >&2
        return 1
    fi
    if [ "$TMAX_REQUIRE_FULL_GPU_ALLOCATION" = "1" ] && {
       [ "$capacity" -ne "$allocated_gpus" ] ||
       [ "${required_gpus}" -ne "$allocated_gpus" ];
    }; then
        echo "ERROR: ${TMAX_TRAIN_MODE} actor (${actor_gpus}) / rollout (${ROLLOUT_NUM_GPUS}) use ${required_gpus} GPUs and must use all ${allocated_gpus} allocated GPUs" >&2
        echo "  Set TMAX_REQUIRE_FULL_GPU_ALLOCATION=0 only for an intentional under-allocation experiment." >&2
        return 1
    fi
    if [ "$((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE))" -ne 0 ]; then
        echo "ERROR: ROLLOUT_NUM_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE" >&2
        return 1
    fi
    if [ "$((ACTOR_NUM_GPUS_PER_NODE % ACTOR_TENSOR_MODEL_PARALLEL_SIZE))" -ne 0 ] && \
       [ "${TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL}" != "1" ]; then
        echo "ERROR: actor GPUs per node ${ACTOR_NUM_GPUS_PER_NODE} must be divisible by tensor parallel size ${ACTOR_TENSOR_MODEL_PARALLEL_SIZE}" >&2
        echo "  Set TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL=1 only when TP ranks intentionally span nodes." >&2
        return 1
    fi
    actor_parallel_size="$((ACTOR_TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))"
    if [ "$((actor_gpus % actor_parallel_size))" -ne 0 ]; then
        echo "ERROR: total actor GPUs ${actor_gpus} must be divisible by tensor*context parallel size ${ACTOR_TENSOR_MODEL_PARALLEL_SIZE}*${CONTEXT_PARALLEL_SIZE}=${actor_parallel_size}" >&2
        return 1
    fi
    if [ "$((SEQ_LENGTH % (2 * CONTEXT_PARALLEL_SIZE)))" -ne 0 ]; then
        echo "ERROR: SEQ_LENGTH=${SEQ_LENGTH} must be divisible by 2*CONTEXT_PARALLEL_SIZE=$((2 * CONTEXT_PARALLEL_SIZE)) for Megatron context-parallel slicing" >&2
        return 1
    fi
    if [ "${ROLLOUT_MAX_RESPONSE_LEN}" -gt "${TMAX_MAX_TOTAL_RESPONSE_LEN}" ]; then
        echo "ERROR: per-turn ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN} exceeds cumulative TMAX_MAX_TOTAL_RESPONSE_LEN=${TMAX_MAX_TOTAL_RESPONSE_LEN}" >&2
        return 1
    fi
    if [ "${TMAX_TRAIN_PACK_LENGTH}" -ne "$((ROLLOUT_MAX_PROMPT_LEN + TMAX_MAX_TOTAL_RESPONSE_LEN))" ] || \
       [ "${SEQ_LENGTH}" -ne "${TMAX_TRAIN_PACK_LENGTH}" ]; then
        echo "ERROR: TMax requires SEQ_LENGTH=prompt+total_response=${TMAX_TRAIN_PACK_LENGTH}, got SEQ_LENGTH=${SEQ_LENGTH}" >&2
        return 1
    fi
    if [ "$((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE))" -lt "${TMAX_TRAIN_PACK_LENGTH}" ] && \
       [ "${TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP}" -ne 1 ]; then
        echo "ERROR: trainer token capacity MAX_TOKENS_PER_GPU*CP=$((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE)) is smaller than the complete TMax pack ${TMAX_TRAIN_PACK_LENGTH}" >&2
        echo "  Set TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP=1 only when the harness bounds individual trajectories independently; oversize samples run alone." >&2
        return 1
    fi
    if [ "$((rollout_product % NUM_STEPS_PER_ROLLOUT))" -ne 0 ]; then
        echo "ERROR: rollout batch product must be divisible by NUM_STEPS_PER_ROLLOUT" >&2
        return 1
    fi
    expected_global_batch="$((rollout_product / NUM_STEPS_PER_ROLLOUT))"
    global_batch="${GLOBAL_BATCH_SIZE:-${expected_global_batch}}"
    if ! [[ "$global_batch" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: GLOBAL_BATCH_SIZE must be a positive integer, got ${global_batch}" >&2
        return 1
    fi
    if [ "$global_batch" -ne "$expected_global_batch" ]; then
        echo "ERROR: GLOBAL_BATCH_SIZE=${global_batch} must equal ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT/NUM_STEPS_PER_ROLLOUT=${expected_global_batch}" >&2
        return 1
    fi
    actor_dp="$((actor_gpus / actor_parallel_size))"
    if [ "$((global_batch % actor_dp))" -ne 0 ]; then
        echo "ERROR: global batch ${global_batch} must be divisible by actor DP ${actor_dp}" >&2
        return 1
    fi
    train_rollouts_per_dp="$((global_batch / actor_dp))"
    min_train_rollouts_per_dp="${TMAX_MIN_TRAIN_ROLLOUTS_PER_DP:-8}"
    if [ "$train_rollouts_per_dp" -lt "$min_train_rollouts_per_dp" ]; then
        echo "ERROR: global batch gives only ${train_rollouts_per_dp} trajectories per actor DP rank; require at least ${min_train_rollouts_per_dp}" >&2
        return 1
    fi
}
if ! _tmax_validate_resource_topology; then
    return 1 2>/dev/null || exit 1
fi
unset -f _tmax_validate_resource_topology

export TMAX_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS:-1800}"
export TMAX_ENABLE_GRACEFUL_EXIT="${TMAX_ENABLE_GRACEFUL_EXIT:-1}"
export POLAR_FULLY_ASYNC="${POLAR_FULLY_ASYNC:-true}"
export TMAX_PROFILE_DISABLE_CHECKPOINT="${TMAX_PROFILE_DISABLE_CHECKPOINT:-0}"
if ! [[ "${TMAX_PROFILE_DISABLE_CHECKPOINT}" =~ ^[01]$ ]]; then
    echo "ERROR: TMAX_PROFILE_DISABLE_CHECKPOINT must be 0 or 1" >&2
    return 1 2>/dev/null || exit 1
fi
if [ "${TMAX_PROFILE_DISABLE_CHECKPOINT}" = "1" ] && \
   [ "${TMAX_ENABLE_GRACEFUL_EXIT}" != "0" ]; then
    echo "ERROR: TMAX_PROFILE_DISABLE_CHECKPOINT=1 requires TMAX_ENABLE_GRACEFUL_EXIT=0" >&2
    return 1 2>/dev/null || exit 1
fi
if [ "${TMAX_TRAIN_MODE}" = "colocate" ]; then
    if [ "${POLAR_FULLY_ASYNC}" != "false" ]; then
        echo "ERROR: TMAX_TRAIN_MODE=colocate requires POLAR_FULLY_ASYNC=false" >&2
        return 1 2>/dev/null || exit 1
    fi
    if [ "${TMAX_ENABLE_GRACEFUL_EXIT}" != "0" ]; then
        echo "ERROR: TMAX_TRAIN_MODE=colocate requires TMAX_ENABLE_GRACEFUL_EXIT=0; sync train.py has no graceful lifecycle support" >&2
        return 1 2>/dev/null || exit 1
    fi
fi
export TMAX_MIN_ASYNC_LEVEL="${TMAX_MIN_ASYNC_LEVEL:-4}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-${TMAX_MIN_ASYNC_LEVEL}}"
if ! [[ "${TMAX_MIN_ASYNC_LEVEL}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${POLAR_MAX_ASYNC_LEVEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TMAX_MIN_ASYNC_LEVEL and POLAR_MAX_ASYNC_LEVEL must be positive integers" >&2
    return 1 2>/dev/null || exit 1
fi
if [ "${POLAR_FULLY_ASYNC}" = "true" ] && \
   [ "${POLAR_MAX_ASYNC_LEVEL}" -lt "${TMAX_MIN_ASYNC_LEVEL}" ]; then
    echo "[tmax env] raising async level ${POLAR_MAX_ASYNC_LEVEL} -> ${TMAX_MIN_ASYNC_LEVEL} to keep the rollout worker pool supplied" >&2
    export POLAR_MAX_ASYNC_LEVEL="${TMAX_MIN_ASYNC_LEVEL}"
fi
# The official 9B training backend allows 1,200 seconds of active agent work.
# This explicit train-only override is applied while rendering rollout tasks;
# eval datasets retain their own metadata.agent_timeout values unchanged.
export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS="${TMAX_TRAIN_AGENT_TIMEOUT_SECONDS:-1200}"
# Leave room outside active agent work for the 120-second TMax verifier plus
# container startup, READY queueing, postprocessing, and teardown.
export TMAX_TRAIN_TASK_TIMEOUT_RESERVE_SECONDS="${TMAX_TRAIN_TASK_TIMEOUT_RESERVE_SECONDS:-600}"
# The HTTP envelope must outlive the larger infrastructure/session budget.
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-3600}"
export POLAR_TASK_TIMEOUT_FLOOR_SECONDS="${POLAR_TASK_TIMEOUT_FLOOR_SECONDS:-1800}"
export TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS:-1800}"
if [ "${TMAX_ENABLE_GRACEFUL_EXIT}" = "1" ] && {
   ! [[ "${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS}" =~ ^[0-9]+$ ]] ||
   ! [[ "${TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS}" =~ ^[0-9]+$ ]] ||
   [ "${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS}" -lt "${TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS}" ];
}; then
    echo "ERROR: TMAX_GRACEFUL_EXIT_BUFFER_SECONDS must be at least ${TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS}" >&2
    echo "  The buffer must cover an in-flight rollout plus the synchronous distributed checkpoint." >&2
    return 1 2>/dev/null || exit 1
fi
# Match the released TMax recipe: finish all 32 trajectories in a prompt group,
# then keep only groups with non-zero reward variance.  A positive completion
# fraction enables Polar's local straggler cancellation and is therefore an
# explicit throughput experiment rather than the paper-faithful default.
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0}"
export POLAR_EARLY_STOP_GRACE_SESSIONS="${POLAR_EARLY_STOP_GRACE_SESSIONS:-0}"
# Deliberately use `-` rather than `:-` so a persisted empty value can disable
# dynamic sampling for a legacy run without being replaced by the new default.
export TMAX_DYNAMIC_SAMPLING_FILTER_PATH="${TMAX_DYNAMIC_SAMPLING_FILTER_PATH-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU="${TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU:-16}"
export TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU="${TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU:-16}"
export TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU="${TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU:-8}"
export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-$((ROLLOUT_NUM_GPUS * 2))}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-$((ROLLOUT_NUM_GPUS * TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU))}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-$((ROLLOUT_NUM_GPUS * TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU))}"
# Popen creation runs off the gateway event loop, but concurrent Apptainer
# clone/exec handshakes still create host-level fork pressure. Live TMax load
# needs about seven spawns/s; two admitted spawns sustain roughly twice that
# rate while keeping bursts bounded.
export POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY="${POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY:-2}"
# Completion records arrive in bursts when many concurrent agent turns finish.
# Sixteen independent writers keep Lustre metadata operations off the gateway
# loop; the queue covers more than two complete 576x20-turn waves before its
# lossless backpressure path activates.
export POLAR_COMPLETION_QUEUE_SIZE="${POLAR_COMPLETION_QUEUE_SIZE:-32768}"
export POLAR_COMPLETION_WRITE_WORKERS="${POLAR_COMPLETION_WRITE_WORKERS:-16}"
export POLAR_COMPLETION_BATCH_SIZE="${POLAR_COMPLETION_BATCH_SIZE:-16}"
export POLAR_COMPLETION_WRITE_MAX_ATTEMPTS="${POLAR_COMPLETION_WRITE_MAX_ATTEMPTS:-3}"
export POLAR_COMPLETION_RETRY_BACKOFF_SECONDS="${POLAR_COMPLETION_RETRY_BACKOFF_SECONDS:-0.1}"

_tmax_validate_async_capacity() {
    local name value active_sessions min_active_sessions min_run_workers min_postrun_workers
    if ! [[ "${POLAR_APPTAINER_PERSISTENT_BROKER}" =~ ^[01]$ ]]; then
        echo "ERROR: POLAR_APPTAINER_PERSISTENT_BROKER must be 0 or 1, got ${POLAR_APPTAINER_PERSISTENT_BROKER}" >&2
        return 1
    fi
    for name in \
        TMAX_TRAIN_AGENT_TIMEOUT_SECONDS TMAX_TRAIN_TASK_TIMEOUT_RESERVE_SECONDS \
        POLAR_REQUEST_TIMEOUT POLAR_TASK_TIMEOUT_FLOOR_SECONDS \
        TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU \
        TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU \
        TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU \
        POLAR_MAX_INIT_WORKERS POLAR_MAX_RUN_WORKERS POLAR_MAX_POSTRUN_WORKERS \
        POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY \
        POLAR_APPTAINER_BROKER_START_CONCURRENCY \
        POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY \
        POLAR_APPTAINER_BROKER_START_TIMEOUT_SEC \
        POLAR_APPTAINER_DIRECT_EXEC_RETRIES; do
        value="${!name}"
        if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: ${name} must be a positive integer, got ${value}" >&2
            return 1
        fi
    done
    if ! [[ "${POLAR_EARLY_STOP_GRACE_SESSIONS}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: POLAR_EARLY_STOP_GRACE_SESSIONS must be a non-negative integer" >&2
        return 1
    fi
    if [ "${POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY}" -gt 1024 ]; then
        echo "ERROR: POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY must be at most 1024" >&2
        return 1
    fi
    if [ "${POLAR_APPTAINER_BROKER_START_CONCURRENCY}" -gt 64 ]; then
        echo "ERROR: POLAR_APPTAINER_BROKER_START_CONCURRENCY must be at most 64" >&2
        return 1
    fi
    if [ "${POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY}" -gt 64 ]; then
        echo "ERROR: POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY must be at most 64" >&2
        return 1
    fi
    for name in \
        POLAR_APPTAINER_BROKER_RETRY_BACKOFF_SEC \
        POLAR_APPTAINER_BROKER_RETRY_BACKOFF_MAX_SEC; do
        value="${!name}"
        if ! [[ "$value" =~ ^[0-9]+$ ]]; then
            echo "ERROR: ${name} must be a non-negative integer, got ${value}" >&2
            return 1
        fi
    done
    if [ "${POLAR_APPTAINER_BROKER_RETRY_BACKOFF_MAX_SEC}" -lt \
         "${POLAR_APPTAINER_BROKER_RETRY_BACKOFF_SEC}" ]; then
        echo "ERROR: POLAR_APPTAINER_BROKER_RETRY_BACKOFF_MAX_SEC must be at least POLAR_APPTAINER_BROKER_RETRY_BACKOFF_SEC" >&2
        return 1
    fi
    if [ "${POLAR_REQUEST_TIMEOUT}" -lt "${POLAR_TASK_TIMEOUT_FLOOR_SECONDS}" ]; then
        echo "ERROR: POLAR_REQUEST_TIMEOUT must be at least POLAR_TASK_TIMEOUT_FLOOR_SECONDS" >&2
        return 1
    fi
    if [ "${POLAR_TASK_TIMEOUT_FLOOR_SECONDS}" -lt \
         "$((TMAX_TRAIN_AGENT_TIMEOUT_SECONDS + TMAX_TRAIN_TASK_TIMEOUT_RESERVE_SECONDS))" ]; then
        echo "ERROR: POLAR_TASK_TIMEOUT_FLOOR_SECONDS must cover TMAX_TRAIN_AGENT_TIMEOUT_SECONDS plus TMAX_TRAIN_TASK_TIMEOUT_RESERVE_SECONDS" >&2
        return 1
    fi

    if [ "${POLAR_FULLY_ASYNC}" != "true" ]; then
        return 0
    fi
    active_sessions="$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT * POLAR_MAX_ASYNC_LEVEL))"
    min_active_sessions="$((ROLLOUT_NUM_GPUS * TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU))"
    min_run_workers="$((ROLLOUT_NUM_GPUS * TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU))"
    min_postrun_workers="$((ROLLOUT_NUM_GPUS * TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU))"
    if [ "$active_sessions" -lt "$min_active_sessions" ]; then
        echo "ERROR: fully-async capacity ${active_sessions} sessions is too small for ${ROLLOUT_NUM_GPUS} rollout GPUs; require at least ${min_active_sessions}" >&2
        echo "  Increase ROLLOUT_BATCH_SIZE or POLAR_MAX_ASYNC_LEVEL to prevent sparse decode and GPU idle time." >&2
        return 1
    fi
    if [ "${POLAR_MAX_RUN_WORKERS}" -lt "$min_run_workers" ]; then
        echo "ERROR: POLAR_MAX_RUN_WORKERS=${POLAR_MAX_RUN_WORKERS} is too small for ${ROLLOUT_NUM_GPUS} rollout GPUs; require at least ${min_run_workers}" >&2
        return 1
    fi
    if [ "${POLAR_MAX_POSTRUN_WORKERS}" -lt "$min_postrun_workers" ]; then
        echo "ERROR: POLAR_MAX_POSTRUN_WORKERS=${POLAR_MAX_POSTRUN_WORKERS} is too small for ${ROLLOUT_NUM_GPUS} rollout GPUs; require at least ${min_postrun_workers}" >&2
        return 1
    fi
}
if ! _tmax_validate_async_capacity; then
    return 1 2>/dev/null || exit 1
fi
unset -f _tmax_validate_async_capacity

export TMAX_AGENT_HARNESS="${TMAX_AGENT_HARNESS:-${POLAR_AGENT_HARNESS:-mini_swe_agent}}"
export POLAR_AGENT_HARNESS="${TMAX_AGENT_HARNESS}"
if [ "${TMAX_AGENT_HARNESS}" = "spilot_router" ] || \
   [ "${TMAX_AGENT_HARNESS}" = "controller_v3" ]; then
    for _tmax_spilot_isolation_name in \
        POLAR_APPTAINER_NO_MOUNT_HOSTFS \
        POLAR_APPTAINER_NO_MOUNT_TMP \
        POLAR_APPTAINER_ISOLATE_PID \
        POLAR_APPTAINER_ISOLATE_IPC \
        POLAR_APPTAINER_CLEANENV; do
        if [ "${!_tmax_spilot_isolation_name:-}" != 1 ]; then
            echo "ERROR: formal SPilot requires ${_tmax_spilot_isolation_name}=1 at allocation startup" >&2
            return 1 2>/dev/null || exit 1
        fi
    done
    unset _tmax_spilot_isolation_name
fi
export POLAR_AGENT_MODEL_NAME="${POLAR_AGENT_MODEL_NAME:-Qwen/Qwen3.5-9B}"
# Qwen3.5's tokenizer supports interleaved reasoning natively. Explicit model
# kwargs keep that behavior through LiteLLM and the gateway; the qwen3 parser
# preserves reasoning_content across tool turns instead of flattening it into
# visible content. These values also match the released TMax rollout recipe.
export POLAR_AGENT_STEP_LIMIT="${POLAR_AGENT_STEP_LIMIT:-64}"
export POLAR_AGENT_COST_LIMIT="${POLAR_AGENT_COST_LIMIT:-0}"
export POLAR_AGENT_TEMPERATURE="${POLAR_AGENT_TEMPERATURE:-1.0}"
export POLAR_AGENT_TOP_P="${POLAR_AGENT_TOP_P:-1.0}"
export POLAR_AGENT_MAX_TOKENS="${POLAR_AGENT_MAX_TOKENS:-16384}"
export POLAR_AGENT_ENABLE_THINKING="${POLAR_AGENT_ENABLE_THINKING:-true}"
export SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"
case "${POLAR_AGENT_ENABLE_THINKING}" in
    true|false) ;;
    *)
        echo "ERROR: POLAR_AGENT_ENABLE_THINKING must be true or false" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
if ! [[ "${POLAR_AGENT_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: POLAR_AGENT_MAX_TOKENS must be a positive integer" >&2
    return 1 2>/dev/null || exit 1
fi
case "${TMAX_AGENT_HARNESS}" in
    controller_v3|mini_swe_agent|spilot_router|vanillux2)
        export POLAR_AGENT_PATH="${MINI_SWE_AGENT_CONTAINER_DIR}/bin:/opt/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        printf -v POLAR_AGENT_RUNTIME_VOLUME '        - %s:%s:ro' \
            "${MINI_SWE_AGENT_RUNTIME_DIR}" "${MINI_SWE_AGENT_CONTAINER_DIR}"
        export POLAR_AGENT_RUNTIME_VOLUME
        ;;
    codex)
        export POLAR_AGENT_PATH="/opt/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        export POLAR_AGENT_RUNTIME_VOLUME=""
        ;;
    *)
        echo "ERROR: unsupported TMAX_AGENT_HARNESS=${TMAX_AGENT_HARNESS}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

_tmax_validate_spilot_episode_admission() {
    local name value expected_qwen expected_gpt expected_agent expected_task
    local expected_request wall_seconds required_buffer_seconds required_wall_seconds
    local runtime_gateway_count
    if [ "${TMAX_AGENT_HARNESS}" != "spilot_router" ]; then
        return 0
    fi

    runtime_gateway_count=1
    if [ "${POLAR_MULTI_GATEWAY}" = "1" ]; then
        runtime_gateway_count="${POLAR_GATEWAY_COUNT_OVERRIDE:-${NUM_NODES}}"
    fi

    # Generic/legacy SPilot entrypoints fail safe to the pre-admission
    # behavior. The canonical SPilot wrapper opts in explicitly and persists
    # the complete contract in run state.
    export SPILOT_EPISODE_ADMISSION_ENABLED="${SPILOT_EPISODE_ADMISSION_ENABLED:-false}"
    export SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT="${SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT:-${runtime_gateway_count}}"
    if ! [[ "${SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT must be a positive integer" >&2
        return 1
    fi
    case "${SPILOT_EPISODE_ADMISSION_ENABLED}" in
        true)
            for name in \
                SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS \
                SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT \
                SPILOT_QWEN_MAX_ACTIVE_EPISODES \
                SPILOT_GPT_MAX_ACTIVE_EPISODES \
                SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES \
                SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES \
                SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY \
                SPILOT_GPT_GATEWAY_MAX_CONCURRENCY \
                SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES \
                SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES; do
                value="${!name:-}"
                if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
                    echo "ERROR: ${name} must be a positive integer when SPilot episode admission is enabled" >&2
                    return 1
                fi
            done
            if [ "${SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS}" -gt 86400 ]; then
                echo "ERROR: SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS must be at most 86400" >&2
                return 1
            fi
            if [ "${SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT}" -ne "${runtime_gateway_count}" ]; then
                echo "ERROR: SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT must equal runtime gateway count ${runtime_gateway_count}" >&2
                return 1
            fi
            if [ $((SPILOT_QWEN_MAX_ACTIVE_EPISODES % SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT)) -ne 0 ] || \
               [ $((SPILOT_GPT_MAX_ACTIVE_EPISODES % SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT)) -ne 0 ]; then
                echo "ERROR: both SPilot aggregate model-pool caps must be divisible by all ${SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT} gateways" >&2
                return 1
            fi
            expected_qwen="$((SPILOT_QWEN_MAX_ACTIVE_EPISODES / SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT))"
            expected_gpt="$((SPILOT_GPT_MAX_ACTIVE_EPISODES / SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT))"
            if [ "${SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES}" -ne "${expected_qwen}" ] || \
               [ "${SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES}" -ne "${expected_gpt}" ]; then
                echo "ERROR: SPilot per-gateway active-episode caps do not match the exact aggregate split" >&2
                return 1
            fi
            if [ "${SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY}" -ne "${expected_qwen}" ] || \
               [ "${SPILOT_GPT_GATEWAY_MAX_CONCURRENCY}" -ne "${expected_gpt}" ]; then
                echo "ERROR: each SPilot pool HTTP max_concurrency must equal its local episode cap" >&2
                return 1
            fi
            if [ "${SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES}" -ne "${SPILOT_QWEN_MAX_ACTIVE_EPISODES}" ] || \
               [ "${SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES}" -ne "${SPILOT_GPT_MAX_ACTIVE_EPISODES}" ]; then
                echo "ERROR: persisted SPilot effective caps must equal the configured aggregate caps" >&2
                return 1
            fi
            # The runner's internal total remains 3,000 seconds and credits
            # measured queue time. Provider admission has its own explicit wait
            # budget and therefore must not depend on the gateway worker-pool
            # shape used by the controlled Qwen3.5 training comparison. A
            # saturated provider queue may exceed Q; that admission expiry is
            # an infrastructure-masked outcome, not a synthetic reward sample.
            expected_agent="$((3300 + SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS))"
            expected_task="$((4500 + SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS + expected_agent))"
            expected_request="$((5100 + SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS + expected_agent))"
            if [ "${TMAX_TRAIN_AGENT_TIMEOUT_SECONDS}" -ne "${expected_agent}" ] || \
               [ "${POLAR_TASK_TIMEOUT_FLOOR_SECONDS}" -ne "${expected_task}" ] || \
               [ "${POLAR_REQUEST_TIMEOUT}" -ne "${expected_request}" ]; then
                echo "ERROR: SPilot admission timeout formula requires agent/task/request=${expected_agent}/${expected_task}/${expected_request}" >&2
                return 1
            fi
            if [ "${TMAX_PROFILE_DISABLE_CHECKPOINT}" = "0" ]; then
                if [ "${PARTITION}" != backfill ]; then
                    echo "ERROR: canonical SPilot admission requires PARTITION=backfill because batch is limited to four hours" >&2
                    return 1
                fi
                # Once Slime enters its graceful window it must have enough
                # time for the longest in-flight request plus one hour reserved
                # for the final checkpoint and process teardown. The allocation
                # itself must also have room for one complete request before
                # that window.
                required_buffer_seconds="$((POLAR_REQUEST_TIMEOUT + 3600))"
                if [ "${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS}" -lt "${required_buffer_seconds}" ]; then
                    echo "ERROR: SPilot TMAX_GRACEFUL_EXIT_BUFFER_SECONDS must cover POLAR_REQUEST_TIMEOUT + 3600 (${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS} < ${required_buffer_seconds})" >&2
                    return 1
                fi
                if ! wall_seconds="$(tmax_slurm_duration_seconds "${WALL_TIME}")"; then
                    echo "ERROR: unsupported SPilot WALL_TIME=${WALL_TIME}" >&2
                    return 1
                fi
                required_wall_seconds="$((TMAX_GRACEFUL_EXIT_BUFFER_SECONDS + POLAR_REQUEST_TIMEOUT))"
                if [ "${wall_seconds}" -lt "${required_wall_seconds}" ]; then
                    echo "ERROR: SPilot WALL_TIME must cover POLAR_REQUEST_TIMEOUT + TMAX_GRACEFUL_EXIT_BUFFER_SECONDS (${wall_seconds} < ${required_wall_seconds})" >&2
                    return 1
                fi
            else
                echo "[tmax env] disposable profile: skipping durable admission wall/checkpoint reserve checks" >&2
            fi
            ;;
        false)
            export SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS="${SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS:-0}"
            export SPILOT_QWEN_MAX_ACTIVE_EPISODES="${SPILOT_QWEN_MAX_ACTIVE_EPISODES:-0}"
            export SPILOT_GPT_MAX_ACTIVE_EPISODES="${SPILOT_GPT_MAX_ACTIVE_EPISODES:-0}"
            export SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES="${SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES:-null}"
            export SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES="${SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES:-null}"
            export SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY="${SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY:-32}"
            export SPILOT_GPT_GATEWAY_MAX_CONCURRENCY="${SPILOT_GPT_GATEWAY_MAX_CONCURRENCY:-32}"
            export SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES="${SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES:-0}"
            export SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES="${SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES:-0}"
            if ! [[ "${SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
               ! [[ "${SPILOT_GPT_GATEWAY_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: disabled SPilot HTTP max_concurrency values must remain positive integers" >&2
                return 1
            fi
            if [ "${SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS}" != 0 ] || \
               [ "${SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES}" != null ] || \
               [ "${SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES}" != null ]; then
                echo "ERROR: disabled SPilot episode admission requires zero wait and null caps for both aliases" >&2
                return 1
            fi
            ;;
        *)
            echo "ERROR: SPILOT_EPISODE_ADMISSION_ENABLED must be true or false" >&2
            return 1
            ;;
    esac
}
if ! _tmax_validate_spilot_episode_admission; then
    return 1 2>/dev/null || exit 1
fi
unset -f _tmax_validate_spilot_episode_admission

# Missing SIFs are fatal by default. The dual-eval contract rejects partial
# mode: every task in the deterministic train+holdout population must have a
# usable SIF before submission.
export TMAX_ONLY_READY="${TMAX_ONLY_READY:-0}"
export TMAX_TRAIN_START_INDEX="${TMAX_TRAIN_START_INDEX:-0}"
# Optional fail-closed complement selector. The file's task names are removed
# from the complete deterministic source population before SIF readiness is
# applied, so a fixed holdout cannot leak into an all-ready training set.
export TMAX_EXCLUDE_DATA="${TMAX_EXCLUDE_DATA:-}"
export TMAX_MAX_TASKS="${TMAX_MAX_TASKS:-14501}"
export TMAX_EVAL_ENABLED="${TMAX_EVAL_ENABLED:-1}"
# Keep the holdout/data-integrity contract independent from whether this
# training run schedules baseline, periodic, or final evals.
export TMAX_TRAINING_EVAL_ENABLED="${TMAX_TRAINING_EVAL_ENABLED:-${TMAX_EVAL_ENABLED}}"
export TMAX_EVAL_SOURCE="${TMAX_EVAL_SOURCE:-tmax}"
if [[ "${TMAX_TRAIN_START_INDEX}" =~ ^[0-9]+$ ]] && \
   [[ "${TMAX_MAX_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    _tmax_default_eval_start="$((TMAX_TRAIN_START_INDEX + TMAX_MAX_TASKS))"
else
    _tmax_default_eval_start=0
fi
export TMAX_EVAL_START_INDEX="${TMAX_EVAL_START_INDEX:-${_tmax_default_eval_start}}"
unset _tmax_default_eval_start
export TMAX_EVAL_MAX_TASKS="${TMAX_EVAL_MAX_TASKS:-100}"
export TMAX_EVAL_DATASET_NAME="${TMAX_EVAL_DATASET_NAME:-tmax_holdout}"
export TMAX_EVAL_SAMPLES_PER_PROMPT="${TMAX_EVAL_SAMPLES_PER_PROMPT:-1}"
if [[ "${TMAX_EVAL_START_INDEX}" =~ ^[0-9]+$ ]] && \
   [[ "${TMAX_EVAL_MAX_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    _tmax_default_total_tasks="$((TMAX_EVAL_START_INDEX + TMAX_EVAL_MAX_TASKS))"
else
    _tmax_default_total_tasks=14601
fi
export TMAX_TOTAL_TASKS="${TMAX_TOTAL_TASKS:-${_tmax_default_total_tasks}}"
unset _tmax_default_total_tasks
if [ -z "${TMAX_REQUIRE_EXACT_TOTAL_TASKS+x}" ]; then
    if [ "${TMAX_TOTAL_TASKS}" = "14601" ]; then
        TMAX_REQUIRE_EXACT_TOTAL_TASKS=1
    else
        # A shorter temporary window (for example the first built 1,000 SIFs)
        # is still selected exactly, but may live inside the full 14,601-task
        # source tree while the remaining SIFs are being built.
        TMAX_REQUIRE_EXACT_TOTAL_TASKS=0
    fi
fi
export TMAX_REQUIRE_EXACT_TOTAL_TASKS

# Terminal-Bench 2.0 is a second, independently reported eval dataset. Keeping
# TMAX_EVAL_* for the TMax holdout preserves old single-eval launch contracts
# and makes a 900/100 run a three-variable override.
if [ -z "${TMAX_EXTERNAL_EVAL_ENABLED+x}" ]; then
    if [ "${TMAX_EVAL_ENABLED}" = "1" ] && [ "${TMAX_EVAL_SOURCE}" = "tmax" ]; then
        TMAX_EXTERNAL_EVAL_ENABLED=1
    else
        TMAX_EXTERNAL_EVAL_ENABLED=0
    fi
fi
export TMAX_EXTERNAL_EVAL_ENABLED
export TMAX_EXTERNAL_EVAL_SOURCE="${TMAX_EXTERNAL_EVAL_SOURCE:-harbor}"
export TMAX_EXTERNAL_EVAL_MAX_TASKS="${TMAX_EXTERNAL_EVAL_MAX_TASKS:-89}"
export TMAX_EXTERNAL_EVAL_DATASET_NAME="${TMAX_EXTERNAL_EVAL_DATASET_NAME:-terminal_bench_2_0}"
# Periodic eval uses one attempt per task so it does not consume five complete
# Terminal-Bench passes at every checkpoint. For a paper-comparable final
# evaluation, explicitly set this to 5. Every accounted attempt contributes
# one vote to the aggregate reward metric.
export TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT="${TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT:-1}"
export TMAX_EXTERNAL_EVAL_TEMPERATURE="${TMAX_EXTERNAL_EVAL_TEMPERATURE:-0.7}"
export TMAX_EXTERNAL_EVAL_TOP_P="${TMAX_EXTERNAL_EVAL_TOP_P:-0.95}"
export TMAX_EXTERNAL_EVAL_MAX_RESPONSE_LEN="${TMAX_EXTERNAL_EVAL_MAX_RESPONSE_LEN:-16384}"
if [ -z "${TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES+x}" ]; then
    TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES="$((TMAX_EXTERNAL_EVAL_MAX_TASKS * TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT))"
fi
export TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES
# Cluster-local, revision-pinned Terminal-Bench 2.0 assets. Harbor's immutable
# ``terminal-bench@2.0`` registry entry pins all 89 tasks to the commit below;
# do not substitute the repository's moving main branch or TB2.1 task files.
# Both paths remain overrideable so another Harbor dataset can use the same
# generic preparer.
export TMAX_HARBOR_EVAL_TASKS_DIR="${TMAX_HARBOR_EVAL_TASKS_DIR:-${POLAR_DATA_ROOT}/benchmarks/terminal-bench-2.0/terminal-bench}"
export TMAX_HARBOR_EVAL_IMAGE_DIR="${TMAX_HARBOR_EVAL_IMAGE_DIR:-${POLAR_DATA_ROOT}/benchmarks/terminal-bench-2.0/enroot-images}"
export TMAX_HARBOR_EVAL_REVISION="${TMAX_HARBOR_EVAL_REVISION:-terminal-bench@2.0@69671fbaac6d67a7ef0dfec016cc38a64ef7a77c}"
export TMAX_HARBOR_EVAL_AGENT_TIMEOUT_CAP="${TMAX_HARBOR_EVAL_AGENT_TIMEOUT_CAP:-900}"
export TMAX_HARBOR_EVAL_VERIFIER_TIMEOUT_CAP="${TMAX_HARBOR_EVAL_VERIFIER_TIMEOUT_CAP:-600}"
export TMAX_HARBOR_EVAL_TIMEOUT_OVERHEAD="${TMAX_HARBOR_EVAL_TIMEOUT_OVERHEAD:-120}"
if [ -z "${TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT+x}" ]; then
    TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT="${TMAX_HARBOR_EVAL_AGENT_STEP_LIMIT:-64}"
fi
export TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT
# Backward-compatible generic Harbor preparation knob. The external dataset's
# eval-config metadata override remains authoritative if both are supplied.
export TMAX_HARBOR_EVAL_AGENT_STEP_LIMIT="${TMAX_HARBOR_EVAL_AGENT_STEP_LIMIT:-${TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT}}"
# A very large interval plus Slime's mandatory final-iteration trigger yields
# exactly two comparable points: the pre-train baseline and final model.
export TMAX_EVAL_INTERVAL="${TMAX_EVAL_INTERVAL:-1000000}"
if [ -z "${TMAX_EVAL_MIN_VALID_SAMPLES+x}" ]; then
    if [ "${TMAX_EVAL_ENABLED}" = "1" ] && \
       [[ "${TMAX_EVAL_MAX_TASKS}" =~ ^[1-9][0-9]*$ ]] && \
       [[ "${TMAX_EVAL_SAMPLES_PER_PROMPT}" =~ ^[1-9][0-9]*$ ]]; then
        TMAX_EVAL_MIN_VALID_SAMPLES="$((TMAX_EVAL_MAX_TASKS * TMAX_EVAL_SAMPLES_PER_PROMPT))"
    else
        TMAX_EVAL_MIN_VALID_SAMPLES=1
    fi
fi
export TMAX_EVAL_MIN_VALID_SAMPLES
export TMAX_EVAL_TEMPERATURE="${TMAX_EVAL_TEMPERATURE:-0.2}"
export TMAX_EVAL_TOP_P="${TMAX_EVAL_TOP_P:-1.0}"
export TMAX_EVAL_MAX_RESPONSE_LEN="${TMAX_EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}"
export TMAX_PREPARE_EVAL_DATA="${TMAX_PREPARE_EVAL_DATA:-1}"

_tmax_validate_generation_limits() {
    local name value response_limit
    for name in \
        TMAX_MODEL_MAX_CONTEXT_LENGTH SGLANG_CONTEXT_LENGTH \
        ROLLOUT_MAX_PROMPT_LEN ROLLOUT_MAX_RESPONSE_LEN \
        TMAX_MAX_TOTAL_RESPONSE_LEN TMAX_TRAIN_PACK_LENGTH \
        POLAR_AGENT_MAX_TOKENS POLAR_AGENT_STEP_LIMIT \
        TMAX_EVAL_MAX_RESPONSE_LEN TMAX_EXTERNAL_EVAL_MAX_RESPONSE_LEN \
        TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT TMAX_HARBOR_EVAL_AGENT_STEP_LIMIT; do
        value="${!name}"
        if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: ${name} must be a positive integer, got ${value}" >&2
            return 1
        fi
    done
    if [ "${SGLANG_CONTEXT_LENGTH}" -gt "${TMAX_MODEL_MAX_CONTEXT_LENGTH}" ]; then
        echo "ERROR: SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH} exceeds Qwen3.5-9B native context ${TMAX_MODEL_MAX_CONTEXT_LENGTH}" >&2
        return 1
    fi
    if [ "${TMAX_TRAIN_PACK_LENGTH}" -gt "${SGLANG_CONTEXT_LENGTH}" ]; then
        echo "ERROR: TMAX_TRAIN_PACK_LENGTH=${TMAX_TRAIN_PACK_LENGTH} exceeds SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}" >&2
        return 1
    fi
    for name in ROLLOUT_MAX_RESPONSE_LEN POLAR_AGENT_MAX_TOKENS \
        TMAX_EVAL_MAX_RESPONSE_LEN \
        TMAX_EXTERNAL_EVAL_MAX_RESPONSE_LEN; do
        response_limit="${!name}"
        if [ $((ROLLOUT_MAX_PROMPT_LEN + response_limit)) -gt "${SGLANG_CONTEXT_LENGTH}" ]; then
            echo "ERROR: ROLLOUT_MAX_PROMPT_LEN + ${name} exceeds SGLANG_CONTEXT_LENGTH (${ROLLOUT_MAX_PROMPT_LEN} + ${response_limit} > ${SGLANG_CONTEXT_LENGTH})" >&2
            return 1
        fi
    done
    for name in TMAX_EVAL_TEMPERATURE TMAX_EXTERNAL_EVAL_TEMPERATURE; do
        value="${!name}"
        if ! [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
           ! awk -v value="$value" 'BEGIN { exit !(value >= 0) }'; then
            echo "ERROR: ${name} must be a finite non-negative number, got ${value}" >&2
            return 1
        fi
    done
    for name in TMAX_EVAL_TOP_P TMAX_EXTERNAL_EVAL_TOP_P; do
        value="${!name}"
        if ! [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
           ! awk -v value="$value" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
            echo "ERROR: ${name} must be in (0, 1], got ${value}" >&2
            return 1
        fi
    done
}
if ! _tmax_validate_generation_limits; then
    return 1 2>/dev/null || exit 1
fi
unset -f _tmax_validate_generation_limits

_tmax_validate_dataset_split() {
    local name value train_end eval_end eval_total complement_mode=0
    for name in TMAX_TRAIN_START_INDEX TMAX_EVAL_START_INDEX; do
        value="${!name}"
        if ! [[ "$value" =~ ^[0-9]+$ ]]; then
            echo "ERROR: ${name} must be a non-negative integer, got ${value}" >&2
            return 1
        fi
    done
    if ! [[ "${TMAX_MAX_TASKS}" =~ ^[1-9][0-9]*$|^-1$ ]]; then
        echo "ERROR: TMAX_MAX_TASKS must be a positive integer or -1, got ${TMAX_MAX_TASKS}" >&2
        return 1
    fi
    if [ -n "${TMAX_EXCLUDE_DATA}" ]; then
        complement_mode=1
        if [ "${TMAX_TRAIN_START_INDEX}" -ne 0 ] || [ "${TMAX_MAX_TASKS}" != "-1" ]; then
            echo "ERROR: TMAX_EXCLUDE_DATA requires TMAX_TRAIN_START_INDEX=0 and TMAX_MAX_TASKS=-1" >&2
            return 1
        fi
        if [ "${TMAX_EVAL_ENABLED}" != "1" ] || [ "${TMAX_EVAL_SOURCE}" != "tmax" ]; then
            echo "ERROR: TMAX_EXCLUDE_DATA requires an enabled TMax eval dataset" >&2
            return 1
        fi
        if [ "${TMAX_EXTERNAL_EVAL_ENABLED}" != "0" ]; then
            echo "ERROR: TMAX_EXCLUDE_DATA complement mode requires offline external benchmarks" >&2
            return 1
        fi
        if [ ! -s "${TMAX_EXCLUDE_DATA}" ]; then
            echo "ERROR: TMAX_EXCLUDE_DATA is missing or empty: ${TMAX_EXCLUDE_DATA}" >&2
            return 1
        fi
    fi
    for name in TMAX_TOTAL_TASKS TMAX_EVAL_MAX_TASKS TMAX_EVAL_INTERVAL \
        TMAX_EVAL_SAMPLES_PER_PROMPT TMAX_EVAL_MIN_VALID_SAMPLES \
        TMAX_EXTERNAL_EVAL_MAX_TASKS TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT \
        TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES; do
        value="${!name}"
        if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: ${name} must be a positive integer, got ${value}" >&2
            return 1
        fi
    done
    for name in TMAX_EVAL_ENABLED TMAX_TRAINING_EVAL_ENABLED \
        TMAX_EXTERNAL_EVAL_ENABLED \
        TMAX_REQUIRE_EXACT_TOTAL_TASKS; do
        value="${!name}"
        if ! [[ "$value" =~ ^[01]$ ]]; then
            echo "ERROR: ${name} must be 0 or 1, got ${value}" >&2
            return 1
        fi
    done
    if [ "${TMAX_TRAINING_EVAL_ENABLED}" = "1" ] && \
       [ "${TMAX_EVAL_ENABLED}" != "1" ]; then
        echo "ERROR: TMAX_TRAINING_EVAL_ENABLED=1 requires TMAX_EVAL_ENABLED=1" >&2
        return 1
    fi
    if ! [[ "${TMAX_PREPARE_EVAL_DATA}" =~ ^[01]$ ]]; then
        echo "ERROR: TMAX_PREPARE_EVAL_DATA must be 0 or 1, got ${TMAX_PREPARE_EVAL_DATA}" >&2
        return 1
    fi
    if [ "${TMAX_EVAL_ENABLED}" = "1" ]; then
        if [ "${TMAX_EVAL_SOURCE}" != "tmax" ] && [ "${TMAX_EVAL_SOURCE}" != "harbor" ]; then
            echo "ERROR: TMAX_EVAL_SOURCE must be tmax or harbor, got ${TMAX_EVAL_SOURCE}" >&2
            return 1
        fi
        if [ "${TMAX_EVAL_SOURCE}" = "tmax" ] && \
           [ "${TMAX_MAX_TASKS}" = "-1" ] && [ "${complement_mode}" -ne 1 ]; then
            echo "ERROR: TMAX_EVAL_ENABLED=1 requires either a finite training window or TMAX_EXCLUDE_DATA" >&2
            return 1
        fi
        if [ "${complement_mode}" -ne 1 ]; then
            train_end="$((TMAX_TRAIN_START_INDEX + TMAX_MAX_TASKS))"
            if [ "${TMAX_EVAL_SOURCE}" = "tmax" ] && \
               [ "${TMAX_EVAL_START_INDEX}" -lt "${train_end}" ]; then
                echo "ERROR: fixed eval window begins at ${TMAX_EVAL_START_INDEX}, before the training window ends at ${train_end}" >&2
                return 1
            fi
        fi
        if [ -z "${TMAX_EVAL_DATASET_NAME}" ]; then
            echo "ERROR: TMAX_EVAL_DATASET_NAME must be non-empty" >&2
            return 1
        fi
        eval_total="$((TMAX_EVAL_MAX_TASKS * TMAX_EVAL_SAMPLES_PER_PROMPT))"
        if [ "${TMAX_EVAL_MIN_VALID_SAMPLES}" -gt "${eval_total}" ]; then
            echo "ERROR: TMAX_EVAL_MIN_VALID_SAMPLES=${TMAX_EVAL_MIN_VALID_SAMPLES} exceeds configured eval sample count ${eval_total}" >&2
            return 1
        fi
    fi
    if [ "${TMAX_EXTERNAL_EVAL_ENABLED}" = "1" ]; then
        if [ "${TMAX_EVAL_ENABLED}" != "1" ]; then
            echo "ERROR: TMAX_EXTERNAL_EVAL_ENABLED=1 requires TMAX_EVAL_ENABLED=1" >&2
            return 1
        fi
        if [ "${TMAX_EVAL_SOURCE}" != "tmax" ] || \
           [ "${TMAX_EXTERNAL_EVAL_SOURCE}" != "harbor" ]; then
            echo "ERROR: dual eval requires TMAX source=tmax plus external source=harbor" >&2
            return 1
        fi
        if [ "${TMAX_ONLY_READY}" != "0" ]; then
            echo "ERROR: the full TMax train/holdout contract forbids TMAX_ONLY_READY=1" >&2
            return 1
        fi
        if [ "${TMAX_TRAIN_START_INDEX}" -ne 0 ]; then
            echo "ERROR: the full TMax split must begin at deterministic index 0" >&2
            return 1
        fi
        train_end="$((TMAX_TRAIN_START_INDEX + TMAX_MAX_TASKS))"
        eval_end="$((TMAX_EVAL_START_INDEX + TMAX_EVAL_MAX_TASKS))"
        if [ "${TMAX_EVAL_START_INDEX}" -ne "${train_end}" ] || \
           [ "${eval_end}" -ne "${TMAX_TOTAL_TASKS}" ]; then
            echo "ERROR: TMax split must cover exactly ${TMAX_TOTAL_TASKS} tasks without a gap: train=[0,${train_end}), holdout=[${TMAX_EVAL_START_INDEX},${eval_end})" >&2
            return 1
        fi
        if [ "${TMAX_EXTERNAL_EVAL_DATASET_NAME}" = "${TMAX_EVAL_DATASET_NAME}" ] || \
           [ -z "${TMAX_EXTERNAL_EVAL_DATASET_NAME}" ]; then
            echo "ERROR: TMax holdout and external eval dataset names must be non-empty and distinct" >&2
            return 1
        fi
        eval_total="$((TMAX_EXTERNAL_EVAL_MAX_TASKS * TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT))"
        if [ "${TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES}" -gt "${eval_total}" ]; then
            echo "ERROR: TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES=${TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES} exceeds configured external eval sample count ${eval_total}" >&2
            return 1
        fi
    fi
}
if ! _tmax_validate_dataset_split; then
    return 1 2>/dev/null || exit 1
fi
unset -f _tmax_validate_dataset_split
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-tmax-mini-swe-qwen35-9b-4n-full-async}"
export RUN_ID="${RUN_ID:-${EXPERIMENT_NAME}-$(date -u +%Y%m%dT%H%M%SZ)}"
export JOB_NAME="${JOB_NAME:-polar-tmax-${RUN_ID}}"
export TMAX_TRAIN_DATA="${TMAX_TRAIN_DATA:-${POLAR_DATA_ROOT}/runs/${RUN_ID}/tmax-train.jsonl}"
export TMAX_EVAL_DATA="${TMAX_EVAL_DATA:-${POLAR_DATA_ROOT}/runs/${RUN_ID}/${TMAX_EVAL_DATASET_NAME}-eval.jsonl}"
export TMAX_EXTERNAL_EVAL_DATA="${TMAX_EXTERNAL_EVAL_DATA:-${POLAR_DATA_ROOT}/runs/${RUN_ID}/${TMAX_EXTERNAL_EVAL_DATASET_NAME}-eval.jsonl}"
export TMAX_EVAL_CONFIG_PATH="${TMAX_EVAL_CONFIG_PATH:-${POLAR_DATA_ROOT}/runs/${RUN_ID}/tmax-eval-config.json}"
export TMAX_DATA_INTEGRITY_MANIFEST="${TMAX_DATA_INTEGRITY_MANIFEST:-${POLAR_DATA_ROOT}/runs/${RUN_ID}/tmax-data-integrity.json}"
export SAVE_DIR="${SAVE_DIR:-${POLAR_DATA_ROOT}/ckpt/${RUN_ID}}"
export TRAINING_COMPLETE_MARKER="${TRAINING_COMPLETE_MARKER:-${SAVE_DIR}/TRAINING_COMPLETE}"
export FINAL_EVAL_COMPLETE_MARKER="${FINAL_EVAL_COMPLETE_MARKER:-${SAVE_DIR}/FINAL_EVAL_COMPLETE}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/tmax_slime_grpo/current_run.env}"
export TMAX_SUBMIT_RECEIPT_FILE="${TMAX_SUBMIT_RECEIPT_FILE:-${POLAR_DATA_ROOT}/runs/${RUN_ID}/submit/last_submission.env}"
# All SPilot-repo experiments log to the shared SPilot project by default;
# in-flight logical runs keep their serialized project from run state.
export WANDB_PROJECT="${WANDB_PROJECT:-SPilot}"
export WANDB_GROUP="${WANDB_GROUP:-tmax-mini-swe-qwen35-9b-full-async-8t24r}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_ALWAYS_USE_TRAIN_STEP="${WANDB_ALWAYS_USE_TRAIN_STEP:-1}"
export TMAX_REQUIRE_WANDB="${TMAX_REQUIRE_WANDB:-1}"
export TMAX_TRAIN_ABI_PREFLIGHT="${TMAX_TRAIN_ABI_PREFLIGHT:-1}"
if ! [[ "${TMAX_TRAIN_ABI_PREFLIGHT}" =~ ^[01]$ ]]; then
    echo "ERROR: TMAX_TRAIN_ABI_PREFLIGHT must be 0 or 1" >&2
    return 1 2>/dev/null || exit 1
fi
export GPU_MONITOR_ENABLED="${GPU_MONITOR_ENABLED:-1}"
export GPU_MONITOR_NODE_ROLE="${GPU_MONITOR_NODE_ROLE:-rank}"

_polar_load_export_from_zshrc() {
    local name="$1"
    local line value
    if [ -n "${!name:-}" ] || [ ! -f "$HOME/.zshrc" ]; then
        return
    fi
    line="$(grep -E "^export ${name}=" "$HOME/.zshrc" 2>/dev/null | tail -n 1 || true)"
    [ -n "$line" ] || return
    value="${line#export ${name}=}"
    eval "export ${name}=${value}"
}

_polar_load_export_from_zshrc WANDB_API_KEY
_polar_load_export_from_zshrc HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# The converted Megatron checkpoint contains trusted non-tensor metadata. PyTorch
# 2.6+ otherwise changes legacy torch.load() call sites to weights-only loading.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export https_proxy="${https_proxy:-http://cw-dfw-cs-001-container-cache:3128}"
export http_proxy="${http_proxy:-${https_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export no_proxy="${no_proxy:-${NO_PROXY:-}}"
export NO_PROXY="${NO_PROXY:-${no_proxy}}"

mkdir -p \
    "${POLAR_DATA_ROOT}/agent_cli" \
    "${POLAR_DATA_ROOT}/checkpoints" \
    "$(dirname "${MINI_SWE_AGENT_RUNTIME_DIR}")" \
    "${POLAR_DATA_ROOT}/runs" \
    "${POLAR_DATA_ROOT}/ckpt"

echo "[tmax env] nodes=${NUM_NODES} gpus/node=${SLURM_GPUS} partition=${PARTITION} no_instance=${POLAR_APPTAINER_NO_INSTANCE} network=${POLAR_SANDBOX_NETWORK}"
echo "[tmax env] dataset=${TMAX_DATASET_DIR} sif_dir=${APPTAINER_IMAGE_DIR} train_data=${TMAX_TRAIN_DATA}"
echo "[tmax env] mode=${TMAX_TRAIN_MODE} actor=${ACTOR_NUM_NODES}x${ACTOR_NUM_GPUS_PER_NODE}/tp${ACTOR_TENSOR_MODEL_PARALLEL_SIZE}/cp${CONTEXT_PARALLEL_SIZE} rollout_gpus=${ROLLOUT_NUM_GPUS}/tp${ROLLOUT_NUM_GPUS_PER_ENGINE} batch=${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT}/${NUM_STEPS_PER_ROLLOUT} fully_async=${POLAR_FULLY_ASYNC}/${POLAR_MAX_ASYNC_LEVEL} active_sessions=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT * POLAR_MAX_ASYNC_LEVEL)) run_workers=${POLAR_MAX_RUN_WORKERS}"
echo "[tmax env] harness=${TMAX_AGENT_HARNESS} runtime=${MINI_SWE_AGENT_RUNTIME_DIR}"
echo "[tmax env] train_sqsh=${POLR_TRAIN_SQSH} slime=${SLIME_DIR} ref_load=${REF_LOAD} run_id=${RUN_ID} sglang_base_port=${SLIME_ROLLOUT_BASE_PORT:-allocation-scoped}"
