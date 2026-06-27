#!/usr/bin/env bash
# Route A (Polar drives agents) overrides for cw-dfw / jiaruiy / nvr_lpr_llm.
# Base = current Polar + Slime v0.3.0 + SGLang 0.5.13. Sandbox = Apptainer SIF,
# nested in Pyxis via POLAR_APPTAINER_NO_INSTANCE=1 (ephemeral exec --overlay).
#
# Usage:
#   source examples/swegym_slime_grpo/env.cwdfw.sh
#   python3 examples/swegym_slime_grpo/build_sifs.py --jobs 4     # build <instance_id>.sif
#   bash    examples/swegym_slime_grpo/submit_slurm.sh            # launch training
#
# Lines marked TODO point at things YOU provision first (see launch steps).
# ──────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
USER_ROOT="$(dirname "${SPILOT_ROOT}")"
# Unified PERSISTENT data root for this project: per-task SIFs, agent-CLI tree,
# converted weights, runs/checkpoints. (Ephemeral build scratch stays on node-local
# /tmp — see build_sifs.py --cache-root — so it never lands here or in the repo.)
export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"

# ── Slurm ───────────────────────────────────────────────────────────────
export ACCOUNT="${ACCOUNT:-nvr_lpr_llm}"
export PARTITION="${PARTITION:-interactive}"
export SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-H100}"
export SLURM_GPUS="${SLURM_GPUS:-8}"
export NUM_NODES="${NUM_NODES:-2}"
export WALL_TIME="${WALL_TIME:-4:00:00}"

# ── Training container ──────────────────────────────────────────────────
# RECOMMENDED (simplest): run directly in your flappydora igpu image (already has
# a WORKING system apptainer 1.5.1 + skopeo + fuse-overlayfs, verified nested) and
# reuse a training venv on lustre (mounted into the container). NO train.sqsh build.
DEFAULT_TRAIN_SQSH="${SPILOT_ROOT}/container/polar_train.sqsh"
if [ ! -f "${DEFAULT_TRAIN_SQSH}" ] && [ -f "${USER_ROOT}/spilot-router/container/polar_train.sqsh" ]; then
    DEFAULT_TRAIN_SQSH="${USER_ROOT}/spilot-router/container/polar_train.sqsh"
fi
export POLR_TRAIN_SQSH="${POLR_TRAIN_SQSH:-${DEFAULT_TRAIN_SQSH}}"
export POLR_TRAIN_VENV="${POLR_TRAIN_VENV:-${USER_ROOT}/.python/polar}"
export TRAIN_CONTAINER_MOUNTS="/lustre/fsw:/lustre/fsw"               # makes the lustre venv + SIFs visible in-container
export POLAR_APPTAINER_BIN=/usr/bin/apptainer                         # flappydora system apptainer
export POLAR_APPTAINER_NO_INSTANCE=1                                  # MUST stay 1 (nested instance:// fails here)
# OPTIONAL (reproducible, self-contained): instead bake a train.sqsh via
# build_training_sqsh.sh (BASE_SQSH=flappydora, venv->/opt/polr_venv), then set
# POLR_TRAIN_SQSH=/path/to/that.sqsh and POLR_TRAIN_VENV=/opt/polr_venv.
export BASE_SQSH="${BASE_SQSH:-docker://flappydora/ubuntu22.04-cuda13.3:latest}"

# ── Sandbox SIFs (stable scheme: <APPTAINER_IMAGE_DIR>/<instance_id>.sif) ──
# build_sifs.py writes <instance_id>.sif here (skopeo->apptainer build; pull is proxy-blocked).
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${POLAR_DATA_ROOT}/swegym_sifs}"
export AGENT_CLI_DIR="${AGENT_CLI_DIR:-${POLAR_DATA_ROOT}/agent_cli/opt_node}"

# ── Model / HF (tokens already in your ~/.zshrc) ────────────────────────
export HF_HOME="${HF_HOME:-${USER_ROOT}/.cache/huggingface}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
export REF_LOAD="${REF_LOAD:-$POLAR_DATA_ROOT/checkpoints/Qwen3.5-4B_torch_dist}"   # convert_weights.sh writes here; run.sh loads it
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-$REF_LOAD}"                                # convert_weights.sh output dir
# HF_HOME stays a shared global cache (not per-project); point it under $POLAR_DATA_ROOT if you want it self-contained.

_polar_load_export_from_zshrc() {
    local name="$1"
    local line value
    if [ -n "${!name:-}" ] || [ ! -f "$HOME/.zshrc" ]; then
        return
    fi
    line="$(grep -E "^export ${name}=" "$HOME/.zshrc" 2>/dev/null | tail -n 1 || true)"
    if [ -z "$line" ]; then
        return
    fi
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

# Bring-up profile: 2 H100 nodes total, one Megatron actor node and one
# Slime/SGLang rollout node. Override NUM_NODES=4 ACTOR_NUM_NODES=3 to scale
# back to the full 4-node profile after the 2-node run is healthy.
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-9}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export SEQ_LENGTH="${SEQ_LENGTH:-2048}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-16000}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-2}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-2400}"
export POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS:-900}"
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}"
export POLAR_APPTAINER_DIRECT_EXEC_RETRIES="${POLAR_APPTAINER_DIRECT_EXEC_RETRIES:-3}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-swegym-slime-grpo-qwen35-4b-2n8h100}"
export RUN_ID="${RUN_ID:-$EXPERIMENT_NAME}"
export SAVE_DIR="${SAVE_DIR:-$POLAR_DATA_ROOT/ckpt/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-polar-swegym-grpo}"
export WANDB_GROUP="${WANDB_GROUP:-swegym-qwen35-4b-async-grpo}"
export GPU_MONITOR_ENABLED="${GPU_MONITOR_ENABLED:-1}"

# ── Slime / Megatron ───────────────────────────────────────────────────────
# Keep the dependency line used by this repository; do not silently switch to
# hao/nrt's older Slime checkout.
export SLIME_DIR="${SLIME_DIR:-${SPILOT_ROOT}/src/slime}"
export MEGATRON_DIR="${MEGATRON_DIR:-${SPILOT_ROOT}/src/Megatron-LM}"

# Keep import/conversion helpers under per-user process limits on login and
# avoid CPU oversubscription inside Slurm GPU jobs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ── proxy (for skopeo in build_sifs.py) ─────────────────────────────────
export https_proxy="${https_proxy:-http://cw-dfw-cs-001-container-cache:3128}"
export http_proxy="${http_proxy:-$https_proxy}"

mkdir -p "$POLAR_DATA_ROOT"/swegym_sifs "$POLAR_DATA_ROOT"/agent_cli "$POLAR_DATA_ROOT"/checkpoints "$POLAR_DATA_ROOT"/runs "$POLAR_DATA_ROOT"/ckpt 2>/dev/null || true
echo "[env.cwdfw] route A | ACCOUNT=$ACCOUNT -C=$SLURM_CONSTRAINT NODES=$NUM_NODES GPUS/NODE=$SLURM_GPUS NO_INSTANCE=$POLAR_APPTAINER_NO_INSTANCE"
echo "[env.cwdfw] DATA_ROOT=$POLAR_DATA_ROOT  (SIFs/CLI/weights/runs all under here; build scratch on /tmp)"
echo "[env.cwdfw] SIF_DIR=$APPTAINER_IMAGE_DIR (<instance_id>.sif)  TRAIN_SQSH=$POLR_TRAIN_SQSH"
echo "[env.cwdfw] SLIME_DIR=$SLIME_DIR  HF_CHECKPOINT=$HF_CHECKPOINT  RUN_ID=$RUN_ID WANDB_MODE=$WANDB_MODE"
echo "[env.cwdfw] rollout batch=${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT} max_async=${POLAR_MAX_ASYNC_LEVEL} task_timeout=${POLAR_TASK_TIMEOUT_SECONDS}s max_tokens/gpu=${MAX_TOKENS_PER_GPU} apptainer_retries=${POLAR_APPTAINER_DIRECT_EXEC_RETRIES}"
