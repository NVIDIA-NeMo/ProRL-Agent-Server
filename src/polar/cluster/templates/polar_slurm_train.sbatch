#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────────
# Polar SLURM Training Job Script
#
# Orchestrates distributed RL training: Polar services (rollout + gateway) for
# agent session management, and Slime/Ray/SGLang/Megatron for GRPO training.
#
# GPU layout (single trainer node, 8 GPUs):
#   GPU 0..ROLLOUT_GPUS-1  – SGLang inference (Slime/Ray-managed, weight-synced)
#   GPU ROLLOUT_GPUS..7     – Megatron GRPO training (TP=TP_SIZE, DP=auto)
#
# Polar services (CPU-only) run on the host using POLAR_VENV.
# Training commands run inside the training SIF via apptainer exec.
#
# Required env vars (pass via --export):
#   POLAR_CODE         – Path to ProRL-Agent-Server on cluster
#   POLAR_WORKSPACE    – Base workspace path on cluster
#   POLAR_CONFIG_PATH  – Path to polar_config.yaml (bridge config)
#   PROMPT_DATA        – Path to JSONL training data
#   HF_CHECKPOINT      – HuggingFace model checkpoint name
#   TRAIN_SIF          – Path to training SIF image
# ────────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ -z "${POLAR_CODE:-}" ]; then
    echo "FATAL: POLAR_CODE not set." >&2; exit 1
fi
if [ -z "${POLAR_WORKSPACE:-}" ]; then
    echo "FATAL: POLAR_WORKSPACE not set." >&2; exit 1
fi

# Source environment setup
TEMPLATE_DIR="${POLAR_CODE}/src/polar/cluster/templates"
source "${TEMPLATE_DIR}/env.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
TP_SIZE="${TP_SIZE:-2}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
TRAIN_NUM_ROLLOUTS="${TRAIN_NUM_ROLLOUTS:-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
TOTAL_GPUS=$((ACTOR_GPUS + ROLLOUT_GPUS))

# Training SIF — must be pre-built via 'polar cluster build-sif --example train'
TRAIN_SIF="${TRAIN_SIF:-${POLAR_SIFS}/train-slime-grpo.sif}"
if [ ! -f "${TRAIN_SIF}" ]; then
    echo "FATAL: Training SIF not found: ${TRAIN_SIF}" >&2
    echo "  Build with: polar cluster build-sif -c cluster.yaml --example train" >&2
    exit 1
fi

# HF cache on lustre (shared, persistent across jobs)
HF_HOME="${POLAR_WORKSPACE}/hf_cache"
mkdir -p "${HF_HOME}"

# Apptainer exec prefix for GPU commands.
# PYTHONNOUSERSITE prevents user-installed packages (e.g. torch in ~/.local)
# from shadowing the NGC container's system packages.
# LD_LIBRARY_PATH includes CUDA compat for forward-compatibility with older drivers.
# HF_HOME on lustre so the container can access downloaded models.
APPTAINER_EXEC="apptainer exec --nv --writable-tmpfs --no-home \
  --bind ${POLAR_WORKSPACE}:${POLAR_WORKSPACE} \
  --env PYTHONNOUSERSITE=1 \
  --env LD_LIBRARY_PATH=/usr/local/cuda/compat:\${LD_LIBRARY_PATH:-} \
  --env HF_HOME=${HF_HOME} \
  --env HF_HUB_OFFLINE=0 \
  ${TRAIN_SIF}"

# ── Job directory ─────────────────────────────────────────────────────────────
JOB_DIR="${POLAR_RESULTS}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}/logs"
echo "[polar_train] Job directory: ${JOB_DIR}"
echo "[polar_train] Job ID: ${SLURM_JOB_ID}"
echo "[polar_train] Nodes: ${SLURM_JOB_NODELIST}"
echo "[polar_train] Training SIF: ${TRAIN_SIF}"

# ── Discover hostnames ────────────────────────────────────────────────────────
HOSTNAMES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
TRAINER_NODE="${HOSTNAMES[0]}"
NUM_NODES="${#HOSTNAMES[@]}"
echo "[polar_train] Trainer node: ${TRAINER_NODE}"
echo "[polar_train] Allocated ${NUM_NODES} node(s): ${HOSTNAMES[*]}"

# ── Derived paths ─────────────────────────────────────────────────────────────
HF_CKPT_BASENAME="${HF_CHECKPOINT##*/}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${POLAR_WORKSPACE}/checkpoints/${HF_CKPT_BASENAME}_torch_dist}"
SAVE_DIR="${TRAIN_SAVE_DIR:-${JOB_DIR}/checkpoints}"
mkdir -p "${SAVE_DIR}"
export SAVE_DIR

ROLLOUT_SAVE_DIR="${JOB_DIR}/rollout_results"
mkdir -p "${ROLLOUT_SAVE_DIR}"

# ── Generate topology.yaml ────────────────────────────────────────────────────
# Points gateway at SGLang router (on trainer node), NOT at vLLM
TOPOLOGY_PATH="${JOB_DIR}/topology.yaml"
SGLANG_URL="http://${TRAINER_NODE}:${SGLANG_ROUTER_PORT}"

python -m polar.cluster.topology \
    --output "${TOPOLOGY_PATH}" \
    --save-dir "${ROLLOUT_SAVE_DIR}" \
    --sglang-base-url "${SGLANG_URL}"

export POLAR_TOPOLOGY="${TOPOLOGY_PATH}"
echo "[polar_train] Generated topology: ${TOPOLOGY_PATH}"
cat "${TOPOLOGY_PATH}"

# ── Patch polar_config.yaml with actual rollout URL ───────────────────────────
PATCHED_POLAR_CONFIG="${JOB_DIR}/polar_config.yaml"
POLAR_ROLLOUT_URL="http://${TRAINER_NODE}:${ROLLOUT_PORT}"
sed "s|polar_rollout_url:.*|polar_rollout_url: \"${POLAR_ROLLOUT_URL}\"|" \
    "${POLAR_CONFIG_PATH}" > "${PATCHED_POLAR_CONFIG}"
echo "[polar_train] Patched polar_config.yaml: polar_rollout_url → ${POLAR_ROLLOUT_URL}"

# ── PID tracking for cleanup ──────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo "[polar_train] Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[polar_train] Killing PID ${pid}"
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "[polar_train] Cleanup complete."
}
trap cleanup EXIT INT TERM

source "${TEMPLATE_DIR}/wait_for_service.sh"

# ── Step 1: Start Polar services (CPU, on host using POLAR_VENV) ──────────────
echo "[polar_train] Starting Polar rollout server on ${TRAINER_NODE}:${ROLLOUT_PORT}..."
python -m polar.cli serve_rollout -c "${TOPOLOGY_PATH}" \
    > "${JOB_DIR}/logs/rollout.log" 2>&1 &
PIDS+=($!)

wait_for_service "http://${TRAINER_NODE}:${ROLLOUT_PORT}/health" "Rollout" 30 2 || exit 1

# Start gateway(s) on all nodes
for ((i=0; i<NUM_NODES; i++)); do
    NODE_ID="node-$(printf '%02d' ${i})"
    GW_PORT=$((GATEWAY_BASE_PORT + i))
    GW_HOST="${HOSTNAMES[$i]}"

    echo "[polar_train] Starting gateway ${NODE_ID} on ${GW_HOST}:${GW_PORT}..."
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
done

for ((i=0; i<NUM_NODES; i++)); do
    GW_PORT=$((GATEWAY_BASE_PORT + i))
    GW_HOST="${HOSTNAMES[$i]}"
    wait_for_service "http://${GW_HOST}:${GW_PORT}/health" \
        "Gateway node-$(printf '%02d' ${i})" 30 2 || exit 1
done

echo "[polar_train] Polar services ready."

# ── Step 2: Convert weights if needed (inside container) ──────────────────────
# Pre-download the HF model on the host (compute nodes may lack internet,
# and the container uses --no-home so ~/.cache is not available).
if [[ ! -d "${HF_CHECKPOINT}" ]]; then
    HF_LOCAL_DIR="${POLAR_WORKSPACE}/hf_models/${HF_CKPT_BASENAME}"
    if [[ ! -d "${HF_LOCAL_DIR}" ]]; then
        echo "[polar_train] Downloading HF model: ${HF_CHECKPOINT} -> ${HF_LOCAL_DIR}"
        HF_HUB_OFFLINE=0 HF_HOME="${HF_HOME}" python -c "
from huggingface_hub import snapshot_download
snapshot_download('${HF_CHECKPOINT}', local_dir='${HF_LOCAL_DIR}')
"
        echo "[polar_train] Download complete."
    else
        echo "[polar_train] HF model already cached at ${HF_LOCAL_DIR}"
    fi
    HF_CHECKPOINT="${HF_LOCAL_DIR}"
fi

if [ ! -d "${TORCH_DIST_DIR}/release" ]; then
    echo "[polar_train] Converting HF weights: ${HF_CHECKPOINT} -> ${TORCH_DIST_DIR}"
    mkdir -p "${TORCH_DIST_DIR}"
    ${APPTAINER_EXEC} bash -c "
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export PYTHONPATH='/opt/Megatron-LM:/opt/polar/src:\${PYTHONPATH:-}'
        torchrun --nproc_per_node 1 \
            /opt/slime/tools/convert_hf_to_torch_dist.py \
            ${MODEL_ARGS} \
            --hf-checkpoint '${HF_CHECKPOINT}' \
            --save '${TORCH_DIST_DIR}' \
            --tensor-model-parallel-size 1 \
            --pipeline-model-parallel-size 1 \
            --context-parallel-size 1 \
            --expert-model-parallel-size 1 \
            --expert-tensor-parallel-size 1
    " > "${JOB_DIR}/logs/convert_weights.log" 2>&1
    echo "[polar_train] Weight conversion complete."
else
    echo "[polar_train] Weights already converted at ${TORCH_DIST_DIR}"
fi

# ── Step 3+4: Start Ray and launch training (single container session) ────────
# Ray start daemonizes processes in the container's PID/network namespace.
# ray job submit must run in the SAME apptainer exec invocation so it can
# reach the Ray dashboard via localhost.  Separate invocations create isolated
# namespaces, causing DNS/socket errors (OSError 107, Transport endpoint).
echo "[polar_train] Starting Ray + training (single container session)..."
echo "[polar_train]   Steps: ${TRAIN_NUM_ROLLOUTS}"
echo "[polar_train]   Batch: ${ROLLOUT_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} samples"
echo "[polar_train]   Global batch size: ${GLOBAL_BATCH_SIZE}"
echo "[polar_train]   Actor GPUs: ${ACTOR_GPUS} (TP=${TP_SIZE}), Rollout GPUs: ${ROLLOUT_GPUS}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/opt/Megatron-LM:/opt/polar/src\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

${APPTAINER_EXEC} bash -c "
    set -euo pipefail

    # Verify critical SGLang dependencies are importable
    echo '[polar_train] Verifying runtime dependencies...'
    python3 -c 'import openai; import pybase64; import partial_json_parser; import sglang; print(\"All SGLang deps OK\")' || {
        echo '[polar_train] ERROR: Missing SGLang dependencies in SIF. Rebuild with: polar cluster build-sif --example train --force' >&2
        exit 1
    }

    # Clean up any stale Ray processes
    ray stop --force 2>/dev/null || true
    sleep 2

    # Start Ray head node (daemonizes)
    ray start --head \
        --node-ip-address '${TRAINER_NODE}' \
        --port '${RAY_PORT}' \
        --dashboard-host 0.0.0.0 \
        --dashboard-port '${RAY_DASHBOARD_PORT}' \
        --num-gpus '${TOTAL_GPUS}' \
        --disable-usage-stats

    echo '[polar_train] Ray cluster started. Dashboard: http://${TRAINER_NODE}:${RAY_DASHBOARD_PORT}'

    # Wait for Ray dashboard to be ready
    for i in \$(seq 1 30); do
        if curl -sf http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version >/dev/null 2>&1; then
            echo '[polar_train] Ray dashboard is ready.'
            break
        fi
        echo '[polar_train] Waiting for Ray dashboard... ('\$i'/30)'
        sleep 2
    done

    # Submit training job (uses localhost since we're in the same namespace)
    export PYTHONPATH='/opt/Megatron-LM:/opt/polar/src:\${PYTHONPATH:-}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1

    ray job submit \
        --address='http://127.0.0.1:${RAY_DASHBOARD_PORT}' \
        --runtime-env-json='${RUNTIME_ENV_JSON}' \
        -- python3 /opt/slime/train_async.py \
        --actor-num-nodes 1 \
        --actor-num-gpus-per-node '${ACTOR_GPUS}' \
        --rollout-num-gpus '${ROLLOUT_GPUS}' \
        --rollout-num-gpus-per-engine 1 \
        ${MODEL_ARGS} \
        --hf-checkpoint '${HF_CHECKPOINT}' \
        --ref-load '${TORCH_DIST_DIR}' \
        --load '${SAVE_DIR}' \
        --save '${SAVE_DIR}' \
        --save-interval 5 \
        --update-weights-interval 1 \
        --rollout-function-path polar.slime.rollout.generate_rollout_polar_async \
        --custom-rm-path polar.slime.reward.reward_func \
        --custom-config-path '${PATCHED_POLAR_CONFIG}' \
        --prompt-data '${PROMPT_DATA}' \
        --input-key prompt \
        --label-key label \
        --metadata-key metadata \
        --rollout-shuffle \
        --reward-key score \
        --num-rollout '${TRAIN_NUM_ROLLOUTS}' \
        --rollout-batch-size '${ROLLOUT_BATCH_SIZE}' \
        --n-samples-per-prompt '${N_SAMPLES_PER_PROMPT}' \
        --rollout-max-response-len 8192 \
        --rollout-max-prompt-len 4096 \
        --global-batch-size '${GLOBAL_BATCH_SIZE}' \
        --rollout-global-dataset \
        --disable-rollout-trim-samples \
        --tensor-model-parallel-size '${TP_SIZE}' \
        --sequence-parallel \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --recompute-granularity full \
        --recompute-method uniform \
        --recompute-num-layers 1 \
        --use-dynamic-batch-size \
        --max-tokens-per-gpu 8192 \
        --advantage-estimator external \
        --use-rollout-logprobs \
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
        --attention-backend flash \
        --no-gradient-accumulation-fusion \
        ${WANDB_PROJECT:+--use-wandb --wandb-project '${WANDB_PROJECT}'} \
        ${WANDB_EXP_NAME:+--wandb-exp-name '${WANDB_EXP_NAME}'} \
        ${WANDB_GROUP:+--wandb-group '${WANDB_GROUP}'} \
        --sglang-router-ip 0.0.0.0 \
        --sglang-router-port '${SGLANG_ROUTER_PORT}' \
        ${EXTRA_TRAIN_ARGS:-}
" 2>&1 | tee "${JOB_DIR}/logs/training.log"

TRAIN_EXIT=${PIPESTATUS[0]}

# ── Results ───────────────────────────────────────────────────────────────────
echo ""
echo "[polar_train] ════════════════════════════════════════════════════════"
echo "[polar_train] Training finished (exit code: ${TRAIN_EXIT})"
echo "[polar_train] Checkpoints: ${SAVE_DIR}"
echo "[polar_train] Rollout results: ${ROLLOUT_SAVE_DIR}"
echo "[polar_train] Logs: ${JOB_DIR}/logs/"
echo "[polar_train] ════════════════════════════════════════════════════════"

exit ${TRAIN_EXIT}
