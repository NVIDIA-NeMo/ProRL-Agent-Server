#!/bin/bash
#SBATCH --job-name=cua_planner
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs_single/planner-%j.out

# ============================================================================
# Planner Node: Runs Qwen3-VL-235B vLLM server (tp=8, all 8 GPUs)
# ============================================================================
# Requires COORD_FILE env var (set by run.sh wrapper)
# Writes hostname to COORD_FILE so the actor job can discover this node.
# ============================================================================

if [ -z "$COORD_FILE" ]; then
    echo "[Planner] ERROR: COORD_FILE not set. Use run.sh or pass via --export."
    exit 1
fi

CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/bcui/images/cua-vllm-0.13.0.sqsh"
PLANNER_MODEL="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking"

# Write hostname for actor discovery
PLANNER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "$PLANNER_HOST" > "$COORD_FILE"
echo "[Planner] Wrote hostname '$PLANNER_HOST' to $COORD_FILE"

# Cleanup coordination file on exit
cleanup() {
    echo "[Planner] Cleaning up coordination file..."
    rm -f "$COORD_FILE"
}
trap cleanup EXIT

echo "[Planner] Job ID:   $SLURM_JOB_ID"
echo "[Planner] Node:     $PLANNER_HOST"

# Launch Planner vLLM server (runs in foreground until cancelled or time limit)
srun --nodes=1 \
     --ntasks=1 \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     bash -c "
      echo '[Planner] Launching Qwen3-VL-235B on $(hostname)'
      vllm serve $PLANNER_MODEL \
        --api-key gen \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --limit-mm-per-prompt.video 0 \
        --limit-mm-per-prompt.image 3 \
        --async-scheduling \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.9
     "
