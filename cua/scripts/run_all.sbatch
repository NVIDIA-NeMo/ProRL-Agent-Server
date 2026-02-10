#!/bin/bash
#SBATCH --job-name=cua_all
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/run_all-%j.out

# ============================================================================
# Consolidated 2-Node CUA Data Collection Script
# ============================================================================
# Node 1 (Planner): Runs Qwen3-VL-235B vLLM server (tp=8, all 8 GPUs)
# Node 2 (Actor):   Runs UI-TARS-1.5-7B vLLM server (tp=4) + data collection VMs
#
# This replaces the previous 3-node setup (2 GPU + 1 CPU) by co-locating
# VMs on the Actor node which has spare GPU/CPU capacity.
# ============================================================================

# Combined container with vLLM/CUDA + QEMU/KVM dependencies
# CONTAINER_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/cua_vllm.sqsh"
# CONTAINER_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/vllm-0.13.0.sqsh"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/bcui/images/cua-vllm-0.13.0.sqsh"

# Project paths
PROJECT_ROOT="/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server"
PROJECT_DIR="$PROJECT_ROOT/cua"

# Model paths
PLANNER_MODEL="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking"
# PLANNER_MODEL="/lustre/fsw/portfolios/nvr/users/bcui/models/qwen-3-vl-4b-thinking"
ACTOR_MODEL="ByteDance-Seed/UI-TARS-1.5-7B"

PLANNER_PORT=8000
ACTOR_PORT=8000

# Max parallel VMs (reduced from 24 to account for shared resources with vLLM)
MAX_PARALLEL=${MAX_PARALLEL:-16}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-10000}

# --- 1. Get Node List and Assign Roles ---
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
PLANNER_NODE=$(echo "$ALL_NODES" | head -n 1)
ACTOR_NODE=$(echo "$ALL_NODES" | head -n 2 | tail -n 1)

echo "[run_all] Job ID:        $SLURM_JOB_ID"
echo "[run_all] Planner Node:  $PLANNER_NODE"
echo "[run_all] Actor Node:    $ACTOR_NODE"
echo "[run_all] Max Parallel:  $MAX_PARALLEL"
echo "[run_all] Max Trajs:     $MAX_TRAJECTORIES"

# TODO: add back --enable-expert-parallel \
# --enable-expert-parallel \
# --- 2. Launch Planner vLLM Server (Node 1, all 8 GPUs) ---
srun -w "$PLANNER_NODE" \
     --nodes=1 \
     --ntasks=1 \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     --output=logs/planner.out \
     bash -c "
      echo '[Planner] Launching Qwen3-VL-235B on $PLANNER_NODE'
      vllm serve $PLANNER_MODEL \
        --api-key gen \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --limit-mm-per-prompt.video 0 \
        --limit-mm-per-prompt.image 3 \
        --async-scheduling \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.9
     " &
PLANNER_SRUN_PID=$!

# --- 3. Launch Actor vLLM + Data Collection (Node 2) ---
# The Actor node runs both the vLLM server AND the data collection VMs.
# /dev/kvm is mounted for QEMU/KVM hardware virtualization.
srun -w "$ACTOR_NODE" \
     --nodes=1 \
     --ntasks=1 \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre,/dev/kvm:/dev/kvm \
     --container-writable \
     --output=logs/actor_and_collect.out \
     bash -c "
      set -e

      echo '[Actor] Launching UI-TARS-1.5-7B on $ACTOR_NODE'
      vllm serve $ACTOR_MODEL \
        --api-key gen \
        --tensor-parallel-size 4 \
        --limit-mm-per-prompt.image 5 \
        --limit-mm-per-prompt.video 0 \
        --max-model-len 65536 &
      VLLM_PID=\$!

      # --- Health Check: Wait for both vLLM servers ---
      echo '[Actor] Waiting for vLLM servers to become healthy...'

      wait_for_server() {
          local host=\$1
          local port=\$2
          local name=\$3
          local max_wait=600  # 10 minutes
          local elapsed=0

          while [ \$elapsed -lt \$max_wait ]; do
              if curl -sf http://\${host}:\${port}/health > /dev/null 2>&1; then
                  echo \"[Actor] \$name server healthy at \${host}:\${port}\"
                  return 0
              fi
              sleep 10
              elapsed=\$((elapsed + 10))
              if [ \$((elapsed % 60)) -eq 0 ]; then
                  echo \"[Actor] Still waiting for \$name (\${elapsed}s)...\"
              fi
          done

          echo \"[Actor] ERROR: \$name server did not start within \${max_wait}s\"
          return 1
      }

      # Wait for local Actor server
      wait_for_server localhost $ACTOR_PORT Actor
      ACTOR_OK=\$?

      # Wait for remote Planner server
      wait_for_server $PLANNER_NODE $PLANNER_PORT Planner
      PLANNER_OK=\$?

      if [ \$ACTOR_OK -ne 0 ] || [ \$PLANNER_OK -ne 0 ]; then
          echo '[Actor] ERROR: Server health check failed. Aborting.'
          kill \$VLLM_PID 2>/dev/null
          exit 1
      fi

      echo '[Actor] Both servers healthy. Starting data collection...'

      # --- Launch Data Collection ---
      cd $PROJECT_DIR
      source cua_env_reqs/bin/activate
      export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH

      python parallel_collect_trajectories.py \
          --planner_node $PLANNER_NODE \
          --actor_node localhost \
          --max_parallel $MAX_PARALLEL \
          --max_trajectories $MAX_TRAJECTORIES

      COLLECT_EXIT=\$?
      echo \"[Actor] Data collection finished with exit code \$COLLECT_EXIT\"

      # Cleanup
      kill \$VLLM_PID 2>/dev/null
      exit \$COLLECT_EXIT
     " &
ACTOR_SRUN_PID=$!

# --- 4. Wait for completion ---
# The Actor srun will finish when data collection completes (or fails).
# The Planner srun runs indefinitely until we kill it.
echo "[run_all] Waiting for Actor node (data collection) to finish..."
wait $ACTOR_SRUN_PID
ACTOR_EXIT=$?

echo "[run_all] Actor node exited with code $ACTOR_EXIT. Stopping Planner..."
kill $PLANNER_SRUN_PID 2>/dev/null
wait $PLANNER_SRUN_PID 2>/dev/null

echo "[run_all] All done. Exit code: $ACTOR_EXIT"
exit $ACTOR_EXIT
