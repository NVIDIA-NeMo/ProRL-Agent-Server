# CUA Data Collection

Runs a **planner** model (Qwen3-VL-235B, tp=8) on a non-reserved GPU node and one or more **actor** nodes (UI-TARS-1.5-7B, tp=4) on reserved GPU nodes with KVM-accelerated Linux VMs for data collection. All nodes use the combined `cua-vllm-0.13.0.sqsh` container image (vLLM/CUDA + QEMU/KVM).

**Why `enroot exec`?** The srun enroot container loses `/dev/kvm` write access. Running `enroot exec` from outside the container retains it. Actor nodes use the SSH+enroot holder-job pattern for this reason.

## Automated Full Run

```bash
cd scripts
bash run.sh
```

This:
1. Submits a planner sbatch job (Qwen3-VL-235B, 8 GPUs, non-reserved)
2. Waits for the planner to start and write its hostname to a coordination file
3. Launches `NUM_ACTORS` actor instances, each submitting its own holder job on a reserved node
4. Each actor starts UI-TARS-1.5-7B vLLM + data collection VMs inside its container
5. Waits for all actors to finish, then cancels the planner

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_ACTORS` | 1 | Number of actor nodes to launch |
| `MAX_PARALLEL` | 16 | Parallel VMs per actor node |
| `MAX_TRAJECTORIES` | 10000 | Trajectories to collect per actor |

Each actor independently collects up to `MAX_TRAJECTORIES`. With `NUM_ACTORS=3, MAX_TRAJECTORIES=500`, you get up to 1500 total.

### Examples

```bash
# Single actor, defaults
bash run.sh

# 3 actors, 4 parallel VMs each, 500 trajectories each
NUM_ACTORS=3 MAX_PARALLEL=4 MAX_TRAJECTORIES=500 bash run.sh
```

### Logs

All logs go to `scripts/logs_multi_thread/`:

| File | Contents |
|------|----------|
| `planner-<jobid>.out` | Planner vLLM server output |
| `actor_launcher_<N>.log` | Actor N lifecycle (job submission, container polling, SSH exec) |
| `actor_<N>-<jobid>.out` | Actor N vLLM + data collection output |
| `vllm_actor_<N>.log` | Actor N vLLM server detailed logs |

### SLURM Configuration

| Component | Account | Partition | Reservation |
|-----------|---------|-----------|-------------|
| Planner | `nvr_lacr_llm` | `interactive` | none |
| Actor | `llmservice_fm_vision` | `interactive` | `sla_res_osworld_agent_vlm` |

## Interactive Debugging

### 1. Start the Planner vLLM Server

In a separate terminal, submit the planner on its own 8-GPU node:

```bash
IMAGE="/lustre/fsw/portfolios/nvr/users/bcui/images/cua-vllm-0.13.0.sqsh"
PLANNER_MODEL="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking"

srun --job-name=planner \
    --account=nvr_lacr_llm \
    --partition=interactive \
    --gpus-per-node=8 \
    --nodes=1 --ntasks-per-node=1 \
    --time=04:00:00 --exclusive \
    --container-image=$IMAGE \
    --container-mounts=/lustre:/lustre \
    bash -c "vllm serve $PLANNER_MODEL \
        --api-key gen \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --limit-mm-per-prompt.video 0 \
        --limit-mm-per-prompt.image 3 \
        --async-scheduling \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.9"
```

Note which node it lands on via `squeue -u $USER` — you'll need this as `$PLANNER_NODE` later (e.g. `pool0-2838`).

### 2. Get an Interactive Shell with KVM

```bash
cd scripts
bash debug_interactive.sh
```

This allocates a GPU interactive node (8 GPUs) with the combined container image. The script:
1. Submits a background `sleep infinity` job to reserve the node
2. Waits for the enroot container to be ready
3. SSHs into the node and runs `enroot exec` to enter the container

When you exit the shell (`exit`), the script automatically cancels the SLURM job via a cleanup trap.

### 3. Start the Actor vLLM Server

Inside the container shell, start UI-TARS-1.5-7B in the background:

```bash
vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
    --api-key gen \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt.image 5 \
    --limit-mm-per-prompt.video 0 \
    --max-model-len 65536 &
```

Wait for it to be healthy:
```bash
curl http://localhost:8000/health
```

### 4. Run Debug Data Collection

Still inside the container:

```bash
cd /lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server/cua
source cua_env_reqs/bin/activate
export PYTHONPATH=/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server:$PYTHONPATH

python debug_collect_trajectories.py \
    --planner_node $PLANNER_NODE \
    --actor_node localhost
```

`$PLANNER_NODE` is the node from step 1. The actor is `localhost` since it's running on the same node.

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `run.sh` | Multi-actor launcher (1 planner + N actors) |
| `run_planner.sbatch` | Planner vLLM server sbatch job |
| `run_actor_and_vm.sh` | Single actor launcher (SSH+enroot pattern) |
| `run_all.sbatch` | Legacy consolidated 2-node sbatch (kept for reference) |
| `debug_interactive.sh` | Interactive GPU shell with KVM |
| `debug_check_kvm.sbatch` | Verify KVM works on GPU nodes (inside container) |
| `check_kvm_cpu.sbatch` | Verify KVM on CPU nodes (legacy) |
| `check_kvm_bash.sh` | Quick KVM write-permission test |
| `run_models.sbatch` | Start both model servers on 2 GPU nodes (standalone) |
