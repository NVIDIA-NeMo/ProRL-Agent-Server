# Polar on SLURM

Deploy Polar agent rollout jobs on a SLURM cluster.

## Prerequisites

- SSH access to a SLURM login node
- SLURM account with GPU allocation
- Shared filesystem (Lustre, GPFS, etc.) accessible from all nodes
- Apptainer on the cluster (for running agent containers)
- Python 3.10+ on the cluster

## Quick Start

### 1. Configure

```bash
cp examples/slurm/cluster.yaml.example my-cluster.yaml
# Edit my-cluster.yaml with your cluster details
```

Key fields to set:
- `slurm.login_node` — SSH hostname of your login node
- `slurm.account` — your SLURM account string
- `slurm.partition` — SLURM partition name
- `paths.workspace` — base directory on shared filesystem

### 2. One-Time Setup

```bash
polar cluster setup -c my-cluster.yaml
```

This syncs code to the cluster and:
- Creates directories (sif_images/, results/, apptainer_cache/)
- Creates a Python venv and installs Polar
- Verifies Apptainer and CUDA are accessible

### 3. Build SIF Images

```bash
# Build a single harness
polar cluster build-sif -c my-cluster.yaml --example calculator --harness opencode

# Build multiple harnesses
polar cluster build-sif -c my-cluster.yaml --example calculator --harness opencode,codex,swe_agent
```

### 4. Start Services

```bash
# Start vLLM + rollout + gateway (waits until ready)
polar cluster serve -c my-cluster.yaml

# Override model or resources
polar cluster serve -c my-cluster.yaml --model "Qwen/Qwen3.5-72B" --nodes 2

# Preview without submitting
polar cluster serve -c my-cluster.yaml --dry-run
```

Once ready, the command prints the job ID and a sample `submit-task` command.

### 5. Submit Tasks

```bash
# Submit against the running service (use job ID from step 4)
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example calculator --harness opencode

# Multiple harnesses can reuse the same running service
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example calculator --harness codex
```

### 6. Stop Services

```bash
scancel JOB_ID
```

### 7. One-Shot Mode (Alternative)

If you prefer a single command that starts services, runs tasks, and exits:

```bash
polar cluster launch -c my-cluster.yaml --example calculator --harness opencode
```

Services stop after tasks complete.

### 8. Monitor

```bash
# Check job status
polar cluster status -c my-cluster.yaml

# Or directly via SSH
ssh <login-node> squeue -u \$USER
```

### 9. Sync Results

```bash
# Sync everything
polar cluster sync -c my-cluster.yaml

# Sync a specific job
polar cluster sync -c my-cluster.yaml --job-id 12345

# Sync only code changes
polar cluster sync -c my-cluster.yaml --code-only
```

## CLI Commands

### `polar cluster serve`

Start vLLM + rollout + gateway services and wait until ready.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Qwen/Qwen3.5-27B | Model name for vLLM |
| `--nodes` | 1 | Number of SLURM nodes |
| `--gpus` | 8 | GPUs per node |
| `--time` | 04:00:00 | Job time limit |
| `--no-sync` | — | Skip rsync to cluster |
| `--no-wait` | — | Submit and return immediately |
| `--wait-timeout` | 600 | Seconds to wait for services |
| `--dry-run` | — | Print sbatch command only |

### `polar cluster submit-task`

Submit tasks to a running service.

| Option | Default | Description |
|--------|---------|-------------|
| `--job-id` | (required) | SLURM job ID from `serve` |
| `--example` | calculator | Task: `calculator` or `swegym` |
| `--harness` | opencode | Agent: `opencode`, `codex`, `swe_agent`, etc. |
| `--num-rollouts` | 4 | Rollouts per task |
| `--timeout-seconds` | 900 | Per-session timeout |
| `--instance-id` | — | Specific SWE-Gym instance (repeatable) |

### `polar cluster launch` (one-shot)

Starts services, runs tasks, and exits in one SLURM job.

| Option | Default | Description |
|--------|---------|-------------|
| `--example` | calculator | Task: `calculator` or `swegym` |
| `--harness` | opencode | Agent: `opencode`, `codex`, `swe_agent`, etc. |
| `--model` | Qwen/Qwen3.5-27B | Model name for vLLM |
| `--nodes` | 1 | Number of SLURM nodes |
| `--gpus` | 8 | GPUs per node |
| `--time` | 04:00:00 | Job time limit |
| `--num-rollouts` | 4 | Rollouts per task |
| `--timeout-seconds` | 900 | Per-session timeout |
| `--instance-id` | — | Specific SWE-Gym instance (repeatable) |
| `--no-sync` | — | Skip rsync to cluster |
| `--dry-run` | — | Print sbatch command only |

## Architecture

### Two-Phase (serve + submit-task)

The `serve` command submits a SLURM job that:

```
1. Discover hostnames (scontrol show hostnames)
2. Generate topology.yaml
3. Start vLLM server (GPU, ~2-5min to load model)
4. Start rollout service (CPU)
5. Start gateway node(s) (CPU, manages Apptainer containers)
6. Write .services_ready sentinel
7. Wait indefinitely (until scancel or SLURM timeout)
```

The `submit-task` command discovers the running service via the sentinel file and submits tasks.

### One-Shot (launch)

The `launch` command bundles everything into one SLURM job (steps 1-5 above + submit tasks + wait + cleanup).

For multi-node jobs:
- Node 0: vLLM + rollout + gateway
- Node 1+: additional gateway nodes (via `srun --overlap`)

## Backend Support

The cluster config supports multiple backends via the `backend` field:

| Backend | Status | Description |
|---------|--------|-------------|
| `slurm` | Supported | SSH + sbatch job submission |
| `local` | Stub | Run services locally (use example scripts) |
| `k8s`   | Planned | Kubernetes pod scheduling |
