# Polar on SLURM

Deploy Polar agent rollout jobs on a SLURM cluster using Apptainer containers and vLLM inference.

## Prerequisites

- SSH access to a SLURM login node
- SLURM account with GPU allocation
- Shared filesystem (Lustre, GPFS, etc.) accessible from all nodes
- Apptainer on the cluster (for running agent containers)
- Python 3.10+ on the cluster

## Install Polar

From a checkout of this repository (laptop or cluster login node):

```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -e .
polar --help
```

After `polar cluster setup`, the cluster workspace venv also provides the `polar` command.

## Quick Start

### 1. Configure

```bash
cp examples/slurm/cluster.yaml.example my-cluster.yaml
# Edit my-cluster.yaml with your cluster details
```

Key fields to set:
- `slurm.login_node` -- SSH hostname of your login node
- `slurm.account` -- your SLURM account string
- `slurm.partition` -- SLURM partition name
- `paths.workspace` -- base directory on shared filesystem

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

# Build multiple harnesses at once
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

## Example Workflows

### Calculator (Quick Validation)

Build SIFs and submit tasks for one or more harnesses:

```bash
polar cluster build-sif -c my-cluster.yaml \
    --example calculator --harness opencode,codex,swe_agent

polar cluster serve -c my-cluster.yaml

# Use the job ID printed by serve
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example calculator --harness opencode
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example calculator --harness codex
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example calculator --harness swe_agent
```

**Supported calculator harnesses:**

| Harness | API | vLLM Compatible | Status |
|---------|-----|-----------------|--------|
| opencode | OpenAI Chat | Yes | Verified |
| codex | OpenAI Responses | Yes | Verified |
| swe_agent | OpenAI Chat | Yes | Verified |
| qwen_code | OpenAI Chat | Yes | Available |
| openhands_sdk | OpenAI Chat | Yes | Available |
| claude_code | Anthropic Messages | No (needs native API) | Local only |
| gemini_cli | Google Generative AI | No (needs native API) | Local only |

### SWE-Gym (10 Curated Tasks)

Each SWE-Gym instance needs its own SIF:

```bash
# Build all 10 sample SIFs
polar cluster build-sif -c my-cluster.yaml \
    --example swegym --harness swe_agent

# Start services and submit
polar cluster serve -c my-cluster.yaml
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example swegym --harness swe_agent \
    --timeout-seconds 2400

# Or a single instance
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example swegym --harness swe_agent \
    --timeout-seconds 2400 --instance-id getmoto__moto-7365
```

Sample instances:

| Instance | Repo |
|----------|------|
| `getmoto__moto-7365` | getmoto/moto |
| `python__mypy-10392` | python/mypy |
| `conan-io__conan-13721` | conan-io/conan |
| `iterative__dvc-1809` | iterative/dvc |
| `dask__dask-10441` | dask/dask |
| `pydantic__pydantic-8072` | pydantic/pydantic |
| `pandas-dev__pandas-58335` | pandas-dev/pandas |
| `facebookresearch__hydra-1783` | facebookresearch/hydra |
| `bokeh__bokeh-13636` | bokeh/bokeh |
| `Project-MONAI__MONAI-2238` | Project-MONAI/MONAI |

### SWE-bench Verified (500 Tasks)

Full benchmark evaluation with per-instance containers:

```bash
# Cache dataset (once)
python -c "from examples.swebench_verified.dataset import load_swebench_verified; load_swebench_verified()"

# Build per-instance SIFs
polar cluster build-sif -c my-cluster.yaml \
    --example swebench_verified --harness opencode \
    --instance-id django__django-15098

# Start services and submit
polar cluster serve -c my-cluster.yaml
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example swebench_verified --harness opencode \
    --timeout-seconds 3600 --instance-id django__django-15098
```

### SWE-Gym Slime GRPO (RL Training)

Distributed RL training using Polar for rollout and Slime + Megatron for GRPO training.
See [examples/swegym_slime_grpo/README.md](../swegym_slime_grpo/README.md) for detailed setup.

```bash
# Build training SIF (uses slimerl/slime Docker image as base)
polar cluster build-sif -c my-cluster.yaml --example train

# Submit training job
polar cluster train -c my-cluster.yaml \
    --polar-config examples/swegym_slime_grpo/polar_config.yaml \
    --prompt-data examples/swegym_slime_grpo/swegym_10_tasks.jsonl \
    --num-rollouts 5
```

## CLI Commands

### `polar cluster setup`

One-time cluster initialization.

### `polar cluster build-sif`

Build Apptainer SIF images on the cluster.

| Option | Default | Description |
|--------|---------|-------------|
| `--example` | calculator | Task type: `calculator`, `swegym`, `swebench_verified`, `train` |
| `--harness` | opencode | Agent: `opencode`, `codex`, `swe_agent`, etc. |
| `--instance-id` | -- | Specific instance (repeatable, for swegym/swebench) |
| `--force` | -- | Rebuild even if SIF exists |

### `polar cluster serve`

Start vLLM + rollout + gateway services and wait until ready.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Qwen/Qwen3.5-27B | Model name for vLLM |
| `--nodes` | 1 | Number of SLURM nodes |
| `--gpus` | 8 | GPUs per node |
| `--time` | 04:00:00 | Job time limit |
| `--no-sync` | -- | Skip rsync to cluster |
| `--no-wait` | -- | Submit and return immediately |
| `--wait-timeout` | 600 | Seconds to wait for services |
| `--dry-run` | -- | Print sbatch command only |

### `polar cluster submit-task`

Submit tasks to a running service.

| Option | Default | Description |
|--------|---------|-------------|
| `--job-id` | (required) | SLURM job ID from `serve` |
| `--example` | calculator | Task: `calculator`, `swegym`, `swebench_verified` |
| `--harness` | opencode | Agent: `opencode`, `codex`, `swe_agent`, etc. |
| `--num-rollouts` | 4 | Rollouts per task |
| `--timeout-seconds` | 900 | Per-session timeout |
| `--instance-id` | -- | Specific instance (repeatable) |

### `polar cluster launch` (one-shot)

Starts services, runs tasks, and exits in one SLURM job.

| Option | Default | Description |
|--------|---------|-------------|
| `--example` | calculator | Task: `calculator`, `swegym`, `swebench_verified` |
| `--harness` | opencode | Agent: `opencode`, `codex`, `swe_agent`, etc. |
| `--model` | Qwen/Qwen3.5-27B | Model name for vLLM |
| `--nodes` | 1 | Number of SLURM nodes |
| `--gpus` | 8 | GPUs per node |
| `--time` | 04:00:00 | Job time limit |
| `--num-rollouts` | 4 | Rollouts per task |
| `--timeout-seconds` | 900 | Per-session timeout |
| `--instance-id` | -- | Specific instance (repeatable) |
| `--no-sync` | -- | Skip rsync to cluster |
| `--dry-run` | -- | Print sbatch command only |

### `polar cluster train`

Submit RL training job (Slime + Megatron GRPO).

| Option | Default | Description |
|--------|---------|-------------|
| `--polar-config` | (required) | Path to polar_config.yaml |
| `--prompt-data` | (required) | Path to JSONL training data |
| `--hf-checkpoint` | Qwen/Qwen3-4B | HuggingFace model checkpoint |
| `--num-rollouts` | 5 | Training steps |
| `--no-sync` | -- | Skip rsync to cluster |
| `--no-wait` | -- | Submit and return immediately |
| `--dry-run` | -- | Print sbatch command only |

### `polar cluster status`

Show job status and running services.

### `polar cluster sync`

Sync code and results between local and cluster.

| Option | Default | Description |
|--------|---------|-------------|
| `--results-only` | -- | Only sync results (no code) |
| `--code-only` | -- | Only sync code (no results) |
| `--job-id` | -- | Sync a specific job's results |

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

## Results Structure

```
polar-serve_<jobid>/
  topology.yaml
  .services_ready
  logs/
    vllm.log
    rollout.log
    gateway_node-00.log
  rollout_results/
    task_<id>/ses_<id>.json
  tasks/
    request.json          # (calculator)
    response.json         # (calculator)
    manifest.json         # (swegym/swebench)
    <instance_id>/        # (swegym/swebench: one dir per instance)
      request.json
      response.json
    summary.json          # (swegym/swebench)
```

## Tuning Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--num-rollouts` | 4 | Number of parallel sessions per task |
| `--timeout-seconds` | 900 | Per-session timeout. Use 2400 for swegym, 3600 for swebench |
| `max_model_len` | 16384 | In cluster.yaml `model` section. Increase for complex tasks |
| `max_num_seqs` | 64 | Max concurrent vLLM sequences |
| `gpu_memory_utilization` | 0.90 | Reduce if OOM |
| `tensor_parallel_size` | 8 | Match `gpus_per_node` |
| `tool_call_parser` | qwen3_xml | Must match model. qwen3_xml for Qwen3.5, hermes for others |

## Switching Models

Edit your `cluster.yaml`:

```yaml
model:
  name: "Qwen/Qwen3.5-27B"         # Change to your model
  tensor_parallel_size: 8            # Adjust for model size
  tool_call_parser: "qwen3_xml"      # Match model's tool call format
```

Or override on the command line:

```bash
polar cluster serve -c my-cluster.yaml --model "Qwen/Qwen3.5-72B"
```

**Note**: The model must be cached in `$HF_HOME` on the cluster (compute nodes run with `HF_HUB_OFFLINE=1`). To cache a new model:

```bash
huggingface-cli download Qwen/Qwen3.5-72B
```

## Backend Support

The cluster config supports multiple backends via the `backend` field:

| Backend | Status | Description |
|---------|--------|-------------|
| `slurm` | Supported | SSH + sbatch job submission |
| `local` | Stub | Run services locally (use example scripts) |
| `k8s`   | Planned | Kubernetes pod scheduling |
