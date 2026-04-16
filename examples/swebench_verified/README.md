# SWE-bench Verified Example

Evaluate Polar agent harnesses on the full [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) benchmark (500 human-validated tasks).

Each task runs an agent inside a per-instance container with the repo at `base_commit`, then grades the resulting patch via `swebench.harness.grading`.


## Installation

```bash
uv venv
uv pip install -e .
uv pip install --prerelease=allow sglang==0.5.10
bash scripts/patch/patch_sglang.sh
```

Install host-side evaluator dependencies (swebench grading + HuggingFace datasets):

```bash
bash examples/swebench_verified/setup_host.sh
```

## Quick Start

### 1. Start SGLang backends

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --tp-size 2 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mem-fraction-static 0.7 \
    --context-length 262144 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=2,3 uv run python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port 8001 \
    --tp-size 2 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mem-fraction-static 0.7 \
    --context-length 262144 \
    --trust-remote-code
```

### 2. Start Polar services

```bash
uv run polar serve_rollout -c examples/swebench_verified/topology.yaml
uv run polar serve_gateway -c examples/swebench_verified/topology.yaml --node-id localhost-node-01
uv run polar serve_gateway -c examples/swebench_verified/topology.yaml --node-id localhost-node-02
```

### 3. Build runtime images

```bash
# Build all 500
uv run python examples/swebench_verified/build_images.py

# Or build a subset
uv run python examples/swebench_verified/build_images.py --max-tasks 10
```

### 4. Submit tasks

```bash
# Run all 500 tasks for pass@1
uv run python examples/swebench_verified/submit_swebench_tasks.py \
  --harness claude_code \
  --topology examples/swebench_verified/topology.yaml \
  --runtime-backend docker \
  --num-samples 1 \
  --max-concurrent 4 \
  --max-tasks 10

# pass@8 for first 10 tasks
uv run python examples/swebench_verified/submit_swebench_tasks.py \
  --harness claude_code \
  --topology examples/swebench_verified/topology.yaml \
  --runtime-backend docker \
  --num-samples 8 \
  --max-concurrent 4 \
  --max-tasks 10
```

Supported harness names: `claude_code`, `codex`, `opencode`, `openhands_sdk`

## Cluster Deployment (SLURM)

For running on a SLURM cluster with Apptainer containers and vLLM inference.
See [examples/slurm/README.md](../slurm/README.md) for full documentation.

### 1. Configure

```bash
cp examples/slurm/cluster.yaml.example my-cluster.yaml
# Edit my-cluster.yaml with your cluster details
```

### 2. Populate Dataset Cache

The task runner needs the full SWE-bench Verified dataset cached locally.
Run once (requires `datasets` library):

```bash
python -c "from examples.swebench_verified.dataset import load_swebench_verified; load_swebench_verified()"
```

### 3. Build SIF Images

Each SWE-bench Verified instance needs a per-instance SIF:

```bash
# Build SIF for a specific instance + harness
polar cluster build-sif -c my-cluster.yaml \
    --example swebench_verified --harness opencode \
    --instance-id django__django-15098

# Build multiple instances
polar cluster build-sif -c my-cluster.yaml \
    --example swebench_verified --harness opencode \
    --instance-id django__django-15098 \
    --instance-id sympy__sympy-18835
```

### 4. Start Services

```bash
polar cluster serve -c my-cluster.yaml
```

Once services are ready, the command prints the job ID.

### 5. Submit Tasks

```bash
# Submit a single instance (use job ID from step 4)
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example swebench_verified --harness opencode \
    --timeout-seconds 3600 --instance-id django__django-15098

# Submit multiple instances
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example swebench_verified --harness opencode \
    --timeout-seconds 3600 \
    --instance-id django__django-15098 \
    --instance-id sympy__sympy-18835
```

### 6. Stop Services

```bash
scancel JOB_ID
```

### 7. Collect Results

```bash
polar cluster sync -c my-cluster.yaml
```

**One-shot alternative** — start services, run tasks, and exit in one command:

```bash
polar cluster launch -c my-cluster.yaml \
    --example swebench_verified --harness opencode \
    --timeout-seconds 3600 --instance-id django__django-15098
```
