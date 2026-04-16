# Calculator Example

Simple "create a python calculator" rollout example using `polar`.

The example recommends 2 x H100 or comparable GPUs on a local machine:

- one rollout service on `:8080`
- two gateway nodes on `:8100` and `:8101`
- two local SGLang backends on `:8000` and `:8001`
- one topology file at [topology.yaml](topology.yaml)

## Installation

```bash
uv venv
uv pip install -e .
uv pip install --prerelease=allow sglang==0.5.10
bash scripts/patch/patch_sglang.sh
```

The patch supports TITO in sglang for OAI Chat Completion.

## Quick Start

### 1. Start SGLang backends

Start two SGLang servers, one per GPU group:

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
uv run polar serve_rollout -c examples/calculator/topology.yaml
uv run polar serve_gateway -c examples/calculator/topology.yaml --node-id localhost-node-01
uv run polar serve_gateway -c examples/calculator/topology.yaml --node-id localhost-node-02
```

### 3. Build the shared runtime image

Build once for all harnesses:

```bash
uv run python examples/calculator/build_image.py
```

### 4. Submit tasks


```bash
uv run python examples/calculator/submit_calculator_task.py \
  --harness claude_code \
  --topology examples/calculator/topology.yaml \
  --runtime-backend docker \
  --num-samples 8
```

Supported harness names:

- `claude_code`
- `codex`
- `gemini_cli`
- `opencode`
- `openhands_sdk`
- `qwen_code`
- `swe_agent`

## Runtime Layout

The shared runtime image includes:

- Node.js
- Python 3
- git
- a non-root `polar` user

Each rollout then prepares a fresh workspace by:

- installing the harness CLI or SDK for that run
- creating `/polar/session/workspace`
- uploading `calculator.py` and `test_calculator.py`
- initializing a git repo used by the evaluator

## Cluster Deployment (SLURM)

For running on a SLURM cluster with Apptainer containers and vLLM inference.
See [examples/slurm/README.md](../slurm/README.md) for full documentation.

### 1. Configure

```bash
cp examples/slurm/cluster.yaml.example my-cluster.yaml
# Edit my-cluster.yaml with your cluster details
```

### 2. One-Time Setup

```bash
polar cluster setup -c my-cluster.yaml
```

### 3. Build SIF Image

```bash
# Single harness:
polar cluster build-sif -c my-cluster.yaml --example calculator --harness opencode

# Multiple harnesses:
polar cluster build-sif -c my-cluster.yaml --example calculator --harness opencode,codex,swe_agent
```

### 4. Start Services

```bash
polar cluster serve -c my-cluster.yaml
```

Once services are ready, the command prints the job ID and a sample `submit-task` command.

### 5. Submit Tasks

```bash
# Use the job ID from step 4
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example calculator --harness opencode

# Multiple harnesses against the same running service
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example calculator --harness codex
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
polar cluster launch -c my-cluster.yaml --example calculator --harness opencode
```
