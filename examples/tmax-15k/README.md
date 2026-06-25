# TMax-15K-Harbor Example

Run Polar agent harnesses on [TMax-15K-Harbor](https://hub.harborframework.com/datasets/tmax/TMax-15K-Harbor/latest)
— 15k compositional terminal-agent tasks from [TMax](https://github.com/hamishivi/tmax),
each a self-contained container with a programmatic verifier. Each task runs an
agent inside its container, then the **`harbor`** evaluator scores it exactly
as TMax does: inject the task's `tests/`, run `bash /tests/test.sh` (which runs
`pytest test_final_state.py` and writes `0`/`1` to `/logs/verifier/reward.txt`),
and read that reward back.

## Prerequisites

Polar + an inference backend (vLLM shown), Docker, and the **Harbor CLI** (used
once to pull the dataset — it is a TMax dependency, not a Polar one):

```bash
uv pip install harbor          # for `harbor download` (or use the tmax checkout's env)
```

This example assumes 1 node **8×H100** — two inference servers (tensor-parallel 4 each).

## Quick Start

### 1. Pull the dataset (task dirs, not images)

Harbor hub serves task directories; this fetches them to a local folder:

```bash
harbor download 'tmax/TMax-15K-Harbor@latest' --export --output-dir ~/tmax15k
```

Each task dir has `instruction.md`, `task.toml`, `environment/Dockerfile`, and
`tests/` (the verifier, kept out of the image so the agent can't read it).

### 2. Build runtime images

Per task we build the sandbox from its `environment/Dockerfile`, then layer
Node.js (`runtime/Dockerfile`) so the harness CLI can run inside it:

```bash
uv run python examples/tmax-15k/build_images.py --dataset-dir ~/tmax15k --max-tasks 10
```

### 3. Start two inference servers (Qwen3.6-27B)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve Qwen/Qwen3.6-27B --port 8000 \
  --tensor-parallel-size 4 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run vllm serve Qwen/Qwen3.6-27B --port 8001 \
  --tensor-parallel-size 4 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

### 4. Start Polar

```bash
uv run polar serve_rollout -c examples/tmax-15k/topology.vllm.yaml
uv run polar serve_gateway -c examples/tmax-15k/topology.vllm.yaml --node-id localhost-node-01
uv run polar serve_gateway -c examples/tmax-15k/topology.vllm.yaml --node-id localhost-node-02
```

### 5. Submit tasks

The gateway rewrites the harness's `--model-name` to the served `Qwen/Qwen3.6-27B`.
Supported harnesses: `codex`, `claude_code`, `opencode`, `qwen_code`, `pi`, `hermes`, `mini_swe_agent`.

```bash
# pass@4 over the first 10 tasks
uv run python examples/tmax-15k/submit_tmax_tasks.py --dataset-dir ~/tmax15k --harness hermes --max-tasks 10 --num-samples 4
```

Use Apptainer instead of Docker with `--runtime-backend apptainer` (this still
reads images from a local docker daemon). For nodes **without Docker**, see
[Docker-free runs](#docker-free-runs-apptainer-on-slurm) below.

### 6. (Optional) Watch in the dashboard

```bash
uv run polar dashboard -c examples/tmax-15k/topology.vllm.yaml   # http://127.0.0.1:8090
```

## Docker-free runs (Apptainer on Slurm)

Slurm nodes without Docker can't `docker build` or pull `docker-daemon:` images.
Build once on a docker-capable box, snapshot each runtime image to a `.sif`, copy
them over, and launch the `.sif` directly — Polar's `ApptainerRuntime` needs only
`apptainer` (set `POLAR_APPTAINER_BIN` if your cluster calls it `singularity`):

```bash
# on a box WITH docker — build images, then snapshot them to .sif
uv run python examples/tmax-15k/build_images.py --dataset-dir ~/tmax15k --max-tasks 10
uv run python examples/tmax-15k/prepare_apptainer_images.py \
  --dataset-dir ~/tmax15k --image-dir ~/tmax15k-sif --max-tasks 10

# copy ~/tmax15k-sif/ to the cluster, then on Slurm (no docker needed):
uv run python examples/tmax-15k/submit_tmax_tasks.py --dataset-dir ~/tmax15k \
  --harness hermes --max-tasks 10 \
  --runtime-backend apptainer --apptainer-image-dir ~/tmax15k-sif
```

The dataset dir must also be on the cluster: `submit` reads each task's
`instruction.md`, and the `harbor` evaluator uploads its `tests/` into the
container — only the *images* become `.sif`.