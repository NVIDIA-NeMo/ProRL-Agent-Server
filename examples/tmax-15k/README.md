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

### Parallel SIF builds on Slurm

For a larger TMax slice, submit CPU Slurm array jobs that deterministically shard
the selected tasks. By default each shard translates the task's constrained
`environment/Dockerfile` into an Apptainer definition and builds the `.sif`
directly, so Slurm nodes do not need Docker. This runs inside
`flappydora/ubuntu22.04-cuda13.3:latest` via Slurm's `--container-image` and
uses Apptainer/Singularity inside that container. On this cluster fakeroot is
not available through Pyxis, but the build container runs as root, so the direct
Apptainer build path works without Docker:

```bash
TMAX_SIF_BUILD_SHARDS=1000 \
TMAX_SIF_BUILD_ARRAY_PARALLEL=50 \
TMAX_SIF_BUILD_JOBS_PER_TASK=1 \
bash examples/tmax-15k/submit_build_sifs_slurm.sh
```

Useful knobs:

```bash
TMAX_DATA_ROOT=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data
TMAX_DATASET_DIR=$TMAX_DATA_ROOT/tmax-15k
APPTAINER_IMAGE_DIR=$TMAX_DATA_ROOT/tmax-15k-sif
TMAX_SIF_PYTHON_BIN=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/.python/polar/bin/python
POLAR_APPTAINER_BIN=/usr/bin/apptainer  # auto-detects apptainer/singularity by default
TMAX_SIF_BUILDER=direct-apptainer   # default; use docker-daemon only on docker-capable nodes
TMAX_SIF_APPTAINER_FAKEROOT=0       # default; fakeroot is not available in Pyxis here
TMAX_SIF_BASE_SIF=/path/ubuntu22.sif # optional; avoids Apptainer docker bootstrap pulls
TMAX_SIF_BUILD_LAUNCHER=sbatch-container
TMAX_SIF_BUILD_CONTAINER_IMAGE=flappydora/ubuntu22.04-cuda13.3:latest
TMAX_SIF_BUILD_CONTAINER_MOUNTS=/lustre/fsw:/lustre/fsw
TMAX_SIF_MAX_TASKS=100              # only the first 100 tasks before sharding
TMAX_SIF_BUILD_SHARDS=1000          # default; keeps each cpu_short array task small
TMAX_SIF_BUILD_ARRAY_PARALLEL=50    # default; concurrent cpu_short array tasks
TMAX_SIF_BUILD_PARTITION=cpu_short  # default
TMAX_SIF_BUILD_TIME=4:00:00         # default; cpu_short max on this cluster
TMAX_SIF_BUILD_MEM=64G              # default: 32G
TMAX_SIF_BUILD_FORCE=1              # rebuild existing .sif files
TMAX_SIF_BUILD_FORCE_DOCKER=1       # also rebuild docker images
TMAX_SIF_BUILD_DRY_RUN=1            # print the generated sbatch script
```

To run one shard manually:

```bash
TMAX_SIF_NUM_SHARDS=64 TMAX_SIF_SHARD_INDEX=0 \
bash examples/tmax-15k/build_sif_shard.sh
```

## Train with Slime GRPO

Once the task SIFs are available, use the dependency-compatible training entry
under [`examples/tmax_slime_grpo`](../tmax_slime_grpo/README.md). It generates
Slime JSONL rows from the local dataset, launches task SIFs through Polar, and
uses the Harbor verifier reward for GRPO. A partial-image smoke run is:

```bash
TMAX_ONLY_READY=1 TMAX_MAX_TASKS=64 RUN_ID=tmax-smoke-64 \
bash examples/tmax_slime_grpo/submit_slurm.sh
```
