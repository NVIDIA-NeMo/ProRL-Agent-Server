# SWE-Gym Example

This example scales the same packaged workflow up to the curated 10-task SWE-Gym sample.
It is set up to mirror the local model and topology used by [examples/calculator](../calculator/README.md),
but runs the sample through the `swe_agent` rollout scaffold in an `apptainer` runtime.

Use it when you want:

- task-specific runtime images
- patch-based evaluation on a clean replay runtime
- one batch manifest plus per-task request and response files
- the same 10 sampled SWE-Gym tasks and evaluator flow

## Topology

The example uses the same local cluster shape as the calculator demo, driven by [topology.yaml](topology.yaml).

Install Polar and SGLang:

```bash
uv pip install -e .
uv pip install --upgrade sglang
bash scripts/patch/patch_sglang.sh
```

Start the SGLang backends:

  ```bash
  CUDA_VISIBLE_DEVICES=0 uv run python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --port 8000 --tp-size 1 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3 --tool-call-parser qwen3_coder

  CUDA_VISIBLE_DEVICES=1 uv run python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --port 8001 --tp-size 1 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  ```

Then start the polar services:

```bash
uv run polar serve_rollout -c examples/swegym/topology.yaml
uv run polar serve_gateway -c examples/swegym/topology.yaml --node-id localhost-node-01
uv run polar serve_gateway -c examples/swegym/topology.yaml --node-id localhost-node-02
```

## Host Setup

Install the host-side evaluator dependency once:

```bash
bash examples/swegym/setup_host.sh
```

## Build Images

```bash
bash examples/swegym/swe_agent/setup.sh
```

## Submit A Sample

Run one sampled task with one rollout:

```bash
uv run python examples/swegym/swe_agent/submit_tasks.py \
  --instance-id getmoto__moto-7365 \
  --max-tasks 1 \
  --num-samples 4
```

Run the full 10-task sample with 4 rollouts:

```bash
uv run python examples/swegym/swe_agent/submit_tasks.py --num-samples 4 --max-tasks 10
```

## Outputs

```text
examples/swegym/<harness>/batches/<timestamp>/
  manifest.json
  summary.json
  <instance-id>/
    request.json
    response.json
```

## Notes

- The sample is text-only SWE-Gym data.
- The submit helper extracts patches from `/polar/session/workspace`, then replays them onto `/testbed` for grading.
- `swe_agent` uses a dedicated `polar-sweagent` environment inside the derived image.
- `openhands_sdk` only builds on benchmark images whose native Python is already compatible.

## Cluster Deployment (SLURM)

For running on a SLURM cluster with Apptainer containers and vLLM inference.
See [examples/slurm/README.md](../slurm/README.md) for full documentation.

### 1. Configure

```bash
cp examples/slurm/cluster.yaml.example my-cluster.yaml
# Edit my-cluster.yaml with your cluster details
```

### 2. Build SIF Images

```bash
polar cluster build-sif -c my-cluster.yaml --example swegym --harness swe_agent
```

### 3. Start Services

```bash
polar cluster serve -c my-cluster.yaml
```

Once services are ready, the command prints the job ID.

### 4. Submit Tasks

```bash
# All 10 sample instances (use job ID from step 3)
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example swegym --harness swe_agent \
    --timeout-seconds 2400

# Or a single instance
polar cluster submit-task -c my-cluster.yaml \
    --job-id JOB_ID --example swegym --harness swe_agent \
    --timeout-seconds 2400 --instance-id getmoto__moto-7365
```

### 5. Stop Services

```bash
scancel JOB_ID
```

### 6. Collect Results

```bash
polar cluster sync -c my-cluster.yaml
```

**One-shot alternative** — start services, run tasks, and exit in one command:

```bash
polar cluster launch -c my-cluster.yaml --example swegym --harness swe_agent \
    --timeout-seconds 2400
```
