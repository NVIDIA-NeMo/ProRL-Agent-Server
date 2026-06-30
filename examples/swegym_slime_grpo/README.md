# SWE-Gym Slime GRPO

End-to-end **training** example: train **Qwen3.5-4B** with async **GRPO** on
**SWE-Gym** tasks, using **Polar** for agent rollouts and **Slime** for training.
Targets a single node with 8 × {H100, H200, B200} (4 GPUs train, 4 serve).

> Unlike the rollout demos (calculator / count_stars / swebench_verified), this
> path serves the model with **SGLang**: Slime owns the inference engines and
> syncs the freshly trained weights into them every step (GPU-to-GPU NCCL).

## Prerequisites

Install Polar and SGLang per [Polar installation](../../README.md#installation).

Make sure to install Polar's optional swebench dependency for evaluation.
```
uv pip install -e ".[swebench]"
```

## Quick Start

Log into wandb and one command sets everything up and starts training:

```bash
export WANDB_API_KEY=<your-key>
bash examples/swegym_slime_grpo/launch_e2e.sh
```

It clones Slime + Megatron-LM, applies the Slime router-token metadata patch and
the SGLang 0.5.13 token-metadata patch, installs the training-stack extras
(Transformer Engine; Flash Linear Attention; flash-attn on B200), builds the
293-task SWE-Gym JSONL, pulls the Apptainer images + shared agent CLIs, converts
the Qwen weights to torch_dist, then hands off to `run.sh` (Polar services + Ray
+ the Slime training job).

## (Optional) Watch rollouts in the dashboard

While training runs, start the dashboard **from the repo root** (so its
`./rollout_results` path matches the rollout server's) to inspect live agent
sessions, trajectories, and the rewards feeding each training step:

```bash
uv run polar dashboard -c tmp/swegym_slime_grpo/topology.yaml
```

Open <http://127.0.0.1:8090>. (`tmp/swegym_slime_grpo/topology.yaml` is the
rendered topology that `run.sh` writes at launch.)

## Files

| File | Purpose |
|---|---|
| `launch_e2e.sh` | One-shot entry: setup + run |
| `run.sh` | Launches Polar services + Ray + Slime training job |
| `convert_weights.sh` | HF checkpoint → Megatron torch_dist |
| `model_args.sh` | Default Qwen3.5-4B Megatron args; set `MODEL_ARGS_FILE` for another architecture |
| `topology.yaml` | Polar topology template (`${SGLANG_ROUTER_BASE_URL}` filled at runtime) |
| `polar_config.yaml` | Polar bridge config template (`${AGENT_CLI_DIR}`, `${APPTAINER_IMAGE_DIR}` filled at runtime) |
| `prepare_data.py` | Builds `swegym_train_293.jsonl` |
| `prepare_apptainer_images.py` | Pulls per-task SIF images, builds shared Node + agent CLI dir |
| `sample_tasks.py` | Dataset helpers (HF fetch, registry image lookup) |

## Common knobs

| What you want to tune | Where |
|---|---|
| Train/rollout GPU split, batch size, KL coef, LR | `run.sh` (env vars near top + Slime args at bottom) |
| Which agent harness (qwen_code / claude_code / codex / opencode / pi) | `polar_config.yaml` → `agent.harness` |
| Per-task timeout, async level, callback host | `polar_config.yaml` → `polar_*` keys |
| Gateway/rollout host & port, model served | `topology.yaml` |
| Which SWE-Gym dataset / split | `sample_tasks.py` → `DATASET_NAME`, `DATASET_SPLITS` |
| Model architecture args | `MODEL_ARGS_FILE` (defaults to `model_args.sh`) |
