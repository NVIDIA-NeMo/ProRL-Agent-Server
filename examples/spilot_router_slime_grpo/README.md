# SPilot Router Slime-GRPO

This example reuses the hardened TMax/Slime launcher while replacing the agent
with a trainable Qwen3.5-9B Router and a fixed frozen pool:

- `pool/qwen3.6-27b` -> `nvidia/qwen/qwen3.6-27b`
- `pool/gpt-5.5` -> `openai/openai/gpt-5.5`

The default experiment requests one H100 on each of eight nodes and an
exclusive Slime rollout boundary of 200, producing optimizer iterations 0
through 199. Four cross-node GPUs form the TP4 learner and the other four GPUs
host independent TP1 Router rollout engines. Each synchronous GRPO step
contains one prompt x eight samples (eight episodes, 1,600 episodes total).
Fully-async training prefetch is disabled for this first run; the async cap of
four is used to overlap fixed-evaluation tasks across the rollout engines.
The integrated baseline/final evaluation uses 32 held-out TMax tasks; the
larger Terminal-Bench evaluation is intentionally left for a separate run.

The Router keeps the full 67,584-token trajectory ceiling, but the learner
accumulates groups in at most 24,576-token dynamic microbatches. This leaves
headroom for Adam state and the TP4 FP32 vocabulary loss without changing the
eight-episode effective batch. A 64-token log-probability chunk further bounds
individual FP32 allocations.

## Credential handling

Load `NVIDIA_API_KEY` and `NVIDIA_BASE_URL` into the submission shell first.
The wrapper copies them to `POLAR_*` variables because the shared launcher only
serializes that namespace into its private mode-0600 job environment. The key
is never rendered into topology YAML or passed on a command line.

The same wrapper creates a fresh high-entropy control-plane token for each
allocation. It authenticates Slime task submission and gateway dispatch, while
per-session Router and model-pool capabilities are generated only after a
trusted dispatch. None of these values are written to run state or agent logs.

## Launch sequence

Start with the one-node, one-step smoke run. It uses four actor GPUs, four
Router rollout GPUs, eight trajectories, no dynamic reward filtering, and no
separate holdout evaluation:

```bash
bash examples/spilot_router_slime_grpo/submit_smoke.sh
```

After the smoke run produces one trainable Router trajectory and checkpoint,
start the checkpoint-aware watcher for the default 200-step experiment. A
single allocation is only four hours, so the watcher is the normal launch path:

```bash
bash examples/spilot_router_slime_grpo/watch_training.sh --relaunch --loop
```

For a deliberate single-allocation diagnostic, the submit wrapper remains
available:

```bash
bash examples/spilot_router_slime_grpo/submit_slurm.sh
```

The SPilot run state lives under `runs/spilot_router_slime_grpo/` and never
contains the NVIDIA key.

`cost_penalty_lambda` is deliberately `0.0`; pool usage is logged but does not
shape reward in this first experiment. Invalid Router actions receive reward
zero, and only gateway-provenanced `router_policy` completions are trainable.
