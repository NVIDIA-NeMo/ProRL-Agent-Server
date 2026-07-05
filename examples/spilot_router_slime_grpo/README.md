# SPilot Router Slime-GRPO

This example reuses the hardened TMax/Slime launcher while replacing the agent
with a trainable Qwen3.5-9B Router and a fixed frozen pool:

- `pool/qwen3.6-27b` -> `nvidia/qwen/qwen3.6-27b`
- `pool/gpt-5.5` -> `openai/openai/gpt-5.5`

The default experiment uses eight 8xH100 nodes and an exclusive Slime rollout
boundary of 200, producing optimizer iterations 0 through 199. One node hosts
the learner; the remaining 56 GPUs host independent TP1 Router rollout engines.
Each step contains 8 prompts x 8 samples (64 episodes, 12,800 episodes total).
Fully-async prefetch is disabled for this first run because the frozen remote
pool, capped at 32 concurrent requests per model, is the throughput bottleneck.
The integrated baseline/final evaluation uses 32 held-out TMax tasks; the
larger Terminal-Bench evaluation is intentionally left for a separate run.

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
