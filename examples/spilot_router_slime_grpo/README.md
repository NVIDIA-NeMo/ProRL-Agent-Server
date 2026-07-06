# SPilot Router Slime-GRPO

This example reuses the hardened TMax/Slime launcher while replacing the agent
with a trainable Qwen3.5-9B Router and a fixed frozen pool:

- `pool/qwen3.6-27b` -> `nvidia/qwen/qwen3.6-27b`
- `pool/gpt-5.5` -> `openai/openai/gpt-5.5`

The default experiment is a controlled comparison with
`tmax-8n64-qwen35-9b-lr1e6-b8n32-noeval-fresh-20260702T074934Z`. It requests
eight H100s on each of eight nodes and an exclusive Slime rollout boundary of
200, producing optimizer iterations 0 through 199. Two nodes (16 GPUs) host
the TP4 learner and the remaining 48 GPUs host independent TP1 Router rollout
engines. Each GRPO step contains eight prompts x 32 samples (256 accepted
episodes, 51,200 accepted episodes over 200 steps). Fully-async prefetch uses
the same three-policy-version cap as the reference.

The reference train JSONL, ordering, 100-task holdout, initial weights,
optimizer, `1e-6` learning rate, DPPO/TV settings, token ceilings, GPU Adam,
sample-level loss reduction, and checkpoint interval are pinned unchanged.
Only the Router-specific harness, action builder, evaluator, model-pool calls,
and the longer wall-clock budget required for up to two remote calls differ.
Each frozen pool call uses the same Vanillux2 coding protocol as the direct
Qwen reference: 64 agent steps, a 65,536-token cumulative response budget,
120-second shell commands, 64 consecutive format errors, 10,000-character
head/tail observations, and five transient model attempts. Solve and verify
calls share task files but receive separate persistent-shell state directories.
Baseline and final evaluation run synchronously before step 0 and after step
199 on the same 100 held-out tasks, so evaluation cannot overlap or perturb
the reference-compatible optimizer schedule. Terminal-Bench remains a separate
experiment.

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
