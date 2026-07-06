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

## Paired forced-route evaluation

`forced_route_eval.py` measures the frozen candidates on an identical fixed
task slice without loading, calling, or training the Router. It submits one
Qwen and one GPT task for every selected JSONL row, auto-submits after the
single mini-SWE call, and reuses the normal `spilot_harbor` evaluator. The
evaluation-only builder emits no traces even if a pool completion were ever
persisted accidentally.

The safest path is the dedicated services-only Slurm entrypoint. It requests
one node and no GPU, enters the same proven Pyxis image used by TMax, renders
fresh allocation-local configs, and starts only rollout, one gateway, and the
gateway/proxy UDS bridge. The evaluator runs on the same node, after which the
entrypoint tears all three services down. It never starts Ray, Slime, SGLang,
or a Router actor.

For example, compare 32 paired holdout tasks on cw-dfw. The output directory
must not already exist. The submitter loads the NVIDIA credential from the
current shell or `~/.zshrc`, writes only a mode-0600 submission envelope, and
passes its path (not the key) to Slurm. The allocated-node entrypoint sources
and immediately deletes that file. It generates the control-plane token in
memory and never persists it:

```bash
REPO=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/ProRL-Agent-Server
DATA_ROOT=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data
RUN_ID=qwen-vs-gpt-holdout-$(date -u +%Y%m%dT%H%M%SZ)
RESULT_ROOT=${DATA_ROOT}/runs/spilot-forced-route-eval

ACCOUNT=nvr_lpr_llm \
PARTITION=backfill,batch \
SLURM_CONSTRAINT=H100 \
CPUS_PER_TASK=32 \
POLAR_SLURM_MEM_PER_NODE=128G \
WALL_TIME=12:00:00 \
FORCED_EVAL_GPUS=0 \
bash "${REPO}/examples/spilot_router_slime_grpo/submit_forced_route_eval.sh" \
  --i-understand-eval-only \
  --run-id "${RUN_ID}" \
  --data "${DATA_ROOT}/runs/tmax-14598r-14498t100h-20260701T011143Z/tmax_holdout-eval.jsonl" \
  --data-root "${DATA_ROOT}" \
  --pool-base-url https://inference-api.nvidia.com/v1 \
  --start-index 0 --max-tasks 32 \
  --seed 20260706 --max-concurrency 4 \
  --output-dir "${RESULT_ROOT}/${RUN_ID}"
```

There is no software GPU dependency because both candidates are remote. If a
site partition refuses a zero-GPU Pyxis allocation, resubmit with
`FORCED_EVAL_GPUS=1`; that GPU remains unused. Do not use an old job's
rendered config: it contains stale compute-node URLs and job-local `/tmp` UDS
paths.

Before that paid run, use the same command with `PARTITION=interactive`,
`WALL_TIME=02:00:00`, `--max-tasks 1`, and `--max-concurrency 2` as the allocation
smoke. A 32-pair run can require many waves of full coding-agent work, so the
two-hour interactive limit is not a safe full-benchmark wall time.

For an already running, deliberately managed Polar allocation, the low-level
evaluator remains available. Pass that allocation's freshly rendered
`polar_config.yaml`, not the `${...}` template:

```bash
python examples/spilot_router_slime_grpo/forced_route_eval.py \
  --i-understand-eval-only \
  --run-id qwen-vs-gpt-holdout-v1 \
  --data /abs/path/tmax_holdout-eval.jsonl \
  --polar-config /abs/path/job-123/polar_config.yaml \
  --rollout-url http://rollout-host:18080 \
  --start-index 0 \
  --max-tasks 32 \
  --seed 20260706 \
  --max-concurrency 4 \
  --output-dir /abs/path/forced-route-qwen-vs-gpt
```

In the low-level form, the submission shell must already contain
`POLAR_CONTROL_PLANE_TOKEN`; model credentials remain in the running Polar
service. The output directory is created exclusively and contains:

- `manifest.json`: immutable data/config hashes, range, seed, candidates, and
  the no-actor contract;
- `results.jsonl`: one row per task/candidate/replicate with fixed-denominator
  reward, lifecycle status, pool status, and run/eval/end-to-end latency;
- `summary.json`: per-candidate accuracy/validity/latency plus paired wins,
  losses, ties, and GPT-minus-Qwen reward delta.

Use `--forward-seed-to-pool` only after confirming that every endpoint accepts
an OpenAI-compatible `seed` field. Without it, the seed still fixes task order,
candidate interleaving, slot assignment, pair identity, and the audit manifest.
