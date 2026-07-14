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
In-training holdout evaluation is disabled (`TMAX_TRAINING_EVAL_ENABLED=0`) to
match the no-eval reference run. Frozen-candidate calibration and any final
holdout evaluation are separate experiments and cannot perturb the optimizer
schedule. Terminal-Bench also remains separate.

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
start the checkpoint-aware watcher for the default 200-step experiment. The
canonical allocation requests two days on `backfill`; the watcher remains the
normal launch path so a requeue or infrastructure interruption can resume from
the last validated checkpoint:

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

`cost_penalty_lambda` defaults to `0.0`; runs can opt into success-gated cost
shaping through `SPILOT_COST_PENALTY_LAMBDA`. Invalid Router actions receive
reward zero, and only gateway-provenanced `router_policy` completions are
trainable.

For training diagnostics, `rollout/raw_reward` is Slime's legacy
trace/sample-weighted series and can include masked or early-stop placeholder
samples. Treat the one-session-one-vote `polar/reward_mean` as the aggregate
quality signal and use `polar/spilot_router/reward_candidate_c0_mean` and
`polar/spilot_router/reward_candidate_c1_mean` for the frozen candidates. The
C-index is the lexical alias order: for this pool C0 is GPT-5.5 and C1 is
Qwen3.6-27B. Interpret those rewards together with each candidate's timeout and
admission-failure metrics.

Cost-shaped experiments also publish one-session-one-vote decomposition under
`polar/spilot_router/` so performance and efficiency do not have to be inferred
from the combined reward:

- `accuracy_outcome_mean` is the raw Harbor task outcome before Router validity
  and cost shaping;
- `total_cost_{mean,median,min,max,total}` is the actual accepted-session pool
  spend;
- `cost_penalty_fraction_mean` is the evaluator's applied penalty fraction;
- `cost_penalty_reward_delta_mean` is the reward removed by cost shaping; and
- `cost_adjusted_reward_mean` is the final reward used for training.

Every family has an `*_accounted_session_count`; malformed or missing metadata
is omitted instead of becoming a fabricated zero. These counts are independent:
for example, a containment-filtered session can still contribute raw accuracy
and cost while being absent from the training-reward denominator. The same
values are split by initial slot (`m0`/`m1`) and stable candidate
(`candidate_c0`/`candidate_c1`).
Per-call cost is additionally reported by stable candidate and by `solve` versus
`verify`, with `pool_cost_reconciliation_delta` comparing the call ledger to
accepted-session `total_cost`.

## Paired forced-route evaluation

`forced_route_eval.py` measures the frozen candidates on an identical fixed
task slice without loading, calling, or training the Router. It submits one
Qwen and one GPT task for every selected JSONL row, auto-submits after the
single mini-SWE call, and reuses the normal `spilot_harbor` evaluator. The
evaluation-only builder emits no traces even if a pool completion were ever
persisted accidentally.

`--include-qwen35-baseline` optionally adds a third, evaluation-only route:
`pool/qwen3.5-9b-baseline` -> `nvidia/qwen/qwen3.5-9b`. The `pool/` alias keeps
the baseline inside the same strict lease and one-shot call-capability boundary
as the two primary candidates. This switch does not modify the two-candidate
training topology or any training action space.

The safest path is the dedicated services-only Slurm entrypoint. It requests
one node and no GPU, enters the same proven Pyxis image used by TMax, renders
fresh allocation-local configs, and starts rollout, one gateway, the
gateway/proxy UDS bridge, and a loopback tokenizer-only service. The latter
loads the Qwen3.5 chat template/tokenizer assets but no model weights; it gives
Vanillux2 the same cumulative-budget token counts normally supplied by the
training actor. The evaluator runs on the same node, after which the entrypoint
tears all services down. It never starts Ray, Slime, SGLang, generation, or a
Router actor.

Before starting any service, the launcher hashes the live implementation,
copies `src/` and this complete example into a read-only allocation snapshot,
and verifies the copy byte-for-byte. Every child command and `PYTHONPATH` then
uses that snapshot. A secret-free semantic identity additionally binds the
candidate alias/model/base URL and admission caps, tokenizer content,
mini-SWE and `agent_cli` runtimes, container/executable identities, and Python
dependency versions. It is verified before evaluation, before formal metrics,
and after clean service teardown.

For example, compare 32 paired holdout tasks on cw-dfw. A new benchmark requires
a fresh run id and output/service/submit paths; an explicit `--resume` keeps the
same immutable run/output but always creates fresh service and submit paths.
The submitter loads the NVIDIA credential from the current shell or `~/.zshrc`,
writes only a mode-0600 submission envelope, and
passes its path (not the key) to Slurm. The allocated-node entrypoint sources
and immediately deletes that file. It generates the control-plane token in
memory and never persists it:

```bash
REPO=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/ProRL-Agent-Server
DATA_ROOT=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data
RUN_ID=qwen-vs-gpt-holdout-$(date -u +%Y%m%dT%H%M%SZ)
RESULT_ROOT=${DATA_ROOT}/runs/spilot-forced-route-eval

ACCOUNT=nvr_lpr_llm \
PARTITION=backfill \
SLURM_CONSTRAINT=H100 \
CPUS_PER_TASK=32 \
POLAR_SLURM_MEM_PER_NODE=128G \
FORCED_EVAL_GPUS=0 \
bash "${REPO}/examples/spilot_router_slime_grpo/submit_forced_route_eval.sh" \
  --i-understand-eval-only \
  --run-id "${RUN_ID}" \
  --data "${DATA_ROOT}/runs/tmax-14598r-14498t100h-20260701T011143Z/tmax_holdout-eval.jsonl" \
  --data-root "${DATA_ROOT}" \
  --pool-base-url https://inference-api.nvidia.com/v1 \
  --start-index 0 --max-tasks 32 \
  --seed 20260706 --max-concurrency 4 \
  --pool-timeout-seconds 1200 \
  --output-dir "${RESULT_ROOT}/${RUN_ID}"
```

Add `--include-qwen35-baseline` to that command for the three-candidate matrix.
The immutable plan, source/asset identity, spend ledger, and walltime bound all
record the switch and exact endpoint mapping.

If `WALL_TIME` is omitted, the submitter preflights the exact selected JSONL
slice. It computes the outer task envelope as the larger of the selected rows'
maximum `timeout_seconds` and configured agent timeout plus their maximum
`verifier_timeout`, then applies
`ceil(max_tasks * candidate_count * replicates * max_paid_attempts / max_concurrency)`
waves and a
30-minute service/scheduling margin. An explicitly set walltime is rejected
before `sbatch` if it cannot cover that bound. For the documented holdout
slice (`timeout_seconds=840`, `verifier_timeout=120`) at concurrency 4, the
32-task minimum is 56,520 seconds (`15:42:00`). The default global cap is one
paid attempt per task/candidate work item; `--max-paid-attempts-per-work N`
with `N > 1` is an explicit opt-in to full paid retries and increases the
walltime bound accordingly. Override the margin with
`FORCED_EVAL_SLURM_MARGIN_SECONDS` only when the site overhead is understood.
With the optional baseline and the same documented settings, the minimum is
83,880 seconds (`23:18:00`).

There is no software GPU dependency because both candidates are remote. If a
site partition refuses a zero-GPU Pyxis allocation, resubmit with
`FORCED_EVAL_GPUS=1`; that GPU remains unused. Do not use an old job's
rendered config: it contains stale compute-node URLs and job-local `/tmp` UDS
paths.

Before that paid run, use the same command with `PARTITION=interactive`,
`--max-tasks 1`, and `--max-concurrency 2` as the allocation
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
  --semantic-identity /abs/path/job-123/semantic_identity.json \
  --rollout-url http://rollout-host:18080 \
  --start-index 0 \
  --max-tasks 32 \
  --seed 20260706 \
  --max-concurrency 4 \
  --output-dir /abs/path/forced-route-qwen-vs-gpt
```

In the low-level form, the submission shell must already contain
`POLAR_CONTROL_PLANE_TOKEN`; model credentials remain in the running Polar
service. The semantic identity must be the one produced before those services
started; hand-written identity files are unsupported. A new output directory
starts a run. Repeating the exact command
with `--resume` against the same live allocation resumes an interrupted run:
the evaluator
validates the immutable plan, keeps already persisted task ids, reattaches to
any task accepted remotely just before the interruption, verifies its echoed
plan/work/row fingerprints, and submits only missing work. Session
`ERROR`/`TIMEOUT`, missing forced acknowledgements or candidate calls,
failed/ambiguous calls, local transport failures, and evaluator-induced
cancellations remain pending rather than becoming zero-reward benchmark rows.
Only a complete attributable candidate call, including an explicit matching
candidate timeout, is benchmark data. The default paid-attempt cap is one; a
larger cap is an explicit retry-spend opt-in. An allocation-wide owner lease
and per-pass output lock prevent concurrent resumers from overwriting one
another. Changing the run id, data/config hash,
implementation, prompt/row identity, range, seed, candidate
mapping, or concurrency requires a fresh output directory. The directory
contains:

- `manifest.json`: immutable data/semantic-config/code/SIF/verifier hashes,
  range, seed, candidates, the no-actor contract, separate allocation-attempt
  metadata, and an explicit `collection.status` of `partial` or `complete`;
- `results.jsonl`: a sorted, duplicate-free snapshot replaced and `fsync`ed
  after every authoritative `completed` task/candidate/replicate, with fixed-denominator
  reward, lifecycle status, pool status, and run/eval/end-to-end latency;
- `summary.json`: current collection counts and explicitly labeled
  `collected_only` diagnostics while partial. Formal candidate means and
  paired deltas appear under `final_metrics` only after the complete expected
  matrix and all content/teardown integrity checks pass; otherwise it is null
  with a `withheld_*` status. `paired` remains the backward-compatible
  GPT-minus-Qwen3.6 comparison; `paired_comparisons` additionally reports GPT
  minus Qwen3.5 baseline and Qwen3.6 minus Qwen3.5 baseline when enabled.

Use `--forward-seed-to-pool` only after confirming that every endpoint accepts
an OpenAI-compatible `seed` field. Without it, the seed still fixes task order,
candidate interleaving, slot assignment, pair identity, and the audit manifest.

The immutable plan also records a declared content manifest for the forced and
shared runners, trajectory builder/evaluators, gateway, rollout and runtime
implementation, every selected task SIF, and every verifier tests tree. Missing
or changed content changes the plan and task IDs, so old rows cannot be mixed
into a resumed benchmark. Allocation-local ports, UDS roots, raw rendered
config paths, and hostnames are recorded separately as attempt metadata and do
not change the semantic plan. If an evaluator pass exits with code 3 (pending
work), the services-only launcher performs a bounded number of `--resume`
passes against the same live services; it stops rather than restart unless the
remaining Slurm walltime can cover all missing waves plus shutdown margin.
Paid calls use gateway episode admission with each candidate capped by the
requested evaluation concurrency and a bounded positive wait budget. The
launcher continuously checks structured gateway health; a retained unreaped
episode, repeated health failure, premature service exit, forced kill, or
non-clean gateway/rollout/UDS teardown invalidates and withholds final metrics.

To resume in a fresh allocation, rerun the submit command with the same run id,
dataset slice, seed, candidates, concurrency and output directory, adding
`--resume`. The wrapper generates a fresh service directory automatically; any
semantic config, implementation, SIF, verifier, or row-content change fails
closed before a result can be reused.
