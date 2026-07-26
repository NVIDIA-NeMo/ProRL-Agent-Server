# TMax Slime GRPO

Short GPU-allocation experiments are documented in
[`profile/README.md`](profile/README.md). The profiler compares fully async
train/rollout splits with collocation without touching production run state or
writing model checkpoints.

This example trains Qwen3.5-9B with Slime GRPO while Polar runs each TMax
trajectory in its task-specific Apptainer SIF. The Harbor evaluator injects
the task's `tests/` directory into the live sandbox and returns its 0/1 reward.

The integration keeps this repository's Slime v0.3.0 contract. In particular,
compact trace samples continue to use `Sample.group_id`; the older dependency
settings from `hao/nrt` are not imported.

## Data contract

`prepare_data.py` joins each dataset task with exactly one image:

```text
task_000000_c19dda5b/
  instruction.md
  tests/
  task.toml

tmax-15k-sif/
  task_000000_c19dda5b.sif
```

The generated JSONL contains the instruction plus task name, tests path,
source agent timeout, verifier timeout, workdir, and image path. Training
payloads explicitly override only the active agent budget to the official
1,200-second 9B backend timeout; eval payloads retain their dataset timeouts.
Full mode fails before Slurm submission if any selected image is missing.

## Cluster setup

Defaults are in `env.cwdfw.sh`:

- dataset: `/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data/tmax-15k`
- SIFs: `/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data/tmax-15k-sif`
- Python: `/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/.python/polar/bin/python`
- training image: `data/container/flappydora-ubuntu22.04-cuda13.3.sqsh`
- Slurm partitions: `backfill,batch` for the default four-node, four-hour job.
  A deliberate `WALL_TIME=2:00:00 TMAX_MIN_WALL_TIME=2:00:00` run can use
  `batch_short`; it is deliberately not mixed with normal-QoS partitions by
  default, because on cw-dfw a live
  `QOSGrpNodeLimit` can block the whole comma-separated request and the mixed
  request receives a lower effective priority. Jobs of at most two nodes and
  sixteen GPUs add `interactive`;
  if they request more than two hours they use `interactive,backfill,batch`.
  Three- or four-node jobs use `batch_short` only at two hours or less, while
  larger or longer jobs use `backfill,batch`. The launcher rejects an explicit
  comma-separated list containing `interactive` above its 2-node/16-GPU cap or
  `batch_short` above its 4-node/2-hour cap, because one inadmissible partition
  can block the entire multi-partition request instead of falling back.
- model: a local `data/checkpoints/Qwen3.5-9B` Hugging Face snapshot plus its
  `Qwen3.5-9B_torch_dist` release checkpoint. `model_args.sh` pins the matching
  32-layer, hidden-size-4096, untied-embedding Megatron architecture.
- topology: four 8xH100 nodes, eight trainer GPUs on the first node
  (TP=4, CP=1, DP=2, Megatron sequence parallel) and 24 independent TP=1
  SGLang engines on the remaining 24 GPUs. CP stays at one because the pinned
  Qwen3.5 GatedDeltaNet implementation does not propagate recurrent state
  across context-parallel ranks.
- batch: eight prompts by 32 trajectories in one optimizer step with global
  batch 256 (actor TP=4, CP=1, DP=2, 128 trajectories per DP rank per step)
- trajectory length: 2,048 prompt tokens, at most 16,384 generated tokens per
  model turn, and 65,536 accumulated multi-turn response tokens. The trainer
  `seq_length` and dynamic sample cap are 67,584 so the full prompt plus
  response is accepted instead of being silently clipped.
- scheduler: fully async level 4 by default, with at most 1,024 active sessions,
  48 init / 384 run / 192 postrun workers, and a separately bounded
  completed-result backlog; this gives each rollout GPU about 42 admitted
  sessions and 16 active run-worker slots. Explicit `a2` matrix settings use
  async level 2, including the 9B baseline whose 512/24 in-flight density
  matches the official 1,024/48 recipe. The single gateway owns the complete worker
  pool, preventing slow Harbor verification from retaining sandboxes behind an
  undersized postrun pool. Blocking Popen creation runs outside the gateway
  event loop and is globally admitted two at a time. TMax uses the
  original single-gateway fresh `apptainer exec` path with
  `POLAR_APPTAINER_PERSISTENT_BROKER=0`; every command reuses the session
  overlay but starts a fresh container process. Direct exec retries up to three
  times. Broker mode remains available for controlled comparisons only.
- sampling policy: the paper-faithful default waits for all 32 trajectories in
  each prompt group (`POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0`) and applies
  `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`
  to training rollouts. All-zero and all-one groups are recorded as filtered,
  their reservations are consumed with a distinct outcome, and a fresh prompt
  group is sampled until the full 8x32 training batch contains mixed rewards.
  Evaluation uses its separate one-shot path and is never dynamically filtered.
  Setting a positive completion fraction remains available as an explicit,
  non-paper straggler-throughput experiment.
- failure credit: completed but unsolved/model-budget trajectories receive
  reward zero and remain trainable. A parser-invalid model action also remains
  trainable with reward zero when tokens and old-policy log-probabilities are
  aligned, so successful siblings give it a negative centered/LOO advantage.
  Infrastructure ERROR/TIMEOUT and unaligned traces are zero-gradient; a
  conversion failure replaces only that session with a masked placeholder and
  never terminates the rest of the training batch.
- trainer balancing: dynamic microbatches use `--balance-data` so long
  dynamic-history traces do not leave one DP rank as the synchronization tail
- 9B memory guard: the dynamic sample cap is 67,584 tokens with full
  recomputation, TP=4 sequence parallelism, and CP=1. SGLang uses TP=1, 70%
  static memory, and Qwen3.5-9B's native 262,144-token inference context.
  `ROLLOUT_MAX_RESPONSE_LEN=16384` remains a per-turn cap; the Vanillux2
  adapter separately enforces the 65,536 cumulative response budget with the
  serving tokenizer.
- wall time: four hours minimum by default; an explicit two-hour allocation
  can use `batch_short`. In either case the watcher checkpoints before the
  30-minute drain reserve and continues the same run in another allocation
  until training completes
- W&B axes: model, rollout, Polar, performance, and non-eval timing metrics use
  `train/step`; delayed eval metrics use `eval/train_step`; each GPU sidecar uses
  its own `polar_tmax_system/node_N/train_step` so concurrent writers cannot
  move the canonical training axis backwards.
- HTTP logs: high-frequency Uvicorn access logs and per-completion gateway
  request logs are off by default, and SGLang's HTTP logger defaults to
  `warning`; errors, session lifecycle events, and task summaries remain
  enabled. Set `POLAR_UVICORN_ACCESS_LOG=1` and
  `SGLANG_LOG_LEVEL_HTTP=info` only for a short request-level debugging run.
- nested sandbox mode: `POLAR_APPTAINER_NO_INSTANCE=1`
- nested execution strategy: `POLAR_APPTAINER_PERSISTENT_BROKER=0` for the
  current TMax matrix; set it to `1` only for an explicitly named broker arm.
- sandbox isolation: disable automatic hostfs and outer `/tmp` mounts, use
  separate PID/IPC namespaces, and clean the outer Slurm/PMI environment so a
  task cannot delete sibling session files or join the training PMIx step.
  Rootless Apptainer cannot create a bridge network on this cluster, so TMax
  uses its supported `network=none` mode: every exec gets a private localhost,
  while a job-private Unix socket carries authenticated model traffic back to
  Polar. A separate proxy-socket bind is mounted only for
  `allow_internet=true` tasks and carries HTTP(S) through the cluster proxy;
  offline sandboxes cannot open that socket directly. Fixed services embedded
  in SIF environment scripts therefore cannot collide or leak state across
  concurrent replicas.
- checkpoint cadence: every ten completed rollouts (`SAVE_INTERVAL=10`), plus the graceful-exit save
- checkpoint formats: every save writes a complete HF safetensors export to
  `${SAVE_DIR}/hf/iter_XXXXXXX` (`SAVE_HF_ENABLED=1` by default when
  `HF_CHECKPOINT` is a local snapshot), including the frozen vision/mtp
  tensors copied from the origin snapshot, staged and published atomically
  with an `.export_complete.json` marker — so `export_hf_checkpoint.sh`
  conversion jobs are only needed for pre-existing runs. The Megatron
  torch_dist checkpoint (fp32 optimizer state, ~10x the HF export size) is
  still written by default because it is the only exact resume point;
  `SAVE_MEGATRON=0` drops it for disposable runs, and `SAVE_RETAIN_INTERVAL`
  bounds its disk usage while keeping resumability by pruning older
  torch_dist iterations
- SGLang ports: a different 320-port block is derived from each Slurm job id
  and kept strictly below the node's kernel ephemeral-port range. This avoids
  outbound connections taking a checked-but-not-yet-bound SGLang port during
  parallel engine startup. The router defaults to port 8680, below the cluster's
  ephemeral range and outside all default engine blocks; unsafe or overlapping
  explicit overrides fail before model startup.
- allocation guard: request graceful exit 30 minutes before the actual Slurm end time

TMax uses separate active-agent and infrastructure budgets.
`TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=1200` matches the official 9B
`backend_timeout` and overrides `sample.metadata.agent_timeout` only for
training rollouts. TMax holdout and Terminal-Bench eval payloads keep their
own pinned metadata timeouts unchanged. `POLAR_TASK_TIMEOUT_FLOOR_SECONDS`
defaults to 1,800 seconds and the submitted session budget is
`max(sample.metadata.timeout_seconds, floor)`, so a dataset row with a larger
infrastructure budget is never shortened. The effective agent timeout is
copied into trusted task metadata and starts only when a RUN worker begins; it
caps setup, model/tool execution, and agent postprocessing even when container
startup or READY queueing consumed part of the larger session budget. Harbor
verification remains independently capped by `verifier_timeout`. The total
session deadline still wins if it expires first. `POLAR_REQUEST_TIMEOUT=3600`
keeps the Slime HTTP envelope outside these task budgets. Both values and the
postrun worker ratio are persisted in `run_state.env` for exact continuation.

"Fully async" means bounded background prefetch, not continuous GPU decode.
Agent shell/tool work, sandbox initialization, and Harbor postrun evaluation are
CPU phases. GPU utilization should therefore be evaluated over the steady
rollout interval, not across model startup or graceful teardown. Lower
`TMAX_MIN_ASYNC_LEVEL` only together with a correspondingly smaller run-worker
pool; lower `TMAX_MIN_WALL_TIME` only for a deliberate short validation run.
The 8+24 split dedicates one full node to TP4/DP2 training and the remaining
three nodes to rollout. Level 4 remains the generic default; explicitly named
`a2` settings use level 2. The gateway node
also carries Apptainer/FUSE system-CPU load, so check head-node CPU and callback
latency before raising it further. For a
healthy steady-state run, `perf/wait_time_ratio` should move below roughly 0.2,
`polar/scheduler/active_sessions` should stay well supplied (up to 1,024), and the completed
buffer should be neither permanently empty (rollout supply is too slow) nor
permanently full (rollout is overprovisioned or samples are becoming stale).

### Timing metrics

Business/performance metrics use the W&B `train/step` axis; evaluation uses
`eval/train_step`; each high-frequency `polar_tmax_system/node_N/*` GPU sidecar
uses `polar_tmax_system/node_N/train_step`. The trainer's main rank atomically
publishes each completed optimizer step for every node's sidecar; multiple GPU
samples can therefore share one X value. Raw GPU CSVs
retain Unix wall time and the sampled train step so startup, teardown, and
within-step idle intervals remain available for offline analysis. Every stage
duration is under the top-level `timing/*` namespace: `timing/session_ms/*`,
`timing/pipeline_ms/*`, and `timing/inference/*_ms_*` are milliseconds, while
Slime `timing/*_time`, `timing/service_window`, and
`timing/service_time_max` are seconds. `perf/*` is reserved for throughput,
TFLOPS, transfer rates, and ratios; `polar/*` is reserved for business counts,
rates, rewards, and failure classifications.

After a trainer batch commits successfully, the rollout logger saves two
representative complete message trajectories at every tenth zero-based
`train/step` (10, 20, 30, ...): the longest highest-reward session and the
longest lowest-reward session when both exist. Each step is written as a
two-line, mode-0600 JSONL file under
`${RUN_DIR}/trajectory_examples/trajectory_examples_step_XXXXXX.jsonl` and as
the W&B Table `examples/rollout_trajectories` on the same `train/step`. Message
content is not truncated; credential-shaped strings are recursively redacted,
and W&B receives a hash instead of the raw session ID. Configure the cadence
and count with `POLAR_ROLLOUT_EXAMPLE_INTERVAL` and
`POLAR_ROLLOUT_EXAMPLE_COUNT`. Local and W&B telemetry are fail-open, so an
NFS, serialization, or W&B error only emits a warning and cannot fail training.

- `timing/startup_*` covers submit-to-Pyxis entry (including Slurm queueing),
  batch-shell-to-container entry, container preflight, all Ray ranks and GPU
  resources becoming visible, Polar rollout/gateway/UDS readiness, placement
  group scheduling, SGLang engine health plus router registration, trainer
  model initialization, trainer/rollout wiring, initial weight sync, and the
  first successful optimizer step. The async launcher starts SGLang and trainer
  model loading concurrently; `startup_parallel_model_initialization_time` is
  the wall-clock join of those branches, while each branch retains its own
  duration. Launcher markers use real Unix-nanosecond timestamp pairs across
  processes and local phases use monotonic clocks; missing or reversed marker
  pairs are omitted rather than estimated. Startup metrics are logged at the
  allocation's first pending `train/step` (zero for a fresh run, the resumed
  nonzero step otherwise).
- `timing/session_ms/*` includes rollout-server dispatch/result wait, gateway
  register/init/ready/run/postrun queues, main/eval container startup and
  prepare, agent setup/exec/postprocess, trajectory build, evaluation,
  teardown, and both gateway and rollout-pipeline end-to-end spans.
  Container startup is exposed directly as `container_start_mean` and
  `eval_container_start_mean`; the equivalent `runtime_validation_mean` names
  remain for compatibility with older runs.
- `timing/runtime_exec/*` measures outer Apptainer/Docker exec durations,
  including container entry overhead. `timing/mini_swe_command/*` measures
  command durations inside mini-swe-agent. Fixed categories include
  `git_diff`, `git_status`, tests, package installs, builds, and
  filesystem/shell work. Metrics explicitly distinguish per-session totals
  from per-command means; matching counts, failures, and timeouts remain under
  `polar/runtime_exec/*` and `polar/mini_swe_command/*`. Raw command text is
  never sent to W&B.
- The idempotent TMax container-prepare command retries transient Apptainer
  failures up to three times with exponential backoff. Harbor verifier uploads
  use the same three-attempt policy and preserve tar/Apptainer stderr when all
  attempts fail; deterministic failures still return an error and are masked
  from training.
- `timing/inference/*` reports sanitized SGLang e2e, API-dispatch, queue,
  forward, prefill, decode, and inference-service durations (mean/p95/max
  where available). Non-duration token, request-load, and profiled-completion
  statistics remain under `polar/inference/*`. Durations are emitted only when
  SGLang supplies an explicit duration or a valid timestamp pair; Polar does
  not invent a forward duration from the e2e envelope. SGLang 0.5.13's
  `decode_throughput` is intentionally discarded because its non-streaming
  first-token timestamp is taken at final response delivery and produces
  impossible throughput values for the gateway's forced non-streaming calls.
- Trainer metrics include rollout service/handoff, sample conversion, DP split,
  generate e2e, fused Megatron forward-backward dispatch, optimizer dispatch,
  weight sync, and checkpoint save. Exact current-stream CUDA phase timing is
  optional via `SLIME_PROFILE_CUDA_PHASES=1`; it is off by default because the
  required synchronization changes the overlap being measured.
- Evaluation quality metrics use `eval/<dataset>/*`; their duration metrics use
  `timing/eval/<dataset>/*`. Both use `eval/train_step`, because a concurrent
  baseline may finish after later optimizer steps have already been logged.

The aggregate `init`, `run`, `postrun`, and `e2e` values are envelopes. Their
named subphases are nested and must not be added together. Likewise, the outer
runtime exec includes mini-swe-agent's inner command time. Trainer `train_time`
contains reference/log-prob and actor work, actor time contains fused
forward-backward plus optimizer work, and rollout `service_window` can overlap
the driver handoff wait. Startup branch and end-to-end metrics are also
overlapping envelopes, not an additive accounting table.

The launcher validates GPU capacity, full use of every requested GPU, actor
DP/global-batch divisibility, and minimum fully-async sessions/run workers per
rollout GPU before submission. For example, overriding only `NUM_NODES=1` is
rejected because the default 8 actor + 24 rollout GPUs cannot fit. A one-node
experiment must also set
`ACTOR_NUM_NODES=1 ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=4
ROLLOUT_NUM_GPUS_PER_ENGINE=2` explicitly. Set
`TMAX_REQUIRE_FULL_GPU_ALLOCATION=0` only for a deliberate under-allocation
experiment that will not be subject to the cluster's idle-GPU policy.

## Prepare Qwen3.5-9B weights

Download the immutable model snapshot into the shared path, then convert only
its text backbone to a Megatron release checkpoint using the same model args as
training:

```bash
cd /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/ProRL-Agent-Server
source examples/tmax_slime_grpo/env.cwdfw.sh
hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir "${HF_CHECKPOINT}"
bash examples/tmax_slime_grpo/convert_weights.sh
```

The TMax conversion wrapper writes to a job-specific staging directory,
validates the release checkpoint, and only then renames it to `${REF_LOAD}`.
It must run inside the training container on a GPU node. The launcher requires
`${HF_CHECKPOINT}` and
`${REF_LOAD}/latest_checkpointed_iteration.txt`; it fails before model startup
if either is missing. Changing between 4B and 9B is a new run and checkpoint
lineage—never resume a 4B `SAVE_DIR` with the 9B architecture.

Prepare the separate mini-swe-agent runtime once. This does not modify the
training venv or repository dependency files:

```bash
cd /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/ProRL-Agent-Server
bash examples/tmax_slime_grpo/prepare_mini_swe_agent.sh
```

The runtime is a relocatable Python 3.12 venv mounted read-only at
`/opt/polar-mini-swe-agent`; it is not installed into each task image. Re-run
the command after updating this checkout; this updates only the shared portable
runtime and does **not** rebuild any task SIF. The injected runner also connects
LiteLLM directly to the job-private gateway Unix socket and exposes the optional
HTTP-proxy socket on container loopback for mini-SWE action subprocesses. The
timed local environment also
sanitizes control-heavy command output and keeps a bounded head/tail excerpt
before mini-SWE JSON-escapes it, preventing binary output from expanding a
single model turn beyond the SGLang context. The preparer automatically rebuilds
the older non-relocatable layout whose global `PYTHONPATH` could leak Python
3.12 packages into Python 3.10 task commands.

## Shared training environment ABI

The Polar training venv is paired with Transformer Engine 2.16.1 and requires
`nvidia-cublas==13.3.0.5`. Torch 2.11's default CUDA-13.0 dependency resolver
can otherwise replace it with 13.1.0.3, which lacks a symbol required by
Transformer Engine. Any `uv pip install` targeting the shared training venv
must therefore use the checked-in override:

```bash
cd /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/ProRL-Agent-Server
UV_OVERRIDE="$PWD/examples/tmax_slime_grpo/training-uv-overrides.txt" \
  uv pip install --python /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/.python/polar/bin/python \
  <packages...>
```

Submission performs a real `torch -> transformer_engine.pytorch -> Megatron`
import before calling `sbatch`, and every allocated node repeats the check
inside the training image. A future ABI regression therefore fails before GPU
submission, or at the container boundary if the image itself differs.

## Training data selection

The default full run uses the deterministic first 14,501 TMax tasks for
training and the final 100 as `tmax_holdout`; the two windows cover all 14,601
tasks without a gap or overlap. Submission requires every selected SIF before
allocating GPUs. While the full SIF set is being rebuilt, the validated prefix
run uses the first 900 for training and the next 100 for holdout:

```bash
RUN_ID=tmax-900t100h-tb20-$(date -u +%Y%m%dT%H%M%SZ) \
TMAX_MAX_TASKS=900 \
TMAX_EVAL_START_INDEX=900 \
TMAX_EVAL_MAX_TASKS=100 \
TMAX_EVAL_INTERVAL=56 \
bash examples/tmax_slime_grpo/submit_slurm.sh
```

With eight unique prompts and 32 trajectories per prompt, every rollout batch
is one global-batch-256 optimizer step. This prefix has `ceil(900/8)=113`
rollout batches and 113 optimizer `train/step` values (zero-based final step
112). Interval 56 gives comparable eval points at baseline, mid-training, and
the final update.

The second fixed eval is the complete 89-task Terminal-Bench 2.0 set. It is
reported separately from `tmax_holdout`; both are evaluated against the same
model snapshot.
The default cluster paths point at Harbor's immutable `terminal-bench@2.0`
release commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c` and its exact
89 tagged enroot `.sqsh` images. Apptainer runs these SquashFS images directly;
they do not need to be rebuilt as SIFs. Override
`TMAX_HARBOR_EVAL_TASKS_DIR` and `TMAX_HARBOR_EVAL_IMAGE_DIR` to use another
local Harbor dataset. The run also requires an online W&B key and generates a
timestamped run id.

Fixed eval sends dataset-specific temperature, top-p, response cap, and other
sampling overrides through mini-SWE-agent into LiteLLM/SGLang. The TMax holdout
keeps its independent `temperature=0.2`, `top_p=1.0`, and (with the current
paper-aligned rollout defaults) a 16,384-token per-turn cap;
Terminal-Bench 2.0 uses `temperature=0.7`, `top_p=0.95`, a 16,384-token per-turn
cap, and a 64-step agent budget. Each prompt receives the same deterministic
seed at every eval point. Eval waits for every configured
sample and requires all 100 TMax and all 89 Terminal-Bench samples to be valid
by default; lower a corresponding minimum only when deliberately accepting an
incomplete evaluation.
Valid/error counts are logged before an insufficient eval fails, so no
final-eval or training-complete marker is written for a non-comparable result.
The metrics are `eval/tmax_holdout/reward_mean`,
`eval/terminal_bench_2_0/reward_mean`, and the valid-sample-weighted
`eval/aggregate/reward_weighted_mean`; all use `eval/train_step`. With complete
evals, the aggregate weights are exactly 100:89. The 64-step Terminal-Bench
budget is pinned in its eval config and remains independent of any future
`POLAR_AGENT_STEP_LIMIT` training override.

Periodic eval deliberately defaults to one Terminal-Bench attempt per task.
The paper reports the average of five rollouts per prompt, which costs five
full benchmark passes. Use the existing explicit override for a dedicated
paper-grade final evaluation rather than multiplying every periodic eval:

```bash
TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT=5 \
bash examples/tmax_slime_grpo/submit_slurm.sh
```
The inline training eval deliberately caps agent/verifier wall time at 900/600
seconds so it fits the resumable two-hour training allocation. These caps and
the tasks' original official timeouts are both pinned in every eval row; this is
therefore a fixed Terminal-Bench-2.0-derived external eval, not an uncapped
official leaderboard submission. Set both timeout caps to `0` in a dedicated,
long-walltime eval job when official timeout parity is required.

The train and eval JSONLs must resolve to different canonical filesystem paths;
symlink or `..` aliases fail before either file can be written. Submission
validates each deterministic window and rejects any overlapping `task_name`.
Rank 0 repeats both strict validations when the allocation actually starts,
including when `TMAX_PREPARE_DATA=0` and `TMAX_PREPARE_EVAL_DATA=0`, because a
queued job must not trust mutable files checked hours earlier. It then writes
`data/runs/${RUN_ID}/tmax-data-integrity.json`, containing exact SHA-256 hashes
and row counts, and copies that manifest into the Ray runtime through
`POLAR_EVAL_DATA_INTEGRITY_B64`. `TMAX_TRAIN_DATA_SHA256` and
`TMAX_EVAL_DATA_SHA256` expose the exact manifest digests and are persisted in
run state, so a resumed checkpoint cannot silently accept changed prompt or
metadata bytes merely because task order and SIF paths still match, and a later
allocation cannot silently re-pin changed eval content. The generic Polar eval
bridge hashes the bytes it parses on every baseline/final evaluation and fails
before submission if the file changed or an eval dataset is absent from the
manifest.

The temporary 900/100 prefix requires only the first 1,000 SIFs. The default
14,501/100 run requires all 14,601. Re-running the builder is incremental:
existing non-empty regular SIFs are locked and skipped, while failures leave no
published partial target.

```bash
source examples/tmax_slime_grpo/env.cwdfw.sh
TMAX_SIF_MAX_TASKS=-1 \
TMAX_SIF_BUILD_SHARDS=1000 \
TMAX_SIF_BUILD_ARRAY_PARALLEL=100 \
TMAX_SIF_BUILD_JOBS_PER_TASK=4 \
TMAX_SIF_BUILD_MEM=64G \
TMAX_SIF_BASE_SIF="${POLAR_DATA_ROOT}/container/ubuntu-22.04-base.sif" \
bash examples/tmax-15k/submit_build_sifs_slurm.sh
```

```bash
SUBMIT_BACKEND=sbatch bash examples/tmax_slime_grpo/submit_slurm.sh
```

For an intentional partial-data smoke test, set `TMAX_ONLY_READY=1` together
with a small `TMAX_MAX_TASKS`. Missing images are dropped only from that fixed
prefix; ready tasks after the prefix never backfill it.

Use `TMAX_PREPARE_DATA=0` to reuse the run-specific generated file under
`data/runs/${RUN_ID}/tmax-train.jsonl`. Set `TMAX_AGENT_HARNESS=codex` only when
the shared Codex tree has been prepared separately.

To validate data, shared paths, and the generated Slurm command without
submitting a job, add `SUBMIT_DRY_RUN=1`.

`submit_slurm.sh` intentionally delegates the generic Pyxis/sbatch machinery to
`examples/swegym_slime_grpo/submit_slurm.sh`, and TMax `run.sh` similarly reuses
the generic Ray/Slime launcher. Before delegation, the wrapper supplies TMax's
prompt data, topology, Polar config, training script, and a TMax banner label;
no SWE-Gym dataset or harness is selected.

## Seed a new run from an existing checkpoint

`LOAD_DIR` can seed a new `RUN_ID` and `SAVE_DIR` without copying a distributed
checkpoint. The saved rollout data-source cursor is part of that checkpoint, so
reuse the exact prompt JSONL that produced it:

```bash
RUN_ID=tmax-recovery-01 \
SAVE_DIR="${POLAR_DATA_ROOT}/ckpt/tmax-recovery-01" \
LOAD_DIR="${POLAR_DATA_ROOT}/ckpt/tmax-old-run" \
TMAX_TRAIN_DATA="${POLAR_DATA_ROOT}/runs/tmax-old-run/tmax-train.jsonl" \
TMAX_PREPARE_DATA=0 \
bash examples/tmax_slime_grpo/run_every_10_minutes.sh
```

The seed directory is used only while the new save directory is empty. After
the first new checkpoint, all later allocations load `SAVE_DIR`, so a watcher
cannot accidentally jump back to the seed iteration. The epoch target remains
derived from the prompt count (for 900 training prompts and batch eight, final rollout id 112),
because the checkpoint restores the already-consumed data-source position.
For fully-async prefetch, checkpoints save the cursor immediately before the
earliest outstanding reservation. A restart may replay a small suffix that was
completed out of order, but it cannot skip a prompt merely because that prompt
was prefetched before the model checkpoint committed.

## Full training

The default full run requires all 14,601 SIFs. After they are present:

```bash
TMAX_ONLY_READY=0 \
TMAX_MAX_TASKS=14501 \
TMAX_EVAL_START_INDEX=14501 \
TMAX_EVAL_MAX_TASKS=100 \
RUN_ID=tmax-qwen35-9b-full \
SUBMIT_BACKEND=sbatch \
bash examples/tmax_slime_grpo/submit_slurm.sh
```

The launcher rejects a missing SIF, a non-14,601 source tree, any train/holdout
overlap, or a changed eval manifest before it requests GPUs.

Checkpoints are written under `${POLAR_DATA_ROOT}/ckpt/${RUN_ID}`, rendered
Polar configs and per-node GPU CSVs under `${POLAR_DATA_ROOT}/runs/${RUN_ID}`,
and reward, scheduler, staleness, and GPU metrics are attached to the same W&B
run id.

## Four-hour job continuation

Use the watcher as the entry point when training must span multiple Slurm
allocations. It immediately submits the first job, checks every ten minutes,
and reuses the same `RUN_ID`, checkpoint directory, prompt JSONL, and W&B run:

```bash
cd /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/ProRL-Agent-Server
bash examples/tmax_slime_grpo/run_every_10_minutes.sh
```

Keep the watcher in `tmux` or another persistent login-node session. Its state
is stored in `data/runs/tmax_slime_grpo/current_run.env`; a direct
`submit_slurm.sh` invocation also updates that file, so the watcher can be
started after an initial manual submission. Use an explicit id to start or
select another logical run:

```bash
RUN_ID=tmax-qwen35-9b-run-02 \
bash examples/tmax_slime_grpo/run_every_10_minutes.sh
```

The state file intentionally pins topology and batch parameters. A run created
with the previous 2-node, batch-5 defaults will therefore stay on those values.
Use a new `RUN_ID` for the 4-node defaults; changing batch size in place also
changes the epoch/checkpoint iteration meaning and must not silently resume the
old run.

At the default four-hour wall time, Slime stops at the first completed rollout
after the deadline computed from `SLURM_JOB_END_TIME` (3h30m by default),
synchronously saves model, optimizer, scheduler, and rollout dataset state,
then exits successfully. The watcher submits the next allocation unless
`TRAINING_COMPLETE` exists or the final checkpoint iteration has been reached.
Adjust the reserve with
`TMAX_GRACEFUL_EXIT_BUFFER_SECONDS`; set `TMAX_ENABLE_GRACEFUL_EXIT=0` to
disable it.

The default 1,800-second reserve is deliberate: the deadline is checked after
an in-flight rollout, and saving the distributed model is synchronous. Buffers
below `TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS` are rejected unless that safety
floor is explicitly changed for a controlled experiment.

The watcher tracks the exact job id recorded in an atomic submission receipt
and reconciles a missing receipt against the exact `RUN_ID`-specific job name;
jobs from another run cannot make this one appear active. Any three consecutive
terminal jobs or submission attempts without a newer complete model+rollout
checkpoint stop automatic submission, even when Slurm reports different states
or exit codes. After inspecting and fixing the cause, restart deliberately with:

```bash
TMAX_WATCH_RESET_FAILURES=1 bash examples/tmax_slime_grpo/run_every_10_minutes.sh
```

For a status-only check that never submits:

```bash
bash examples/tmax_slime_grpo/watch_training.sh
```
