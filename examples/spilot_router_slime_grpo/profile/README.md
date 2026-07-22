# SPilot train/rollout profiling

This directory contains a short end-to-end profiling suite for two decisions:

1. whether the same GPU budget is better spent on synchronous Slime
   `--colocate` or disjoint fully-async train/rollout pools;
2. for fully async, where to put the train/rollout split and how much async
   depth is actually useful.

The jobs keep the model, dataset, effective GRPO batch (`8 x 32 = 256`), four
gateway ranks, provider admission limits, and gateway worker limits fixed.
Only GPU placement and async depth change.  Arms are serialized by default so
they do not compete for the same remote model-pool quota.

## Quick start

Planning is read-only:

```bash
cd src/ProRL-Agent-Server
bash examples/spilot_router_slime_grpo/profile/submit_profile.sh plan
```

Start with the smallest useful comparison:

```bash
PROFILE_STEPS=3 \
bash examples/spilot_router_slime_grpo/profile/submit_profile.sh submit \
  async-16t16r-l1 async-16t16r-l3 collocate-16shared
```

For the direct `8 train / 32 rollout` comparison, use the controlled four-
gateway matrix below.  The 40-GPU arm allocates a fifth Ray node but does not
start a fifth gateway, so provider admission remains exactly `8 / 32` across
all three arms:

```bash
PROFILE_STEPS=3 \
bash examples/spilot_router_slime_grpo/profile/submit_profile.sh submit \
  async-16t16r-l3 async-8t32r-l3 collocate-32shared
```

Then test the resource split only if fully async is still competitive:

```bash
PROFILE_STEPS=3 \
bash examples/spilot_router_slime_grpo/profile/submit_profile.sh submit \
  async-8t24r-l3 async-16t16r-l3 async-8t8r-l3
```

`submit_profile.sh list` shows all arms and overrides.  Every arm gets a new
`RUN_ID`, `SAVE_DIR`, run-state file, and W&B run.  A suite manifest is written
under:

```text
data/runs/spilot_router_slime_grpo/profile/<PROFILE_ID>/manifest.tsv
```

The suite is intentionally disposable: training eval, graceful resume, and
full checkpoint writes are disabled.  Do not start `watch_training.sh` for
these jobs.  `--save` remains configured so rollout metric journals have a
stable location, but `--save-interval` is omitted, including the normally
forced final checkpoint. W&B defaults to offline mode; all telemetry needed by
the local summarizer remains in the run directory.

## Arms

| Arm | Allocation | Actor | Rollout | Scheduling |
| --- | ---: | ---: | ---: | --- |
| `async-16t16r-l1` | 4 x 8 = 32 GPU | 16, TP4 | 16, TP1 | fully async, depth 1 |
| `async-16t16r-l3` | 4 x 8 = 32 GPU | 16, TP4 | 16, TP1 | fully async, depth 3 |
| `async-8t32r-l3` | 5 x 8 = 40 GPU | 8, TP4 | 32, TP1 | fully async, depth 3; 4 gateways |
| `async-8t24r-l3` | 4 x 8 = 32 GPU | 8, TP4 | 24, TP1 | fully async, depth 3 |
| `async-8t8r-l3` | 4 x 4 = 16 GPU | 8, TP4 | 8, TP1 | fully async, depth 3 |
| `async-4t12r-l3` | 4 x 4 = 16 GPU | 4, TP4 | 12, TP1 | fully async, depth 3 |
| `collocate-16shared` | 4 x 4 = 16 GPU | 16, TP4 | 16, TP1 | synchronous, shared GPUs |
| `collocate-32shared` | 4 x 8 = 32 GPU | 32, TP4 | 32, TP1 | synchronous, shared GPUs |

The clean fixed-32-GPU ratio choices are 8/24 and 16/16.  A nominal 24/8
actor split is invalid for global batch 256 because TP4 gives actor DP6, which
does not divide 256.  A nominal 16/8 four-node arm also cannot be represented
cleanly with homogeneous per-node Slurm GRES: it either leaves eight allocated
GPUs idle or changes the gateway-node count.

`async-8t32r-l3` is intentionally not a fixed-budget comparison: it uses 40
GPUs versus 32 for `async-16t16r-l3`.  Use accepted work per allocated GPU-hour
alongside wall throughput.  `collocate-32shared` is the equal-32-GPU-budget
collocate baseline; `collocate-16shared` instead matches the logical 16-GPU
width of each stage in the disjoint 16/16 arm.

`async-4t12r-l3` and `async-8t8r-l3` are useful cost probes, but first verify
that the smaller learner does not OOM at the production sequence-length
distribution.

## Use the same starting point

The default is the original Qwen3.5-9B release seed and the fixed formal
training JSONL. The submitter unconditionally pins the 9B HF assets, reference
release, model arguments, agent model name, and `LOAD_DIR`, so stale Qwen4 or
numbered-checkpoint variables in the submitting shell cannot change an arm.
Each fresh run starts at rollout zero and has a unique save/state directory.
To compare from a trained policy, all arms must use the same numbered checkpoint
and its exact data-source JSONL:

```bash
PROFILE_LOAD_DIR=/absolute/path/to/checkpoint \
PROFILE_TRAIN_DATA=/absolute/path/to/the/exact/train.jsonl \
PROFILE_STEPS=3 \
bash examples/spilot_router_slime_grpo/profile/submit_profile.sh submit \
  async-16t16r-l3 collocate-16shared
```

If the checkpoint tracker is `N`, the script sets Slime's exclusive
`TMAX_NUM_ROLLOUT` boundary to `N + 1 + PROFILE_STEPS`.  It refuses a numbered
seed without `PROFILE_TRAIN_DATA`.

For a final decision rather than a directional probe, collect at least six
post-warmup optimizer steps and two repetitions (for one warmup step, use at
least `PROFILE_STEPS=7 PROFILE_REPEATS=2`). Repetitions are serialized by the
standard launcher.

The fast profile defaults to `PROFILE_STEPS=3`: one warmup plus two steady
steps, matching the strict cross-suite report contract. It also uses the same
memory and completion policy in every arm:

- `PROFILE_MAX_TOKENS_PER_GPU=24576` bounds each dynamic microbatch and every
  individual trajectory. The bridge preserves the complete prompt and clips
  only the response to the longest exact causal prefix that fits, retaining
  aligned rollout log probabilities for every kept response token. The 24K
  default is shared with direct TMax so the 9B collocate arm remains within
  physical H100 memory and the cross-suite report uses one token contract;
- `PROFILE_MIN_COMPLETE_ACCEPT_FRACTION=0.5` and
  `PROFILE_EARLY_STOP_GRACE_SESSIONS=2` make a 32-session prompt group return
  after 18 usable sessions. The old grace of 16 accidentally restored a
  32/32 long-tail barrier.

Early stopping deliberately sends terminal cancellation to remaining session
runtimes. The gateway's retained-lease `fail-allocation` check remains enabled:
it is a containment invariant, not a profiling guard to bypass. The shorter
18-session threshold reduces exposure to naturally timed-out stragglers, but a
real runtime-destruction or admission-release failure will still invalidate the
arm instead of silently overbooking the model pool.

## Summarize results

The parser uses only the Python standard library.  It accepts either `job-*`
directories or logical run directories and tolerates incomplete jobs:

```bash
python examples/spilot_router_slime_grpo/profile/profile_summary.py \
  --warmup-steps 1 \
  --log-root /path/to/data/logs/slurm \
  --log-timezone UTC \
  --json /tmp/spilot-profile.json \
  /path/to/data/runs/spilot-prof-async-16t16r-l1-* \
  /path/to/data/runs/spilot-prof-async-16t16r-l3-* \
  /path/to/data/runs/spilot-prof-collocate-16shared-*
```

Slime's unqualified timestamps are interpreted in the explicit IANA timezone
and converted to UTC before they are matched to GPU `sample_time` epochs. The
CW-DFW compute nodes used by this launcher emit both Slime and nvidia-smi
timestamps in UTC, even when a login shell is in `America/Los_Angeles`, so the
profile/report launchers default to `UTC`. The parser does not infer timezone
offsets from the telemetry values.

The compact table reports steady-state step time, trainer wait fraction,
accepted sessions/s, queue-backlog slope, policy staleness, aggregate/role GPU
utilization, and accepted sessions per allocated GPU-hour.  The JSON includes
per-step values, p50/p95 summaries, accepted groups/sessions/tokens, queue
gauges and trends, GPU memory/power/temperature, source paths, and warnings for
missing telemetry.

Run its self-contained tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python examples/spilot_router_slime_grpo/profile/test_profile_summary.py
```

## Decision rule

Use accepted trainable work, not raw SGLang token throughput, as the numerator.
For each arm:

- reject a run whose queue grows throughout the steady window, whose drop/error
  fraction changes materially, or whose staleness/importance-ratio guardrail is
  worse than the training policy allows;
- among the remaining arms, keep configurations within 95% of the best
  accepted-session or accepted-token throughput;
- choose the one with the highest accepted work per allocated GPU-hour; use
  lower mean/p95 staleness as the tie-breaker.

Interpret the controlled comparisons in this order:

1. `async-16t16r-l1` vs `async-16t16r-l3` isolates pipeline depth.  Choose the
   smallest depth within 95% of maximum throughput; more in-flight work is not
   free because it raises sample age.
2. `async-8t24r-l3` vs `async-16t16r-l3` locates the fixed-32-GPU balance.  A
   high trainer wait ratio does not by itself prove that more rollout GPUs
   help: if rollout GPU utilization stays near zero, the remote provider or
   CPU agent path is the bottleneck.
3. `async-8t8r-l3` checks whether half of the GPUs preserve nearly all useful
   throughput.  This is particularly important for Router training, where
   remote candidate calls can dominate local decode.
4. `collocate-16shared` has the same logical 16-train/16-rollout capacity as
   the 32-GPU disjoint arm but pays offload/onload and loses cross-batch
   overlap.  Compare both wall throughput and GPU-hour efficiency; compare it
   to `async-8t8r-l3` for a strict 16-GPU-budget decision.

Three steps are a fast systems probe, not a quality verdict.  If collocate and
async are close, extend only the top two from the same checkpoint and compare
held-out improvement per wall hour before changing the production topology.

## Launcher hooks

Two default-off hooks are shared with the production launcher:

- `TMAX_TRAIN_MODE=fully_async|colocate` selects `train_async.py` or
  `train.py --colocate` and changes topology accounting from `actor + rollout`
  to `max(actor, rollout)` only in collocate mode. The shared launcher also
  removes CUDA `expandable_segments` in colocate mode because TorchMemorySaver
  does not support it; fully async retains the existing allocator;
- `TMAX_PROFILE_DISABLE_CHECKPOINT=1` omits all model-checkpoint CLI arguments
  (including `--save`) and async lifecycle markers, and requires graceful exit
  to be disabled.  This avoids Megatron's `--save`/`--save-interval` invariant
  and Slime's forced final checkpoint.  Rollout, W&B, and GPU telemetry remain
  available; the disposable profile does not write the durable async rollout
  metric journal under the model save directory.

Production defaults remain `fully_async` and checkpointing enabled.
