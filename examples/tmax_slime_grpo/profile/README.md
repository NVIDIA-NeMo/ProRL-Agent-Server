# TMax GPU allocation profiling

This directory contains the disposable systems probe for the original TMax
`mini_swe_agent` training path. It uses the same Qwen3.5-9B release seed,
prompt JSONL, global batch, and telemetry contract for every arm.

The default first-stage sweep is:

| Arm | Allocation | Purpose |
| --- | ---: | --- |
| `async-16t16r-l4` | 32 GPU | equal train/rollout split |
| `async-8t24r-l4` | 32 GPU | current production allocation baseline |
| `async-8t32r-l4` | 40 GPU | rollout-heavy absolute-throughput ceiling |
| `collocate-32shared` | 32 GPU | shared-GPU/offload alternative |

`async-4t28r-l4` is available as a follow-up arm if the first sweep remains
strongly rollout-bound. It is not part of the default sweep because a
four-GPU actor has only TP4/DP1 and can become the next bottleneck.

## Plan before submitting

Planning is read-only:

```bash
PROFILE_STEPS=3 \
PROFILE_LOAD_DIR=/abs/Qwen3.5-9B_torch_dist \
PROFILE_TRAIN_DATA=/abs/tmax-train.jsonl \
bash examples/tmax_slime_grpo/profile/submit_profile.sh plan
```

The table must show `4x8`, `4x8`, `5x8`, and `4x8` allocations. The 5-node
arm is intentional: disjoint fully-async pools require 8 train + 32 rollout
GPU, or 40 GPU total.

## Submit

Only the explicit `submit` action mutates state or calls Slurm:

```bash
PROFILE_ID=tmax-gpu-profile-$(date -u +%Y%m%dT%H%M%SZ) \
PROFILE_STEPS=3 \
PROFILE_LOAD_DIR=/abs/Qwen3.5-9B_torch_dist \
PROFILE_TRAIN_DATA=/abs/tmax-train.jsonl \
PROFILE_PARTITION=backfill,batch \
PROFILE_DEPENDENCY_KIND=afterany \
bash examples/tmax_slime_grpo/profile/submit_profile.sh submit
```

Set `PROFILE_AFTER_JOB_ID` to serialize this suite after another profiling
batch. Each arm receives its own run state, W&B run, rollout directory, and
GPU CSVs. Evaluation, graceful checkpointing, and all model-checkpoint CLI
arguments are disabled. `submit` also requires an explicit checkpoint and
prompt JSONL, clears inherited 4B/model/data variables, and pins the complete
Qwen3.5-9B model contract for every arm. Before the first submission it hashes
the prompt JSONL and writes a batch `profile_contract.env`; the same contract
is copied to each run as `profile.env`, which makes the exact data SHA256
available to the strict summary even with evaluation disabled. It rechecks the
file before submitting every arm.

All default arms use `POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5`, grace 2, and a
32,768-token dynamic trainer cap. Since a valid TMax trajectory can still be a
complete 67,584-token pack, oversize individual samples are admitted alone;
the lower cap limits aggregate microbatch pressure without truncating the task.
The submission also carries the sanctioned four-hour idle-GPU reaper exemption.
Do not use `watch_training.sh` for these runs.

`collocate-32shared` is statically valid but has not yet been demonstrated for
the 9B TMax workload. For a lower-risk first launch, submit it alone with
`PROFILE_STEPS=1` before the three-step measurement suite.

## Summarize

The wrapper invokes the common parser and must be given the external Slurm log
directory because job stdout is not stored inside the logical run directory:

```bash
python examples/tmax_slime_grpo/profile/profile_summary.py \
  --warmup-steps 1 \
  --log-root /abs/data/logs/slurm \
  --log-timezone UTC \
  --json /abs/profile/summary.json \
  /abs/run-1 /abs/run-2 /abs/run-3 /abs/run-4
```

The timezone is explicit because Slime perf lines use compute-node wall time,
while GPU CSV `sample_time` is a UTC Unix epoch. The parser converts both onto
one UTC timeline; it never guesses a whole-hour offset from utilization data.

## Monitor and final report

`submit_monitor.sh` creates a serialized `cpu_short` relay chain and atomically
updates one status JSON. Reaching a relay's time budget is an expected,
successful handoff to the next segment; terminal failure in a monitored target
still makes the segment fail. Pass all GPU arm IDs plus the final report job ID:

```bash
bash examples/tmax_slime_grpo/profile/submit_monitor.sh \
  /abs/report/monitor-status.json \
  spilot-16-16=123 spilot-8-32=124 spilot-collocate=125 \
  tmax-16-16=126 tmax-8-24=127 tmax-8-32=128 tmax-collocate=129 \
  report=130
```

Set `MONITOR_ACCOUNT` for monitor relays and `REPORT_ACCOUNT` for the final
report. Both default to `ACCOUNT`, then `SBATCH_ACCOUNT`, then `nvr_lpr_llm`.
For example, prefix either command with `MONITOR_ACCOUNT=my_account` or
`REPORT_ACCOUNT=my_account`.

Schedule the self-contained HTML report after the last TMax arm with:

```bash
POLAR_DATA_ROOT=/abs/data \
bash examples/tmax_slime_grpo/profile/submit_report.sh \
  129 /abs/spilot/manifest.tsv /abs/tmax/manifest.tsv /abs/report
```

The report bundle contains HTML and Markdown reports, the strict analysis
JSON, both suite summaries, and arm/GPU-role audit CSVs. An arm is ranked only
when the Ray job succeeded, all expected train and rollout step records are
present, strict trainable-session provenance exists, and the model/checkpoint/
data/batch/code fingerprint is consistent within its suite. SPilot Router and
TMax remain separate workload blocks; their raw rates are never pooled.

`REPORT_COMPLETE` is written only when each manifest contains exactly the
four comparison arms (SPilot async depth 3, TMax async depth 4, plus each
suite's 32-GPU colocate arm), all eight analyzed arms are contract-valid,
both suites have a measured or directional candidate, and every report
artifact exists. A failed gate preserves the diagnostic bundle, writes
`REPORT_INCOMPLETE.json` with explicit reasons, and makes the report job exit
nonzero.

With three optimizer steps, only two intervals remain after warmup. Treat the
result as a directional systems recommendation, not a statistically
significant training-quality result. Rank both steady steps/hour and accepted
sessions per allocated GPU-hour; never call the 40-GPU arm optimal from raw
wall throughput alone.
