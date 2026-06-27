# TMax Slime GRPO

This example trains Qwen3.5-4B with Slime GRPO while Polar runs each TMax
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
verifier timeout, workdir, and image path. Full mode fails before Slurm
submission if any selected image is missing.

## Cluster setup

Defaults are in `env.cwdfw.sh`:

- dataset: `/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data/tmax-15k`
- SIFs: `/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data/tmax-15k-sif`
- Python: `/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/.python/polar/bin/python`
- training image: `data/container/flappydora-ubuntu22.04-cuda13.3.sqsh`
- Slurm partition: `batch` (`interactive` is limited to two nodes per user)
- topology: eight 8xH100 nodes, five actor nodes and three rollout nodes
- batch: five prompts by eight trajectories, global batch 40 (actor TP=2, DP=20)
- scheduler: fully async with a two-batch prefetch window (at most 80 sessions)
- W&B axis: model, rollout, Polar, and performance metrics use `train/step`
- nested sandbox mode: `POLAR_APPTAINER_NO_INSTANCE=1`
- checkpoint cadence: every completed rollout (`SAVE_INTERVAL=1`)
- four-hour allocation guard: checkpoint and exit at 3h45m by default

Prepare the separate mini-swe-agent runtime once. This does not modify the
training venv or repository dependency files:

```bash
cd /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/src/ProRL-Agent-Server
bash examples/tmax_slime_grpo/prepare_mini_swe_agent.sh
```

## Partial-data training

The default run uses the first 80 tasks whose SIFs already exist, requires an
online W&B key, and generates a timestamped run id:

```bash
SUBMIT_BACKEND=sbatch bash examples/tmax_slime_grpo/submit_slurm.sh
```

Use `TMAX_PREPARE_DATA=0` to reuse the run-specific generated file under
`data/runs/${RUN_ID}/tmax-train.jsonl`. Set `TMAX_AGENT_HARNESS=codex` only when
the shared Codex tree has been prepared separately.

To validate data, shared paths, and the generated Slurm command without
submitting a job, add `SUBMIT_DRY_RUN=1`.

## Full training

After all 14,601 SIFs are present:

```bash
TMAX_ONLY_READY=0 \
TMAX_MAX_TASKS=-1 \
RUN_ID=tmax-qwen35-4b-full \
SUBMIT_BACKEND=sbatch \
bash examples/tmax_slime_grpo/submit_slurm.sh
```

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
RUN_ID=tmax-qwen35-4b-run-02 \
bash examples/tmax_slime_grpo/run_every_10_minutes.sh
```

At the default four-hour wall time, Slime stops at the first completed rollout
after 3h45m, synchronously saves model, optimizer, scheduler, and rollout
dataset state, then exits successfully. The watcher submits the next
allocation unless `TRAINING_COMPLETE` exists or the final checkpoint iteration
has been reached. Adjust the reserve with
`TMAX_GRACEFUL_EXIT_BUFFER_SECONDS`; set `TMAX_ENABLE_GRACEFUL_EXIT=0` to
disable it.

For a status-only check that never submits:

```bash
bash examples/tmax_slime_grpo/watch_training.sh
```
