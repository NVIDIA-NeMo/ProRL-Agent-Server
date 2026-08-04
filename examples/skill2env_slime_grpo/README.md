# Skill2Env Slime GRPO with rubric PRM

This experiment fine-tunes Qwen3.5-4B on the locally available Skill2Env
tasks. Polar executes each task in its matching SIF, runs the deterministic
Harbor verifier, and then uses `harbor_rubric` to assign per-trace process
rewards with `azure/openai/gpt-5.3-codex` through NVIDIA Inference Hub's
Responses API.

The reward for trace `i` is:

```text
outcome_reward + 0.2 * judge_score_i / 5
```

The API credential is read from `PRM_KEY` in `~/.bashrc` at submission time.
It is copied only into the mode-0600 Slurm environment file and is never
written to the experiment config or run state.

Run a one-update, two-node smoke experiment:

```bash
bash examples/skill2env_slime_grpo/run_smoke.sh
```

Submit the complete one-epoch experiment over every task that currently has a
matching SIF:

```bash
bash examples/skill2env_slime_grpo/run_4node_full.sh
bash examples/skill2env_slime_grpo/run_4node_prefix_merging.sh
bash examples/skill2env_slime_grpo/run_8node_full.sh
```

The four-node job uses 8 learner GPUs (TP4/DP2) and 24 TP1 rollout engines,
with 8 prompts x 8 trajectories = global batch 64. The eight-node job uses
16 learner GPUs (TP4/DP4) and 48 TP1 rollout engines, with 16 prompts x 8
trajectories = global batch 128. Each topology has its own run-state file, so
both can run concurrently. Full jobs always use the `batch` partition (never
`backfill`) and request that partition's maximum four-hour wall time. Tool outputs are
included in the rubric judge input by default for this experiment; set
`PRM_INCLUDE_TOOL_OUTPUTS=false` to recover the original response-only judge.
Long trajectories are judged in chunks of at most 32 traces by default; set
`PRM_MAX_TRACES_PER_CALL=0` to restore a single call per trajectory.

`run_4node_prefix_merging.sh` is the four-node timing comparison. It stitches
an append-only agent request chain into one token-level trace while preserving
sampled-token log probabilities and zero-masking interstitial/tool-result
tokens. The rubric judge still receives matched tool outputs exactly once.

The implementation, controlled A/B comparison, smoke evidence, and full-job
topology are documented in [tool_output_prm_report.md](tool_output_prm_report.md).

Both commands regenerate a deterministic JSONL from the intersection of
`skill2env_batch_5` tasks and `skill2env_batch_5_sif` images before submission.
Tasks with unsupported runtime semantics and the API-designer task whose SIF
failed a real Apptainer mount in the smoke run are excluded.
Set `SKILL2ENV_PREPARE_DATA=0` to reuse an already generated, non-empty JSONL
when the source Lustre directory is temporarily slow.
