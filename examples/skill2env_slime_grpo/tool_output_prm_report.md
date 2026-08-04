# Tool-output rubric PRM report

Date: 2026-07-24

## Question

Does giving the rubric judge the observed result of each tool call produce a
more evidence-grounded process reward than showing only the assistant response
that issued the call?

## Implementation

`harbor_rubric` now accepts two optional settings:

```yaml
judge_include_tool_outputs: true
judge_tool_output_max_chars: 12000
judge_max_traces_per_call: 32
```

The first option defaults to `false`, so existing evaluators retain their old
response-only behavior. When enabled, the evaluator:

1. reads tool calls from each trace's `response_messages`;
2. finds the corresponding result in later cumulative `prompt_messages` by
   exact `tool_call_id`;
3. attaches the first observed result to the trace that issued the call;
4. avoids duplicating results repeated in subsequent cumulative prompts;
5. truncates each result independently; and
6. divides the overall trace-section budget across traces when necessary so
   every `trace_N` id remains visible to the judge; and
7. optionally splits long trajectories into bounded judge calls, then restores
   the chunk scores to their original global trace order.

An unobserved final tool call is left without a fabricated result.

## Controlled A/B re-judge

Both arms used the same saved nine-trace `summarize-meeting` trajectory, task
instruction, rubric, verifier outcome (`0.48334`), Responses endpoint, and
`openai/openai/gpt-5.1-codex`. A minimal endpoint request returned HTTP 200
before the comparison.

| Arm | Judge prompt chars | Parsed scores |
|---|---:|---|
| response only | 11,208 | `[2, 3, 3, 2, 2, -3, 1, 1, -2]` |
| response + tool output | 24,123 | `[3, 4, 4, 3, 2, -2, 1, 2, -1]` |

The response-only judge emitted an extra nonexistent `trace_9`; the parser
correctly ignored it. The tool-output judge emitted exactly `trace_0` through
`trace_8`.

The tool-output rationales were more evidence-grounded:

- directory inspection moved from 2 to 3 after the judge saw the actual file
  listing;
- reading context and transcript moved from 3 to 4 after successful contents
  were visible;
- reading the artifact index moved from 2 to 3 because its actual metadata was
  available;
- the faulty write remained negative (`-3` versus `-2`), with both arms
  identifying multi-owner actions and unsupported dates;
- post-write inspection moved from 1 to 2 because the judge could see the
  generated document rather than only the `cat` command.

The absolute scores are not deterministic and the two prompts differ by
design, so this is a qualitative grounding check rather than a statistical
claim. The important result is that the enabled arm can distinguish a command
from its observed success/failure and cite concrete output evidence.

## Automated verification

The focused suite covers default-off compatibility, exact call/result
association, no cumulative duplication, per-result truncation, preservation of
all trace ids under the total prompt cap, Responses payload behavior, partial
judge retries, reward blending, and Skill2Env runtime/data behavior.

```text
77 passed
```

The first end-to-end smoke exposed a long-trajectory edge case: 9 of 82
speculatively completed sessions with 42--64 traces omitted contiguous middle
scores (58 missing scores total) even after retries. Short trajectories were
complete. Setting `judge_max_traces_per_call: 32` addresses this without
changing the default behavior for other experiments. A real 64-trace session
that previously missed 14 scores was re-judged as two 32-trace chunks and
returned 64/64 scores with zero missing values. Call/result association is
built over the complete trajectory before slicing, so a call at a chunk's final
trace still receives the result observed in the following trace's prompt.
After adding chunk and boundary coverage, the focused suite reports:

```text
80 passed
```

## End-to-end training smoke

Slurm job `14241172` completed with exit code 0 in 34m58s. The accepted batch
contained 4 groups / 32 sessions, all 32 were successful and trainable, with
979 trainable traces and no parser-invalid, timeout, or terminal-error session.
Mean accepted reward was `0.393761` (standard deviation `0.333036`).

The learner log-prob guard passed (`masked_mean_abs_diff=0.00777721`, threshold
`0.5`). Step 0 then completed an actual dPPO update with:

```text
train/loss             0.02639867
train/ppo_kl           0.00032786
train/grad_norm        0.12188621
train/global_batch_size 32
```

Iteration 0 was successfully saved under
`local_data/ckpt/skill2env-qwen35-4b-rubric-prm-smoke-20260722T110715Z`.
Ray workers printed connection-reset/NCCL messages during process teardown after
the checkpoint, but Slurm recorded `COMPLETED 0:0`; these were shutdown noise,
not a failed rollout, optimization step, or checkpoint.

## Full training topologies

The primary full launcher uses eight 8xH100 nodes:

- learner: 2 nodes / 16 GPUs, TP4 and DP4;
- rollout: 6 nodes / 48 independent TP1 engines;
- one Polar gateway per node;
- 16 prompts x 8 trajectories = global batch 128;
- all 823 currently ready Skill2Env rows, one epoch (the launcher counts the
  prepared JSONL at submission time);
- GRPO advantage estimator with dPPO policy loss; and
- tool-output rubric PRM enabled.

The proportional four-node launcher uses the same model, dataset, judge, and
one-epoch schedule, with all GPU-dependent batch topology halved:

- learner: 1 node / 8 GPUs, TP4 and DP2;
- rollout: 3 nodes / 24 independent TP1 engines;
- 8 prompts x 8 trajectories = global batch 64; and
- its own run-state file so it can coexist with the eight-node run.

The smoke gate passed. The original full run was submitted to `backfill` as
job `14242111`, but it remained pending on priority and was cancelled before
startup on 2026-07-23 at the user's request.

The launcher now fixes eight-node full runs to `batch`, uses that partition's
four-hour limit, and rejects `PARTITION=backfill` before submission. The
replacement full run was submitted at 2026-07-23 02:29 PDT:

```text
Slurm job: 14277208
run id:    skill2env-qwen35-4b-rubric-prm-tool-output-8n-full-20260723T092832Z
account:   nvr_lpr_llm
partition: batch
request:   8 nodes x 8 H100, 04:00:00
```

The proportional four-node run was submitted at 2026-07-23 03:41 PDT:

```text
Slurm job: 14278458
run id:    skill2env-qwen35-4b-rubric-prm-tool-output-4n-full-20260723T104106Z
account:   nvr_lpr_llm
partition: batch
request:   4 nodes x 8 H100, 04:00:00
```

Both jobs received their requested allocation and exercised the real
rollout/judge/training path. They were later cancelled by the cluster's
occupied-idle job reaper rather than by an application exception:

| Job | Runtime | Progress before cancellation | Scheduler reason |
|---|---:|---|---|
| `14277208` (8 node) | 1h25m55s | completed optimizer step 0; began rollout/train iteration 1 | 48/64 GPUs reported idle for 30m |
| `14278458` (4 node) | 50m30s | log-prob guard passed; actor train reached microbatch 247/260 for step 0 | 11/32 GPUs reported idle for 30m |

The eight-node job is the requested full-training startup proof. Its first
accepted batch contained 16 groups / 128 sessions. All 128 sessions were
successful and trainable, with no timeout, terminal-error, or fully-masked
session. Mean accepted reward was `0.523726` (standard deviation `0.271050`),
and the batch contained 3,796 trainable traces. The trainer/rollout log-prob
guard passed with `masked_mean_abs_diff=0.00845339` against the `0.5`
threshold. The resulting dPPO update reported:

```text
train/loss              0.02092314
train/ppo_kl            0.00031643
train/grad_norm         0.05403605
train/global_batch_size 128
```

The next eight-node batch also passed the log-prob guard
(`masked_mean_abs_diff=0.00727092`) before the scheduler cancellation. Rollout
cursor states for iterations 0 and 1 were written, but no resumable model
checkpoint pointer was committed before cancellation.

The four-node run likewise reached real actor training: its first log-prob
guard passed with `masked_mean_abs_diff=0.00896798`. It was cancelled at
microbatch 247/260, before step-0 metrics or a resumable model checkpoint were
committed.

These cancellations expose a cluster-policy incompatibility with the
colocated fully-asynchronous topology: while the learner performs a long
update, many rollout GPUs become idle long enough for the occupied-idle reaper
to reclaim the allocation. They do not contradict the smoke result or the
eight-node evidence that rubric-judged trajectories correctly enter and update
the learner.

## Four-node prefix-merging timing

Job `14314119` ran the same four-node topology with
`builder.strategy: prefix_merging`. Successful sessions merged their
append-only completion chains into normally one trainable trace; the observed
completed-session distribution during the first batch was predominantly one
trace, with a small number of two-chain/two-trace sessions. For example, one
session merged all 16 of its raw completions into one trace with
`chains_reconstructed_full=1` and `completions_merged=16`.

The first complete optimizer step produced:

| Measurement | Per-request 4-node baseline | Prefix merging |
|---|---:|---:|
| rollout groups / trajectories | 8 / 64 | 8 / 64 |
| rollout collection | 318.5s | 226.5s |
| learner microbatches | 260 | 15 |
| learner log-prob pass | 572.0s | 140.3s |
| full learner train | approximately 42m from measured baseline progress | 433.6s |

Prefix merging therefore reduced the measured learner time to about 17% of
the old projected four-node time, approximately a 5.8x speedup. The step
completed normally with `train_rollout_logprob_abs_diff=0.0`, loss
`-0.00185875`, PPO KL `0.00067145`, and global batch size 64.

This timing run also revealed two external-data/service issues unrelated to
prefix merging:

- one PostgreSQL task group had a corrupt SIF and was dropped after Apptainer
  reported a squashfs read failure; and
- the configured `openai/openai/gpt-5.1-codex` judge began returning HTTP 404
  because that model was deprecated. Its scores therefore fell back to the
  verifier outcome in this timing run.

The timing job was cancelled after step 0 to avoid spending more allocation on
the deprecated judge. `PRM_MODEL` now defaults to the endpoint-verified
`azure/openai/gpt-5.3-codex` (HTTP 200 in the submission preflight), and a
replacement valid-PRM run was submitted:

```text
Slurm job: 14314401
run id:    skill2env-qwen35-4b-gpt53-rubric-prm-prefix-merging-4n-full-20260724T095914Z
account:   nvr_lpr_llm
partition: batch
request:   4 nodes x 8 H100, 04:00:00
```
