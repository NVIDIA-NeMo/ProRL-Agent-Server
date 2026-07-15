# Trajectory Evaluators

Evaluators score a built `Trajectory` into an `EvalResult` (an outcome reward
and/or per-trace rewards, plus metadata). The gateway then merges that reward
onto the trajectory's traces.

## Main files

- `base.py`: the evaluator contract (`async evaluate(trajectory, **runtime) -> EvalResult`).
- `session_completed.py`: reward by terminal status.
- `test_on_output.py`: apply the agent's changes and grade test output.
- `swebench_harness.py`: grade a patch with the SWE-bench harness.
- `harbor.py`: run a Harbor task's `tests/test.sh` in the live runtime.
- `harbor_rubric.py`: Harbor outcome plus a rubric-based LLM judge process reward.
- `_patch_utils.py`: `BasePatchEvaluator` — the shared extract → filter → apply →
  test flow both grading evaluators build on.

## Built-in strategies

**`session_completed`** — reward `1.0` if the session reached `COMPLETED`, else
`0.0`. Needs no runtime; handy as a smoke-test signal.

**`test_on_output`** — for custom/toy tasks. It extracts the agent's git diff,
(optionally) applies it on a fresh runtime, runs a test command, and **grades by
matching parsed test output — not the exit code**: it reads
`PASSED`/`FAILED`/`ERROR`/`SKIPPED <node>` lines and rewards `1.0` only when the
parsed result **exactly equals** the expected map.

| config key | required | meaning |
|---|---|---|
| `test_command` | yes | the command to run |
| `expected_output_json` | yes | `{node: "PASSED", ...}` the output must match |
| `repo_dir` | no | where the diff/test run (default `/testbed`) |
| `patch_command` | no | how to extract the diff (default a `git diff`) |
| `test_timeout` / `apply_timeout` | no | timeouts |
| `exclude_patterns` | no | paths to drop from the diff |

**`swebench_harness`** — grades real SWE-bench-style patches with the SWE-bench
(or SWE-Gym) harness. Takes an `instance` dict plus the same patch config keys.

**`harbor`** — injects a Harbor task's `tests/` directory into the agent's live
runtime, runs `bash /tests/test.sh`, and reads the reward back (the Harbor
verifier contract). Requires `refresh_runtime: false`. Config: `tests_dir`
(required), `verifier_timeout`, `tests_target`, `verifier_dir`, `test_command`.

**`harbor_rubric`** — everything `harbor` does, plus a rubric-based process
reward when the task ships `tests/rubric.md`. One judge call per rollout: an
OpenAI-compatible endpoint sees the task's `instruction.md`, a unified meta
rubric (efficiency, no reward hacking, outcome-independent step judging), the
task rubric, the verifier's raw scoring (`reward.json` when present), and every
trace's `response_messages` in chronological order tagged `trace_0`,
`trace_1`, … (assumes the `per_request` builder; warns otherwise). The judge
returns one JSON object mapping each trace id to an integer score in `[-5, 5]`
and each trace gets `outcome_reward + rubric_coefficient * (score / 5)`. Fails
open: no rubric, a judge failure, or a missing trace id falls back to the
outcome reward. Extra config: `judge_base_url` and `judge_model` (required),
`rubric_coefficient` (default `0.2`), `judge_api_key_env` (default
`JUDGE_API_KEY`, resolved from the evaluator `env` then the process env),
`judge_timeout`, `judge_max_retries`, `judge_temperature`,
`max_section_chars`.

Both grading evaluators need a live runtime (and a `fresh_eval_runtime` when the
task sets `refresh_runtime`); an empty diff scores `0.0`.

## Adding an evaluator

Implement the base contract, return an `EvalResult`, and register the name in
`registry.py` (or pass a `"module:ClassName"` import path). Keep external
services, GPUs, and large datasets out of default unit tests.
