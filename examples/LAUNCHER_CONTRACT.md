# Launcher environment contract

The production launch chain is three hops of `exec`/snapshot, and its interface
is process environment, not arguments:

```
spilot_router_slime_grpo/run.sh   (sets templates, execs ->)
  tmax_slime_grpo/run.sh          (sources env.cwdfw.sh, lifecycle.sh, run_state.sh;
                                   snapshots swegym run.sh into $RUN_DIR/shared_run.sh, execs ->)
    swegym_slime_grpo/run.sh      (the 2,100-line shared launcher: Ray, slime CLI, gateway)
```

Every variable below is SET in an upstream layer and READ inside the shared
launcher after the final `exec` — i.e. it crosses a process-image boundary with
no declaration. Renaming, unsetting, or conditionally skipping any of them in an
upstream file is an interface change to the shared launcher. When adding a new
`SPILOT_*` control it must also be added to the shared launcher's template
whitelist (see `swegym_slime_grpo/run.sh` placeholder list) or rendering fails.

| Variable | Set in |
|---|---|
| `POLAR_AGENT_COST_LIMIT` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_AGENT_ENABLE_THINKING` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_AGENT_HARNESS` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_AGENT_MAX_TOKENS` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_AGENT_MODEL_NAME` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_AGENT_PATH` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_AGENT_RUNTIME_VOLUME` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_AGENT_STEP_LIMIT` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_AGENT_TEMPERATURE` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_AGENT_TOP_P` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_APPTAINER_BIN` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_APT_HTTP_SOURCE_POLICY` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_CANDIDATE_POOL_HEALTH_GATE_ENABLED` | spilot_router_slime_grpo/experiment_defaults.sh |
| `POLAR_CANDIDATE_POOL_HEALTH_MIN_COMPLETION_FRACTION` | spilot_router_slime_grpo/experiment_defaults.sh |
| `POLAR_CANDIDATE_POOL_HEALTH_MIN_OBSERVED_SESSIONS` | spilot_router_slime_grpo/experiment_defaults.sh |
| `POLAR_COMPLETION_BATCH_SIZE` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_COMPLETION_QUEUE_SIZE` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_COMPLETION_RETRY_BACKOFF_SECONDS` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_COMPLETION_WRITE_MAX_ATTEMPTS` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_COMPLETION_WRITE_WORKERS` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_CONFIG_TEMPLATE` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/run.sh |
| `POLAR_EARLY_STOP_GRACE_SESSIONS` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_EVAL_DATA_INTEGRITY_B64` | tmax_slime_grpo/run.sh |
| `POLAR_FULLY_ASYNC` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_GATEWAY_COUNT_OVERRIDE` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_MAX_ASYNC_LEVEL` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_MAX_INIT_WORKERS` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh, tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_MAX_POSTRUN_WORKERS` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh, tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_MAX_RUN_WORKERS` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh, tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_MIN_COMPLETE_ACCEPT_FRACTION` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_MULTI_GATEWAY` | spilot_router_slime_grpo/experiment_defaults.sh |
| `POLAR_REQUEST_TIMEOUT` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `POLAR_ROLLOUT_SAVE_DIR` | tmax_slime_grpo/run.sh |
| `POLAR_SANDBOX_NETWORK` | tmax_slime_grpo/env.cwdfw.sh |
| `POLAR_SHARED_SCRIPT_DIR` | tmax_slime_grpo/run.sh |
| `POLAR_TASK_TIMEOUT_FLOOR_SECONDS` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `POLAR_TRAIN_PROJECT_ROOT` | tmax_slime_grpo/run.sh |
| `SPILOT_COST_NORMALIZER` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh |
| `SPILOT_COST_PENALTY_LAMBDA` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh |
| `SPILOT_EPISODE_ADMISSION_ENABLED` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_GPT_COST_WEIGHT` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh |
| `SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_GPT_GATEWAY_MAX_CONCURRENCY` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_GPT_MAX_ACTIVE_EPISODES` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_LATENCY_NORMALIZER` | spilot_router_slime_grpo/experiment_defaults.sh |
| `SPILOT_LATENCY_PENALTY_LAMBDA` | spilot_router_slime_grpo/experiment_defaults.sh |
| `SPILOT_MAX_POOL_CALLS` | spilot_router_slime_grpo/experiment_defaults.sh |
| `SPILOT_QWEN_COST_WEIGHT` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh |
| `SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_QWEN_MAX_ACTIVE_EPISODES` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `SPILOT_ROUTER_OBS_MAX_CHARS` | spilot_router_slime_grpo/experiment_defaults.sh |
| `SPILOT_ROUTING_MODE` | spilot_router_slime_grpo/experiment_defaults.sh |
| `SPILOT_SLOT_LABEL_MODE` | spilot_router_slime_grpo/experiment_defaults.sh |
| `TMAX_AGENT_HARNESS` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_CONCURRENT_PRETRAIN_EVAL` | spilot_router_slime_grpo/experiment_defaults.sh |
| `TMAX_DYNAMIC_SAMPLING_FILTER_PATH` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_ENABLE_FP32_LM_HEAD` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_BUNDLE_SHA256` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/run.sh |
| `TMAX_EVAL_CONFIG_PATH` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_DATA` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_DATASET_NAME` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_DATA_SHA256` | spilot_router_slime_grpo/experiment_defaults.sh |
| `TMAX_EVAL_ENABLED` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_INTERVAL` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_MAX_RESPONSE_LEN` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_MIN_VALID_SAMPLES` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN` | spilot_router_slime_grpo/experiment_defaults.sh |
| `TMAX_EVAL_SAMPLES_PER_PROMPT` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_TEMPERATURE` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EVAL_TOP_P` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EXTERNAL_EVAL_DATA` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EXTERNAL_EVAL_DATASET_NAME` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EXTERNAL_EVAL_ENABLED` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EXTERNAL_EVAL_MAX_RESPONSE_LEN` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EXTERNAL_EVAL_TEMPERATURE` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_EXTERNAL_EVAL_TOP_P` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_MAX_TOTAL_RESPONSE_LEN` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_MODEL_MAX_CONTEXT_LENGTH` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_NUM_ROLLOUT` | spilot_router_slime_grpo/experiment_defaults.sh, spilot_router_slime_grpo/submit_lambda02.sh |
| `TMAX_OPTIMIZER_CPU_OFFLOAD` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_OVERRIDE_OPT_PARAM_SCHEDULER` | spilot_router_slime_grpo/experiment_defaults.sh |
| `TMAX_PROFILE_DISABLE_CHECKPOINT` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT` | spilot_router_slime_grpo/experiment_defaults.sh |
| `TMAX_TRAINING_EVAL_ENABLED` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_TRAIN_AGENT_TIMEOUT_SECONDS` | spilot_router_slime_grpo/experiment_defaults.sh, tmax_slime_grpo/env.cwdfw.sh, tmax_slime_grpo/run_state.sh |
| `TMAX_TRAIN_MODE` | tmax_slime_grpo/env.cwdfw.sh |
| `TMAX_TRAIN_PACK_LENGTH` | tmax_slime_grpo/env.cwdfw.sh |

Generated 2026-07-21 from the current tree: 144 prefixed variables are
referenced by the shared launcher; the 96 above are produced upstream in the
same chain. Regenerate with the snippet in this file's git blame commit message.
