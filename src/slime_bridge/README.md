# Slime Bridge

`slime_bridge` connects [Slime](https://github.com/THUDM/slime)'s RL training
loop to a running Polar rollout server over HTTP. It lives **outside** the
`polar` package because Slime, Ray, Megatron, and torch are installed separately
— Polar depends on none of them.

## How it fits

Slime calls one entry point, `generate_rollout_polar_async`, wired in via
`--rollout-function-path`. From there the bridge:

- submits async task batches to `polar_rollout_url` (or a node derived from
  `polar_topology_path`) and collects each result through a local callback
  listener with a polling safety net;
- tracks rollout ids and policy versions, stamps Polar scheduler metadata
  (`group_id`, `policy_version`, `rollout_step`) onto every task, and keeps
  async admission bounded to the current request or an explicitly enabled
  fixed fully-async prefetch window;
- converts each Polar `Trajectory` back into Slime `Sample`s (one per trace,
  grouped with `rollout_id` so all traces from a trajectory count once), keeping
  an exact causal prefix of overlong traces and dropping traces whose full prompt
  leaves no trainable-token budget;
- computes dynamic-trace leave-one-trajectory-out advantages and zeroes out
  failed/aborted trajectories.

## Main files

- `config.py`: `PolarSlimeConfig` + `resolve_polar_slime_config`; also renders the
  task payload, the instruction, and the topology that points gateways at Slime's
  SGLang router.
- `rollout.py`: the async worker (submit → callback/poll → convert), the
  evaluation path, the acceptance filters, and the Slime entry point.
- `_messages.py`: prompt/message flattening shared by rollout + adapter.
- `adapter.py`: convert a Polar `SessionResult` into Slime `Sample`s.
- `data_source.py`: `CeilEpochRolloutDataSourceWithBuffer` — rounds the epoch
  length up so the dataset tail isn't skipped.
- `reward.py`: reward hook that reads the reward Polar already embedded.
- `reward_post_process.py`: trajectory-aware, group-normalized reward shaping.

## What the bridge owns

- Turn Slime samples + prompts into Polar task requests and submit async batches.
- Optionally apply `polar_task_timeout_floor` to the rendered task budget while
  preserving any larger dataset timeout. A validated
  `sample.metadata.agent_timeout` is forwarded separately; gateways start that
  active-agent budget at RUN rather than spending it in INIT/READY queues.
- Track rollout ids / policy versions and bound async admission.
- When `polar_fully_async: true`, keep a fixed
  `rollout_batch_size * polar_max_async_level` prefetch window warm across
  rollout boundaries; completed samples retain policy-staleness metadata.
- Filter unusable groups (zero trainable tokens, too few completed samples,
  logprob errors) with per-category metrics.
- Convert Polar trajectories back into Slime samples; compute dynamic-trace
  advantages.
- Run the evaluation path over `eval_datasets` and emit W&B metrics.

## Optional eval-data integrity pin

Launchers can set `POLAR_EVAL_DATA_INTEGRITY_B64` to a base64-encoded JSON
manifest. The bridge remains backward compatible when it is unset. When set,
every configured eval dataset must have a canonical-path entry, and the bridge
SHA-256 hashes the exact bytes it parses on every evaluation. A changed or
unlisted file fails before any Polar task is submitted. The version-1 envelope
is intentionally launcher-agnostic:

```json
{
  "schema_version": 1,
  "algorithm": "sha256",
  "datasets": [
    {"name": "holdout", "path": "/abs/eval.jsonl", "sha256": "<64 hex>"}
  ]
}
```

Keep the encoded JSON in the worker environment rather than passing only a
mutable manifest path; an on-disk copy may still be retained for experiment
auditing. The TMax launcher generates both forms automatically.

## Slime installation

Install Slime from the THUDM git checkout (not the unrelated PyPI `slime`
package). The SWE-Gym Slime GRPO example automates this with `launch_e2e.sh`; the
manual equivalent from the repository root is:

```bash
git clone --branch v0.3.0 --depth 1 https://github.com/THUDM/slime.git slime
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
bash scripts/patch/patch_slime_router_tokens.sh

uv pip install -e .
uv pip install -e slime
uv pip install -e Megatron-LM
```

The patch command above is only for the legacy SWE-Gym bootstrap flow. A
source-locked launcher such as TMax must instead pin a Slime commit that
already contains the adapter support. Do not apply the patch in place to a
source-locked checkout: the training watcher intentionally rejects dirty or
revision-drifted sources before it submits a replacement job.

Use `SLIME_DIR=/path/to/slime` and `MEGATRON_DIR=/path/to/Megatron-LM` for
checkouts outside the repository root. Run the patch command with the same
`SLIME_DIR` value before installing Slime. The patch preserves exact
SGLang-native prompt/output token ids and token-level logprobs in Slime's
OpenAI-compatible adapter response, so Polar does not retokenize trajectories
locally. The Slime training environment provides the heavy dependencies
(e.g. `torch`); Polar does not add them.
