# Configuration

`polar.config` loads and validates the single `topology.yaml` that describes a
whole Polar deployment: one **rollout** server plus one or more **gateway**
nodes (each with its own inference backend). `TopologyConfig.load()` is the entry
point every `polar` command uses.

## Mental model

One file, two halves:

- `rollout:` — the central orchestrator that clients submit tasks to.
- `gateway:` — the worker fleet. `gateway.nodes[]` is a list; each entry is an
  independent gateway process with its own ports, worker pools, and inference
  endpoint.

The schema is **strict and immutable**: unknown keys are rejected
(`extra="forbid"`, so a typo fails fast) and every model is frozen after load.
Convenience defaulting fills the gaps — a blank `public_url` is derived from
`host:port` (mapping `0.0.0.0`/`::` → `127.0.0.1`), and `gateway.rollout_server_url`
falls back to `rollout.public_url` when omitted.

## Main files

- `topology.py`: the Pydantic models (`TopologyConfig`, `RolloutServiceConfig`,
  `GatewayConfig`, `GatewayNodeConfig`), `load()`, and the URL/selection helpers.
- `__init__.py`: package exports.

## Schema

**`rollout`** — `RolloutServiceConfig`

| field | type | default |
|---|---|---|
| `host` | str | `0.0.0.0` |
| `port` | int | `8080` |
| `public_url` | str | derived from `host:port` |
| `save_dir` | str? | `None` (no result persistence) |
| `dispatch_poll_interval_seconds` | float | `1.0` |
| `callback_grace_seconds` | float | `120.0` |

**`gateway`** — `GatewayConfig`

| field | type | default |
|---|---|---|
| `heartbeat_interval_seconds` | int | `30` |
| `rollout_server_url` | str? | `rollout.public_url` |
| `nodes` | list | **required**, ≥1, unique ids |
| `completion_persistence` | block | `enabled` |

`completion_persistence` controls the async on-disk capture of model calls:
`enabled` (`true`), `max_field_bytes` (`1048576`), `queue_size` (`1024`).

**`gateway.nodes[]`** — `GatewayNodeConfig`

| field | type | default |
|---|---|---|
| `id` | str | hostname (must be unique) |
| `host` / `port` | str / int | `0.0.0.0` / `8081` |
| `public_url` | str | derived from `host:port` |
| `model_served` | str | `""` |
| `inference.engine` | `sglang` \| `vllm` | `sglang` |
| `inference.scheduler` | `none` \| `thunderagent` | `none` |
| `inference.base_url` | str | `http://127.0.0.1:8000` |
| `max_init_workers` | int | `4` |
| `max_run_workers` | int | `2` |
| `max_postrun_workers` | int | `4` |
| `default_runtime` | `RuntimeSpec`? | `None` |

## Example

```yaml
rollout:
  host: 127.0.0.1
  port: 8080
  public_url: http://127.0.0.1:8080
  save_dir: ./rollout_results

gateway:
  heartbeat_interval_seconds: 30
  nodes:
    - id: localhost-node-01
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      model_served: Qwen/Qwen3.5-4B
      max_init_workers: 8
      max_run_workers: 4
      max_postrun_workers: 4
      inference:
        engine: sglang   # or vllm
        scheduler: none  # or thunderagent; opt-in
        base_url: http://127.0.0.1:8000
```

## ThunderAgent scheduler

ThunderAgent runs as an external OpenAI-compatible proxy; Polar does not start
or configure it. `inference.engine` remains `sglang` or `vllm` (matching the
backend behind ThunderAgent), while `inference.base_url` points to the
ThunderAgent listener. The default `scheduler: none` keeps the direct inference
path unchanged. Polar uses `<gateway-node-id>:<session-id>` as the stable
ThunderAgent program identity, so gateways sharing one proxy remain isolated.

Install the external ThunderAgent checkout in editable mode if needed:

```bash
python -m pip install -e /path/to/ThunderAgent
```

For example, with a vLLM server on port 8000, start ThunderAgent with the fixed
rollout settings:

```bash
thunderagent \
  --backend-type vllm \
  --backends http://127.0.0.1:8000 \
  --port 9000 \
  --router tr \
  --metrics \
  --acting-token-weight 0.0
```

Then opt the gateway node in and point it at that proxy:

```yaml
inference:
  engine: vllm
  scheduler: thunderagent
  base_url: http://127.0.0.1:9000
```

When gateways share a ThunderAgent instance, coordinate each weight update as
one shared operation: pause and drain all gateways, call
`POST /weight_sync/begin` once per ThunderAgent instance, update the weights,
call `POST /weight_sync/end` once per instance, and then resume all gateways.
Do not map each gateway's pause/resume pair to its own ThunderAgent begin/end
pair, which can release the shared barrier too early.

## Reachable URLs and multi-node

`public_url`s must be reachable by whoever calls them: the rollout server calls
each node's `public_url`; each node calls back to `rollout_server_url` and its
own `inference.base_url`. Locally the derived `127.0.0.1` URLs work; for
multi-host deployments set explicit reachable URLs.

`polar serve_gateway` requires `--node-id` when the topology has more than one
node, so a gateway process always starts with the right ports, worker limits,
and inference endpoint.
