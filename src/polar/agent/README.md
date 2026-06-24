# Agent Harnesses

In Polar, an **agent harness** is whatever launches your agent inside a prepared
runtime. The key idea is that you **do not integrate agents into Polar** — you
run them unmodified. A harness only has to:

1. start the agent process, and
2. let the agent's LLM calls flow through the gateway proxy.

Polar injects the proxy endpoints as environment variables
(`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_API_URL`, and matching
`*_API_KEY`s set to the session id). The gateway serves the model, rewrites the
request to the served model, and **captures the trajectory** from the wire-level
calls. So the harness never parses transcripts or implements agent logic — that
all lives in the agent.

The public task field is `agent`, validated by `models.AgentSpec`.

## Three ways to run an agent

| Path | When to use | How |
|---|---|---|
| **Preset** | A popular agent we already ship a launcher for | `agent.harness: "<name>"` |
| **`shell`** | Any agent you can express as a shell command | `agent.harness: "shell"` + `agent.custom_shell` |
| **`import_path`** | Your own harness class, kept in your repo | `agent.import_path: "your.module:YourHarness"` |

Presets are **conveniences, not integrations** — each is a thin `BaseHarness`
(a few dozen lines) that writes the agent's config and emits its run command.
If your agent isn't listed below, you don't add code to Polar: reach for `shell`
or `import_path`.

## Presets

API type names match `polar.gateway.detection.APIType`: `anthropic`,
`openai_chat`, `openai_responses`, and `google`. *Streaming* is the wire style
the agent sends to the proxy. *Version* is the external CLI/SDK release verified
end-to-end by the [calculator example](../../../examples/calculator/README.md);
examples may pin their own, but these are the known-good ones.

| Preset | API type | Streaming | Verified version |
|---|---|---|---|
| `claude_code` | `anthropic` | `true` | `@anthropic-ai/claude-code@2.1.111` |
| `codex` | `openai_responses` | `true` | `@openai/codex@0.125.0` |
| `gemini_cli` | `google` | `true` | `@google/gemini-cli@0.38.1` |
| `opencode` | `openai_chat` | `true` | `opencode-ai@1.4.6` |
| `openclaw` | `openai_chat` | `true` | `openclaw@2026.5.27` |
| `openhands_sdk` | `openai_chat` | `false` | `openhands-sdk==1.17.0` ¹ |
| `hermes` | `openai_chat` | `true` | `hermes-agent==0.15.1` |
| `pi` | `openai_chat` | `false` | `@mariozechner/pi-coding-agent@0.67.68` |
| `qwen_code` | `openai_chat` | `true` | `@qwen-code/qwen-code@0.14.5` |
| `shell` | set by `agent.custom_shell` | set by `agent.custom_shell` | — |

¹ Install `openhands-tools==1.17.0` at the same version. `1.18+` needs Python
3.13 (a transitive `lmnr` pin is unsatisfiable on 3.12); pin `1.17.0` on a
Python 3.12 image.

Each preset routes to the proxy a little differently because each agent reads a
different env var / config key — e.g. `gemini_cli` maps `GOOGLE_API_*` onto the
CLI's `GEMINI_API_KEY`/`GOOGLE_GEMINI_BASE_URL`; `openclaw` and `hermes` write the
gateway URL into their config files because they don't read `OPENAI_BASE_URL`;
`codex` writes the gateway URL into the Codex config so the default OpenAI
provider sends Responses API traffic through Polar. The per-file comments
explain each piece — that glue is the *only* reason a preset is more than five
lines.

## The harness contract

A harness receives the task instruction, a runtime execution helper, the model
name, environment, settings, and optional MCP servers. It returns an
`AgentRunResult` with status `completed`, `failed`, or `timeout`.

- The harness starts the agent process.
- Polar owns runtime setup, the model proxy endpoints, completion capture, and
  evaluation.

```python
class BaseHarness:
    async def setup(self, runtime) -> None:        # write config, install nothing heavy
        ...
    def run_steps(self, instruction) -> list[ExecInput]:   # the command(s) to run the agent
        ...
```

### Anatomy of a preset

A preset is just those two methods. For example, a CLI agent that already reads
`OPENAI_BASE_URL`/`OPENAI_API_KEY` needs almost nothing:

```python
class MyAgentHarness(BaseHarness):
    def run_steps(self, instruction: str) -> list[ExecInput]:
        return [ExecInput(command=f"myagent --yolo -p {shlex.quote(instruction)}")]
```

`setup()` is where a preset writes a config file (MCP servers, a custom provider
base URL, skills). `run_steps()` returns the shell command(s); the injected proxy
env vars are merged in automatically. Note `setup()` runs *before* the proxy env
is available, so anything that needs `$OPENAI_BASE_URL` must be written inside a
`run_steps()` command (see `openclaw`/`hermes`/`pi`).

## Bring your own agent

You don't need a preset. Two no-Polar-code paths:

**`shell`** — wrap any command. Requires `agent.custom_shell`; cannot be combined
with MCP servers or a skills path.

```yaml
agent:
  harness: shell
  custom_shell:
    command: "my-agent run --task {{INSTRUCTION}} 2>&1 | tee $AGENT_LOG_DIR/agent.txt"
```

**`import_path`** — keep your harness class in your own repo and point at it:

```yaml
agent:
  import_path: "my_pkg.harness:MyAgentHarness"
```

The import path must resolve to a `BaseHarness` subclass.

## Main files

- `base.py` — the harness contract.
- `models.py` — `AgentSpec`, `MCPServerSpec`, `AgentRunResult`.
- `factory.py` — preset name lookup and `import_path` loading.
- `presets/` — the ready-made launchers in the table above.
