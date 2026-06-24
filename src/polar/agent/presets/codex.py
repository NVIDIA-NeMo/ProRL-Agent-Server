"""Codex CLI harness — https://github.com/openai/codex"""

from __future__ import annotations

import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR
from polar.runtime.models import ExecInput

DEFAULT_CODEX_VERSION = "0.125.0"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_MODEL_NAME = "gpt-5.5"


class CodexHarness(BaseHarness):
    """Run OpenAI Codex CLI in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        # Keep credentials (auth.json, config.toml) outside the log dir so log
        # rotation or archival can't clobber them. Absolute path — $HOME won't
        # expand in docker exec -e.
        self._codex_home = f"{RUNTIME_SESSION_DIR}/.codex"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._codex_home}")

        expected_version = self._expected_version()
        if expected_version:
            result = await runtime.exec(
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                "if ! command -v codex >/dev/null 2>&1; then "
                "echo 'codex CLI not found; install @openai/codex in runtime.prepare' >&2; "
                "exit 127; "
                "fi; "
                "installed=$(codex --version | awk 'NF {print $NF; exit}'); "
                f"if [ \"$installed\" != {shlex.quote(expected_version)} ]; then "
                f"echo 'codex version mismatch: expected {expected_version}, got '\"$installed\" >&2; "
                "exit 1; "
                "fi"
            )
            if result.return_code != 0:
                output = result.stderr or result.stdout or "codex version check failed"
                raise RuntimeError(output.strip())

        # Host-uploaded files keep the host UID, which blocks codex's
        # exec_command-based edits (cat/tee/open) on a non-root container
        # user. Other harnesses survive by rm+recreating the file. Best-effort;
        # a no-op on images without sudo.
        workdir = runtime.spec.workdir or runtime.runtime_session_dir
        await runtime.exec(
            f'sudo chown -R "$(id -u):$(id -g)" {shlex.quote(workdir)} 2>/dev/null || true'
        )

        # Register MCP servers via TOML config
        if self.mcp_servers:
            toml_lines: list[str] = []
            for server in self.mcp_servers:
                toml_lines.append(f'[mcp_servers."{server.name}"]')
                if server.transport == "stdio":
                    toml_lines.append(f'command = "{server.command}"')
                    if server.args:
                        args_str = ", ".join(f'"{a}"' for a in server.args)
                        toml_lines.append(f"args = [{args_str}]")
                else:
                    toml_lines.append(f'url = "{server.url}"')
                    toml_lines.append(f'type = "{server.transport}"')
            toml_content = "\n".join(toml_lines)
            await runtime.exec(
                f"cat > {self._codex_home}/config.toml << 'POLARCFG'\n{toml_content}\nPOLARCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p $HOME/.agents/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* $HOME/.agents/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        env: dict[str, str] = {
            **self.env,
            "CODEX_HOME": self._codex_home,
        }

        # Match Harbor's Codex harness: Codex keeps the default OpenAI provider
        # and reads the proxy from config.toml. Codex then sends Responses API
        # traffic to $OPENAI_BASE_URL/v1/responses, which Polar captures.
        flags: list[str] = [
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
        ]
        model = _cli_model_name(self.model_name)
        flags.append(f"--model {shlex.quote(model)}")
        flags.extend(["--json", "--enable unified_exec"])

        for key, cli in [
            ("reasoning_effort", "-c model_reasoning_effort"),
            ("reasoning_summary", "-c model_reasoning_summary"),
        ]:
            value = self.settings.get(key)
            if key == "reasoning_effort" and value is None:
                value = DEFAULT_REASONING_EFFORT
            if value is not None:
                flags.append(f"{cli}={shlex.quote(str(value))}")

        flags_str = " ".join(flags)
        return [
            # Write synthetic auth.json so codex picks up OPENAI_API_KEY
            ExecInput(
                command=(
                    f"mkdir -p {self._codex_home} && "
                    f'printf \'{{"OPENAI_API_KEY": "%s"}}\' "$OPENAI_API_KEY" '
                    f"> {self._codex_home}/auth.json && "
                    'if [ -n "${OPENAI_BASE_URL:-}" ]; then '
                    f"cat >> {self._codex_home}/config.toml <<POLARCODEX\n"
                    'openai_base_url = "${OPENAI_BASE_URL}"\n'
                    "POLARCODEX\n"
                    "fi"
                ),
                env=env,
            ),
            ExecInput(
                command=(
                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                    f"codex exec {flags_str} -- {escaped} "
                    f"2>&1 </dev/null | tee {RUNTIME_AGENT_LOG_DIR}/codex.txt"
                ),
                env=env,
            ),
        ]

    def postrun_steps(self) -> list[ExecInput]:
        return [
            ExecInput(
                command=(
                    'if [ -d "$CODEX_HOME/sessions" ]; then '
                    f"rm -rf {RUNTIME_AGENT_LOG_DIR}/sessions && "
                    f'cp -R "$CODEX_HOME/sessions" {RUNTIME_AGENT_LOG_DIR}/sessions; '
                    "fi"
                ),
                env={"CODEX_HOME": self._codex_home},
            )
        ]

    def _expected_version(self) -> str | None:
        value = self.settings.get("version", DEFAULT_CODEX_VERSION)
        if value in (None, ""):
            return None
        return str(value)


def _cli_model_name(model_name: str | None) -> str:
    model = model_name or DEFAULT_MODEL_NAME
    for prefix in ("openai/", "anthropic/", "google/", "gcp/google/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model
