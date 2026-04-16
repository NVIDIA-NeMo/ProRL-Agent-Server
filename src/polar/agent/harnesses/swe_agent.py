"""SWE-Agent harness — https://github.com/SWE-agent/SWE-agent"""

from __future__ import annotations

import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentRunResult, AgentSpec
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from polar.runtime.models import ExecInput


class SweAgentHarness(BaseHarness):
    """Run SWE-Agent CLI against a task."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._problem_statement_path = f"{RUNTIME_AGENT_LOG_DIR}/problem_statement.md"
        self._repo_path = str(self.settings.get("repo_path") or "/polar/session/workspace")
        self._shell_preamble = str(self.settings.get("shell_preamble") or "").strip()

    async def setup(self, runtime: BaseRuntime) -> None:
        pass

    async def postprocess(
        self,
        runtime: BaseRuntime,
        result: AgentRunResult,
    ) -> None:
        del result

        repo = shlex.quote(self._repo_path)
        logs = shlex.quote(RUNTIME_AGENT_LOG_DIR)
        await runtime.exec(
            "\n".join(
                [
                    f'TRAJ_FILE="$(find {repo}/trajectories -type f -name \'*.traj\' 2>/dev/null | sort | tail -n 1)"',
                    'if [ -n "$TRAJ_FILE" ] && [ -f "$TRAJ_FILE" ]; then',
                    f'  cp "$TRAJ_FILE" {logs}/swe-agent.trajectory.json 2>/dev/null || true',
                    "fi",
                    f'PATCH_FILE="$(find {repo}/trajectories -type f -name \'*.patch\' 2>/dev/null | sort | tail -n 1)"',
                    'if [ -n "$PATCH_FILE" ] && [ -f "$PATCH_FILE" ]; then',
                    f'  cp "$PATCH_FILE" {logs}/swe-agent.patch 2>/dev/null || true',
                    f'  if git -C {repo} diff --quiet --cached -- && git -C {repo} diff --quiet --; then',
                    f'    (cd {repo} && git apply "$PATCH_FILE") || true',
                    "  fi",
                    "fi",
                ]
            )
        )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        model = self.model_name or "openai/gpt-5.4"
        env: dict[str, str] = {**self.env}

        flags: list[str] = []
        resolved_keys: set[str] = set()
        for key, cli in [
            ("per_instance_cost_limit", "--agent.model.per_instance_cost_limit"),
            ("total_cost_limit", "--agent.model.total_cost_limit"),
            ("max_input_tokens", "--agent.model.max_input_tokens"),
            ("temperature", "--agent.model.temperature"),
            ("top_p", "--agent.model.top_p"),
        ]:
            value = self.settings.get(key)
            if value is not None:
                flags.append(f"{cli}={shlex.quote(str(value))}")
                resolved_keys.add(key)

        # Local/vLLM-served models are commonly unknown to LiteLLM cost tables.
        # Defaulting these limits to zero avoids aborting after the first turn.
        if "per_instance_cost_limit" not in resolved_keys:
            flags.append("--agent.model.per_instance_cost_limit=0")
        if "total_cost_limit" not in resolved_keys:
            flags.append("--agent.model.total_cost_limit=0")
        if "max_input_tokens" not in resolved_keys:
            flags.append("--agent.model.max_input_tokens=0")

        flags_str = (" " + " ".join(flags)) if flags else ""
        preamble = f"{self._shell_preamble} && " if self._shell_preamble else ""

        safe_instruction = instruction.replace("'", "'\"'\"'")

        return [
            ExecInput(
                command=(
                    f"cat > {self._problem_statement_path} << 'POLARINST'\n{safe_instruction}\nPOLARINST\n"
                    f"{preamble}"
                    'set -o pipefail && '
                    'export OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL="$OPENAI_BASE_URL" && '
                    f"sweagent run "
                    f"--agent.model.name={shlex.quote(model)} "
                    f"--problem_statement.path={shlex.quote(self._problem_statement_path)} "
                    f"--env.deployment.type=local "
                    f"--env.repo.path={shlex.quote(self._repo_path)}"
                    f"{flags_str} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/swe-agent.txt"
                ),
                env=env,
            )
        ]

    def postrun_steps(self) -> list[ExecInput]:
        return [
            ExecInput(
                command=(
                    f'TRAJ_FILE="$(find {shlex.quote(self._repo_path)}/trajectories -type f -name \'*.traj\' '
                    f'2>/dev/null | sort | tail -n 1)" && '
                    f'[ -n "$TRAJ_FILE" ] && cp "$TRAJ_FILE" '
                    f"{RUNTIME_AGENT_LOG_DIR}/swe-agent.trajectory.json 2>/dev/null || true"
                ),
            )
        ]
