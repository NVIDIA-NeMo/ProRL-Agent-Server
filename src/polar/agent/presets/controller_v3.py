"""Dynamic Controller V3 harness for the pinned Mini-SWE-Agent 2.4 runtime."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path

from polar.agent.base import BaseHarness
from polar.agent.models import AgentRunResult, AgentSpec
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR
from polar.runtime.models import ExecInput


CONTROLLER_V3_RUNNER_PATH = f"{RUNTIME_SESSION_DIR}/controller_v3_runner.py"
CONTROLLER_V3_MODULE_PATH = f"{RUNTIME_SESSION_DIR}/oracle_controller_v3.py"
CONTROLLER_V3_CONFIG_PATH = f"{RUNTIME_SESSION_DIR}/controller_v3_one_vote.yaml"
CONTROLLER_V3_TRAJECTORY_PATH = f"{RUNTIME_AGENT_LOG_DIR}/mini-swe-agent.traj.json"
_DEFAULT_PYTHON = "/opt/polar-mini-swe-agent/python/bin/python3.10"
_ROUTER_CAPABILITY = "POLAR_ROUTER_CAPABILITY"
_POOL_CAPABILITY = "POLAR_MODEL_POOL_CAPABILITY"


class ControllerV3Harness(BaseHarness):
    """Run one-vote Controller V3 without modifying the immutable runtime."""

    _SUPPORTED_SETTINGS = frozenset(
        {"runner_python", "step_limit", "cost_limit", "wall_time_limit_seconds"}
    )

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        if not self.model_name:
            raise ValueError("controller_v3 requires agent.model_name")
        unknown = sorted(set(self.settings) - self._SUPPORTED_SETTINGS)
        if unknown:
            raise ValueError(
                f"Unsupported controller_v3 settings: {', '.join(unknown)}"
            )

    @staticmethod
    def _sources() -> dict[str, Path]:
        directory = Path(__file__).parent
        return {
            CONTROLLER_V3_RUNNER_PATH: directory / "controller_v3_runner.py",
            CONTROLLER_V3_MODULE_PATH: directory / "oracle_controller_v3.py",
            CONTROLLER_V3_CONFIG_PATH: directory / "controller_v3_one_vote.yaml",
        }

    async def setup(self, runtime: BaseRuntime) -> None:
        for destination, source in self._sources().items():
            if not source.is_file():
                raise RuntimeError(f"Controller V3 source is missing: {source}")
            await runtime.upload_file(str(source), destination)

    def run_steps(self, instruction: str) -> list[ExecInput]:
        python = str(self.settings.get("runner_python") or _DEFAULT_PYTHON)
        env = {
            **self.env,
            "CONTROLLER_V3_TASK_B64": base64.b64encode(
                instruction.encode("utf-8")
            ).decode("ascii"),
            "CONTROLLER_V3_POLICY_MODEL": "openai/router/policy",
            "MSWEA_CONFIGURED": "true",
            "MSWEA_COST_TRACKING": "ignore_errors",
            "MSWEA_GLOBAL_CONFIG_DIR": "/polar/session/.mini-swe-agent",
        }
        for key, setting in (
            ("CONTROLLER_V3_STEP_LIMIT", "step_limit"),
            ("CONTROLLER_V3_COST_LIMIT", "cost_limit"),
            ("CONTROLLER_V3_WALL_TIME_LIMIT_SECONDS", "wall_time_limit_seconds"),
        ):
            if setting in self.settings:
                env[key] = str(self.settings[setting])

        sources = self._sources()
        return [
            ExecInput(
                protected_argv=[python, CONTROLLER_V3_RUNNER_PATH],
                protected_env_keys=[_ROUTER_CAPABILITY, _POOL_CAPABILITY],
                protected_file_digests={
                    destination: hashlib.sha256(source.read_bytes()).hexdigest()
                    for destination, source in sources.items()
                },
                env=env,
            )
        ]

    async def postprocess(
        self, runtime: BaseRuntime, result: AgentRunResult
    ) -> None:
        host_path = runtime.resolve_host_path(CONTROLLER_V3_TRAJECTORY_PATH)
        if host_path is None or not host_path.is_file():
            return
        try:
            document = json.loads(host_path.read_text())
            usage = document["info"]["oracle_controller_v3"]["usage"]["large"]
            cost = float(usage["cost"])
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("invalid GPT cost")
            result.metadata["controller_v3_cost"] = {
                key: usage[key]
                for key in (
                    "cost",
                    "input_tokens",
                    "cached_input_tokens",
                    "uncached_input_tokens",
                    "output_tokens",
                    "pricing",
                )
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result.metadata["controller_v3_cost_error"] = str(exc)
