"""``harbor`` evaluator — score a rollout with a Harbor task's programmatic verifier.

Harbor tasks (any ``*-Harbor`` dataset on the hub — TMax-15K, Terminal-Bench,
TB-Lite, …) ship their verifier *alongside* but deliberately *outside* the task
image: a ``tests/`` directory holding ``test.sh`` (plus whatever it drives, e.g.
``pytest test_final_state.py``). Harbor grades every task the same way — inject
that directory into the container the agent just used, run ``bash /tests/test.sh``
(which writes a reward to ``/logs/verifier/reward.txt``), then read it back.

This evaluator reproduces that contract against Polar's live runtime, so the
score matches what Harbor computes. Unlike the SWE-bench / ``test_on_output``
evaluators it does **not** extract or replay a git diff: a Harbor verifier
inspects the *final state* of the container, so grading must run in the same
runtime the agent operated in. Submit with ``refresh_runtime: false`` (the
default) so the agent's runtime is handed to ``evaluate`` as ``runtime``.

Config schema (:class:`~polar.trajectory.models.EvaluatorSpec.config`)
----------------------------------------------------------------------
- ``tests_dir`` *(str, required)* — host path to this task's ``tests/`` directory
  (``<dataset>/<task>/tests``); uploaded into ``tests_target`` in the runtime.
- ``verifier_timeout`` *(float, default 120)* — seconds for ``test_command``,
  clamped to the session-wide budget. Matches Harbor's ``[verifier].timeout_sec``.
- ``tests_target`` *(str, default ``/tests``)* — where the verifier is injected.
- ``verifier_dir`` *(str, default ``/logs/verifier``)* — where ``test.sh`` writes.
- ``test_command`` *(str, default ``bash /tests/test.sh``)* — verifier entrypoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory


class HarborEvaluator(BaseTrajectoryEvaluator):
    """Grade a rollout by running a Harbor ``tests/test.sh`` in the live runtime."""

    MODE = "harbor"

    def __init__(
        self,
        *,
        tests_dir: str,
        verifier_timeout: float = 120.0,
        tests_target: str = "/tests",
        verifier_dir: str = "/logs/verifier",
        test_command: str = "bash /tests/test.sh",
    ) -> None:
        self.tests_dir = str(tests_dir).strip()
        if not self.tests_dir:
            raise ValueError("harbor evaluator requires a non-empty 'tests_dir'")
        if not Path(self.tests_dir).is_dir():
            raise FileNotFoundError(f"harbor evaluator tests_dir does not exist: {self.tests_dir}")
        self.verifier_timeout = float(verifier_timeout)
        if self.verifier_timeout <= 0:
            raise ValueError("verifier_timeout must be greater than 0")
        self.tests_target = tests_target.rstrip("/") or "/tests"
        self.verifier_dir = verifier_dir.rstrip("/") or "/logs/verifier"
        self.test_command = test_command.strip()
        if not self.test_command:
            raise ValueError("harbor evaluator requires a non-empty 'test_command'")

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        rt = runtime.get("runtime")
        if not isinstance(rt, BaseRuntime):
            raise RuntimeError(
                "harbor evaluator requires a live runtime; submit with "
                "refresh_runtime=false so the agent's runtime reaches the evaluator"
            )

        artifacts_dir = Path(runtime["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        env = runtime.get("env")
        eval_env = env if isinstance(env, dict) else {}
        cap = runtime.get("timeout_seconds")
        test_timeout = self.verifier_timeout if cap is None else min(self.verifier_timeout, float(cap))

        # 1. Inject the verifier into the container the agent just used.
        await rt.exec(
            f"rm -rf {self.tests_target} {self.verifier_dir} && "
            f"mkdir -p {self.tests_target} {self.verifier_dir}",
            env=eval_env,
        )
        await rt.upload_dir(self.tests_dir, self.tests_target)
        await rt.exec(f"chmod -R +x {self.tests_target} 2>/dev/null || true", env=eval_env)

        # 2. Run the verifier (writes 0/1 to reward.txt, the Harbor contract).
        result = await rt.exec(self.test_command, env=eval_env, timeout_sec=test_timeout)
        test_output = (result.stdout or "") + (result.stderr or "")
        test_output_path = artifacts_dir / "verifier.stdout.log"
        test_output_path.write_text(test_output)

        # 3. Read the reward back, clamped to [0, 1] (mirrors Harbor's reward parsing).
        reward = await self._read_reward(rt, eval_env)

        metadata: dict[str, Any] = {
            "mode": self.MODE,
            "resolved": reward >= 1.0,
            "reward": reward,
            "verifier_exit_code": result.return_code,
            "verifier_timeout": result.return_code == -1,
            "test_output_path": str(test_output_path),
        }
        return EvalResult(outcome_reward=reward, metadata=metadata)

    async def _read_reward(self, rt: BaseRuntime, env: dict[str, str]) -> float:
        text = await rt.exec(f"cat {self.verifier_dir}/reward.txt 2>/dev/null", env=env)
        if text.return_code == 0 and (text.stdout or "").strip():
            try:
                return _clamp(float(text.stdout.strip()))
            except ValueError:
                pass
        # Fallback: Harbor also accepts a reward.json (scalar or {name: reward}).
        blob = await rt.exec(f"cat {self.verifier_dir}/reward.json 2>/dev/null", env=env)
        if blob.return_code == 0 and (blob.stdout or "").strip():
            try:
                data = json.loads(blob.stdout)
                if isinstance(data, (int, float)):
                    return _clamp(float(data))
                if isinstance(data, dict) and data:
                    return _clamp(sum(float(v) for v in data.values()) / len(data))
            except (ValueError, TypeError):
                pass
        return 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
