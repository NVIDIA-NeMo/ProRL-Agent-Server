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
- ``upload_attempts`` *(int, default 3)* — attempts for transient verifier upload
  failures before the sample is marked erroneous.
- ``upload_retry_backoff_seconds`` *(float, default 0.1)* — initial exponential
  backoff between upload attempts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

from polar.runtime.base import BaseRuntime, RuntimeDestroyedError
from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory


logger = logging.getLogger(__name__)


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
        upload_attempts: int = 3,
        upload_retry_backoff_seconds: float = 0.1,
        emit_cost_reward: bool = False,
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
        self.upload_attempts = int(upload_attempts)
        if not 1 <= self.upload_attempts <= 10:
            raise ValueError("upload_attempts must be between 1 and 10")
        self.upload_retry_backoff_seconds = float(upload_retry_backoff_seconds)
        if not 0.0 <= self.upload_retry_backoff_seconds <= 60.0:
            raise ValueError("upload_retry_backoff_seconds must be between 0 and 60")
        self.emit_cost_reward = bool(emit_cost_reward)

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
        test_timeout = (
            self.verifier_timeout if cap is None else min(self.verifier_timeout, float(cap))
        )

        # 1. Inject the verifier into the container the agent just used.
        await rt.exec(
            f"rm -rf {self.tests_target} {self.verifier_dir} && "
            f"mkdir -p {self.tests_target} {self.verifier_dir}",
            env=eval_env,
        )
        await self._upload_tests(rt)
        await rt.exec(f"chmod -R +x {self.tests_target} 2>/dev/null || true", env=eval_env)

        # 2. Run the verifier (writes 0/1 to reward.txt, the Harbor contract).
        result = await rt.exec(self.test_command, env=eval_env, timeout_sec=test_timeout)
        test_output = (result.stdout or "") + (result.stderr or "")
        test_output_path = artifacts_dir / "verifier.stdout.log"
        # The verifier awaits a subprocess, so a concurrent session cleanup
        # may remove the directory after the initial mkdir above.
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        test_output_path.write_text(test_output)

        # 3. Read the verifier's reported reward for diagnostics, but only
        # accept it when the verifier itself exited successfully.  A crashed
        # or timed-out verifier can leave a stale/partial reward file behind;
        # treating that as success would train on an invalid evaluation.
        reported_reward = await self._read_reward(rt, eval_env)
        verifier_succeeded = result.return_code == 0
        reward = reported_reward if verifier_succeeded else 0.0

        metadata: dict[str, Any] = {
            "mode": self.MODE,
            "resolved": reward >= 1.0,
            "reward": reward,
            "verifier_reported_reward": reported_reward,
            "verifier_reward_accepted": verifier_succeeded,
            "verifier_exit_code": result.return_code,
            "verifier_timeout": result.return_code == -1,
            "test_output_path": str(test_output_path),
        }
        reward_components = None
        if self.emit_cost_reward:
            cost = _controller_v3_cost(runtime.get("agent_result"))
            reward_components = {
                "harbor_reward": reward,
                "negative_cost": -cost,
            }
            metadata["gpt_cost_usd"] = cost
        return EvalResult(
            outcome_reward=reward,
            outcome_reward_components=reward_components,
            metadata=metadata,
        )

    async def _upload_tests(self, rt: BaseRuntime) -> None:
        """Retry transient Apptainer/tar setup failures before losing a sample."""

        for attempt in range(1, self.upload_attempts + 1):
            try:
                await rt.upload_dir(self.tests_dir, self.tests_target)
                return
            except RuntimeDestroyedError:
                # Cancellation tears down the direct-exec runtime. Retrying it
                # would start fresh Apptainer commands after DELETE was acked.
                raise
            except Exception as exc:
                if rt.destroyed or attempt >= self.upload_attempts:
                    raise
                logger.warning(
                    "Harbor tests upload failed (attempt %d/%d): %s",
                    attempt,
                    self.upload_attempts,
                    exc,
                )
                if self.upload_retry_backoff_seconds > 0.0:
                    await asyncio.sleep(
                        min(
                            self.upload_retry_backoff_seconds * (2 ** (attempt - 1)),
                            60.0,
                        )
                    )

    async def _read_reward(self, rt: BaseRuntime, env: dict[str, str]) -> float:
        text = await rt.exec(f"cat {self.verifier_dir}/reward.txt 2>/dev/null", env=env)
        if text.return_code == 0 and (text.stdout or "").strip():
            try:
                return _clamp(text.stdout.strip())
            except (TypeError, ValueError):
                pass
        # Fallback: Harbor also accepts a reward.json (scalar or {name: reward}).
        blob = await rt.exec(f"cat {self.verifier_dir}/reward.json 2>/dev/null", env=env)
        if blob.return_code == 0 and (blob.stdout or "").strip():
            try:
                data = json.loads(blob.stdout)
                if isinstance(data, (int, float)) and not isinstance(data, bool):
                    return _clamp(data)
                if isinstance(data, dict) and data:
                    values = [_finite_float(value) for value in data.values()]
                    return _clamp(sum(values) / len(values))
            except (ValueError, TypeError):
                pass
        return 0.0


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not a Harbor reward")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Harbor reward must be finite")
    return parsed


def _clamp(value: Any) -> float:
    parsed = _finite_float(value)
    return max(0.0, min(1.0, parsed))


def _controller_v3_cost(agent_result: Any) -> float:
    metadata = getattr(agent_result, "metadata", None)
    if not isinstance(metadata, dict) and isinstance(agent_result, dict):
        metadata = agent_result.get("metadata")
    cost_metadata = (
        metadata.get("controller_v3_cost") if isinstance(metadata, dict) else None
    )
    if not isinstance(cost_metadata, dict):
        raise RuntimeError("Controller V3 GPT cost metadata is missing")
    cost = _finite_float(cost_metadata.get("cost"))
    if cost < 0:
        raise ValueError("Controller V3 GPT cost must be nonnegative")
    return cost
