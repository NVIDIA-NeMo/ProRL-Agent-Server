from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec
from polar.trajectory.evaluator.harbor import HarborEvaluator
from polar.trajectory.models import Trajectory


class HarborRuntime(BaseRuntime):
    def __init__(
        self,
        session_dir: Path,
        artifacts_dir: Path,
        *,
        verifier_return_code: int = 0,
        reported_reward: str | None = "1",
        reported_reward_json: str | None = None,
        upload_failures: int = 0,
        destroy_on_upload_failure: bool = False,
    ) -> None:
        super().__init__(RuntimeSpec(image="image"), "session", session_dir)
        self._test_artifacts_dir = artifacts_dir
        self._verifier_return_code = verifier_return_code
        self._reported_reward = reported_reward
        self._reported_reward_json = reported_reward_json
        self._upload_failures = upload_failures
        self._destroy_on_upload_failure = destroy_on_upload_failure
        self.upload_attempts = 0

    @property
    def runtime_id(self) -> str:
        return "harbor-stub"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        if command == "bash /tests/test.sh":
            shutil.rmtree(self._test_artifacts_dir)
            return ExecResult(
                stdout="passed\n",
                stderr="warning\n",
                return_code=self._verifier_return_code,
            )
        if "reward.txt" in command:
            if self._reported_reward is None:
                return ExecResult(return_code=1)
            return ExecResult(stdout=self._reported_reward, return_code=0)
        if "reward.json" in command:
            if self._reported_reward_json is None:
                return ExecResult(return_code=1)
            return ExecResult(stdout=self._reported_reward_json, return_code=0)
        return ExecResult(return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        pass

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        self.upload_attempts += 1
        if self.upload_attempts <= self._upload_failures:
            if self._destroy_on_upload_failure:
                self._destroyed = True
            raise RuntimeError("transient apptainer upload failure")

    async def download_file(self, remote_path: str, local_path: str) -> None:
        pass

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        pass


@pytest.mark.asyncio
async def test_harbor_recreates_artifacts_dir_after_verifier_wait(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    artifacts_dir = tmp_path / "session" / "artifacts"
    runtime = HarborRuntime(tmp_path / "session", artifacts_dir)
    evaluator = HarborEvaluator(tests_dir=str(tests_dir))

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        runtime=runtime,
        artifacts_dir=artifacts_dir,
    )

    assert result.outcome_reward == 1.0
    assert result.metadata["verifier_reported_reward"] == 1.0
    assert result.metadata["verifier_reward_accepted"] is True
    assert (artifacts_dir / "verifier.stdout.log").read_text() == ("passed\nwarning\n")


@pytest.mark.asyncio
async def test_harbor_rejects_positive_reward_when_verifier_exits_nonzero(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    artifacts_dir = tmp_path / "session" / "artifacts"
    runtime = HarborRuntime(
        tmp_path / "session",
        artifacts_dir,
        verifier_return_code=2,
        reported_reward="1",
    )
    evaluator = HarborEvaluator(tests_dir=str(tests_dir))

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        runtime=runtime,
        artifacts_dir=artifacts_dir,
    )

    assert result.outcome_reward == 0.0
    assert result.metadata["resolved"] is False
    assert result.metadata["reward"] == 0.0
    assert result.metadata["verifier_reported_reward"] == 1.0
    assert result.metadata["verifier_reward_accepted"] is False
    assert result.metadata["verifier_exit_code"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reported_reward", ["nan", "NaN", "inf", "+Infinity", "-inf"])
async def test_harbor_rejects_nonfinite_text_reward(
    tmp_path,
    reported_reward: str,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    artifacts_dir = tmp_path / "session" / "artifacts"
    runtime = HarborRuntime(
        tmp_path / "session",
        artifacts_dir,
        reported_reward=reported_reward,
    )
    evaluator = HarborEvaluator(tests_dir=str(tests_dir))

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        runtime=runtime,
        artifacts_dir=artifacts_dir,
    )

    assert result.outcome_reward == 0.0
    assert result.metadata["resolved"] is False
    assert result.metadata["verifier_reported_reward"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reported_reward_json",
    ["NaN", "Infinity", "-Infinity", "true", '{"score": NaN}', '{"score": true}'],
)
async def test_harbor_rejects_nonfinite_or_boolean_json_reward(
    tmp_path,
    reported_reward_json: str,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    artifacts_dir = tmp_path / "session" / "artifacts"
    runtime = HarborRuntime(
        tmp_path / "session",
        artifacts_dir,
        reported_reward=None,
        reported_reward_json=reported_reward_json,
    )
    evaluator = HarborEvaluator(tests_dir=str(tests_dir))

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        runtime=runtime,
        artifacts_dir=artifacts_dir,
    )

    assert result.outcome_reward == 0.0
    assert result.metadata["resolved"] is False
    assert result.metadata["verifier_reported_reward"] == 0.0


@pytest.mark.asyncio
async def test_harbor_retries_transient_tests_upload(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    artifacts_dir = tmp_path / "session" / "artifacts"
    runtime = HarborRuntime(
        tmp_path / "session",
        artifacts_dir,
        upload_failures=2,
    )
    evaluator = HarborEvaluator(
        tests_dir=str(tests_dir),
        upload_attempts=3,
        upload_retry_backoff_seconds=0,
    )

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        runtime=runtime,
        artifacts_dir=artifacts_dir,
    )

    assert runtime.upload_attempts == 3
    assert result.outcome_reward == 1.0


@pytest.mark.asyncio
async def test_harbor_does_not_retry_upload_after_runtime_destroyed(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    artifacts_dir = tmp_path / "session" / "artifacts"
    runtime = HarborRuntime(
        tmp_path / "session",
        artifacts_dir,
        upload_failures=1,
        destroy_on_upload_failure=True,
    )
    evaluator = HarborEvaluator(
        tests_dir=str(tests_dir),
        upload_attempts=3,
        upload_retry_backoff_seconds=0,
    )

    with pytest.raises(RuntimeError, match="transient apptainer upload failure"):
        await evaluator.evaluate(
            Trajectory(status="COMPLETED"),
            runtime=runtime,
            artifacts_dir=artifacts_dir,
        )

    assert runtime.upload_attempts == 1


@pytest.mark.asyncio
async def test_harbor_raises_after_upload_attempts_exhausted(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    artifacts_dir = tmp_path / "session" / "artifacts"
    runtime = HarborRuntime(
        tmp_path / "session",
        artifacts_dir,
        upload_failures=3,
    )
    evaluator = HarborEvaluator(
        tests_dir=str(tests_dir),
        upload_attempts=3,
        upload_retry_backoff_seconds=0,
    )

    with pytest.raises(RuntimeError, match="transient apptainer upload failure"):
        await evaluator.evaluate(
            Trajectory(status="COMPLETED"),
            runtime=runtime,
            artifacts_dir=artifacts_dir,
        )

    assert runtime.upload_attempts == 3


@pytest.mark.asyncio
async def test_harbor_emits_negative_gpt_cost_component(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    artifacts_dir = tmp_path / "session" / "artifacts"
    runtime = HarborRuntime(tmp_path / "session", artifacts_dir)
    evaluator = HarborEvaluator(
        tests_dir=str(tests_dir),
        emit_cost_reward=True,
    )

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        runtime=runtime,
        artifacts_dir=artifacts_dir,
        agent_result=SimpleNamespace(
            metadata={"controller_v3_cost": {"cost": 0.0123}}
        ),
    )

    assert result.outcome_reward == 1.0
    assert result.outcome_reward_components == {
        "harbor_reward": 1.0,
        "negative_cost": -0.0123,
    }
    assert result.metadata["gpt_cost_usd"] == 0.0123


@pytest.mark.asyncio
async def test_harbor_cost_reward_fails_closed_without_usage(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    artifacts_dir = tmp_path / "session" / "artifacts"
    runtime = HarborRuntime(tmp_path / "session", artifacts_dir)
    evaluator = HarborEvaluator(
        tests_dir=str(tests_dir),
        emit_cost_reward=True,
    )

    with pytest.raises(RuntimeError, match="cost metadata is missing"):
        await evaluator.evaluate(
            Trajectory(status="COMPLETED"),
            runtime=runtime,
            artifacts_dir=artifacts_dir,
            agent_result=SimpleNamespace(metadata={}),
        )
