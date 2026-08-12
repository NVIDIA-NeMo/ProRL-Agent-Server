from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec
from polar.trajectory.evaluator.harbor_rubric import (
    HarborEvaluatorWithRubric,
    _parse_trace_scores,
)
from polar.trajectory.models import Trace, Trajectory

REWARD_JSON = '{"chart_selection": 1, "alt_text_insights": 0}'


class FakeRuntime(BaseRuntime):
    """Runtime stub whose verifier always reports a fixed reward."""

    def __init__(
        self, tmp_path: Path, reward: str = "1.0", reward_json: str = REWARD_JSON
    ) -> None:
        super().__init__(RuntimeSpec(image="fake"), "session-1", tmp_path / "session")
        self.reward = reward
        self.reward_json = reward_json

    @property
    def runtime_id(self) -> str:
        return "fake"

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        if "reward.json" in command:
            return ExecResult(stdout=self.reward_json, return_code=0)
        if "reward.txt" in command:
            return ExecResult(stdout=self.reward, return_code=0)
        return ExecResult(stdout="", return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None: ...

    async def upload_dir(self, local_path: str, remote_path: str) -> None: ...

    async def download_file(self, remote_path: str, local_path: str) -> None: ...

    async def download_dir(self, remote_path: str, local_path: str) -> None: ...


def _make_task_dir(tmp_path: Path, *, with_rubric: bool = True) -> Path:
    task_dir = tmp_path / "task"
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("#!/bin/bash\n")
    (task_dir / "instruction.md").write_text("Build the chart pack.")
    if with_rubric:
        (tests_dir / "rubric.md").write_text("## Must-do\n- do the right thing\n")
    return tests_dir


def _make_evaluator(tests_dir: Path, **overrides: Any) -> HarborEvaluatorWithRubric:
    config: dict[str, Any] = {
        "tests_dir": str(tests_dir),
        "judge_base_url": "http://judge.local/v1",
        "judge_model": "judge-1",
        "rubric_coefficient": 0.2,
    }
    config.update(overrides)
    return HarborEvaluatorWithRubric(**config)


def _make_trajectory(builder: str = "per_request") -> Trajectory:
    return Trajectory(
        status="COMPLETED",
        metadata={"builder": builder},
        traces=[
            Trace(
                prompt_messages=[{"role": "user", "content": "task"}],
                response_messages=[{"role": "assistant", "content": "ALPHA step"}],
            ),
            Trace(
                prompt_messages=[
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "ALPHA step"},
                    {"role": "tool", "content": "observation"},
                ],
                response_messages=[{"role": "assistant", "content": "BETA done"}],
            ),
        ],
    )


def _runtime_kwargs(tmp_path: Path, runtime: FakeRuntime) -> dict[str, Any]:
    return {
        "runtime": runtime,
        "artifacts_dir": tmp_path / "artifacts",
        "env": {},
        "timeout_seconds": None,
    }


def _patch_judge(
    monkeypatch: pytest.MonkeyPatch,
    scores: list[int | None],
    seen_prompts: list[str] | None = None,
) -> None:
    """Stub the single per-rollout judge call with fixed per-trace scores."""

    async def fake_call_judge(
        self: HarborEvaluatorWithRubric,
        client: Any,
        messages: list[dict[str, str]],
        trace_count: int,
    ) -> tuple[list[int | None], dict[str, Any]]:
        assert trace_count == len(scores)
        if seen_prompts is not None:
            seen_prompts.append(messages[-1]["content"])
        return list(scores), {"attempts": []}

    monkeypatch.setattr(HarborEvaluatorWithRubric, "_call_judge", fake_call_judge)


def test_rubric_present_blends_outcome_and_judge_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = _make_task_dir(tmp_path)
    evaluator = _make_evaluator(tests_dir)
    seen_prompts: list[str] = []
    _patch_judge(monkeypatch, [5, -5], seen_prompts)

    result = asyncio.run(
        evaluator.evaluate(
            _make_trajectory(), **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path))
        )
    )

    assert result.outcome_reward == 1.0
    assert result.trace_rewards == pytest.approx([1.0, 0.0])
    assert result.metadata["rubric_applied"] is True
    assert result.metadata["judge_scores"] == [5, -5]
    assert result.metadata["judge_failures"] == 0
    assert result.metadata["judge_model"] == "judge-1"
    assert result.metadata["judge_calibration"] == "trace_behavior_alignment"
    assert "builder_warning" not in result.metadata
    # One judge call for the whole rollout; the debug record is persisted.
    assert len(seen_prompts) == 1
    record = json.loads((tmp_path / "artifacts" / "judge" / "rollout.json").read_text())
    assert record["scores"] == [5, -5]


@pytest.mark.parametrize(
    ("outcome", "score", "expected"),
    [
        (1.0, 0, 0.8),
        (1.0, -4, 0.64),
        (0.0, 5, 0.2),
        (0.0, -1, 0.0),
        (0.5, None, 0.5),
        (1.0, -5, 0.0),
    ],
)
def test_calibrated_reward_is_bounded(
    tmp_path: Path, outcome: float, score: int | None, expected: float
) -> None:
    evaluator = _make_evaluator(_make_task_dir(tmp_path))

    assert evaluator._calibrate_reward(outcome, score) == pytest.approx(expected)


@pytest.mark.parametrize("coefficient", [-0.1, 1.1])
def test_rubric_coefficient_must_be_between_zero_and_one(
    tmp_path: Path, coefficient: float
) -> None:
    with pytest.raises(ValueError, match="rubric_coefficient must be between 0 and 1"):
        _make_evaluator(_make_task_dir(tmp_path), rubric_coefficient=coefficient)


def test_judge_prompt_calibrates_trace_behavior_without_step_assumptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = _make_task_dir(tmp_path)
    evaluator = _make_evaluator(tests_dir)
    seen_prompts: list[str] = []
    _patch_judge(monkeypatch, [0, 0], seen_prompts)

    asyncio.run(
        evaluator.evaluate(
            _make_trajectory(), **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path))
        )
    )

    prompt = seen_prompts[0]
    assert REWARD_JSON in prompt
    assert '<trace id="trace_0">' in prompt
    assert '<trace id="trace_1">' in prompt
    # Response messages retain builder-provided order, without prompt-side turns.
    assert prompt.index("ALPHA step") < prompt.index("BETA done")
    assert "### TOOL\nobservation" not in prompt
    assert "## Meta rubric" in prompt
    assert "Reward hacking or solution gaming is a strict -5" in prompt
    assert "Must-do requirements or Best-practice" in prompt
    assert "explicit Must-avoid violation" in prompt
    assert "Trace ids are labels, not guaranteed chronological steps" in prompt
    assert "do not treat trace ids as solution steps" in prompt
    assert "Do not require a trace to finish the whole task" in prompt
    assert "loops of the same failed action" in prompt
    assert "how critically it contributes" not in prompt
    assert "one agent turn, in chronological order" not in prompt
    assert "Build the chart pack." in prompt


def test_rubric_absent_degrades_to_plain_harbor(tmp_path: Path) -> None:
    tests_dir = _make_task_dir(tmp_path, with_rubric=False)
    evaluator = _make_evaluator(tests_dir)

    result = asyncio.run(
        evaluator.evaluate(
            _make_trajectory(), **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path))
        )
    )

    assert result.outcome_reward == 1.0
    assert result.trace_rewards is None
    assert result.metadata["rubric_applied"] is False


def test_missing_scores_fall_back_to_outcome_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = _make_task_dir(tmp_path)
    evaluator = _make_evaluator(tests_dir)
    _patch_judge(monkeypatch, [3, None])

    result = asyncio.run(
        evaluator.evaluate(
            _make_trajectory(),
            **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path, reward="0")),
        )
    )

    assert result.outcome_reward == 0.0
    assert result.trace_rewards == pytest.approx([0.2 * 3 / 5, 0.0])
    assert result.metadata["judge_scores"] == [3, None]
    assert result.metadata["judge_failures"] == 1


def test_non_per_request_builder_uses_same_trace_behavior_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = _make_task_dir(tmp_path)
    evaluator = _make_evaluator(tests_dir)
    seen_prompts: list[str] = []
    _patch_judge(monkeypatch, [0, 0], seen_prompts)

    result = asyncio.run(
        evaluator.evaluate(
            _make_trajectory(builder="prefix_merging"),
            **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path)),
        )
    )

    assert "builder_warning" not in result.metadata
    assert result.metadata["judge_calibration"] == "trace_behavior_alignment"
    assert "Depending on the builder" in seen_prompts[0]
    assert "a complete rollout, or a parallel branch" in seen_prompts[0]


def test_render_traces_keeps_tool_calls(tmp_path: Path) -> None:
    tests_dir = _make_task_dir(tmp_path)
    evaluator = _make_evaluator(tests_dir)
    traces = [
        Trace(
            prompt_messages=[{"role": "user", "content": "task"}],
            response_messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "run_shell", "arguments": '{"cmd": "ls"}'}}
                    ],
                }
            ],
        ),
    ]

    rendered = evaluator._render_traces(traces)

    assert '<trace id="trace_0">' in rendered
    assert '[tool_call] run_shell({"cmd": "ls"})' in rendered


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"trace_0": {"score": 4, "rationale": "good"}, "trace_1": {"score": -2}}', [4, -2]),
        ('prose {"trace_0": 3, "trace_1": {"score": 12}} after', [3, 5]),
        ('{"scores": {"trace_0": {"score": -99}, "trace_1": {"score": 0}}}', [-5, 0]),
        ('{"trace_1": {"score": 2}}', [None, 2]),
        ('{"trace_0": {"score": "bad"}, "trace_1": {"score": 1}}', [None, 1]),
        ("no json here", [None, None]),
        ('{"other": 1}', [None, None]),
    ],
)
def test_parse_trace_scores(text: str, expected: list[int | None]) -> None:
    assert _parse_trace_scores(text, 2) == expected
