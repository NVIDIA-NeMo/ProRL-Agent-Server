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
    assert result.trace_rewards == pytest.approx([1.2, 0.8])
    assert result.metadata["rubric_applied"] is True
    assert result.metadata["judge_scores"] == [5, -5]
    assert result.metadata["judge_failures"] == 0
    assert result.metadata["judge_model"] == "judge-1"
    assert "builder_warning" not in result.metadata
    # One judge call for the whole rollout; the debug record is persisted.
    assert len(seen_prompts) == 1
    record = json.loads((tmp_path / "artifacts" / "judge" / "rollout.json").read_text())
    assert record["scores"] == [5, -5]


def test_judge_prompt_contains_reward_json_and_trace_ids(
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
    # Chronological response messages, without prompt-side observation turns.
    assert prompt.index("ALPHA step") < prompt.index("BETA done")
    assert "observation" not in prompt
    assert "## Meta rubric" in prompt
    assert "Reward hacking is a strict -5" in prompt
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


def test_non_per_request_builder_records_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = _make_task_dir(tmp_path)
    evaluator = _make_evaluator(tests_dir)
    _patch_judge(monkeypatch, [0, 0])

    result = asyncio.run(
        evaluator.evaluate(
            _make_trajectory(builder="prefix_merging"),
            **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path)),
        )
    )

    assert "prefix_merging" in result.metadata["builder_warning"]


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
