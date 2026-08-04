from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import httpx

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


def test_fallback_rubric_scores_task_without_rubric_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = _make_task_dir(tmp_path, with_rubric=False)
    evaluator = _make_evaluator(
        tests_dir,
        fallback_rubric="## Generic process rubric\n- make efficient, verified progress",
        require_rubric=True,
    )
    seen_prompts: list[str] = []
    _patch_judge(monkeypatch, [2, -1], seen_prompts)

    result = asyncio.run(
        evaluator.evaluate(
            _make_trajectory(), **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path))
        )
    )

    assert result.trace_rewards == pytest.approx([1.08, 0.96])
    assert result.metadata["rubric_applied"] is True
    assert result.metadata["rubric_source"] == "fallback"
    assert "Generic process rubric" in seen_prompts[0]


def test_required_rubric_rejects_missing_task_and_fallback_rubric(
    tmp_path: Path,
) -> None:
    evaluator = _make_evaluator(
        _make_task_dir(tmp_path, with_rubric=False),
        require_rubric=True,
    )

    with pytest.raises(RuntimeError, match="fallback_rubric"):
        asyncio.run(
            evaluator.evaluate(
                _make_trajectory(), **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path))
            )
        )


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


def test_prefix_merging_builder_is_supported(
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

    assert "builder_warning" not in result.metadata


def test_unknown_builder_records_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = _make_task_dir(tmp_path)
    evaluator = _make_evaluator(tests_dir)
    _patch_judge(monkeypatch, [0, 0])

    result = asyncio.run(
        evaluator.evaluate(
            _make_trajectory(builder="unknown"),
            **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path)),
        )
    )

    assert "unknown" in result.metadata["builder_warning"]


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


def test_render_traces_optionally_attaches_matching_tool_outputs(tmp_path: Path) -> None:
    evaluator = _make_evaluator(
        _make_task_dir(tmp_path),
        judge_include_tool_outputs=True,
    )
    traces = [
        Trace(
            response_messages=[
                {
                    "role": "assistant",
                    "content": "inspect",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "function": {"name": "bash", "arguments": '{"cmd":"ls"}'},
                        }
                    ],
                }
            ],
        ),
        Trace(
            # Cumulative prompts repeat old observations; the renderer must
            # attach this result once, to trace_0 rather than trace_1.
            prompt_messages=[
                {"role": "tool", "tool_call_id": "call-a", "content": "a.py\nb.py"},
                {"role": "tool", "tool_call_id": "unrelated", "content": "ignore"},
            ],
            response_messages=[{"role": "assistant", "content": "done"}],
        ),
    ]

    rendered = evaluator._render_traces(traces)

    trace_0, trace_1 = rendered.split('<trace id="trace_1">')
    assert '<tool_output tool_call_id="call-a" tool_name="bash">' in trace_0
    assert "a.py\nb.py" in trace_0
    assert "unrelated" not in rendered
    assert "a.py\nb.py" not in trace_1


def test_render_traces_tool_outputs_default_off(tmp_path: Path) -> None:
    evaluator = _make_evaluator(_make_task_dir(tmp_path))
    traces = [
        Trace(
            response_messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call-a", "function": {"name": "bash", "arguments": "{}"}}
                    ],
                }
            ]
        ),
        Trace(
            prompt_messages=[
                {"role": "tool", "tool_call_id": "call-a", "content": "secret output"}
            ],
            response_messages=[{"role": "assistant", "content": "next"}],
        ),
    ]

    rendered = evaluator._render_traces(traces)

    assert "secret output" not in rendered
    assert "<tool_outputs>" not in rendered


def test_render_prefix_merged_trace_attaches_interstitial_tool_output_once(
    tmp_path: Path,
) -> None:
    evaluator = _make_evaluator(
        _make_task_dir(tmp_path),
        judge_include_tool_outputs=True,
    )
    traces = [
        Trace(
            response_messages=[
                {
                    "role": "assistant",
                    "content": "inspect",
                    "tool_calls": [
                        {
                            "id": "merged-call",
                            "function": {"name": "bash", "arguments": '{"cmd":"ls"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "merged-call",
                    "content": "merged-result.txt",
                },
                {"role": "assistant", "content": "use the result"},
            ]
        )
    ]

    rendered = evaluator._render_traces(traces)

    assert rendered.count("merged-result.txt") == 1
    assert (
        '<tool_output tool_call_id="merged-call" tool_name="bash">'
        in rendered
    )


def test_render_prefix_merged_trace_hides_tool_output_when_disabled(
    tmp_path: Path,
) -> None:
    evaluator = _make_evaluator(_make_task_dir(tmp_path))
    traces = [
        Trace(
            response_messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "merged-call",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "merged-call",
                    "content": "merged secret",
                },
            ]
        )
    ]

    rendered = evaluator._render_traces(traces)

    assert "merged secret" not in rendered
    assert "<tool_outputs>" not in rendered


def test_render_traces_truncates_each_tool_output(tmp_path: Path) -> None:
    evaluator = _make_evaluator(
        _make_task_dir(tmp_path),
        judge_include_tool_outputs=True,
        judge_tool_output_max_chars=20,
    )
    traces = [
        Trace(
            response_messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call-a", "function": {"name": "bash", "arguments": "{}"}}
                    ],
                }
            ]
        ),
        Trace(
            prompt_messages=[
                {"role": "tool", "tool_call_id": "call-a", "content": "A" * 100}
            ],
            response_messages=[{"role": "assistant", "content": "next"}],
        ),
    ]

    rendered = evaluator._render_traces(traces)

    assert "truncated" in rendered
    assert "A" * 100 not in rendered


def test_render_traces_preserves_all_trace_ids_under_total_cap(tmp_path: Path) -> None:
    evaluator = _make_evaluator(
        _make_task_dir(tmp_path),
        judge_include_tool_outputs=True,
        judge_tool_output_max_chars=10_000,
        max_section_chars=1_000,
    )
    traces = []
    for index in range(4):
        call_id = f"call-{index}"
        traces.append(
            Trace(
                prompt_messages=(
                    []
                    if index == 0
                    else [
                        {
                            "role": "tool",
                            "tool_call_id": f"call-{index - 1}",
                            "content": str(index - 1) * 2_000,
                        }
                    ]
                ),
                response_messages=[
                    {
                        "role": "assistant",
                        "content": f"step {index}",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "function": {"name": "bash", "arguments": "{}"},
                            }
                        ],
                    }
                ],
            )
        )

    rendered = evaluator._render_traces(traces)

    assert len(rendered) <= 1_000
    for index in range(4):
        assert f'<trace id="trace_{index}">' in rendered


def test_long_rollout_is_scored_in_configured_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _make_evaluator(
        _make_task_dir(tmp_path),
        judge_max_traces_per_call=2,
    )
    trajectory = Trajectory(
        status="COMPLETED",
        metadata={"builder": "per_request"},
        traces=[
            Trace(response_messages=[{"role": "assistant", "content": f"step {i}"}])
            for i in range(5)
        ],
    )
    call_sizes: list[int] = []

    async def fake_call_judge(
        self: HarborEvaluatorWithRubric,
        client: Any,
        messages: list[dict[str, str]],
        trace_count: int,
    ) -> tuple[list[int | None], dict[str, Any]]:
        call_sizes.append(trace_count)
        return [trace_count] * trace_count, {"request": {}, "attempts": []}

    monkeypatch.setattr(HarborEvaluatorWithRubric, "_call_judge", fake_call_judge)

    result = asyncio.run(
        evaluator.evaluate(
            trajectory,
            **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path)),
        )
    )

    assert call_sizes == [2, 2, 1]
    assert result.metadata["judge_scores"] == [2, 2, 2, 2, 1]
    record = json.loads((tmp_path / "artifacts" / "judge" / "rollout.json").read_text())
    assert record["chunk_count"] == 3
    assert [chunk["global_trace_start"] for chunk in record["chunks"]] == [0, 2, 4]
    assert record["scores"] == [2, 2, 2, 2, 1]


def test_chunk_boundary_keeps_preceding_trace_tool_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _make_evaluator(
        _make_task_dir(tmp_path),
        judge_include_tool_outputs=True,
        judge_max_traces_per_call=1,
    )
    trajectory = Trajectory(
        status="COMPLETED",
        metadata={"builder": "per_request"},
        traces=[
            Trace(
                response_messages=[
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "boundary-call",
                                "function": {"name": "bash", "arguments": "{}"},
                            }
                        ],
                    }
                ]
            ),
            Trace(
                prompt_messages=[
                    {
                        "role": "tool",
                        "tool_call_id": "boundary-call",
                        "content": "boundary result",
                    }
                ],
                response_messages=[{"role": "assistant", "content": "done"}],
            ),
        ],
    )
    judge_prompts: list[str] = []

    async def fake_call_judge(
        self: HarborEvaluatorWithRubric,
        client: Any,
        messages: list[dict[str, str]],
        trace_count: int,
    ) -> tuple[list[int | None], dict[str, Any]]:
        judge_prompts.append(messages[-1]["content"])
        return [1] * trace_count, {"request": {}, "attempts": []}

    monkeypatch.setattr(HarborEvaluatorWithRubric, "_call_judge", fake_call_judge)

    asyncio.run(
        evaluator.evaluate(
            trajectory,
            **_runtime_kwargs(tmp_path, FakeRuntime(tmp_path)),
        )
    )

    assert "boundary result" in judge_prompts[0]
    assert "boundary result" not in judge_prompts[1]


def test_responses_judge_uses_responses_protocol(tmp_path: Path) -> None:
    evaluator = _make_evaluator(
        _make_task_dir(tmp_path),
        judge_base_url="https://judge.local/v1/responses",
        judge_api="responses",
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"trace_0": {"score": 4}}',
                            }
                        ],
                    }
                ]
            },
        )

    async def call() -> tuple[list[int | None], dict[str, Any]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await evaluator._call_judge(
                client,
                [{"role": "user", "content": "grade"}],
                1,
            )

    scores, _record = asyncio.run(call())
    assert scores == [4]
    assert str(seen[0].url) == "https://judge.local/v1/responses"
    assert json.loads(seen[0].content) == {
        "model": "judge-1",
        "input": [{"role": "user", "content": "grade"}],
        "max_output_tokens": 8192,
    }


def test_partial_judge_output_retries_and_keeps_best_result(tmp_path: Path) -> None:
    evaluator = _make_evaluator(
        _make_task_dir(tmp_path),
        judge_api="responses",
        judge_max_retries=1,
    )
    replies = iter(
        [
            '{"trace_0": {"score": 2}}',
            '{"trace_0": {"score": 2}, "trace_1": {"score": 4}}',
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output_text": next(replies)})

    async def call() -> tuple[list[int | None], dict[str, Any]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await evaluator._call_judge(
                client,
                [{"role": "user", "content": "grade"}],
                2,
            )

    scores, record = asyncio.run(call())
    assert scores == [2, 4]
    assert len(record["attempts"]) == 2
    assert "parsed 1/2" in record["attempts"][0]["error"]


def test_rejects_unknown_judge_api(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="judge_api"):
        _make_evaluator(_make_task_dir(tmp_path), judge_api="completions")


def test_rejects_non_boolean_tool_output_option(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="judge_include_tool_outputs"):
        _make_evaluator(
            _make_task_dir(tmp_path),
            judge_include_tool_outputs="true",
        )


def test_rejects_negative_trace_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="judge_max_traces_per_call"):
        _make_evaluator(
            _make_task_dir(tmp_path),
            judge_max_traces_per_call=-1,
        )


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
