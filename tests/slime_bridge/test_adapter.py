from __future__ import annotations

from enum import Enum

import pytest

from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
from polar.trajectory.models import Trace, Trajectory
from slime_bridge import adapter
from slime_bridge.adapter import RolloutLogprobError, session_result_to_samples


class FakeSample:
    class Status(str, Enum):
        COMPLETED = "completed"
        ABORTED = "aborted"
        FAILED = "failed"
        TRUNCATED = "truncated"

    def __init__(
        self,
        *,
        group_index: int,
        index: int,
        rollout_id: int,
        prompt,
        tokens: list[int],
        response: str,
        response_length: int,
        reward,
        loss_mask: list[int],
        rollout_log_probs: list[float],
        status,
        session_id: str,
        metadata: dict,
        remove_sample: bool = False,
    ) -> None:
        self.group_index = group_index
        self.index = index
        self.rollout_id = rollout_id
        self.prompt = prompt
        self.tokens = tokens
        self.response = response
        self.response_length = response_length
        self.reward = reward
        self.loss_mask = loss_mask
        self.rollout_log_probs = rollout_log_probs
        self.status = status
        self.session_id = session_id
        self.metadata = metadata
        self.remove_sample = remove_sample


def _session_result(
    *,
    trace: Trace | None = None,
    traces: list[Trace] | None = None,
    status: SessionStatus = SessionStatus.COMPLETED,
) -> SessionResult:
    if traces is None:
        assert trace is not None
        traces = [trace]
    return SessionResult(
        session_id="session-1",
        task_id="task-1",
        status=status,
        node_id="node-a",
        timing=SessionTiming(init_ms=1.0, run_ms=2.0, postrun_ms=3.0),
        metadata={"policy_version": 5},
        trajectory=Trajectory(
            status="COMPLETED" if status == SessionStatus.COMPLETED else "TIMEOUT",
            metadata={"rollout_step": 7},
            traces=traces,
        ),
    )


def test_session_result_to_samples_converts_trace_to_slime_like_sample(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        loss_mask=[1, 0],
        prompt_messages=[{"role": "user", "content": "Say hi"}],
        response_messages=[{"role": "assistant", "content": "Hi"}],
        response_logprobs=[-0.1, -0.2],
        reward=1.0,
        metadata={"group_id": "group-1"},
    )

    samples = session_result_to_samples(
        _session_result(trace=trace),
        group_index=11,
        trajectory_index=2,
        reward_key="score",
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.group_index == 11
    assert sample.index == 2
    assert sample.rollout_id == 2
    assert sample.prompt == [{"role": "user", "content": "Say hi"}]
    assert sample.tokens == [1, 2, 3, 4]
    assert sample.response == "[assistant] Hi"
    assert sample.response_length == 2
    assert sample.reward == {"score": 1.0}
    assert sample.loss_mask == [1, 0]
    assert sample.rollout_log_probs == [-0.1, -0.2]
    assert sample.status == FakeSample.Status.COMPLETED
    assert sample.metadata["polar"]["group_id"] == "group-1"
    assert sample.metadata["polar"]["policy_version"] == 5
    assert sample.metadata["polar"]["rollout_step"] == 7


def test_session_result_to_samples_shares_rollout_id_across_trace_siblings(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    traces = [
        Trace(
            prompt_ids=[1],
            response_ids=[2],
            loss_mask=[1],
            response_logprobs=[-0.1],
            reward=1.0,
        ),
        Trace(
            prompt_ids=[3],
            response_ids=[4],
            loss_mask=[1],
            response_logprobs=[-0.2],
            reward=2.0,
        ),
    ]

    samples = session_result_to_samples(
        _session_result(traces=traces),
        group_index=11,
        trajectory_index=7,
    )

    assert len(samples) == 2
    assert [sample.index for sample in samples] == [7, 7]
    assert [sample.rollout_id for sample in samples] == [7, 7]
    assert [sample.metadata["polar"]["trace_index"] for sample in samples] == [0, 1]


def test_session_result_to_samples_emits_placeholder_when_trace_is_unusable(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[],
        response_ids=[],
        loss_mask=[],
    )

    samples = session_result_to_samples(
        _session_result(trace=trace),
        group_index=1,
        trajectory_index=2,
    )

    assert len(samples) == 1
    assert samples[0].remove_sample is True
    assert samples[0].rollout_id == 2
    assert samples[0].loss_mask == [0]
    assert samples[0].metadata["polar"]["placeholder"] is True


def test_session_result_to_samples_clips_overlong_prompt_tokens(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1, 2, 3, 4, 5],
        response_ids=[6, 7],
        loss_mask=[1, 1],
        response_logprobs=[-0.1, -0.2],
        reward=1.0,
    )

    samples = session_result_to_samples(
        _session_result(trace=trace),
        group_index=1,
        trajectory_index=2,
        max_tokens=4,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [4, 5, 6, 7]
    assert sample.response_length == 2
    assert sample.loss_mask == [1, 1]
    assert sample.rollout_log_probs == [-0.1, -0.2]
    assert sample.remove_sample is False
    assert sample.metadata["polar"]["token_clipping"] == {
        "original_prompt_tokens": 5,
        "clipped_prompt_tokens": 3,
        "kept_prompt_tokens": 2,
        "response_tokens": 2,
        "max_tokens": 4,
    }


def test_session_result_to_samples_drops_trace_when_response_exceeds_cap(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2, 3, 4],
        loss_mask=[1, 1, 1],
        response_logprobs=[-0.1, -0.2, -0.3],
        reward=1.0,
    )

    samples = session_result_to_samples(
        _session_result(trace=trace),
        group_index=1,
        trajectory_index=2,
        max_tokens=2,
    )

    assert len(samples) == 1
    assert samples[0].remove_sample is True
    assert samples[0].metadata["polar"]["placeholder"] is True


def test_session_result_to_samples_requires_logprobs_for_trainable_tokens(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2],
        loss_mask=[1],
        response_logprobs=None,
    )

    with pytest.raises(RolloutLogprobError, match="missing rollout_log_probs"):
        session_result_to_samples(
            _session_result(trace=trace),
            group_index=1,
            trajectory_index=2,
        )
