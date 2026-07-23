from __future__ import annotations

from enum import Enum

import pytest

from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
from polar.trajectory.models import Trace, Trajectory
from slime_bridge import adapter
from slime_bridge.adapter import (
    RolloutLogprobError,
    session_result_to_placeholder,
    session_result_to_samples,
)


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
    error: str | None = None,
    trajectory_metadata: dict | None = None,
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
            status=str(status),
            metadata={"rollout_step": 7, **(trajectory_metadata or {})},
            traces=traces,
            error=error,
        ),
        error=error,
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


def test_adapter_preserves_named_reward_components(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2],
        loss_mask=[1],
        response_logprobs=[-0.1],
        reward=0.75,
        reward_components={"reward_1": 1.0, "reward_2": 0.5},
    )

    [sample] = session_result_to_samples(
        _session_result(trace=trace),
        group_index=0,
        trajectory_index=0,
        reward_key="score",
    )

    assert sample.reward == {
        "score": 0.75,
        "reward_1": 1.0,
        "reward_2": 0.5,
    }


def test_failed_execution_zeros_named_reward_components(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2],
        loss_mask=[1],
        response_logprobs=[-0.1],
        reward=1.0,
        reward_components={"reward_1": 1.0, "reward_2": 0.5},
    )

    [sample] = session_result_to_samples(
        _session_result(
            trace=trace,
            status=SessionStatus.ERROR,
            error="synthetic failure",
        ),
        group_index=0,
        trajectory_index=0,
        reward_key="score",
    )

    assert sample.reward == {
        "score": 0.0,
        "reward_1": 0.0,
        "reward_2": 0.0,
    }


@pytest.mark.parametrize(
    "malformed_reward",
    [float("nan"), float("inf"), float("-inf"), True, False],
)
def test_adapter_fail_closes_nonfinite_or_boolean_trace_reward(
    monkeypatch,
    malformed_reward,
) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    # Bypass the current schema to model replay of a legacy artifact written
    # before reward validation existed; the adapter remains a second boundary.
    trace = Trace.model_construct(
        prompt_ids=[1],
        response_ids=[2],
        loss_mask=[1],
        prompt_messages=[],
        response_messages=[],
        response_logprobs=[-0.1],
        reward=malformed_reward,
        metadata={},
    )

    sample = session_result_to_samples(
        _session_result(trace=trace),
        group_index=11,
        trajectory_index=2,
        reward_key="score",
    )[0]

    assert sample.reward == {"score": 0.0}


@pytest.mark.parametrize(
    ("session_status", "sample_status", "filter_reason", "error"),
    [
        (
            SessionStatus.ERROR,
            FakeSample.Status.FAILED,
            "session_error",
            "step 0 exited with code -15",
        ),
        (
            SessionStatus.TIMEOUT,
            FakeSample.Status.ABORTED,
            "session_timeout",
            "step 0 timed out",
        ),
    ],
)
def test_failed_execution_cannot_keep_positive_trace_reward(
    monkeypatch,
    session_status,
    sample_status,
    filter_reason,
    error,
) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2],
        loss_mask=[1],
        response_logprobs=[-0.1],
        reward=1.0,
    )

    sample = session_result_to_samples(
        _session_result(trace=trace, status=session_status, error=error),
        group_index=3,
        trajectory_index=4,
        reward_key="score",
    )[0]

    assert sample.status == sample_status
    assert sample.reward == {"score": 0.0}
    assert sample.loss_mask == [0]
    assert sample.remove_sample is True
    assert sample.metadata["polar"]["training_filter"] == {
        "masked": True,
        "reason": filter_reason,
        "detail": error,
        "original_reward": 1.0,
    }


def test_agent_timeout_maps_to_trainable_truncated_negative(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2, 3],
        loss_mask=[1, 0],
        response_logprobs=[-0.1, 0.0],
        # Simulate a stale artifact that still carries the verifier reward;
        # the adapter must fail-close the effective scalar to zero.
        reward=1.0,
        metadata={
            "training_filter": {
                "masked": False,
                "trainable": True,
                "reason": "agent_timeout",
                "detail": "agent execution timeout",
                "original_reward": 1.0,
            }
        },
    )
    result = _session_result(
        trace=trace,
        status=SessionStatus.TIMEOUT,
        error="agent execution timeout",
        trajectory_metadata={
            "agent_result": {
                "status": "timeout",
                "return_code": -1,
                "timeout_source": "agent",
                "timeout_stage": "exec",
            },
            "evaluation": {"verifier_timeout": False},
        },
    )

    sample = session_result_to_samples(
        result,
        group_index=3,
        trajectory_index=4,
        reward_key="score",
    )[0]

    assert sample.status == FakeSample.Status.TRUNCATED
    assert sample.reward == {"score": 0.0}
    assert sample.loss_mask == [1, 0]
    assert sample.rollout_log_probs == [-0.1, 0.0]
    assert sample.remove_sample is False
    assert sample.metadata["polar"]["training_filter"] == {
        "masked": False,
        "trainable": True,
        "reason": "agent_timeout",
        "detail": "agent execution timeout",
        "original_reward": 1.0,
    }


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


def test_session_result_to_samples_trains_aligned_parser_invalid_trace_with_zero_reward(
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    traces = [
        Trace(
            prompt_ids=[1],
            response_ids=[2, 3],
            loss_mask=[1, 1],
            response_logprobs=[-0.1, -0.2],
            response_messages=[
                {
                    "role": "assistant",
                    "content": '<tool_call>\n{"function=bash>\n',
                    "tool_calls": [],
                }
            ],
            finish_reason="tool_calls",
            reward=1.0,
        ),
        Trace(
            prompt_ids=[4],
            response_ids=[5],
            loss_mask=[1],
            response_logprobs=[-0.2],
            response_messages=[{"role": "assistant", "content": "Done", "tool_calls": []}],
            finish_reason="stop",
            reward=1.0,
        ),
    ]

    samples = session_result_to_samples(
        _session_result(traces=traces),
        group_index=11,
        trajectory_index=7,
    )

    assert len(samples) == 2
    invalid, valid = samples
    assert invalid.rollout_id == valid.rollout_id == 7
    assert invalid.reward == {"score": 0.0}
    assert invalid.loss_mask == [1, 1]
    assert invalid.rollout_log_probs == [-0.1, -0.2]
    assert invalid.remove_sample is False
    assert invalid.metadata["polar"]["training_filter"]["trainable"] is True
    assert invalid.metadata["polar"]["training_filter"]["reason"] == "parser_invalid_tool_call"
    assert valid.reward == {"score": 1.0}
    assert valid.loss_mask == [1]
    assert valid.remove_sample is False


def test_session_result_to_samples_drops_unaligned_parser_invalid_trace(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2, 3],
        loss_mask=[1, 1],
        response_logprobs=None,
        response_messages=[
            {
                "role": "assistant",
                "content": '<tool_call>\n{"function=bash>\n',
                "tool_calls": [],
            }
        ],
        finish_reason="tool_calls",
        reward=1.0,
    )

    sample = session_result_to_samples(
        _session_result(trace=trace),
        group_index=11,
        trajectory_index=7,
    )[0]

    assert sample.remove_sample is True
    assert sample.loss_mask == [0]
    assert sample.metadata["polar"]["placeholder"] is True


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


def test_explicit_conversion_placeholder_records_error(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(prompt_ids=[1], response_ids=[2], loss_mask=[1])

    sample = session_result_to_placeholder(
        _session_result(trace=trace),
        group_index=1,
        trajectory_index=2,
        conversion_error="ValueError: broken alignment",
    )

    assert sample.remove_sample is True
    assert sample.reward == {"score": 0.0}
    assert sample.metadata["polar"]["sample_conversion_error"] == ("ValueError: broken alignment")
    assert sample.metadata["polar"]["training_filter"]["reason"] == ("sample_conversion_error")


def test_session_result_to_samples_keeps_exact_causal_prefix(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5, 6, 7],
        loss_mask=[1, 0, 1, 1, 1],
        response_logprobs=[-0.1, 0.0, -0.2, -0.3, -0.4],
        reward=1.0,
    )

    samples = session_result_to_samples(
        _session_result(trace=trace),
        group_index=1,
        trajectory_index=2,
        max_tokens=5,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [1, 2, 3, 4, 5]
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 0, 1]
    assert sample.rollout_log_probs == [-0.1, 0.0, -0.2]
    assert sample.status == FakeSample.Status.TRUNCATED
    assert sample.remove_sample is False
    assert sample.metadata["polar"]["token_clipping"] == {
        "strategy": "causal_prefix",
        "original_total_tokens": 7,
        "original_prompt_tokens": 2,
        "original_response_tokens": 5,
        "kept_prompt_tokens": 2,
        "kept_response_tokens": 3,
        "dropped_response_tokens": 2,
        "max_tokens": 5,
    }


def test_session_result_to_samples_drops_trace_when_prompt_fills_cap(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        loss_mask=[1, 1],
        response_logprobs=[-0.1, -0.2],
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


def test_session_result_to_samples_drops_prefix_with_no_trainable_tokens(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2, 3, 4, 5],
        loss_mask=[0, 0, 1, 1],
        response_logprobs=[0.0, 0.0, -0.1, -0.2],
        reward=1.0,
    )

    samples = session_result_to_samples(
        _session_result(trace=trace),
        group_index=1,
        trajectory_index=2,
        max_tokens=3,
    )

    assert len(samples) == 1
    assert samples[0].remove_sample is True
    assert samples[0].metadata["polar"]["placeholder"] is True


def test_session_result_to_samples_does_not_mutate_trace_when_clipping(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2, 3, 4],
        loss_mask=[1, 1, 1],
        response_logprobs=[-0.1, -0.2, -0.3],
        reward=1.0,
    )

    session_result_to_samples(
        _session_result(trace=trace),
        group_index=1,
        trajectory_index=2,
        max_tokens=2,
    )

    assert trace.response_ids == [2, 3, 4]
    assert trace.loss_mask == [1, 1, 1]
    assert trace.response_logprobs == [-0.1, -0.2, -0.3]


def test_observed_scale_long_trace_stays_trainable_under_20k_cap(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    prompt_len = 1_500
    response_len = 48_000
    trace = Trace(
        prompt_ids=list(range(prompt_len)),
        response_ids=list(range(prompt_len, prompt_len + response_len)),
        loss_mask=[1] * response_len,
        response_logprobs=[-0.1] * response_len,
        reward=1.0,
    )

    samples = session_result_to_samples(
        _session_result(trace=trace),
        group_index=1,
        trajectory_index=2,
        max_tokens=20_000,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.remove_sample is False
    assert sample.status == FakeSample.Status.TRUNCATED
    assert len(sample.tokens) == 20_000
    assert sample.response_length == 18_500
    assert sum(sample.loss_mask) == 18_500
    assert len(sample.rollout_log_probs) == 18_500
    assert sample.reward == {"score": 1.0}


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


@pytest.mark.parametrize("nonfinite_logprob", [float("nan"), float("inf"), float("-inf")])
def test_session_result_to_samples_rejects_nonfinite_logprobs(
    monkeypatch,
    nonfinite_logprob: float,
) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2, 3],
        loss_mask=[1, 1],
        response_logprobs=[-0.1, nonfinite_logprob],
    )

    with pytest.raises(RolloutLogprobError, match="non-finite rollout_log_probs"):
        session_result_to_samples(
            _session_result(trace=trace),
            group_index=1,
            trajectory_index=2,
        )
