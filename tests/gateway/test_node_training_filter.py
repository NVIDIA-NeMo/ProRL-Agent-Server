from __future__ import annotations

from polar.agent.models import AgentRunResult
from polar.gateway.node import GatewayNodeManager
from polar.trajectory.models import EvalResult, EvaluatorSpec, Trace, Trajectory


def _trace(*, content: str, tool_calls: list[dict] | None, finish_reason: str) -> Trace:
    return Trace(
        prompt_ids=[1],
        response_ids=[2, 3],
        loss_mask=[1, 1],
        response_logprobs=[-0.1, -0.2],
        response_messages=[
            {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls or [],
            }
        ],
        finish_reason=finish_reason,
    )


def test_terminal_outcome_zeroes_but_trains_parser_invalid_trace() -> None:
    invalid = _trace(
        content='<tool_call>\n{"function=bash>\n',
        tool_calls=[],
        finish_reason="tool_calls",
    )
    valid = _trace(
        content="Fixed the issue.",
        tool_calls=[],
        finish_reason="stop",
    )
    trajectory = Trajectory(
        status="COMPLETED",
        metadata={"builder": "prefix_merging"},
        traces=[invalid, valid],
    )

    merged = GatewayNodeManager._merge_eval_result(
        trajectory,
        EvalResult(outcome_reward=1.0),
        EvaluatorSpec(strategy="harbor"),
    )

    assert merged.traces[0].reward == 0.0
    assert merged.traces[0].loss_mask == [1, 1]
    assert merged.traces[0].metadata["training_filter"]["trainable"] is True
    assert merged.traces[0].metadata["training_filter"]["reason"] == "parser_invalid_tool_call"
    assert merged.traces[1].reward == 1.0
    assert merged.traces[1].loss_mask == [1, 1]
    assert "training_filter" not in merged.traces[1].metadata
    assert merged.metadata["evaluation"]["parser_invalid_traces_zero_rewarded"] == 1


def test_terminal_outcome_preserves_normal_content_only_final_answer() -> None:
    final_answer = _trace(
        content="The patch is complete and tests pass.",
        tool_calls=[],
        finish_reason="stop",
    )
    trajectory = Trajectory(status="COMPLETED", traces=[final_answer])

    merged = GatewayNodeManager._merge_eval_result(
        trajectory,
        EvalResult(outcome_reward=1.0),
        EvaluatorSpec(strategy="harbor"),
    )

    assert merged.traces[0].reward == 1.0
    assert merged.traces[0].loss_mask == [1, 1]
    assert "parser_invalid_traces_zero_rewarded" not in merged.metadata["evaluation"]


def test_error_execution_discards_positive_verifier_reward_and_masks_trace() -> None:
    trace = _trace(
        content="The task state looks fixed, but the agent process was terminated.",
        tool_calls=[],
        finish_reason="stop",
    )
    trajectory = Trajectory(
        status="ERROR",
        error="step 0 exited with code -15",
        traces=[trace],
    )

    merged = GatewayNodeManager._merge_eval_result(
        trajectory,
        EvalResult(
            outcome_reward=1.0,
            metadata={
                "resolved": True,
                "reward": 1.0,
                "verifier_reported_reward": 1.0,
                "verifier_reward_accepted": True,
                "verifier_exit_code": 0,
            },
        ),
        EvaluatorSpec(strategy="harbor"),
    )

    assert merged.status == "ERROR"
    assert merged.error == "step 0 exited with code -15"
    assert merged.traces[0].reward == 0.0
    assert merged.traces[0].loss_mask == [0, 0]
    assert merged.traces[0].metadata["training_filter"] == {
        "masked": True,
        "reason": "session_error",
        "detail": "step 0 exited with code -15",
        "original_reward": 1.0,
    }

    evaluation = merged.metadata["evaluation"]
    assert evaluation["outcome_reward"] == 0.0
    assert evaluation["discarded_outcome_reward"] == 1.0
    assert evaluation["reward"] == 0.0
    assert evaluation["discarded_reward"] == 1.0
    assert evaluation["resolved"] is False
    assert evaluation["verifier_reported_reward"] == 1.0
    assert evaluation["reward_discarded"] is True
    assert evaluation["reward_discard_reason"] == "session_error"


def test_agent_budget_timeout_is_zero_reward_trainable_negative() -> None:
    trajectory = Trajectory(
        status="TIMEOUT",
        error="agent execution timeout",
        metadata={
            "agent_result": {
                "status": "timeout",
                "return_code": -1,
                "timeout_source": "agent",
                "timeout_stage": "exec",
            }
        },
        traces=[
            _trace(
                content="The model kept working until its agent budget expired.",
                tool_calls=[],
                finish_reason="tool_calls",
            )
        ],
    )

    merged = GatewayNodeManager._merge_eval_result(
        trajectory,
        EvalResult(
            outcome_reward=1.0,
            metadata={
                "resolved": True,
                "reward": 1.0,
                "verifier_reported_reward": 1.0,
                "verifier_reward_accepted": True,
                "verifier_timeout": False,
            },
        ),
        EvaluatorSpec(strategy="harbor"),
    )

    trace = merged.traces[0]
    assert trace.reward == 0.0
    assert trace.loss_mask == [1, 1]
    assert trace.response_logprobs == [-0.1, -0.2]
    assert trace.metadata["training_filter"] == {
        "masked": False,
        "trainable": True,
        "reason": "agent_timeout",
        "detail": "agent execution timeout",
        "original_reward": 1.0,
    }
    assert merged.metadata["evaluation"]["discarded_outcome_reward"] == 1.0
    assert merged.metadata["evaluation"]["outcome_reward"] == 0.0
    assert merged.metadata["evaluation"]["reward_discard_reason"] == "agent_timeout"


def test_session_or_evaluator_timeout_remains_fully_masked() -> None:
    for timeout_source, timeout_stage, verifier_timeout in (
        ("session", "exec", False),
        ("agent", "exec", True),
        ("agent", "postprocess", False),
    ):
        trajectory = Trajectory(
            status="TIMEOUT",
            error="session execution timeout",
            metadata={
                "agent_result": {
                    "status": "timeout",
                    "return_code": -1,
                    "timeout_source": timeout_source,
                    "timeout_stage": timeout_stage,
                }
            },
            traces=[
                _trace(
                    content="Partial work",
                    tool_calls=[],
                    finish_reason="tool_calls",
                )
            ],
        )

        merged = GatewayNodeManager._merge_eval_result(
            trajectory,
            EvalResult(
                outcome_reward=1.0,
                metadata={"verifier_timeout": verifier_timeout},
            ),
            EvaluatorSpec(strategy="harbor"),
        )

        assert merged.traces[0].reward == 0.0
        assert merged.traces[0].loss_mask == [0, 0]
        assert merged.traces[0].metadata["training_filter"]["masked"] is True
        assert merged.traces[0].metadata["training_filter"]["reason"] == "session_timeout"


def test_agent_timeout_without_aligned_logprobs_is_not_trainable() -> None:
    trace = _trace(
        content="Partial work",
        tool_calls=[],
        finish_reason="tool_calls",
    ).model_copy(update={"response_logprobs": None})
    trajectory = Trajectory(
        status="TIMEOUT",
        error="agent execution timeout",
        metadata={
            "agent_result": {
                "status": "timeout",
                "return_code": -1,
                "timeout_source": "agent",
                "timeout_stage": "exec",
            }
        },
        traces=[trace],
    )

    merged = GatewayNodeManager._merge_eval_result(
        trajectory,
        EvalResult(outcome_reward=1.0),
        EvaluatorSpec(strategy="harbor"),
    )

    assert merged.traces[0].reward == 0.0
    assert merged.traces[0].loss_mask == [0, 0]
    assert merged.traces[0].metadata["training_filter"] == {
        "masked": True,
        "trainable": False,
        "reason": "agent_timeout_unaligned",
        "detail": "agent execution timeout",
        "original_reward": 1.0,
    }


def test_gateway_persists_agent_timeout_source_on_trajectory() -> None:
    marked = GatewayNodeManager._attach_agent_result_metadata(
        Trajectory(status="COMPLETED"),
        AgentRunResult(
            status="timeout",
            return_code=-1,
            error="agent execution timeout",
            metadata={"timeout_source": "agent", "timeout_stage": "exec"},
        ),
    )

    assert marked.metadata["agent_result"] == {
        "status": "timeout",
        "return_code": -1,
        "error": "agent execution timeout",
        "timeout_source": "agent",
        "timeout_stage": "exec",
    }
