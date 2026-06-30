from __future__ import annotations

from polar.trajectory.models import Trace
from polar.trajectory.training_filter import (
    mask_parser_invalid_tool_call_trace,
    parser_invalid_tool_call_reason,
    zero_reward_parser_invalid_tool_call_trace,
)


def _trace(
    message: dict,
    *,
    finish_reason: str = "stop",
    metadata: dict | None = None,
) -> Trace:
    return Trace(
        prompt_ids=[1],
        response_ids=[2, 3],
        loss_mask=[1, 1],
        response_logprobs=[-0.1, -0.2],
        response_messages=[message],
        finish_reason=finish_reason,
        reward=1.0,
        metadata=metadata or {},
    )


def test_tool_call_finish_without_structured_call_is_parser_invalid() -> None:
    trace = _trace(
        {"role": "assistant", "content": "trying a tool", "tool_calls": []},
        finish_reason="tool_calls",
    )

    assert parser_invalid_tool_call_reason(trace) == "tool_call_finish_without_structured_call"


def test_literal_tool_protocol_without_structured_call_is_parser_invalid() -> None:
    trace = _trace(
        {
            "role": "assistant",
            "content": 'Checking files.\n\n<tool_call>\n{"function=bash>\n',
            "tool_calls": [],
        }
    )

    assert parser_invalid_tool_call_reason(trace) == "literal_tool_call_without_structured_call"


def test_content_only_answer_and_tool_call_examples_remain_trainable() -> None:
    inline = _trace(
        {
            "role": "assistant",
            "content": "The `<tool_call>` tag in your example is malformed.",
            "tool_calls": [],
        }
    )
    fenced = _trace(
        {
            "role": "assistant",
            "content": "Example:\n```xml\n<tool_call>\n</tool_call>\n```",
            "tool_calls": [],
        }
    )

    assert parser_invalid_tool_call_reason(inline) is None
    assert parser_invalid_tool_call_reason(fenced) is None


def test_structured_tool_call_is_not_filtered() -> None:
    trace = _trace(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                }
            ],
        },
        finish_reason="tool_calls",
    )

    assert parser_invalid_tool_call_reason(trace) is None


def test_parser_invalid_trace_gets_zero_reward_but_keeps_aligned_gradient() -> None:
    trace = _trace(
        {"role": "assistant", "content": "", "tool_calls": []},
        finish_reason="tool_calls",
    )

    filtered, detail = zero_reward_parser_invalid_tool_call_trace(trace)

    assert detail == "tool_call_finish_without_structured_call"
    assert filtered.reward == 0.0
    assert filtered.loss_mask == [1, 1]
    assert filtered.metadata["training_filter"] == {
        "masked": False,
        "trainable": True,
        "reason": "parser_invalid_tool_call",
        "detail": "tool_call_finish_without_structured_call",
        "original_reward": 1.0,
    }
    assert trace.reward == 1.0
    assert trace.loss_mask == [1, 1]
    assert trace.metadata == {}

    compat, _ = mask_parser_invalid_tool_call_trace(trace)
    assert compat.loss_mask == [1, 1]


def test_explicit_tool_parser_error_metadata_is_honored() -> None:
    trace = _trace(
        {"role": "assistant", "content": "ordinary content", "tool_calls": []},
        metadata={"tool_call_validation": {"tool_call_parse_error": "invalid XML"}},
    )

    assert parser_invalid_tool_call_reason(trace) == "explicit_tool_call_parse_error"
