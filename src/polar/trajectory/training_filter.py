"""Training-safety filters for reconstructed trajectory traces.

Tool-capable model servers normally return parsed tool calls in the assistant
message's ``tool_calls`` field.  A parser failure can instead leave the raw
tool protocol in ``content`` while returning no structured call.  Such a trace
must not inherit a positive terminal reward: doing so teaches the malformed
protocol that caused the retry.  When sampled token IDs, log-probabilities,
and the loss mask remain aligned, however, it is a genuine policy failure and
should stay trainable with zero reward.  Only an unaligned trace is
diagnostic-only.

The helpers in this module deliberately use strong signals only.  Ordinary
content-only final answers remain trainable.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from polar.trajectory.models import Trace


_TOOL_CALL_FINISH_REASONS = frozenset({"tool_calls", "tool_use", "function_call"})
_EXPLICIT_ERROR_KEYS = (
    "tool_call_format_error",
    "tool_call_parse_error",
    "tool_parser_error",
    "invalid_tool_call",
)
_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_LITERAL_TOOL_CALL_RE = re.compile(
    r"(?:^|\n)\s*</?tool_call>\s*(?:\n|$)|"
    r"(?:^|\n)\s*<function\s*=.+?>\s*(?:\n|$)",
    re.IGNORECASE,
)


def parser_invalid_tool_call_reason(trace: Trace) -> str | None:
    """Return a stable reason when ``trace`` contains an unparsed tool call.

    Detection prefers explicit structured error metadata.  The fallback
    signals are intentionally narrow:

    * the server says the completion ended with a tool call, but the final
      assistant message contains no structured call; or
    * an assistant message contains a raw tool-protocol marker on its own line
      but that same message has no structured call.

    Inline discussion of ``<tool_call>`` and fenced examples are not treated as
    tool attempts, so normal content-only answers are unaffected.
    """

    existing_filter = trace.metadata.get("training_filter")
    if (
        isinstance(existing_filter, dict)
        and existing_filter.get("reason") == "parser_invalid_tool_call"
    ):
        return str(existing_filter.get("detail") or "annotated_parser_invalid_tool_call")

    if _has_explicit_tool_call_error(trace.metadata):
        return "explicit_tool_call_parse_error"

    assistant_messages = [
        message
        for message in trace.response_messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]

    for message in assistant_messages:
        if _has_explicit_tool_call_error(message) or _has_explicit_tool_call_error(
            message.get("metadata")
        ):
            return "explicit_tool_call_parse_error"

    if trace.finish_reason in _TOOL_CALL_FINISH_REASONS and assistant_messages:
        if not _has_structured_tool_calls(assistant_messages[-1]):
            return "tool_call_finish_without_structured_call"

    for message in assistant_messages:
        if _has_structured_tool_calls(message):
            continue
        content = message.get("content")
        if isinstance(content, str) and _contains_literal_tool_call(content):
            return "literal_tool_call_without_structured_call"

    return None


def zero_reward_parser_invalid_tool_call_trace(
    trace: Trace,
) -> tuple[Trace, str | None]:
    """Zero a malformed policy action without discarding its aligned gradient.

    A malformed tool call is an action sampled by the policy, not an
    infrastructure outage. Preserving its source loss mask lets centered group
    advantages push its probability down when sibling rollouts solve.
    """

    detail = parser_invalid_tool_call_reason(trace)
    if detail is None:
        return trace, None

    metadata = deepcopy(trace.metadata)
    current_filter = metadata.get("training_filter")
    training_filter = dict(current_filter) if isinstance(current_filter, dict) else {}
    training_filter.update(
        {
            "masked": False,
            "trainable": True,
            "reason": "parser_invalid_tool_call",
            "detail": detail,
        }
    )
    training_filter.setdefault("original_reward", trace.reward)
    metadata["training_filter"] = training_filter

    return trace.model_copy(
        update={
            "reward": 0.0,
            "metadata": metadata,
        }
    ), detail


def mask_parser_invalid_tool_call_trace(trace: Trace) -> tuple[Trace, str | None]:
    """Compatibility alias; aligned parser failures are no longer masked."""

    return zero_reward_parser_invalid_tool_call_trace(trace)


def _has_structured_tool_calls(message: dict[str, Any]) -> bool:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        return any(isinstance(tool_call, dict) for tool_call in tool_calls)
    return isinstance(tool_calls, dict) and bool(tool_calls)


def _contains_literal_tool_call(content: str) -> bool:
    without_fenced_examples = _FENCED_CODE_BLOCK_RE.sub("", content)
    return _LITERAL_TOOL_CALL_RE.search(without_fenced_examples) is not None


def _has_explicit_tool_call_error(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False
    containers = [metadata]
    for key in ("tool_call_validation", "parser", "parse_result"):
        nested = metadata.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    return any(
        _is_error_value(container.get(key))
        for container in containers
        for key in _EXPLICIT_ERROR_KEYS
        if key in container
    )


def _is_error_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "none", "null", "ok", "valid"}
    return bool(value)
