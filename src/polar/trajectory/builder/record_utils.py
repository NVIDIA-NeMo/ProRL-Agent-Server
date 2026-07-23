"""Helpers for converting completion records into trajectory traces."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from polar.trajectory.models import CompletionRecord, Trace


def _coerce_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    extracted: list[int] = []
    for item in value:
        try:
            extracted.append(int(item))
        except (TypeError, ValueError):
            return None
    return extracted


def _paired_tokens_from_logprobs_content(
    choice: dict[str, Any],
) -> tuple[list[int], list[float]] | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None

    token_ids: list[int] = []
    token_logprobs: list[float] = []
    for item in content:
        if not isinstance(item, dict):
            return None
        token_id = item.get("token_id")
        logprob = item.get("logprob")
        if token_id is None or logprob is None:
            return None
        try:
            token_ids.append(int(token_id))
            token_logprobs.append(float(logprob))
        except (TypeError, ValueError):
            return None
    return token_ids, token_logprobs


def _paired_tokens_from_sglang_meta(
    choice: dict[str, Any],
) -> tuple[list[int], list[float]] | None:
    meta_info = choice.get("meta_info")
    if not isinstance(meta_info, dict):
        return None
    output_logprobs = meta_info.get("output_token_logprobs")
    if not isinstance(output_logprobs, list) or not output_logprobs:
        return None

    token_ids: list[int] = []
    token_logprobs: list[float] = []
    for item in output_logprobs:
        token_id = None
        logprob = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            logprob = item[0]
            token_id = item[1]
        elif isinstance(item, dict):
            token_id = item.get("token_id")
            logprob = item.get("logprob", item.get("token_logprob"))

        if token_id is None or logprob is None:
            return None
        try:
            token_ids.append(int(token_id))
            token_logprobs.append(float(logprob))
        except (TypeError, ValueError):
            return None
    return token_ids, token_logprobs


def _extract_response_tokens(
    response: dict[str, Any],
    choice: dict[str, Any],
    *,
    tokenizer: Any | None = None,
) -> tuple[list[int], list[float] | None]:
    """Extract response ids and logprobs without mixing unaligned sources."""
    token_ids = choice.get("token_ids", response.get("token_ids"))
    response_ids = _coerce_int_list(token_ids)
    paired_candidates = (
        _paired_tokens_from_logprobs_content(choice),
        _paired_tokens_from_sglang_meta(choice),
    )

    if response_ids is not None:
        for paired in paired_candidates:
            if paired is None:
                continue
            paired_ids, paired_logprobs = paired
            if paired_ids == response_ids:
                return response_ids, paired_logprobs
        return response_ids, None

    for paired in paired_candidates:
        if paired is not None:
            return paired

    reconstructed = _reconstruct_response_tokens(response, choice, tokenizer)
    if reconstructed is not None:
        return reconstructed

    return [], None


def _reconstruct_response_tokens(
    response: dict[str, Any],
    choice: dict[str, Any],
    tokenizer: Any | None,
) -> tuple[list[int], list[float]] | None:
    """Recover IDs from OpenAI token strings when an older SGLang omits IDs."""
    if tokenizer is None:
        return None
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list) or not content:
        return None

    pieces: list[str] = []
    token_logprobs: list[float] = []
    for item in content:
        if not isinstance(item, dict) or not isinstance(item.get("token"), str):
            return None
        try:
            token_logprobs.append(float(item["logprob"]))
        except (KeyError, TypeError, ValueError):
            return None
        pieces.append(item["token"])

    encoded = tokenizer.encode("".join(pieces), add_special_tokens=False)
    token_ids = _token_ids_from_encoding(encoded)
    if token_ids is None or len(token_ids) != len(token_logprobs):
        return None
    expected = _usage_token_count(response, "completion_tokens")
    if expected is not None and expected != len(token_ids):
        return None
    return token_ids, token_logprobs


def _reconstruct_prompt_tokens(
    request: dict[str, Any],
    response: dict[str, Any],
    tokenizer: Any | None,
) -> list[int]:
    if tokenizer is None:
        return []
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    kwargs = request.get("chat_template_kwargs")
    template_kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
    tools = request.get("tools")
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools if isinstance(tools, list) and tools else None,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
    except (TypeError, ValueError):
        return []
    token_ids = _token_ids_from_encoding(encoded)
    if token_ids is None:
        return []
    expected = _usage_token_count(response, "prompt_tokens")
    if expected is not None and expected != len(token_ids):
        return []
    return token_ids


def _token_ids_from_encoding(encoded: Any) -> list[int] | None:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    return _coerce_int_list(encoded)


def _usage_token_count(response: dict[str, Any], key: str) -> int | None:
    usage = response.get("usage")
    value = usage.get(key) if isinstance(usage, dict) else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_prompt_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    return [deepcopy(message) for message in messages if isinstance(message, dict)]


def _extract_tools(request: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    extracted = [deepcopy(tool) for tool in tools if isinstance(tool, dict)]
    return extracted or None


def build_trace_from_completion(
    completion: CompletionRecord,
    *,
    tokenizer: Any | None = None,
) -> Trace:
    """Normalize one stored completion record into a trajectory trace."""

    request = completion.request if isinstance(completion.request, dict) else {}
    response = completion.response if isinstance(completion.response, dict) else {}
    choices = response.get("choices")
    first_choice = (
        choices[0]
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else {}
    )
    prompt_ids = (
        first_choice.get("input_token_ids")
        or first_choice.get("prompt_token_ids")
        or response.get("prompt_token_ids")
    )
    if not isinstance(prompt_ids, list):
        prompt_ids = _reconstruct_prompt_tokens(request, response, tokenizer)
    response_message = first_choice.get("message")
    finish_reason = first_choice.get("finish_reason")

    response_ids, response_logprobs = _extract_response_tokens(
        response,
        first_choice,
        tokenizer=tokenizer,
    )

    return Trace(
        prompt_ids=list(prompt_ids) if isinstance(prompt_ids, list) else [],
        response_ids=response_ids,
        loss_mask=[1] * len(response_ids),
        prompt_messages=_extract_prompt_messages(request),
        response_messages=[deepcopy(response_message)] if isinstance(response_message, dict) else [],
        tools=_extract_tools(request),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        response_logprobs=response_logprobs,
        metadata=deepcopy(completion.metadata),
    )
