from __future__ import annotations

from typing import Any

from openhands.sdk.llm.options.common import apply_defaults_if_absent


def select_chat_options(
    llm, user_kwargs: dict[str, Any], has_tools: bool
) -> dict[str, Any]:
    """Normalize kwargs for Chat Completion API.

    LiteLLM handles model-specific parameter filtering via drop_params=True,
    so we pass configured options and let LiteLLM strip unsupported ones.
    """
    # First pass: apply simple defaults without touching user-supplied values
    defaults: dict[str, Any] = {
        "top_k": llm.top_k,
        "top_p": llm.top_p,
        "temperature": llm.temperature,
        # OpenAI-compatible param is `max_completion_tokens`
        "max_completion_tokens": llm.max_output_tokens,
    }
    out = apply_defaults_if_absent(user_kwargs, defaults)

    # Azure -> uses max_tokens instead
    if llm.model.startswith("azure"):
        if "max_completion_tokens" in out:
            out["max_tokens"] = out.pop("max_completion_tokens")

    # If user didn't set extra_headers, propagate from llm config
    if llm.extra_headers is not None and "extra_headers" not in out:
        out["extra_headers"] = dict(llm.extra_headers)

    # Reasoning effort - pass if configured, LiteLLM handles unsupported via drop_params
    if llm.reasoning_effort is not None:
        out["reasoning_effort"] = llm.reasoning_effort
        # Reasoning models typically ignore temp/top_p, so remove them when
        # reasoning_effort is configured to avoid confusing the model
        out.pop("temperature", None)
        out.pop("top_p", None)

    # Extended thinking - pass thinking config if budget is set and max_output_tokens is known
    if llm.extended_thinking_budget and llm.max_output_tokens is not None:
        # Anthropic throws errors if thinking budget equals or exceeds max output
        # tokens -- force the thinking budget lower if there's a conflict
        budget_tokens = min(llm.extended_thinking_budget, llm.max_output_tokens - 1)
        out["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }
        # Enable interleaved thinking
        # Merge default header with any user-provided headers; user wins on conflict
        existing = out.get("extra_headers") or {}
        out["extra_headers"] = {
            "anthropic-beta": "interleaved-thinking-2025-05-14",
            **existing,
        }
        # Fix litellm behavior
        out["max_tokens"] = llm.max_output_tokens
        # Extended thinking models ignore temp/top_p
        out.pop("temperature", None)
        out.pop("top_p", None)

    # Safety settings - pass if configured, LiteLLM handles unsupported via drop_params
    if llm.safety_settings:
        out["safety_settings"] = llm.safety_settings

    # Tools: if not using native, strip tool_choice so we don't confuse providers
    if not has_tools:
        out.pop("tools", None)
        out.pop("tool_choice", None)

    # Prompt cache retention - pass if configured, LiteLLM drops for unsupported models
    if llm.prompt_cache_retention:
        out["prompt_cache_retention"] = llm.prompt_cache_retention

    # Pass through user-provided extra_body unchanged
    if llm.litellm_extra_body:
        out["extra_body"] = llm.litellm_extra_body

    return out
