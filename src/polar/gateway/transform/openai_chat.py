"""OpenAI Chat Completions transformer with SGLang training enhancements."""

from __future__ import annotations

from typing import Any

from polar.gateway.transform.base import BaseTransformer


def _uses_openai_reasoning_token_limit(model_name: object) -> bool:
    """Return whether the served OpenAI model rejects legacy ``max_tokens``."""

    if not isinstance(model_name, str):
        return False
    model_id = model_name.rsplit("/", 1)[-1].lower()
    if model_id == "gpt-5" or model_id.startswith(("gpt-5.", "gpt-5-")):
        return True
    return any(
        model_id == family or model_id.startswith(f"{family}-") for family in ("o1", "o3", "o4")
    )


class OpenAIChatTransformer(BaseTransformer):
    """Transform OpenAI Chat requests (passthrough + training params)."""

    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        result = body.copy()
        served_model = body.get("_polar_model_served")
        if _uses_openai_reasoning_token_limit(served_model):
            # NVIDIA's OpenAI-reasoning compatibility contract follows the
            # GPT-5 API: max_completion_tokens replaces max_tokens, while
            # temperature and stop are rejected.  An explicit modern field
            # wins if a recursive client config supplied both.
            if "max_completion_tokens" not in result and result.get("max_tokens") is not None:
                result["max_completion_tokens"] = result["max_tokens"]
            result.pop("max_tokens", None)
            result.pop("temperature", None)
            result.pop("stop", None)
        elif "max_tokens" not in result and "max_completion_tokens" in result:
            result["max_tokens"] = result["max_completion_tokens"]
        return self._normalize_request(
            result,
            served_model if isinstance(served_model, str) else None,
        )

    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        result = response.copy()
        if "model" in original_request:
            result["model"] = original_request["model"]
        return result

    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> dict[str, Any]:
        result = chunk.copy()
        if "model" in original_request:
            result["model"] = original_request["model"]
        return result
