"""Portable mini-SWE model adapter for the Tmax Vanillux2 protocol.

This module is copied into the shared mini-SWE runtime as
``polar_mini_swe_vanillux``. Keep it independent of Polar imports because task
containers mount only that portable runtime.
"""

from __future__ import annotations

import os
import time
from typing import Any

import litellm
from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.litellm_model import LitellmModel


VANILLUX2_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell. Working directory "
            "and exported environment variables are preserved between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}

_TOKENIZE_MAX_ATTEMPTS = 5
_TOKENIZE_RETRY_BACKOFF_SECONDS = 0.1
_TOKENIZE_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_TOKENIZE_ERROR_DETAIL_LIMIT = 240
_MAX_TOKENS_FIELD = "max_tokens"
_MAX_COMPLETION_TOKENS_FIELD = "max_completion_tokens"


def _bounded_error_detail(detail: object) -> str:
    text = " ".join(str(detail).split()) or "no error detail"
    if len(text) <= _TOKENIZE_ERROR_DETAIL_LIMIT:
        return text
    return f"{text[: _TOKENIZE_ERROR_DETAIL_LIMIT - 3]}..."


def _http_error_detail(response: Any, status_code: int) -> str:
    try:
        response_text = response.text
    except Exception:
        response_text = ""
    detail = f"HTTP {status_code}"
    if response_text:
        detail = f"{detail}: {response_text}"
    return _bounded_error_detail(detail)


class Vanillux2LitellmModel(LitellmModel):
    """Use Vanillux2's persistent-bash schema and one-action contract."""

    def __init__(
        self,
        *,
        response_token_budget: int = 0,
        **kwargs: Any,
    ) -> None:
        if isinstance(response_token_budget, bool):
            raise ValueError("response_token_budget must be a non-negative integer")
        try:
            parsed_budget = int(response_token_budget)
        except (TypeError, ValueError) as exc:
            raise ValueError("response_token_budget must be a non-negative integer") from exc
        if parsed_budget < 0 or parsed_budget != response_token_budget:
            raise ValueError("response_token_budget must be a non-negative integer")

        super().__init__(**kwargs)
        # mini-SWE recursively merges the protocol YAML with per-candidate
        # model kwargs.  GPT-5-family candidates therefore inherit the YAML's
        # ``max_tokens`` unless it is removed here, leaving both OpenAI token
        # limit fields on one request.  Treat an explicit
        # ``max_completion_tokens`` as authoritative and use the same field
        # for the cumulative-budget clamp.
        if _MAX_COMPLETION_TOKENS_FIELD in self.config.model_kwargs:
            self.config.model_kwargs.pop(_MAX_TOKENS_FIELD, None)
            self.completion_token_limit_field = _MAX_COMPLETION_TOKENS_FIELD
        else:
            self.completion_token_limit_field = _MAX_TOKENS_FIELD
        self.response_token_budget = parsed_budget
        self.initial_prompt_tokens: int | None = None
        self.current_prompt_tokens: int | None = None
        self.used_response_tokens = 0

    def query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        """Clamp this turn to the remaining cumulative trajectory budget.

        The released Tmax protocol defines response length as growth from the
        first rendered prompt.  Counting only API ``completion_tokens`` misses
        tool observations and chat-template interstitials.  Before every model
        call we therefore ask the serving tokenizer to render the complete
        current messages/tools, subtract the first-turn prompt length, and
        bound this turn's ``max_tokens`` by the remainder.
        """

        if self.response_token_budget > 0:
            prepared = self._prepare_messages_for_api(messages)
            current = self._tokenize_prompt(prepared)
            if self.initial_prompt_tokens is None:
                self.initial_prompt_tokens = current
            self.current_prompt_tokens = current
            self.used_response_tokens = max(0, current - self.initial_prompt_tokens)
            remaining = self.response_token_budget - self.used_response_tokens
            if remaining <= 0:
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": "ResponseTokenBudgetExceeded",
                        "extra": {
                            "exit_status": "ResponseTokenBudgetExceeded",
                            "submission": "",
                            "response_token_budget": self.response_token_budget,
                            "used_response_tokens": self.used_response_tokens,
                        },
                    }
                )
            other_limit_field = (
                _MAX_TOKENS_FIELD
                if self.completion_token_limit_field == _MAX_COMPLETION_TOKENS_FIELD
                else _MAX_COMPLETION_TOKENS_FIELD
            )
            kwargs.pop(other_limit_field, None)
            kwargs[self.completion_token_limit_field] = min(
                remaining,
                self._requested_max_tokens(kwargs),
            )
        return super().query(messages, **kwargs)

    def _requested_max_tokens(self, query_kwargs: dict[str, Any]) -> int:
        raw_value = query_kwargs.get(
            self.completion_token_limit_field,
            self.config.model_kwargs.get(
                self.completion_token_limit_field,
                self.response_token_budget,
            ),
        )
        if isinstance(raw_value, bool):
            raise ValueError(f"{self.completion_token_limit_field} must be a positive integer")
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.completion_token_limit_field} must be a positive integer"
            ) from exc
        if value <= 0 or value != raw_value:
            raise ValueError(f"{self.completion_token_limit_field} must be a positive integer")
        return value

    def _tokenize_prompt(self, messages: list[dict[str, Any]]) -> int:
        base_url = (
            os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or ""
        ).rstrip("/")
        if not base_url:
            raise RuntimeError("response_token_budget requires OPENAI_API_BASE or OPENAI_BASE_URL")
        tokenize_url = (
            f"{base_url}/tokenize" if base_url.endswith("/v1") else f"{base_url}/v1/tokenize"
        )
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "tools": [VANILLUX2_BASH_TOOL],
        }
        extra_body = self.config.model_kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            chat_template_kwargs = extra_body.get("chat_template_kwargs")
            if isinstance(chat_template_kwargs, dict):
                payload["chat_template_kwargs"] = chat_template_kwargs
        for key in ("tool_choice", "reasoning_effort"):
            value = self.config.model_kwargs.get(key)
            if value is not None:
                payload[key] = value

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        import httpx

        client = getattr(litellm, "client_session", None)
        owns_client = client is None
        if owns_client:
            client = httpx.Client(trust_env=False)
        try:
            for attempt in range(1, _TOKENIZE_MAX_ATTEMPTS + 1):
                try:
                    response = client.post(
                        tokenize_url,
                        json=payload,
                        headers=headers,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    body = response.json()
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    detail = _http_error_detail(exc.response, status_code)
                    if status_code not in _TOKENIZE_RETRYABLE_STATUS_CODES:
                        raise RuntimeError(
                            "failed to obtain exact prompt length from gateway "
                            f"/v1/tokenize after {attempt}/{_TOKENIZE_MAX_ATTEMPTS} "
                            f"attempts: {detail}"
                        ) from exc
                    retryable_error: Exception = exc
                except (httpx.TransportError, TimeoutError, ConnectionError) as exc:
                    detail = _bounded_error_detail(f"{type(exc).__name__}: {exc}")
                    retryable_error = exc
                except Exception as exc:
                    detail = _bounded_error_detail(f"{type(exc).__name__}: {exc}")
                    raise RuntimeError(
                        "failed to obtain exact prompt length from gateway "
                        f"/v1/tokenize after {attempt}/{_TOKENIZE_MAX_ATTEMPTS} "
                        f"attempts: {detail}"
                    ) from exc
                else:
                    count = body.get("count") if isinstance(body, dict) else None
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        raise RuntimeError(
                            "gateway /v1/tokenize returned an invalid scalar token count"
                        )
                    return count

                if attempt == _TOKENIZE_MAX_ATTEMPTS:
                    raise RuntimeError(
                        "failed to obtain exact prompt length from gateway "
                        f"/v1/tokenize after {attempt}/{_TOKENIZE_MAX_ATTEMPTS} "
                        f"attempts: {detail}"
                    ) from retryable_error
                time.sleep(_TOKENIZE_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        finally:
            if owns_client:
                client.close()

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[VANILLUX2_BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as exc:
            exc.message += (
                " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            )
            raise

    def _parse_actions(self, response: Any) -> list[dict[str, Any]]:
        actions = super()._parse_actions(response)
        if len(actions) == 1:
            return actions

        finish_reason = getattr(response.choices[0], "finish_reason", None)
        content = Template(
            self.config.format_error_template,
            undefined=StrictUndefined,
        ).render(
            actions=actions,
            error=(f"Expected exactly one bash tool call, but received {len(actions)}."),
            finish_reason=finish_reason,
        )
        raise FormatError(
            {
                "role": "user",
                "content": content,
                "extra": {"interrupt_type": "FormatError"},
            }
        )
