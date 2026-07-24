"""HTTP client for forwarding requests to an OpenAI-compatible inference server.

Backend differences (request params, response shape) are isolated in the
``InferenceEngine`` strategy this client holds; the HTTP/streaming/pause logic
here is backend-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx
import orjson

from polar.gateway.engine import InferenceEngine

logger = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """Base class for upstream gateway failures."""


class UpstreamHTTPError(UpstreamError):
    """Raised when the upstream returns a non-2xx status."""

    def __init__(self, status_code: int, body: dict[str, Any] | str | None = None):
        self.status_code = status_code
        self.body = body
        super().__init__(self._build_message(status_code, body))

    @staticmethod
    def _build_message(status_code: int, body: dict[str, Any] | str | None) -> str:
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            message = body.get("message")
            if isinstance(message, str) and message:
                return message
        if isinstance(body, str) and body:
            return body
        return f"Upstream request failed with status {status_code}"


class UpstreamTimeoutError(UpstreamError):
    """Raised when the upstream times out."""


class UpstreamTransportError(UpstreamError):
    """Raised for connection and transport failures."""


class InferenceClient:
    """Direct httpx client to an inference server's OpenAI-compatible API.

    Per-call bound comes from the session's remaining-timeout budget
    (`_await_with_budget` at the gateway node). The internal httpx timeout
    is a high liveness ceiling so that a stuck engine can't pin a request
    past the session deadline. The ``engine`` strategy injects backend-specific
    request params and canonicalizes responses.
    """

    _LIVENESS_TIMEOUT_SECONDS = 900.0
    _WORKER_CACHE_TTL_SECONDS = 5.0

    def __init__(
        self,
        base_url: str,
        engine: InferenceEngine,
        *,
        default_headers: Mapping[str, str] | None = None,
        max_concurrency: int | None = None,
    ):
        if max_concurrency is not None and max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        self.base_url = base_url.rstrip("/")
        self.engine = engine
        # These headers are supplied only by trusted gateway configuration.
        # In particular, callers' sandbox Authorization header (which carries
        # the Polar session ID) is never copied into this client.
        self._default_headers = dict(default_headers or {})
        self._max_concurrency = max_concurrency
        self._completion_semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )
        self._client: httpx.AsyncClient | None = None
        self._generation_paused = False
        self._inflight_generations = 0
        self._generation_condition = asyncio.Condition()
        self._worker_discovery_lock = asyncio.Lock()
        self._worker_urls: tuple[str, ...] = ()
        self._worker_cache_expires_at = 0.0
        self._next_worker_index = 0
        self._next_tokenizer_index = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._LIVENESS_TIMEOUT_SECONDS, connect=30),
                headers=self._default_headers,
            )
        return self._client

    async def _read_error_body(self, response: httpx.Response) -> dict[str, Any] | str | None:
        content = await response.aread()
        if not content:
            return None

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return

        body = await self._read_error_body(response)
        await response.aclose()
        raise UpstreamHTTPError(response.status_code, body)

    @staticmethod
    def _translate_transport_error(exc: httpx.RequestError) -> UpstreamError:
        if isinstance(exc, httpx.TimeoutException):
            return UpstreamTimeoutError("Upstream request timed out")
        return UpstreamTransportError(f"Upstream request failed: {exc}")

    async def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full JSON response."""
        await self._acquire_generation_slot()
        try:
            if self._completion_semaphore is not None:
                async with self._completion_semaphore:
                    return await self._completion(request)
            return await self._completion(request)
        finally:
            await self._release_generation_slot()

    async def _completion(self, request: dict[str, Any]) -> dict[str, Any]:
        client = await self._get_client()
        from copy import deepcopy

        request_copy = deepcopy(request)
        request_copy.pop("stream", None)
        request_copy["stream"] = False
        request_copy = self.engine.prepare_request(request_copy)
        try:
            completion_url = await self._completion_url(client)
            resp = await client.post(
                completion_url,
                json=request_copy,
                headers=self._json_headers(),
            )
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc

        await self._raise_for_status(resp)
        # Completion payloads can contain hundreds of thousands of token and
        # logprob values.  orjson materially shortens this synchronous decode
        # section on the gateway's event-loop thread.
        return self.engine.normalize_response(orjson.loads(resp.content))

    async def responses(self, request: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming OpenAI Responses request for a native upstream."""
        await self._acquire_generation_slot()
        try:
            if self._completion_semaphore is not None:
                async with self._completion_semaphore:
                    return await self._responses(request)
            return await self._responses(request)
        finally:
            await self._release_generation_slot()

    async def _responses(self, request: dict[str, Any]) -> dict[str, Any]:
        client = await self._get_client()
        from copy import deepcopy

        request_copy = deepcopy(request)
        request_copy["stream"] = False
        request_copy = self.engine.prepare_request(request_copy)
        try:
            resp = await client.post(
                self._v1_endpoint("responses"),
                json=request_copy,
                headers=self._json_headers(),
            )
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc

        await self._raise_for_status(resp)
        return orjson.loads(resp.content)

    async def tokenize(self, request: dict[str, Any]) -> dict[str, Any]:
        """Tokenize a prompt with the inference server's exact chat template.

        This deliberately bypasses ``InferenceEngine.prepare_request``: the
        OpenAI-compatible ``/v1/tokenize`` schema accepts messages, tools and
        chat-template kwargs, but not generation-only fields such as logprob
        metadata flags.  Tokenization is read-only and is therefore also not
        counted as an in-flight generation while trainer weight updates pause
        new completion requests.
        """

        client = await self._get_client()
        try:
            tokenize_url = await self._tokenize_url(client)
            resp = await client.post(
                tokenize_url,
                json=request,
                headers=self._json_headers(),
            )
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc

        await self._raise_for_status(resp)
        return orjson.loads(resp.content)

    async def _tokenize_url(self, client: httpx.AsyncClient) -> str:
        """Resolve tokenization to a regular SGLang worker when available."""

        if self.engine.name != "sglang":
            return self._v1_endpoint("tokenize")
        worker_urls = await self._regular_worker_urls(client)
        if not worker_urls:
            return self._v1_endpoint("tokenize")
        index = self._next_tokenizer_index
        self._next_tokenizer_index += 1
        return f"{worker_urls[index % len(worker_urls)]}/v1/tokenize"

    def _json_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", **self._default_headers}

    def _v1_endpoint(self, endpoint: str) -> str:
        """Avoid duplicating ``/v1`` when a provider includes it in base_url."""

        base_path = urlparse(self.base_url).path.rstrip("/")
        if base_path.endswith("/v1"):
            return endpoint.lstrip("/")
        return f"/v1/{endpoint.lstrip('/')}"

    async def _completion_url(self, client: httpx.AsyncClient) -> str:
        """Resolve SGLang routers to regular workers, with a short-lived cache.

        Source SGLang's router does not consistently preserve the token metadata
        Polar needs for TIS. Regular workers expose the same OpenAI endpoint and
        do preserve it, so requests are distributed directly across them. PD
        (prefill/decode) layouts still go through the router because the router
        must coordinate both worker types.
        """

        if self.engine.name != "sglang":
            return self._v1_endpoint("chat/completions")

        worker_urls = await self._regular_worker_urls(client)
        if not worker_urls:
            return self._v1_endpoint("chat/completions")

        index = self._next_worker_index
        self._next_worker_index += 1
        return f"{worker_urls[index % len(worker_urls)]}/v1/chat/completions"

    async def _regular_worker_urls(self, client: httpx.AsyncClient) -> tuple[str, ...]:
        now = time.monotonic()
        if now < self._worker_cache_expires_at:
            return self._worker_urls

        async with self._worker_discovery_lock:
            now = time.monotonic()
            if now < self._worker_cache_expires_at:
                return self._worker_urls

            worker_urls: tuple[str, ...] = ()
            try:
                response = await client.get("/workers")
                if response.is_success:
                    payload = response.json()
                    workers = payload.get("workers") if isinstance(payload, dict) else None
                    if isinstance(workers, list):
                        worker_types = {
                            worker.get("worker_type")
                            for worker in workers
                            if isinstance(worker, dict)
                        }
                        if not worker_types.intersection({"prefill", "decode"}):
                            worker_urls = tuple(
                                url.rstrip("/")
                                for worker in workers
                                if isinstance(worker, dict)
                                and worker.get("worker_type", "regular") == "regular"
                                and isinstance((url := worker.get("url")), str)
                                and url.startswith(("http://", "https://"))
                            )
                else:
                    await response.aclose()
            except (httpx.RequestError, json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "Could not discover SGLang workers at %s; using the router",
                    self.base_url,
                    exc_info=True,
                )

            self._worker_urls = tuple(dict.fromkeys(worker_urls))
            self._worker_cache_expires_at = now + self._WORKER_CACHE_TTL_SECONDS
            return self._worker_urls

    async def _acquire_generation_slot(self) -> None:
        async with self._generation_condition:
            await self._generation_condition.wait_for(lambda: not self._generation_paused)
            self._inflight_generations += 1

    async def _release_generation_slot(self) -> None:
        async with self._generation_condition:
            self._inflight_generations -= 1
            self._generation_condition.notify_all()

    async def pause_generation(self, *, timeout_seconds: float = 300.0) -> dict[str, Any]:
        """Block new generation requests and wait for current inference calls to drain."""
        async with self._generation_condition:
            self._generation_paused = True
            self._generation_condition.notify_all()
            await asyncio.wait_for(
                self._generation_condition.wait_for(lambda: self._inflight_generations == 0),
                timeout=timeout_seconds,
            )
            return self.generation_status()

    async def resume_generation(self) -> dict[str, Any]:
        async with self._generation_condition:
            self._generation_paused = False
            self._generation_condition.notify_all()
            return self.generation_status()

    def generation_status(self) -> dict[str, Any]:
        return {
            "paused": self._generation_paused,
            "inflight": self._inflight_generations,
            "base_url": self.base_url,
            "engine": self.engine.name,
        }

    async def list_models(self) -> dict[str, Any]:
        """Passthrough GET /v1/models."""
        client = await self._get_client()
        try:
            resp = await client.get(self._v1_endpoint("models"))
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        return resp.json()

    async def health(self) -> dict[str, Any]:
        """Passthrough GET /health."""
        client = await self._get_client()
        try:
            resp = await client.get("/health")
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        content = await resp.aread()
        if not content:
            return {"status": "ok"}

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return {"status": "ok"}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"status": "ok", "body": text}

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
