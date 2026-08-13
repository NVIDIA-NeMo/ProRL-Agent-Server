"""HTTP client for forwarding requests to an OpenAI-compatible inference server.

Backend differences (request params, response shape) are isolated in the
``InferenceEngine`` strategy this client holds; the HTTP/streaming/pause logic
here is backend-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal
from urllib.parse import quote

import httpx

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
    """HTTP client to an inference server or opted-in scheduler proxy.

    Per-call bound comes from the session's remaining-timeout budget
    (`_await_with_budget` at the gateway node). The internal httpx timeout
    is a high liveness ceiling so that a stuck engine can't pin a request
    past the session deadline. The ``engine`` strategy injects backend-specific
    request params and canonicalizes responses.
    """

    _LIVENESS_TIMEOUT_SECONDS = 900.0

    def __init__(
        self,
        base_url: str,
        engine: InferenceEngine,
        *,
        scheduler: Literal["none", "thunderagent"] = "none",
        program_namespace: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.engine = engine
        self.scheduler = scheduler
        self.program_namespace = program_namespace
        self._client: httpx.AsyncClient | None = None
        self._generation_paused = False
        self._inflight_generations = 0
        self._generation_condition = asyncio.Condition()
        self._program_condition = asyncio.Condition()
        self._program_inflight: dict[str, int] = {}
        self._programs_may_exist: set[str] = set()
        self._terminal_programs: set[str] = set()
        self._program_release_tasks: dict[str, asyncio.Task[None]] = {}
        self._closing = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._LIVENESS_TIMEOUT_SECONDS, connect=30),
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

    async def completion(
        self,
        request: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full JSON response."""
        if self.scheduler == "thunderagent" and not session_id:
            raise ValueError("ThunderAgent completions require a session_id")
        program_registered = False
        program_id: str | None = None
        slot_acquired = False
        if self.scheduler == "thunderagent":
            assert session_id is not None
            program_id = await self._begin_program_request(session_id)
            program_registered = True
        try:
            await self._acquire_generation_slot()
            slot_acquired = True
            client = await self._get_client()
            from copy import deepcopy

            request_copy = deepcopy(request)
            request_copy.pop("stream", None)
            request_copy["stream"] = False
            request_copy = self.engine.prepare_request(request_copy)
            headers = {"Content-Type": "application/json"}
            if program_id is not None:
                # ThunderAgent checks body fields before X-Session-ID. Reserve
                # the program identity for Polar so callers cannot split one session.
                request_copy.pop("program_id", None)
                extra_body = request_copy.get("extra_body")
                if isinstance(extra_body, dict):
                    extra_body.pop("program_id", None)
                headers["X-Session-ID"] = program_id
            try:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=request_copy,
                    headers=headers,
                )
            except httpx.RequestError as exc:
                raise self._translate_transport_error(exc) from exc

            await self._raise_for_status(resp)
            return self.engine.normalize_response(resp.json())
        finally:
            try:
                if slot_acquired:
                    await self._release_generation_slot()
            finally:
                if program_registered:
                    assert session_id is not None
                    await self._end_program_request(session_id)

    async def release_program(self, session_id: str) -> bool:
        """Release one program from an opted-in ThunderAgent scheduler."""
        if self.scheduler != "thunderagent":
            return False
        async with self._program_condition:
            self._terminal_programs.add(session_id)
            await self._program_condition.wait_for(
                lambda: self._program_inflight.get(session_id, 0) == 0
            )
            if session_id not in self._programs_may_exist:
                return False
            release_task = self._program_release_tasks.get(session_id)
            owns_task = release_task is None
            if release_task is None:
                release_task = asyncio.create_task(
                    self._release_program_upstream(self._program_id(session_id))
                )
                self._program_release_tasks[session_id] = release_task
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self._program_condition:
                if self._program_release_tasks.get(session_id) is release_task:
                    self._program_release_tasks.pop(session_id, None)
            if not owns_task:
                return await self.release_program(session_id)
            raise
        async with self._program_condition:
            if self._program_release_tasks.get(session_id) is release_task:
                self._program_release_tasks.pop(session_id, None)
                self._programs_may_exist.discard(session_id)
        return owns_task

    def _program_id(self, session_id: str) -> str:
        if self.program_namespace:
            return f"{quote(self.program_namespace, safe='')}:{session_id}"
        return session_id

    async def _begin_program_request(self, session_id: str) -> str:
        async with self._program_condition:
            if self._closing or session_id in self._terminal_programs:
                raise UpstreamHTTPError(
                    409,
                    {"error": {"message": f"Session {session_id} has terminated"}},
                )
            self._program_inflight[session_id] = self._program_inflight.get(session_id, 0) + 1
            # Mark before any await to make release safe even if the proxy POST
            # is cancelled after ThunderAgent has accepted it.
            self._programs_may_exist.add(session_id)
            return self._program_id(session_id)

    async def _end_program_request(self, session_id: str) -> None:
        async with self._program_condition:
            remaining = self._program_inflight.get(session_id, 0) - 1
            if remaining > 0:
                self._program_inflight[session_id] = remaining
            else:
                self._program_inflight.pop(session_id, None)
            self._program_condition.notify_all()

    async def _release_program_upstream(self, program_id: str) -> None:
        client = await self._get_client()
        try:
            response = await client.post(
                "/programs/release",
                json={"program_id": program_id},
                timeout=5.0,
            )
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(response)

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
            resp = await client.get("/v1/models")
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

    async def close(self) -> None:
        program_ids: list[str] = []
        if self.scheduler == "thunderagent":
            async with self._program_condition:
                self._closing = True
                program_ids = list(self._programs_may_exist)
            async with self._generation_condition:
                self._generation_paused = False
                self._generation_condition.notify_all()
        if program_ids:
            results = await asyncio.gather(
                *(self.release_program(program_id) for program_id in program_ids),
                return_exceptions=True,
            )
            for program_id, result in zip(program_ids, results):
                if isinstance(result, BaseException):
                    logger.warning(
                        "Failed to release ThunderAgent program %s during shutdown: %s",
                        program_id,
                        result,
                    )
        if self._client and not self._client.is_closed:
            await self._client.aclose()
