"""Session registry and request resolution helpers."""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from polar.rollout.models import SessionResult, SessionStatus

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ROUTER_CAPABILITY_SCOPE = "router_policy"
MODEL_POOL_CAPABILITY_SCOPE = "model_pool"
MODEL_POOL_ADMISSION_CAPABILITY_SCOPE = "model_pool_admission"
ROUTER_CAPABILITY_ENV = "POLAR_ROUTER_CAPABILITY"
MODEL_POOL_CAPABILITY_ENV = "POLAR_MODEL_POOL_CAPABILITY"
MODEL_POOL_ADMISSION_CAPABILITY_ENV = "POLAR_MODEL_POOL_ADMISSION_CAPABILITY"


class InvalidSessionIdError(ValueError):
    """Raised when a caller-provided session id is unsafe for storage."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def clean_session_id(value: str | None) -> str | None:
    """Normalize and validate an external session id."""
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None
    if not SESSION_ID_PATTERN.fullmatch(normalized):
        raise InvalidSessionIdError(
            "Session IDs may only contain letters, numbers, '.', '_' or '-'.",
        )
    return normalized


def generate_session_id() -> str:
    """Generate a new storage-safe session id."""
    return str(uuid.uuid4())


@dataclass
class SessionInfo:
    """In-memory metadata for a known session."""

    session_id: str
    created_at: datetime
    last_activity: datetime
    task_id: str | None = None
    registered: bool = False
    status: str = SessionStatus.REGISTERED
    result: SessionResult | None = None
    metadata: dict[str, Any] | None = None


class SessionRegistry:
    """Thread-safe in-memory session registry."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionInfo] = {}
        # Only capability digests are retained in the registry.  The raw
        # values live in the dispatched ManagedSession long enough to be
        # injected into its runtime and are never returned by status APIs.
        self._capabilities: dict[bytes, tuple[str, str]] = {}
        self._session_capability_digests: dict[str, set[bytes]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        session_id: str,
        *,
        task_id: str | None = None,
        registered: bool = False,
        status: str = SessionStatus.REGISTERED,
        metadata: dict[str, Any] | None = None,
    ) -> SessionInfo:
        session_id = clean_session_id(session_id) or generate_session_id()
        now = _utcnow()
        with self._lock:
            info = self._sessions.get(session_id)
            if info is not None:
                info.last_activity = now
                if task_id is not None:
                    info.task_id = task_id
                info.registered = info.registered or registered
                info.status = status or info.status
                if metadata:
                    info.metadata = {**(info.metadata or {}), **metadata}
                if status in SessionStatus.active():
                    info.result = None
                return info

            info = SessionInfo(
                session_id=session_id,
                created_at=now,
                last_activity=now,
                task_id=task_id,
                registered=registered,
                status=status,
                metadata=dict(metadata or {}),
            )
            self._sessions[session_id] = info
            return info

    def get(self, session_id: str) -> SessionInfo | None:
        with self._lock:
            return self._sessions.get(session_id)

    def issue_capability(self, session_id: str, *, scope: str) -> str:
        """Issue an unguessable, scoped credential for a dispatched session."""

        if not scope:
            raise ValueError("capability scope cannot be empty")
        with self._lock:
            info = self._sessions.get(session_id)
            if info is None or not info.registered:
                raise ValueError("capabilities require a registered session")
            while True:
                token = secrets.token_urlsafe(32)
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                if digest not in self._capabilities:
                    break
            self._capabilities[digest] = (session_id, scope)
            self._session_capability_digests.setdefault(session_id, set()).add(digest)
            return token

    def resolve_capability(self, token: str, *, scope: str) -> SessionInfo | None:
        """Resolve a raw credential only when it carries the requested scope."""

        digest = hashlib.sha256(token.encode("utf-8")).digest()
        with self._lock:
            record = self._capabilities.get(digest)
            if record is None or record[1] != scope:
                return None
            return self._sessions.get(record[0])

    def revoke_capabilities(self, session_id: str) -> None:
        with self._lock:
            digests = self._session_capability_digests.pop(session_id, set())
            for digest in digests:
                self._capabilities.pop(digest, None)

    def update_activity(self, session_id: str) -> None:
        with self._lock:
            info = self._sessions.get(session_id)
            if info is not None:
                info.last_activity = _utcnow()

    def set_status(self, session_id: str, status: str) -> SessionInfo | None:
        with self._lock:
            info = self._sessions.get(session_id)
            if info is not None:
                info.status = status
                info.last_activity = _utcnow()
            return info

    def set_result(self, session_id: str, result: SessionResult) -> SessionInfo | None:
        with self._lock:
            info = self._sessions.get(session_id)
            if info is None:
                return None
            info.result = result
            info.status = result.status
            info.last_activity = _utcnow()
            return info

    def clear_result_payload(self, session_id: str) -> None:
        """Drop the heavy result payload once the callback has been delivered.

        Why: `SessionInfo.result` retains the full trajectory + timing payload
        for every terminated session, which accumulates unbounded over a
        long-running gateway process. We keep `status` and `task_id` for
        debugging visibility but release the heavy payload.
        """
        with self._lock:
            info = self._sessions.get(session_id)
            if info is not None:
                info.result = None

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            digests = self._session_capability_digests.pop(session_id, set())
            for digest in digests:
                self._capabilities.pop(digest, None)

    def active_count(self) -> int:
        terminal = SessionStatus.terminal()
        with self._lock:
            return sum(
                1 for info in self._sessions.values() if info.status not in terminal
            )

    def active_status_counts(self) -> dict[str, int]:
        terminal = SessionStatus.terminal()
        with self._lock:
            counts: dict[str, int] = {}
            for info in self._sessions.values():
                if info.status in terminal:
                    continue
                counts[info.status] = counts.get(info.status, 0) + 1
            return dict(sorted(counts.items()))

    def active_sessions(self) -> list[dict[str, Any]]:
        terminal = SessionStatus.terminal()
        with self._lock:
            now = _utcnow()
            rows: list[dict[str, Any]] = []
            for info in self._sessions.values():
                if info.status in terminal:
                    continue
                rows.append(
                    {
                        "session_id": info.session_id,
                        "task_id": info.task_id,
                        "status": info.status,
                        "age_seconds": max(
                            0.0,
                            (now - info.created_at).total_seconds(),
                        ),
                        "idle_seconds": max(
                            0.0,
                            (now - info.last_activity).total_seconds(),
                        ),
                    }
                )
            rows.sort(key=lambda row: (str(row["task_id"] or ""), str(row["session_id"])))
            return rows


def extract_api_key(headers: dict[str, str]) -> str | None:
    """Extract API key from Authorization, X-Api-Key, or x-goog-api-key headers."""
    lower_headers = {k.lower(): v for k, v in headers.items()}

    auth = lower_headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None

    x_api_key = lower_headers.get("x-api-key") or lower_headers.get("x_api_key")
    if x_api_key:
        return x_api_key.strip() or None

    # Google APIs use x-goog-api-key header
    goog_key = lower_headers.get("x-goog-api-key")
    if goog_key:
        return goog_key.strip() or None

    return None


def resolve_session_id(
    registry: SessionRegistry,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    query_session_id: str | None = None,
) -> str:
    """Resolve session id from explicit ids or auth, else create a new one."""
    lower_headers = {k.lower(): v for k, v in headers.items()}
    explicit_session_id = (
        clean_session_id(lower_headers.get("x-session-id"))
        or clean_session_id(lower_headers.get("x_session_id"))
        or clean_session_id(lower_headers.get("proxy-x-session-id"))
        or clean_session_id(lower_headers.get("proxy_x_session_id"))
        or clean_session_id(query_session_id)
        or clean_session_id(body.get("_proxy_session_id"))
    )
    if explicit_session_id:
        if registry.get(explicit_session_id) is None:
            registry.register(explicit_session_id)
        else:
            registry.update_activity(explicit_session_id)
        return explicit_session_id

    api_key = extract_api_key(headers)
    if api_key and registry.get(api_key) is not None:
        registry.update_activity(api_key)
        return api_key

    return registry.register(generate_session_id()).session_id


class SessionCreateRequest(BaseModel):
    task_id: str | None = None
    session_id: str | None = None


class SessionCreateResponse(BaseModel):
    session_id: str
    task_id: str | None = None
    created_at: datetime
    completion_count: int
    status: str = SessionStatus.REGISTERED


class SessionStatusResponse(BaseModel):
    session_id: str
    task_id: str | None = None
    created_at: datetime
    completion_count: int
    status: str = SessionStatus.REGISTERED
    result: SessionResult | None = None


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool
    messages_deleted: int
