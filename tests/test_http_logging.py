from __future__ import annotations

from types import SimpleNamespace

import pytest
import uvicorn

from polar.gateway import server as gateway_server
from polar.http_logging import uvicorn_access_log_enabled
from polar.rollout import server as rollout_server


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "On"])
def test_access_log_can_be_explicitly_enabled(monkeypatch, value: str) -> None:
    monkeypatch.setenv("POLAR_UVICORN_ACCESS_LOG", value)

    assert uvicorn_access_log_enabled()


@pytest.mark.parametrize("value", [None, "", "0", "false", "debug"])
def test_access_log_is_disabled_by_default(monkeypatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("POLAR_UVICORN_ACCESS_LOG", raising=False)
    else:
        monkeypatch.setenv("POLAR_UVICORN_ACCESS_LOG", value)

    assert not uvicorn_access_log_enabled()


def test_gateway_server_passes_access_log_policy(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.delenv("POLAR_UVICORN_ACCESS_LOG", raising=False)
    monkeypatch.setattr(gateway_server, "configure_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_server,
        "get_state",
        lambda: SimpleNamespace(node=SimpleNamespace(host="127.0.0.1", port=8100)),
    )
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))

    gateway_server.serve("unused.yaml", node_id="node-0")

    assert captured["access_log"] is False


def test_rollout_server_passes_access_log_policy(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("POLAR_UVICORN_ACCESS_LOG", "1")
    monkeypatch.setattr(rollout_server, "configure_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rollout_server,
        "get_state",
        lambda: SimpleNamespace(
            rollout=SimpleNamespace(host="127.0.0.1", port=8080)
        ),
    )
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))

    rollout_server.serve("unused.yaml")

    assert captured["access_log"] is True
