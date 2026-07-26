from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from slime_bridge import rollout


class _Response:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://gateway/control")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("control failed", request=request, response=response)

    def json(self) -> object:
        return self._payload


class _Client:
    calls: list[tuple[str, str]] = []
    nodes: object = []
    pause_fail_url: str | None = None

    def __init__(self, **_: object) -> None:
        pass

    def __enter__(self) -> "_Client":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get(self, url: str) -> _Response:
        self.calls.append(("GET", url))
        return _Response(self.nodes)

    def post(self, url: str, *, params: object = None) -> _Response:
        del params
        action = url.rsplit("/", 1)[-1]
        self.calls.append((action.upper(), url))
        gateway_url = url.split("/admin/inference/", 1)[0]
        if action == "pause" and gateway_url == self.pause_fail_url:
            return _Response({"paused": True, "inflight": 1}, status_code=503)
        return _Response(
            {"paused": action == "pause", "inflight": 0},
        )


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch):
    _Client.calls = []
    _Client.nodes = []
    _Client.pause_fail_url = None
    rollout._WEIGHT_UPDATE_PAUSED_GATEWAYS = ()
    monkeypatch.setattr(rollout.httpx, "Client", _Client)
    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", "trusted-control-token")
    yield
    rollout._WEIGHT_UPDATE_PAUSED_GATEWAYS = ()


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        polar_rollout_url="http://rollout:8080",
        polar_weight_update_pause_timeout=12.0,
    )


def test_weight_update_hooks_pause_and_resume_complete_gateway_fleet() -> None:
    _Client.nodes = [
        {"gateway_url": "http://gateway-0:8100"},
        {"gateway_url": "http://gateway-1:8100"},
        {"gateway_url": "http://gateway-2:8100"},
        {"gateway_url": "http://gateway-3:8100"},
    ]

    rollout.pause_for_weight_update(_args())

    assert rollout._WEIGHT_UPDATE_PAUSED_GATEWAYS == tuple(
        node["gateway_url"] for node in _Client.nodes
    )
    assert sum(method == "PAUSE" for method, _ in _Client.calls) == 4
    assert not any(method == "RESUME" for method, _ in _Client.calls)

    rollout.resume_after_weight_update(_args())

    assert rollout._WEIGHT_UPDATE_PAUSED_GATEWAYS == ()
    assert sum(method == "RESUME" for method, _ in _Client.calls) == 4


def test_weight_update_hooks_are_attached_to_custom_rollout_function() -> None:
    assert rollout.generate_rollout_polar_async.pause_for_weight_update is (
        rollout.pause_for_weight_update
    )
    assert rollout.generate_rollout_polar_async.resume_after_weight_update is (
        rollout.resume_after_weight_update
    )


def test_partial_pause_failure_resumes_every_gateway_and_aborts() -> None:
    _Client.nodes = [
        {"gateway_url": "http://gateway-0:8100"},
        {"gateway_url": "http://gateway-1:8100"},
    ]
    _Client.pause_fail_url = "http://gateway-1:8100"

    with pytest.raises(RuntimeError, match="Failed to pause"):
        rollout.pause_for_weight_update(_args())

    assert rollout._WEIGHT_UPDATE_PAUSED_GATEWAYS == ()
    assert sum(method == "PAUSE" for method, _ in _Client.calls) == 2
    assert sum(method == "RESUME" for method, _ in _Client.calls) == 2


@pytest.mark.parametrize(
    "nodes",
    [
        [],
        [{"node_id": "missing-url"}],
        [
            {"gateway_url": "http://gateway:8100"},
            {"gateway_url": "http://gateway:8100"},
        ],
    ],
)
def test_pause_fails_closed_on_malformed_gateway_discovery(nodes: object) -> None:
    _Client.nodes = nodes

    with pytest.raises(RuntimeError):
        rollout.pause_for_weight_update(_args())

    assert rollout._WEIGHT_UPDATE_PAUSED_GATEWAYS == ()
    assert not any(method == "PAUSE" for method, _ in _Client.calls)


def test_resume_requires_a_successful_pause() -> None:
    with pytest.raises(RuntimeError, match="was not paused"):
        rollout.resume_after_weight_update(_args())
