from __future__ import annotations

import base64
import os
from pathlib import Path
import socket
import sys
import threading

import pytest

from polar.agent.presets import mini_swe_runner


def test_private_task_moves_to_python_argv_but_not_proc_cmdline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = "Use pkill -f polar-danger-marker-e81a safely\n雪 'quoted'"
    monkeypatch.setattr(sys, "argv", ["polar_mini_swe_runner", "--yolo"])
    monkeypatch.setenv(
        "POLAR_MINI_SWE_TASK_B64",
        base64.b64encode(task.encode("utf-8")).decode("ascii"),
    )
    before = Path("/proc/self/cmdline").read_bytes()

    mini_swe_runner._inject_task_from_env()

    assert sys.argv == ["polar_mini_swe_runner", "--yolo", f"--task={task}"]
    assert "POLAR_MINI_SWE_TASK_B64" not in os.environ
    assert Path("/proc/self/cmdline").read_bytes() == before
    assert b"polar-danger-marker-e81a" not in before


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        ("not%base64", "strict base64"),
        (base64.b64encode(b"\xff").decode("ascii"), "UTF-8"),
    ],
)
def test_private_task_rejects_invalid_payload_and_pops_secret_env(
    monkeypatch: pytest.MonkeyPatch,
    encoded: str,
    message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["polar_mini_swe_runner", "--yolo"])
    monkeypatch.setenv("POLAR_MINI_SWE_TASK_B64", encoded)

    with pytest.raises(ValueError, match=message):
        mini_swe_runner._inject_task_from_env()

    assert sys.argv == ["polar_mini_swe_runner", "--yolo"]
    assert "POLAR_MINI_SWE_TASK_B64" not in os.environ


@pytest.mark.parametrize(
    "task_args",
    [("--task=argv-task",), ("--task", "argv-task")],
)
def test_private_task_rejects_ambiguous_env_and_argv(
    monkeypatch: pytest.MonkeyPatch,
    task_args: tuple[str, ...],
) -> None:
    original_argv = ["polar_mini_swe_runner", *task_args]
    monkeypatch.setattr(sys, "argv", original_argv.copy())
    monkeypatch.setenv(
        "POLAR_MINI_SWE_TASK_B64",
        base64.b64encode(b"env-task").decode("ascii"),
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        mini_swe_runner._inject_task_from_env()

    assert sys.argv == original_argv
    assert "POLAR_MINI_SWE_TASK_B64" not in os.environ


def _serve_unix_echo(path: Path) -> tuple[socket.socket, threading.Thread]:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen()

    def serve() -> None:
        connection, _ = server.accept()
        with connection:
            payload = connection.recv(4096)
            connection.sendall(b"echo:" + payload)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server, thread


def test_loopback_proxy_relays_to_unix_socket(tmp_path: Path) -> None:
    socket_path = tmp_path / "proxy.sock"
    upstream, thread = _serve_unix_echo(socket_path)
    try:
        with mini_swe_runner.LoopbackProxy(str(socket_path), 0) as proxy:
            # Port zero asks the kernel for a collision-free test port.
            with socket.create_connection(("127.0.0.1", proxy.port), timeout=2.0) as client:
                client.sendall(b"request")
                assert client.recv(4096) == b"echo:request"
    finally:
        upstream.close()
        thread.join(timeout=2.0)


def test_loopback_proxy_backlog_supports_parallel_package_downloads() -> None:
    assert mini_swe_runner._ThreadingUnixForwardServer.request_queue_size >= 256


def test_model_retry_policy_aborts_permanent_bad_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadRequest(Exception):
        pass

    class FakeModel:
        abort_exceptions = [TimeoutError]

    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {"exceptions": type("Exceptions", (), {"BadRequestError": BadRequest})},
    )
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake_litellm)
    monkeypatch.setitem(
        __import__("sys").modules,
        "minisweagent.models.litellm_model",
        type("FakeModule", (), {"LitellmModel": FakeModel}),
    )

    mini_swe_runner._configure_model_retry_policy()
    mini_swe_runner._configure_model_retry_policy()

    assert FakeModel.abort_exceptions == [BadRequest, TimeoutError]


def test_task_pythonpath_is_restored_only_after_portable_interpreter_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("POLAR_TASK_PYTHONPATH", "/app:/image/python")

    mini_swe_runner._restore_task_pythonpath()

    assert os.environ["PYTHONPATH"] == "/app:/image/python"
    assert "POLAR_TASK_PYTHONPATH" not in os.environ


def test_proxy_configuration_rewrites_all_proxy_variables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "proxy.sock"
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    upstream.bind(str(socket_path))
    upstream.listen()
    monkeypatch.setenv("POLAR_HTTP_PROXY_UDS", str(socket_path))
    monkeypatch.setenv("POLAR_HTTP_PROXY_PORT", "0")
    monkeypatch.setenv("no_proxy", "metadata.internal,localhost")
    monkeypatch.setenv("NO_PROXY", "registry.internal")

    proxy = mini_swe_runner._configure_http_proxy()
    try:
        assert proxy is not None
        assert proxy.port > 0
        expected_proxy_url = f"http://127.0.0.1:{proxy.port}"
        for name in mini_swe_runner._PROXY_ENV_NAMES:
            assert os.environ[name] == expected_proxy_url
        assert os.environ["no_proxy"] == (
            "metadata.internal,localhost,registry.internal,127.0.0.1,::1"
        )
        assert os.environ["NO_PROXY"] == os.environ["no_proxy"]
    finally:
        assert proxy is not None
        proxy.close()
        upstream.close()


def test_offline_task_removes_inherited_proxy_without_starting_forwarder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLAR_ALLOW_INTERNET", "false")
    monkeypatch.setenv("POLAR_HTTP_PROXY_UDS", "/unused/proxy.sock")
    for name in mini_swe_runner._PROXY_ENV_NAMES:
        monkeypatch.setenv(name, "http://inherited-proxy:3128")

    assert mini_swe_runner._configure_http_proxy() is None
    assert all(name not in os.environ for name in mini_swe_runner._PROXY_ENV_NAMES)
    assert "POLAR_HTTP_PROXY_UDS" not in os.environ
    assert "POLAR_HTTP_PROXY_PORT" not in os.environ


def test_apt_http_sources_are_upgraded_to_https_in_session_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    methods = tmp_path / "usr/lib/apt/methods"
    methods.mkdir(parents=True)
    (methods / "https").write_text("method", encoding="utf-8")
    apt_dir = tmp_path / "etc/apt"
    source_dir = apt_dir / "sources.list.d"
    source_dir.mkdir(parents=True)
    source_list = apt_dir / "sources.list"
    source_list.write_text(
        "# deb http://comment.invalid stable main\n"
        "deb http://archive.ubuntu.com/ubuntu jammy main\n"
        "deb-src [arch=amd64] http://security.ubuntu.com/ubuntu jammy-security main\n"
        "deb https://already-secure.invalid stable main\n",
        encoding="utf-8",
    )
    deb822 = source_dir / "debian.sources"
    deb822.write_text(
        "Types: deb\n"
        "URIs: http://deb.debian.org/debian https://already-secure.invalid/debian\n"
        " http://security.debian.org/debian-security\n"
        "Suites: stable stable-security\n"
        "# URIs: http://comment.invalid/debian\n",
        encoding="utf-8",
    )
    ignored = source_dir / "notes.txt"
    ignored.write_text("deb http://ignored.invalid stable main\n", encoding="utf-8")
    monkeypatch.setenv("POLAR_APT_HTTP_SOURCE_POLICY", "https")

    changed = mini_swe_runner._upgrade_apt_sources_to_https(tmp_path)

    assert changed == (source_list, deb822)
    assert source_list.read_text(encoding="utf-8") == (
        "# deb http://comment.invalid stable main\n"
        "deb https://archive.ubuntu.com/ubuntu jammy main\n"
        "deb-src [arch=amd64] https://security.ubuntu.com/ubuntu jammy-security main\n"
        "deb https://already-secure.invalid stable main\n"
    )
    assert deb822.read_text(encoding="utf-8") == (
        "Types: deb\n"
        "URIs: https://deb.debian.org/debian https://already-secure.invalid/debian\n"
        " https://security.debian.org/debian-security\n"
        "Suites: stable stable-security\n"
        "# URIs: http://comment.invalid/debian\n"
    )
    assert "http://ignored.invalid" in ignored.read_text(encoding="utf-8")


@pytest.mark.parametrize("policy", ["", "preserve", "invalid"])
def test_apt_source_upgrade_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: str,
) -> None:
    source = tmp_path / "etc/apt/sources.list"
    source.parent.mkdir(parents=True)
    source.write_text("deb http://example.invalid stable main\n", encoding="utf-8")
    monkeypatch.setenv("POLAR_APT_HTTP_SOURCE_POLICY", policy)

    assert mini_swe_runner._upgrade_apt_sources_to_https(tmp_path) == ()
    assert "http://example.invalid" in source.read_text(encoding="utf-8")


def test_apt_source_upgrade_requires_https_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "etc/apt/sources.list"
    source.parent.mkdir(parents=True)
    source.write_text("deb http://example.invalid stable main\n", encoding="utf-8")
    monkeypatch.setenv("POLAR_APT_HTTP_SOURCE_POLICY", "https")

    assert mini_swe_runner._upgrade_apt_sources_to_https(tmp_path) == ()
    assert "http://example.invalid" in source.read_text(encoding="utf-8")


def test_zero_proxy_port_requests_an_available_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLAR_HTTP_PROXY_PORT", "0")

    assert mini_swe_runner._proxy_port_from_env() == 0


@pytest.mark.parametrize("value", ["bad", "-1", "65536"])
def test_invalid_proxy_port_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("POLAR_HTTP_PROXY_PORT", value)
    with pytest.raises(ValueError, match="POLAR_HTTP_PROXY_PORT"):
        mini_swe_runner._proxy_port_from_env()
