from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import shlex
import socket
import subprocess
import sys
import threading
import time

import pytest

from polar.runtime.apptainer import ApptainerRuntime
from polar.runtime import apptainer_broker
from polar.runtime.models import RuntimeSpec


def _local_broker_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    allow_internet: bool = False,
    launch_environment: dict[str, str] | None = None,
) -> ApptainerRuntime:
    monkeypatch.setenv("POLAR_APPTAINER_NO_INSTANCE", "1")
    monkeypatch.setenv("POLAR_APPTAINER_DIRECT_EXEC_RETRIES", "1")
    runtime = ApptainerRuntime(
        RuntimeSpec(
            backend="apptainer",
            image="unused-in-local-broker-test.sif",
            workdir=str(tmp_path),
            allow_internet=allow_internet,
        ),
        "local-broker-test",
        tmp_path / "session",
    )
    broker_source = Path(apptainer_broker.__file__)
    supervisor_source = broker_source.with_name("apptainer_broker_supervisor.sh")

    def launch_args() -> list[str]:
        environment = {
            "POLAR_ALLOW_INTERNET": "true" if allow_internet else "false",
            **(launch_environment or {}),
        }
        return [
            "env",
            *(f"{key}={value}" for key, value in environment.items()),
            f"POLAR_APPTAINER_BROKER_RUNTIME_DIR={runtime._broker_dir}",  # noqa: SLF001
            f"POLAR_APPTAINER_BROKER_SCRIPT={broker_source}",
            f"POLAR_APPTAINER_BROKER_PYTHON={sys.executable}",
            "bash",
            str(supervisor_source),
        ]

    runtime._broker_launch_args = launch_args  # type: ignore[method-assign]
    return runtime


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


def _process_identity(pid: int) -> tuple[str, str]:
    command_line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    process_name = Path(f"/proc/{pid}/comm").read_text().strip()
    return command_line, process_name


def test_proxy_stop_never_waits_unbounded_after_sigkill() -> None:
    class StuckProcess:
        pid = 123

        def __init__(self) -> None:
            self.terminate_calls = 0
            self.kill_calls = 0
            self.wait_timeouts: list[float] = []

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, *, timeout: float):
            self.wait_timeouts.append(timeout)
            raise subprocess.TimeoutExpired("polar-proxy", timeout)

    process = StuckProcess()
    stopped = apptainer_broker._LoopbackProxy._stop_process(  # noqa: SLF001
        process,  # type: ignore[arg-type]
        terminate_timeout=0.25,
    )

    assert stopped is False
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_timeouts == [0.25, 1.0]


def test_proxy_close_keeps_unreaped_pid_for_supervisor(monkeypatch, tmp_path: Path) -> None:
    pid_path = tmp_path / "proxy.pid"
    pid_path.write_text("123\n")
    proxy = apptainer_broker._LoopbackProxy("unused.sock", 12345, pid_path=pid_path)  # noqa: SLF001
    proxy.process = object()  # type: ignore[assignment]
    monkeypatch.setattr(proxy, "_stop_process", lambda _process, *, terminate_timeout: False)

    proxy.close()

    assert pid_path.read_text() == "123\n"


def test_supervisor_cleans_residual_proxy_after_clean_broker_exit(tmp_path: Path) -> None:
    broker_source = tmp_path / "fake_broker.py"
    broker_source.write_text(
        """\
import os
from pathlib import Path
import subprocess

runtime_dir = Path(os.environ["POLAR_APPTAINER_BROKER_RUNTIME_DIR"])
proxy = subprocess.Popen(
    ["sleep", "30"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
(runtime_dir / "proxy.pid").write_text(f"{proxy.pid}\\n")
(runtime_dir / "spawned_proxy.pid").write_text(f"{proxy.pid}\\n")
"""
    )
    supervisor_source = Path(apptainer_broker.__file__).with_name(
        "apptainer_broker_supervisor.sh"
    )
    environment = {
        **os.environ,
        "POLAR_APPTAINER_BROKER_RUNTIME_DIR": str(tmp_path),
        "POLAR_APPTAINER_BROKER_SCRIPT": str(broker_source),
        "POLAR_APPTAINER_BROKER_PYTHON": sys.executable,
    }
    proxy_pid: int | None = None

    try:
        completed = subprocess.run(
            ["bash", str(supervisor_source)],
            env=environment,
            timeout=5,
            check=False,
        )
        proxy_pid = int((tmp_path / "spawned_proxy.pid").read_text())

        assert completed.returncode == 0
        assert not (tmp_path / "proxy.pid").exists()
        for _ in range(100):
            process_state = Path(f"/proc/{proxy_pid}/stat")
            if not process_state.exists() or process_state.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("residual proxy was not terminated by the supervisor")
    finally:
        if proxy_pid is not None:
            try:
                os.kill(proxy_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_exec_a_python_executable_falls_back_to_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(apptainer_broker.sys, "executable", "")
    monkeypatch.setattr(apptainer_broker.sys, "platform", "linux")

    assert apptainer_broker._current_python_executable() == "/proc/self/exe"


def test_direct_broker_preserves_background_processes_and_recovers_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _local_broker_runtime(monkeypatch, tmp_path)

    async def scenario() -> None:
        await runtime.start()
        try:
            launched = await runtime.exec(
                "sleep 30 >/dev/null 2>&1 & echo $! > background.pid"
            )
            assert launched.return_code == 0

            verified = await runtime.exec(
                'kill -0 "$(cat background.pid)" && echo background-alive'
            )
            assert verified.return_code == 0
            assert verified.stdout == "background-alive\n"

            started_at = time.monotonic()
            timed_out = await runtime.exec("sleep 30", timeout_sec=0.1)
            assert timed_out.return_code == -1
            assert time.monotonic() - started_at < 5.0

            still_ready = await runtime.exec("echo broker-ready")
            assert still_ready.return_code == 0
            assert still_ready.stdout == "broker-ready\n"
            await runtime.exec('kill "$(cat background.pid)"')
        finally:
            await runtime.stop()

    asyncio.run(scenario())


def test_broker_identity_survives_python_pkill_matching_and_preexec_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Simulate pkill selection, then an unexpected between-command exit."""

    runtime = _local_broker_runtime(monkeypatch, tmp_path)

    async def scenario() -> None:
        await runtime.start()
        try:
            broker_pid_path = runtime._broker_dir / "broker.pid"  # noqa: SLF001
            original_pid = int(broker_pid_path.read_text())
            command_line, process_name = _process_identity(original_pid)
            # These are the two process fields matched by ``pkill -f python``
            # and ``pkill python`` respectively.
            assert "python" not in command_line.lower()
            assert "python" not in process_name.lower()
            assert command_line.startswith("polar-broker ")
            assert process_name == "polar-broker"

            os.kill(original_pid, signal.SIGKILL)
            results = await asyncio.gather(
                *(runtime.exec(f"echo recovered-{index}") for index in range(8))
            )
            assert [result.return_code for result in results] == [0] * 8
            assert [result.stdout for result in results] == [
                f"recovered-{index}\n" for index in range(8)
            ]

            replacement_pid = int(broker_pid_path.read_text())
            assert replacement_pid != original_pid
            command_line, process_name = _process_identity(replacement_pid)
            assert "python" not in command_line.lower()
            assert "python" not in process_name.lower()

            summary = runtime.exec_timing_summary()
            assert summary["broker_recovery_count"] == 1
            assert summary["broker_recovery_failure_count"] == 0
            assert summary["broker_preflight_failure_count"] >= 1
            assert summary["broker_supervisor_restart_count"] == 1
            assert summary["broker_generation"] == 2
            assert summary["broker_recovery_ms"] >= 0.0
        finally:
            await runtime.stop()

    asyncio.run(scenario())


def test_direct_broker_cancellation_kills_foreground_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _local_broker_runtime(monkeypatch, tmp_path)

    async def scenario() -> None:
        await runtime.start()
        try:
            command = asyncio.create_task(
                runtime.exec("echo $$ > foreground.pid; exec sleep 30")
            )
            for _ in range(100):
                if (tmp_path / "foreground.pid").exists():
                    break
                await asyncio.sleep(0.02)
            assert (tmp_path / "foreground.pid").exists()
            foreground_pid = int((tmp_path / "foreground.pid").read_text())

            command.cancel()
            with pytest.raises(asyncio.CancelledError):
                await command

            for _ in range(100):
                try:
                    Path(f"/proc/{foreground_pid}").stat()
                except FileNotFoundError:
                    break
                await asyncio.sleep(0.02)
            assert not Path(f"/proc/{foreground_pid}").exists()
        finally:
            await runtime.stop()

    asyncio.run(scenario())


def test_proxy_listener_kill_does_not_kill_control_broker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proxy_socket = tmp_path / "unused-upstream-proxy.sock"
    port_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_socket.bind(("127.0.0.1", 0))
    proxy_port = int(port_socket.getsockname()[1])
    port_socket.close()
    runtime = _local_broker_runtime(
        monkeypatch,
        tmp_path,
        allow_internet=True,
        launch_environment={
            "POLAR_HTTP_PROXY_UDS": str(proxy_socket),
            "POLAR_HTTP_PROXY_PORT": str(proxy_port),
        },
    )

    async def scenario() -> None:
        await runtime.start()
        try:
            proxy_pid = runtime._broker_dir / "proxy.pid"  # noqa: SLF001
            original_proxy_pid = int(proxy_pid.read_text())
            command_line, process_name = _process_identity(original_proxy_pid)
            assert "python" not in command_line.lower()
            assert "python" not in process_name.lower()
            assert command_line.startswith("polar-proxy ")
            assert process_name == "polar-proxy"
            result = await runtime.exec(
                f"old=$(cat {shlex.quote(str(proxy_pid))}); "
                'kill -9 "$old"; '
                "for _ in $(seq 1 100); do "
                f"new=$(cat {shlex.quote(str(proxy_pid))} 2>/dev/null || true); "
                '[ -n "$new" ] && [ "$new" != "$old" ] && break; sleep 0.02; done; '
                '[ -n "$new" ] && [ "$new" != "$old" ] && echo proxy-restarted'
            )
            assert result.return_code == 0
            assert result.stdout == "proxy-restarted\n"
            assert runtime._broker_task is not None  # noqa: SLF001
            assert not runtime._broker_task.done()  # noqa: SLF001

            ready = await runtime.exec("echo control-broker-ready")
            assert ready.return_code == 0
            assert ready.stdout == "control-broker-ready\n"
        finally:
            await runtime.stop()

    asyncio.run(scenario())


def test_supervisor_recovers_when_command_kills_control_broker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _local_broker_runtime(monkeypatch, tmp_path)

    async def scenario() -> None:
        await runtime.start()
        try:
            broker_pid = runtime._broker_dir / "broker.pid"  # noqa: SLF001
            result = await runtime.exec(
                f'kill -9 "$(cat {shlex.quote(str(broker_pid))})"'
            )
            assert result.return_code == 125
            assert result.stderr is not None
            assert "broker recovered" in result.stderr

            ready = await runtime.exec("echo recovered-control-broker")
            assert ready.return_code == 0
            assert ready.stdout == "recovered-control-broker\n"
        finally:
            await runtime.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("teardown_method", ["stop", "cancel"])
def test_stop_and_cancel_do_not_restart_broker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    teardown_method: str,
) -> None:
    runtime = _local_broker_runtime(monkeypatch, tmp_path)

    async def scenario() -> None:
        await runtime.start()
        generation_path = runtime._broker_dir / "broker.generation"  # noqa: SLF001
        restart_path = runtime._broker_dir / "broker.restart_count"  # noqa: SLF001
        broker_pid_path = runtime._broker_dir / "broker.pid"  # noqa: SLF001
        broker_pid = int(broker_pid_path.read_text())
        assert generation_path.read_text().strip() == "1"
        assert restart_path.read_text().strip() == "0"

        await getattr(runtime, teardown_method)()
        await asyncio.sleep(0.2)

        assert runtime.destroyed
        assert generation_path.read_text().strip() == "1"
        assert restart_path.read_text().strip() == "0"
        assert not Path(f"/proc/{broker_pid}").exists()

    asyncio.run(scenario())


def test_broker_proxy_is_shared_and_loopback_bypasses_itself(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proxy_socket = tmp_path / "upstream-proxy.sock"
    upstream, upstream_thread = _serve_unix_echo(proxy_socket)
    port_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_socket.bind(("127.0.0.1", 0))
    proxy_port = int(port_socket.getsockname()[1])
    port_socket.close()

    runtime = _local_broker_runtime(
        monkeypatch,
        tmp_path,
        allow_internet=True,
        launch_environment={
            "POLAR_HTTP_PROXY_UDS": str(proxy_socket),
            "POLAR_HTTP_PROXY_PORT": str(proxy_port),
            "no_proxy": "metadata.internal,localhost",
            "NO_PROXY": "registry.internal",
        },
    )
    python_command = (
        "import os,socket;"
        f"s=socket.create_connection(('127.0.0.1',{proxy_port}),timeout=2);"
        "s.sendall(b'request');print(s.recv(4096).decode());s.close();"
        "print(os.environ['HTTP_PROXY']);print(os.environ['NO_PROXY']);"
        "print(os.environ.get('POLAR_HTTP_PROXY_UDS','missing'));"
        "print(os.environ.get('POLAR_HTTP_PROXY_BROKER_READY','missing'))"
    )

    async def scenario() -> None:
        await runtime.start()
        try:
            result = await runtime.exec(
                f"{shlex.quote(sys.executable)} -c {shlex.quote(python_command)}"
            )
            assert result.return_code == 0
            assert result.stdout is not None
            lines = result.stdout.splitlines()
            assert lines[0] == "echo:request"
            assert lines[1] == f"http://127.0.0.1:{proxy_port}"
            assert lines[2] == (
                "metadata.internal,localhost,registry.internal,127.0.0.1,::1"
            )
            assert lines[3:] == ["missing", "1"]
        finally:
            await runtime.stop()

    try:
        asyncio.run(scenario())
    finally:
        upstream.close()
        upstream_thread.join(timeout=2.0)


def test_offline_broker_child_cannot_restore_proxy_from_request() -> None:
    environment = apptainer_broker.child_environment(
        {
            "POLAR_ALLOW_INTERNET": "false",
            "HTTP_PROXY": "http://outer-proxy:3128",
        },
        {
            "POLAR_ALLOW_INTERNET": "true",
            "POLAR_HTTP_PROXY_UDS": "/escape/proxy.sock",
            "https_proxy": "http://escape-proxy:3128",
        },
        proxy_url=None,
        allow_internet=False,
    )

    assert environment["POLAR_ALLOW_INTERNET"] == "false"
    assert "POLAR_HTTP_PROXY_UDS" not in environment
    assert all(name not in environment for name in apptainer_broker._PROXY_ENV_NAMES)
