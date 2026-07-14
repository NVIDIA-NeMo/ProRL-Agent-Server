from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import signal
import shlex
import shutil
import tarfile
import threading

import pytest

from polar.runtime import apptainer
from polar.runtime.apptainer import ApptainerRuntime
from polar.runtime.base import (
    BaseRuntime,
    RuntimeContainmentError,
    RuntimeDestroyedError,
)
from polar.runtime.models import ExecResult, RuntimeSpec


def _runtime(
    monkeypatch,
    tmp_path: Path,
    *,
    direct: bool,
    persistent_broker: bool | None = None,
) -> ApptainerRuntime:
    for name in (
        "POLAR_APPTAINER_NO_MOUNT_HOSTFS",
        "POLAR_APPTAINER_NO_MOUNT_TMP",
        "POLAR_APPTAINER_CLEANENV",
        "POLAR_APPTAINER_ISOLATE_PID",
        "POLAR_APPTAINER_ISOLATE_IPC",
        "POLAR_APPTAINER_PERSISTENT_BROKER",
        "POLAR_APPTAINER_BROKER_CLEANUP_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(name, raising=False)
    if direct:
        monkeypatch.setenv("POLAR_APPTAINER_NO_INSTANCE", "1")
    else:
        monkeypatch.delenv("POLAR_APPTAINER_NO_INSTANCE", raising=False)
    if persistent_broker is not None:
        monkeypatch.setenv(
            "POLAR_APPTAINER_PERSISTENT_BROKER",
            "1" if persistent_broker else "0",
        )
    monkeypatch.setattr(
        ApptainerRuntime,
        "_resolve_binary",
        staticmethod(lambda: "/usr/bin/apptainer"),
    )
    return ApptainerRuntime(
        RuntimeSpec(
            backend="apptainer",
            image="/images/task.sif",
            workdir="/work",
            kwargs={"volumes": ["/host/cli:/opt/node:ro"]},
        ),
        "session-1",
        tmp_path / "session",
    )


def _direct_launch_args(runtime: ApptainerRuntime) -> list[str]:
    runtime._overlay_dir = runtime.session_dir / "overlay"  # noqa: SLF001
    runtime._overlay_dir.mkdir(parents=True, exist_ok=True)  # noqa: SLF001
    runtime._broker_dir.mkdir(parents=True, exist_ok=True)  # noqa: SLF001
    return runtime._broker_launch_args()  # noqa: SLF001


def _write_fake_proc_process(
    proc_root: Path,
    *,
    pid: int,
    ppid: int,
    session_id: int,
    start_time: int,
    environ: bytes = b"",
) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    # Fields after comm start at field 3. Put starttime at field 22/index 19.
    remainder = ["S", str(ppid), str(pid), str(session_id), *("0" for _ in range(15))]
    remainder.append(str(start_time))
    (process_dir / "stat").write_text(f"{pid} (runtime parent) {' '.join(remainder)}\n")
    (process_dir / "environ").write_bytes(environ)


def test_direct_broker_cleanup_timeout_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "POLAR_APPTAINER_BROKER_CLEANUP_TIMEOUT_SEC"
    monkeypatch.delenv(env_name, raising=False)
    assert apptainer._direct_broker_cleanup_timeout_seconds() == 15.0  # noqa: SLF001

    for valid, expected in (("1", 1.0), ("7.5", 7.5), ("60", 60.0)):
        monkeypatch.setenv(env_name, valid)
        assert (  # noqa: SLF001
            apptainer._direct_broker_cleanup_timeout_seconds() == expected
        )

    for invalid in ("0", "0.99", "60.01", "nan", "inf", "not-a-number"):
        monkeypatch.setenv(env_name, invalid)
        assert (  # noqa: SLF001
            apptainer._direct_broker_cleanup_timeout_seconds() == 15.0
        )


def test_broker_rpc_concurrency_is_independent_of_blocked_default_executor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime._broker_dir.mkdir(parents=True)  # noqa: SLF001
    request_count = 40
    executor_started = threading.Event()
    executor_release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)

    def block_default_executor() -> None:
        executor_started.set()
        executor_release.wait()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)
        blocker = loop.run_in_executor(None, block_default_executor)
        while not executor_started.is_set():
            await asyncio.sleep(0)

        all_requests_received = asyncio.Event()
        release_responses = asyncio.Event()
        handler_tasks: set[asyncio.Task[None]] = set()
        received = 0

        async def handle(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            nonlocal received
            task = asyncio.current_task()
            assert task is not None
            handler_tasks.add(task)
            try:
                assert await reader.readline() == b'{"operation":"ping"}\n'
                received += 1
                if received == request_count:
                    all_requests_received.set()
                await release_responses.wait()
                writer.write(b'{"ok":true}\n')
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_unix_server(
            handle,
            path=str(runtime._broker_socket),  # noqa: SLF001
        )
        requests = [
            asyncio.create_task(
                runtime._broker_rpc(  # noqa: SLF001
                    {"operation": "ping"},
                    socket_timeout=1.0,
                )
            )
            for _ in range(request_count)
        ]
        try:
            # This is both above the usual 32-worker default and runs while
            # the deliberately one-worker default executor is fully blocked.
            await asyncio.wait_for(all_requests_received.wait(), timeout=1.0)
            assert received == request_count
            assert not blocker.done()
            release_responses.set()
            assert await asyncio.gather(*requests) == [{"ok": True}] * request_count
        finally:
            release_responses.set()
            executor_release.set()
            await blocker
            await asyncio.gather(*requests, return_exceptions=True)
            server.close()
            await server.wait_closed()
            if handler_tasks:
                await asyncio.gather(*handler_tasks, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        executor_release.set()
        executor.shutdown(wait=True)


def test_broker_rpc_times_out_waiting_for_response(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime._broker_dir.mkdir(parents=True)  # noqa: SLF001

    async def scenario() -> None:
        request_received = asyncio.Event()
        release_handler = asyncio.Event()

        async def handle(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            try:
                await reader.readline()
                request_received.set()
                await release_handler.wait()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_unix_server(
            handle,
            path=str(runtime._broker_socket),  # noqa: SLF001
        )
        try:
            with pytest.raises(TimeoutError):
                await runtime._broker_rpc(  # noqa: SLF001
                    {"operation": "ping"},
                    socket_timeout=0.05,
                )
            assert request_received.is_set()
        finally:
            release_handler.set()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_broker_rpc_rejects_oversized_response(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime._broker_dir.mkdir(parents=True)  # noqa: SLF001

    async def scenario() -> None:
        handler_done = asyncio.Event()

        async def handle(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            try:
                await reader.readline()
                writer.write(b"x" * (1024 * 1024 + 1))
                await writer.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                handler_done.set()

        server = await asyncio.start_unix_server(
            handle,
            path=str(runtime._broker_socket),  # noqa: SLF001
        )
        try:
            with pytest.raises(RuntimeError, match="response exceeded 1 MiB"):
                await runtime._broker_rpc(  # noqa: SLF001
                    {"operation": "ping"},
                    socket_timeout=1.0,
                )
            await asyncio.wait_for(handler_done.wait(), timeout=1.0)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_broker_rpc_reports_disconnect_before_response(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime._broker_dir.mkdir(parents=True)  # noqa: SLF001

    async def scenario() -> None:
        async def handle(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            await reader.readline()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(
            handle,
            path=str(runtime._broker_socket),  # noqa: SLF001
        )
        try:
            with pytest.raises(
                apptainer._BrokerDisconnectedError,  # noqa: SLF001
                match="without a response",
            ):
                await runtime._broker_rpc(  # noqa: SLF001
                    {"operation": "ping"},
                    socket_timeout=1.0,
                )
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_cancelling_broker_rpc_closes_async_unix_stream(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime._broker_dir.mkdir(parents=True)  # noqa: SLF001

    async def scenario() -> None:
        request_received = asyncio.Event()
        peer_disconnected = asyncio.Event()

        async def handle(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            try:
                await reader.readline()
                request_received.set()
                assert await reader.read() == b""
                peer_disconnected.set()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_unix_server(
            handle,
            path=str(runtime._broker_socket),  # noqa: SLF001
        )
        request = asyncio.create_task(
            runtime._broker_rpc(  # noqa: SLF001
                {"operation": "ping"},
                socket_timeout=None,
            )
        )
        try:
            await asyncio.wait_for(request_received.wait(), timeout=1.0)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            await asyncio.wait_for(peer_disconnected.wait(), timeout=1.0)
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("response", "error_type", "message"),
    [
        (b"not-json\n", ValueError, None),
        (b"[]\n", RuntimeError, "non-object response"),
        (b'{"ok":false,"error":"synthetic"}\n', RuntimeError, "synthetic"),
    ],
)
def test_broker_rpc_validates_json_response(
    monkeypatch,
    tmp_path: Path,
    response: bytes,
    error_type: type[Exception],
    message: str | None,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime._broker_dir.mkdir(parents=True)  # noqa: SLF001

    async def scenario() -> None:
        async def handle(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            await reader.readline()
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(
            handle,
            path=str(runtime._broker_socket),  # noqa: SLF001
        )
        try:
            expectation = pytest.raises(error_type)
            if message is not None:
                expectation = pytest.raises(error_type, match=message)
            with expectation:
                await runtime._broker_rpc(  # noqa: SLF001
                    {"operation": "ping"},
                    socket_timeout=1.0,
                )
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_direct_broker_process_snapshot_tracks_escaped_session_tree(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    session_dir = tmp_path / "session"
    sibling_session = tmp_path / "session-sibling"
    runtime_pid = 8_100_001
    fuse_pid = 8_100_002
    setsid_child_pid = 8_100_003
    grandchild_pid = 8_100_004
    sibling_pid = 8_200_001
    sibling_child_pid = 8_200_002

    # Split ENGINE_CONFIG exactly as Apptainer may do for a large config.
    session_json = f'{{"bindpath":[{{"source":"{session_dir}"}}]}}'.encode()
    split = len(session_json) // 2
    runtime_environ = b"\0".join(
        (
            b"ENGINE_CONFIG_CHUNKS=2",
            b"ENGINE_CONFIG1=" + session_json[:split],
            b"ENGINE_CONFIG2=" + session_json[split:],
        )
    )
    _write_fake_proc_process(
        proc_root,
        pid=runtime_pid,
        ppid=1,
        session_id=runtime_pid,
        start_time=101,
        environ=runtime_environ,
    )
    _write_fake_proc_process(
        proc_root,
        pid=fuse_pid,
        ppid=runtime_pid,
        session_id=runtime_pid,
        start_time=102,
    )
    # A workload may call setsid, so SID membership alone is insufficient.
    _write_fake_proc_process(
        proc_root,
        pid=setsid_child_pid,
        ppid=fuse_pid,
        session_id=setsid_child_pid,
        start_time=103,
    )
    _write_fake_proc_process(
        proc_root,
        pid=grandchild_pid,
        ppid=setsid_child_pid,
        session_id=setsid_child_pid,
        start_time=104,
    )
    _write_fake_proc_process(
        proc_root,
        pid=sibling_pid,
        ppid=1,
        session_id=sibling_pid,
        start_time=201,
        environ=(f'ENGINE_CONFIG1={{"bindpath":[{{"source":"{sibling_session}"}}]}}').encode(),
    )
    _write_fake_proc_process(
        proc_root,
        pid=sibling_child_pid,
        ppid=sibling_pid,
        session_id=sibling_pid,
        start_time=202,
    )

    sessions, processes = apptainer._direct_broker_process_snapshot(  # noqa: SLF001
        session_dir,
        proc_root=proc_root,
    )

    assert sessions == {runtime_pid}
    assert processes == {
        runtime_pid: 101,
        fuse_pid: 102,
        setsid_child_pid: 103,
        grandchild_pid: 104,
    }


def test_containment_snapshot_fails_closed_when_procfs_is_unreadable(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeContainmentError, match="could not read procfs"):
        apptainer._direct_broker_process_snapshot(  # noqa: SLF001
            tmp_path / "missing-proc",
            proc_root=tmp_path / "missing-proc",
        )


def test_containment_snapshot_ignores_process_that_exits_during_stat_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    pid = 8_225_001
    _write_fake_proc_process(
        proc_root,
        pid=pid,
        ppid=1,
        session_id=pid,
        start_time=1,
    )
    process_stat = proc_root / str(pid) / "stat"
    real_read_text = Path.read_text

    def exited_stat(path: Path, *args, **kwargs) -> str:
        if path == process_stat:
            raise ProcessLookupError("synthetic procfs ESRCH")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", exited_stat)

    sessions, processes = apptainer._direct_broker_process_snapshot(  # noqa: SLF001
        tmp_path / "session",
        proc_root=proc_root,
    )

    assert sessions == set()
    assert processes == {}


def test_containment_ownership_scan_ignores_process_that_exits_during_stat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    pid = 8_225_002
    _write_fake_proc_process(
        proc_root,
        pid=pid,
        ppid=1,
        session_id=pid,
        start_time=1,
    )
    process_dir = proc_root / str(pid)
    real_stat = Path.stat

    def exited_process(path: Path, *args, **kwargs):
        if path == process_dir:
            raise ProcessLookupError("synthetic procfs ESRCH")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", exited_process)

    sessions, processes = apptainer._direct_broker_process_snapshot(  # noqa: SLF001
        tmp_path / "session",
        proc_root=proc_root,
    )

    assert sessions == set()
    assert processes == {}


def test_containment_ownership_scan_ignores_process_that_exits_during_environ_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    pid = 8_225_003
    _write_fake_proc_process(
        proc_root,
        pid=pid,
        ppid=1,
        session_id=pid,
        start_time=1,
    )
    process_environ = proc_root / str(pid) / "environ"
    real_read_bytes = Path.read_bytes

    def exited_environ(path: Path) -> bytes:
        if path == process_environ:
            raise ProcessLookupError("synthetic procfs ESRCH")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", exited_environ)

    sessions, processes = apptainer._direct_broker_process_snapshot(  # noqa: SLF001
        tmp_path / "session",
        proc_root=proc_root,
    )

    assert sessions == set()
    assert processes == {}


def test_containment_ownership_scan_fails_closed_on_unreadable_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_fake_proc_process(
        proc_root,
        pid=8_250_001,
        ppid=1,
        session_id=8_250_001,
        start_time=1,
    )
    real_read_bytes = Path.read_bytes

    def unreadable_environ(path: Path) -> bytes:
        if path.name == "environ":
            raise PermissionError("synthetic hidepid")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable_environ)
    with pytest.raises(RuntimeContainmentError, match="process environment"):
        apptainer._direct_broker_process_snapshot(  # noqa: SLF001
            tmp_path / "session",
            proc_root=proc_root,
        )


def test_force_stop_refuses_to_cancel_live_owner_before_sid_is_known(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)

    class LiveOwnerTask:
        def done(self) -> bool:
            return False

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def scenario() -> None:
        task = LiveOwnerTask()
        runtime._broker_task = task  # noqa: SLF001
        monkeypatch.setattr(
            apptainer,
            "_direct_broker_process_snapshot",
            lambda *_args, **_kwargs: (set(), {}),
        )
        monkeypatch.setattr(apptainer.asyncio, "to_thread", immediate_to_thread)
        with pytest.raises(RuntimeContainmentError, match="identify the live"):
            await runtime._force_stop_broker()  # noqa: SLF001
        assert runtime._broker_task is task  # noqa: SLF001
        assert not task.done()

    asyncio.run(scenario())


def test_direct_broker_cleanup_uses_captured_sid_after_parent_exits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    runtime_pid = 8_300_001
    fuse_pid = 8_300_002
    setsid_child_pid = 8_300_003
    sibling_pid = 8_400_001
    owned_pids = {fuse_pid, setsid_child_pid}
    # The runtime parent carrying ENGINE_CONFIG has already exited. Its FUSE
    # helper is now under PID 1 but retains the runtime parent's SID.
    _write_fake_proc_process(
        proc_root,
        pid=fuse_pid,
        ppid=1,
        session_id=runtime_pid,
        start_time=302,
    )
    _write_fake_proc_process(
        proc_root,
        pid=setsid_child_pid,
        ppid=fuse_pid,
        session_id=setsid_child_pid,
        start_time=303,
    )
    # This sibling deliberately has a path with the target as a prefix.
    sibling_session = Path(f"{runtime.session_dir}-sibling")
    _write_fake_proc_process(
        proc_root,
        pid=sibling_pid,
        ppid=1,
        session_id=sibling_pid,
        start_time=401,
        environ=(f'ENGINE_CONFIG1={{"bindpath":[{{"source":"{sibling_session}"}}]}}').encode(),
    )
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if signum == signal.SIGKILL:
            shutil.rmtree(proc_root / str(pid), ignore_errors=True)

    monkeypatch.setattr(apptainer.os, "kill", fake_kill)

    runtime._cleanup_direct_broker_processes(  # noqa: SLF001
        proc_root=proc_root,
        term_grace_seconds=0,
        kill_timeout_seconds=0.2,
        known_sessions={runtime_pid},
    )

    assert {pid for pid, signum in signals if signum == signal.SIGTERM} == owned_pids
    assert {pid for pid, signum in signals if signum == signal.SIGKILL} == owned_pids
    assert all(pid != sibling_pid for pid, _ in signals)
    assert (proc_root / str(sibling_pid)).exists()


def test_force_stop_scans_for_runtime_after_launcher_task_is_gone(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime._broker_was_started = True  # noqa: SLF001
    cleanup_called = threading.Event()
    observed_sessions: list[set[int]] = []

    def fake_snapshot(*_args, **_kwargs):
        return {8_500_001}, {}

    def fake_cleanup(*, known_sessions: set[int]) -> None:
        observed_sessions.append(known_sessions)
        cleanup_called.set()

    monkeypatch.setattr(apptainer, "_direct_broker_process_snapshot", fake_snapshot)
    runtime._cleanup_direct_broker_processes = fake_cleanup  # type: ignore[method-assign]  # noqa: SLF001,E501

    asyncio.run(runtime._force_stop_broker())  # noqa: SLF001

    assert cleanup_called.is_set()
    assert observed_sessions == [{8_500_001}]


def test_failed_cancel_keeps_destruction_unproven_and_stop_retries_cleanup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    cleanup_attempts = 0

    async def flaky_force_stop() -> None:
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise RuntimeContainmentError("synthetic residual process")

    runtime._force_stop_broker = flaky_force_stop  # type: ignore[method-assign]  # noqa: SLF001,E501

    with pytest.raises(RuntimeContainmentError, match="residual process"):
        asyncio.run(runtime.cancel())
    assert runtime.destroyed is False
    with pytest.raises(RuntimeDestroyedError, match="already destroyed"):
        asyncio.run(runtime.exec("echo must-not-run-during-failed-teardown"))

    asyncio.run(runtime.stop())

    assert cleanup_attempts == 2
    assert runtime.destroyed is True


def test_instance_mode_remains_the_default(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=False)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def fake_run(*args: str, **kwargs):
        calls.append((args, kwargs))
        return 0, None, None

    runtime._run_local_command = fake_run  # type: ignore[method-assign]
    asyncio.run(runtime.start())
    asyncio.run(runtime.exec("echo ok"))
    asyncio.run(runtime.stop())

    assert calls[0][0][1:3] == ("instance", "start")
    assert "--overlay" in calls[0][0]
    assert "/images/task.sif" in calls[0][0]
    assert calls[1][0][1:3] == ("exec", f"instance://{runtime.runtime_id}")
    assert calls[2][0][1:3] == ("instance", "stop")
    assert all(kwargs["cwd"] == "/" for _, kwargs in calls)


def test_direct_exec_reuses_overlay_and_skips_instance_lifecycle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    broker_calls: list[tuple[str, str | None, dict[str, str], float | None]] = []

    async def fake_broker_exec(command, *, cwd, env, timeout_sec):
        broker_calls.append((command, cwd, env, timeout_sec))
        return 0, "ok\n", None

    runtime._broker_exec = fake_broker_exec  # type: ignore[method-assign]
    launch = _direct_launch_args(runtime)
    result = asyncio.run(runtime.exec("echo ok", env={"HOME": "/root"}))

    assert launch[:2] == ["/usr/bin/apptainer", "exec"]
    assert "instance" not in launch
    assert "--overlay" in launch
    assert str(tmp_path / "session" / "overlay") in launch
    assert "/images/task.sif" in launch
    assert "/host/cli:/opt/node:ro" in launch
    assert launch[-3:-1] == ["bash", "-lc"]
    assert "apptainer_broker_supervisor.sh" in launch[-1]
    assert result.stdout == "ok\n"
    assert broker_calls == [
        (
            "export HOME=/root; echo ok",
            "/work",
            {"HOME": "/root"},
            None,
        )
    ]


def test_direct_broker_control_sources_are_read_only_and_not_redirectable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime.spec = runtime.spec.model_copy(
        update={
            "env": {
                "POLAR_APPTAINER_BROKER_RUNTIME_DIR": "/task/runtime",
                "POLAR_APPTAINER_BROKER_SCRIPT": "/task/broker.py",
                "POLAR_APPTAINER_BROKER_INTERPRETER_FILE": "/task/python",
                "POLAR_APPTAINER_BROKER_PROCESS_NAME": "task-broker",
            }
        }
    )

    launch = _direct_launch_args(runtime)
    trusted_source = Path(apptainer.__file__).resolve().parent
    assert (
        f"{trusted_source}:{apptainer._BROKER_TRUSTED_RUNTIME_DIR}:ro"  # noqa: SLF001
        in launch
    )
    assert not (runtime._broker_dir / "apptainer_broker.py").exists()  # noqa: SLF001
    assert not (runtime._broker_dir / "apptainer_broker_supervisor.sh").exists()  # noqa: SLF001,E501
    assert (
        f"POLAR_APPTAINER_BROKER_RUNTIME_DIR={apptainer._BROKER_RUNTIME_DIR}"  # noqa: SLF001
        in launch
    )
    assert (
        "POLAR_APPTAINER_BROKER_SCRIPT="
        f"{apptainer._BROKER_TRUSTED_RUNTIME_SCRIPT}"  # noqa: SLF001
        in launch
    )
    assert "POLAR_APPTAINER_BROKER_PROCESS_NAME=polar-broker" in launch
    assert apptainer._BROKER_TRUSTED_RUNTIME_SUPERVISOR in launch[-1]  # noqa: SLF001
    assert "/task/broker.py" not in launch
    assert "/task/runtime" not in launch


def test_legacy_direct_exec_reuses_overlay_without_broker_lifecycle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        monkeypatch,
        tmp_path,
        direct=True,
        persistent_broker=False,
    )
    runtime.spec = runtime.spec.model_copy(
        update={
            "direct_exec_init_command": ". /image/service-hook.sh",
            "env": {"HOME": "/polar/session/home"},
        }
    )
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def fake_run(*args: str, **kwargs):
        calls.append((args, kwargs))
        return 0, "ok\n", None

    runtime._run_local_command = fake_run  # type: ignore[method-assign]

    async def exercise() -> tuple[ExecResult, ExecResult]:
        await runtime.start()
        first = await runtime.exec("echo first")
        second = await runtime.exec("echo second", cwd="/other")
        await runtime.stop()
        return first, second

    first, second = asyncio.run(exercise())

    assert first.stdout == "ok\n"
    assert second.stdout == "ok\n"
    assert runtime.destroyed
    assert len(calls) == 3  # validation + two commands; stop starts no process
    validation, first_exec, second_exec = calls
    overlay = str(tmp_path / "session" / "overlay")
    assert validation[0][:2] == ("/usr/bin/apptainer", "exec")
    assert validation[0][-2:] == ("/images/task.sif", "true")
    for args, kwargs in calls:
        assert "instance" not in args
        assert args[args.index("--overlay") + 1] == overlay
        assert kwargs["cwd"] == "/"
        assert "apptainer_broker_supervisor.sh" not in " ".join(args)
    assert first_exec[0][-3:-1] == ("bash", "-lc")
    assert second_exec[0][-3:-1] == ("bash", "-lc")
    assert first_exec[0][-1].count("service-hook.sh") == 1
    assert second_exec[0][-1].count("service-hook.sh") == 1
    assert "cd /work && echo first" in first_exec[0][-1]
    assert "cd /other && echo second" in second_exec[0][-1]


def test_legacy_direct_cancel_uses_active_process_cleanup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        monkeypatch,
        tmp_path,
        direct=True,
        persistent_broker=False,
    )
    base_cancel_called = False

    async def fake_base_cancel(self: BaseRuntime) -> None:
        nonlocal base_cancel_called
        base_cancel_called = True
        self._destroyed = True  # noqa: SLF001

    async def forbidden_broker_cleanup() -> None:
        raise AssertionError("legacy direct cancellation must not use broker cleanup")

    monkeypatch.setattr(BaseRuntime, "cancel", fake_base_cancel)
    runtime._force_stop_broker = forbidden_broker_cleanup  # type: ignore[method-assign]  # noqa: SLF001,E501

    asyncio.run(runtime.cancel())

    assert base_cancel_called
    assert runtime.destroyed


def test_legacy_direct_upload_and_download_use_fresh_exec(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        monkeypatch,
        tmp_path,
        direct=True,
        persistent_broker=False,
    )
    runtime._overlay_dir = runtime.session_dir / "overlay"  # noqa: SLF001
    runtime._overlay_dir.mkdir(parents=True)  # noqa: SLF001
    runtime._copy_to_bind_mount = lambda *_args: False  # type: ignore[method-assign]
    runtime._copy_from_bind_mount = lambda *_args: False  # type: ignore[method-assign]

    source_file = tmp_path / "source.txt"
    source_file.write_text("upload-file")
    source_dir = tmp_path / "source-dir"
    source_dir.mkdir()
    (source_dir / "nested.txt").write_text("upload-dir")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def fake_run(*args: str, **kwargs):
        calls.append((args, kwargs))
        command = args[-1]
        if "tar -cf " in command:
            tokens = shlex.split(command)
            runtime_archive = tokens[tokens.index("-cf") + 1]
            archive_path = runtime._broker_transfers_dir / Path(runtime_archive).name  # noqa: SLF001,E501
            payload_name = (
                "download.txt" if "download.txt" in command else "nested.txt"
            )
            payload = b"download-file" if payload_name == "download.txt" else b"download-dir"
            with tarfile.open(archive_path, mode="w") as archive:
                info = tarfile.TarInfo(payload_name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return 0, None, None

    runtime._run_local_command = fake_run  # type: ignore[method-assign]
    downloaded_file = tmp_path / "downloaded.txt"
    downloaded_dir = tmp_path / "downloaded-dir"

    async def transfer_all() -> None:
        await runtime.upload_file(str(source_file), "/remote/upload.txt")
        await runtime.upload_dir(str(source_dir), "/remote/upload-dir")
        await runtime.download_file("/remote/download.txt", str(downloaded_file))
        await runtime.download_dir("/remote/download-dir", str(downloaded_dir))

    asyncio.run(transfer_all())

    assert downloaded_file.read_text() == "download-file"
    assert (downloaded_dir / "nested.txt").read_text() == "download-dir"
    assert len(calls) == 6
    overlay = str(runtime._overlay_dir)  # noqa: SLF001
    for args, kwargs in calls:
        assert args[:2] == ("/usr/bin/apptainer", "exec")
        assert args[args.index("--overlay") + 1] == overlay
        assert "instance" not in args
        assert "apptainer_broker_supervisor.sh" not in " ".join(args)
        assert kwargs["cwd"] == "/"


def test_direct_exec_initializer_runs_once_after_broker_is_pinned(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime.spec = runtime.spec.model_copy(
        update={
            "direct_exec_init_command": (
                "if [ -f /.singularity.d/env/99-rogue.sh ]; then "
                ". /.singularity.d/env/99-rogue.sh; fi"
            ),
            "env": {
                "HOME": "/polar/session/home",
                "PATH": "/portable/bin:/usr/bin:/bin",
                "POLAR_APPTAINER_BROKER_PYTHON": "/portable/venv/bin/python",
            },
        }
    )
    broker_commands: list[str] = []

    async def fake_broker_exec(command, **_kwargs):
        broker_commands.append(command)
        return 0, "", ""

    async def fake_start_direct_broker() -> None:
        return None

    runtime._broker_exec = fake_broker_exec  # type: ignore[method-assign]
    runtime._start_direct_broker = fake_start_direct_broker  # type: ignore[method-assign]
    launch = _direct_launch_args(runtime)
    asyncio.run(runtime.start())
    asyncio.run(runtime.exec("echo ok"))
    asyncio.run(runtime.exec("echo again"))

    assert "99-rogue.sh" not in launch[-1]
    assert not any("POLAR_APPTAINER_BROKER_PYTHON=" in argument for argument in launch)
    assert runtime._broker_interpreter_file.read_text() == "/portable/venv/bin/python\n"  # noqa: SLF001
    assert "apptainer_broker_supervisor.sh" in launch[-1]
    assert "cd /work &&" not in launch[-1]
    assert "export HOME=/polar/session/home" in launch[-1]
    assert "export PATH=/portable/bin:/usr/bin:/bin" in launch[-1]
    assert len(broker_commands) == 3
    assert broker_commands[0].count("99-rogue.sh") == 2  # test and source in one hook
    assert broker_commands[1].endswith("echo ok")
    assert broker_commands[2].endswith("echo again")
    assert all("99-rogue.sh" not in command for command in broker_commands[1:])


def test_direct_exec_initializer_is_not_repeated_in_persistent_instance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=False)
    runtime.spec = runtime.spec.model_copy(
        update={"direct_exec_init_command": ". /image/service-hook.sh"}
    )
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, **_kwargs):
        calls.append(args)
        return 0, "", ""

    runtime._run_local_command = fake_run  # type: ignore[method-assign]
    asyncio.run(runtime.start())
    asyncio.run(runtime.exec("echo ok"))
    asyncio.run(runtime.stop())

    command = calls[1]
    assert command[-3:] == ("bash", "-lc", "cd /work && echo ok")
    assert "service-hook.sh" not in command


def test_direct_exec_initializer_rejects_blank_command() -> None:
    with pytest.raises(ValueError, match="direct_exec_init_command must be non-empty"):
        RuntimeSpec(image="/images/task.sif", direct_exec_init_command="  ")


@pytest.mark.parametrize(
    ("allow_internet", "expected_network"),
    [(True, None), (False, "none")],
)
def test_host_fallback_keeps_gateway_uds_when_offline_runtime_forces_none(
    monkeypatch,
    tmp_path: Path,
    allow_internet: bool,
    expected_network: str | None,
) -> None:
    monkeypatch.setenv("POLAR_APPTAINER_NO_INSTANCE", "1")
    monkeypatch.setattr(
        ApptainerRuntime,
        "_resolve_binary",
        staticmethod(lambda: "/usr/bin/apptainer"),
    )
    gateway_bind = "/tmp/polar-job/uds/gateway:/polar/gateway:ro"
    proxy_bind = "/tmp/polar-job/uds/proxy:/polar/proxy:ro"
    runtime = ApptainerRuntime(
        RuntimeSpec(
            backend="apptainer",
            image="/images/task.sif",
            network="host",
            allow_internet=allow_internet,
            env={"POLAR_GATEWAY_UDS": "/polar/gateway/gateway.sock"},
            internet_volumes=[proxy_bind],
            kwargs={"volumes": [gateway_bind]},
        ),
        "session-host-fallback",
        tmp_path / "session",
    )
    launch = _direct_launch_args(runtime)
    assert gateway_bind in launch
    assert (proxy_bind in launch) is allow_internet
    assert runtime.spec.env["POLAR_GATEWAY_UDS"] == "/polar/gateway/gateway.sock"
    if expected_network is None:
        assert "--net" not in launch
        assert "--network" not in launch
    else:
        network_index = launch.index("--network")
        assert launch[network_index + 1] == expected_network


def test_runtime_can_disable_automatic_hostfs_mounts(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    monkeypatch.setenv("POLAR_APPTAINER_NO_MOUNT_HOSTFS", "1")
    options = _direct_launch_args(runtime)
    no_mount_index = options.index("--no-mount")
    assert options[no_mount_index + 1] == "cwd,home,bind-paths,hostfs"


def test_runtime_always_disables_automatic_cwd_and_home_mounts(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    options = _direct_launch_args(runtime)
    no_mount_index = options.index("--no-mount")
    assert options[no_mount_index + 1] == "cwd,home,bind-paths"


def test_runtime_can_isolate_outer_tmp_mount(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    monkeypatch.setenv("POLAR_APPTAINER_NO_MOUNT_HOSTFS", "1")
    monkeypatch.setenv("POLAR_APPTAINER_NO_MOUNT_TMP", "1")
    options = _direct_launch_args(runtime)
    no_mount_index = options.index("--no-mount")
    assert options[no_mount_index + 1] == "cwd,home,bind-paths,hostfs,tmp"
    assert "--pid" not in options
    assert "--ipc" not in options


def test_runtime_can_isolate_pid_and_ipc_namespaces(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    monkeypatch.setenv("POLAR_APPTAINER_ISOLATE_PID", "1")
    monkeypatch.setenv("POLAR_APPTAINER_ISOLATE_IPC", "1")
    options = _direct_launch_args(runtime)
    assert "--pid" in options
    assert "--ipc" in options


def test_runtime_can_clean_outer_scheduler_environment(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    monkeypatch.setenv("POLAR_APPTAINER_CLEANENV", "1")
    broker_calls: list[dict[str, str]] = []

    async def fake_broker_exec(_command, *, env, **_kwargs):
        broker_calls.append(env)
        return 0, "", ""

    runtime._broker_exec = fake_broker_exec  # type: ignore[method-assign]
    launch = _direct_launch_args(runtime)
    asyncio.run(
        runtime.exec(
            "python -V",
            env={"OMP_NUM_THREADS": "1", "https_proxy": "http://proxy:3128"},
        )
    )

    clean_index = launch.index("--cleanenv")
    image_index = launch.index("/images/task.sif")
    assert clean_index < image_index
    assert broker_calls == [{"OMP_NUM_THREADS": "1", "https_proxy": "http://proxy:3128"}]


def test_destructive_task_cannot_inherit_shared_launcher_cwd(monkeypatch, tmp_path: Path) -> None:
    shared_data_root = tmp_path / "shared-data"
    shared_data_root.mkdir()
    sentinel = shared_data_root / "must-survive"
    sentinel.write_text("safe")
    monkeypatch.chdir(shared_data_root)

    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def fake_run(*args: str, **kwargs):
        calls.append((args, kwargs))
        return 0, "", ""

    runtime._run_local_command = fake_run  # type: ignore[method-assign]

    async def fake_ready(_timeout: float) -> None:
        await asyncio.sleep(0)

    runtime._wait_for_broker_ready = fake_ready  # type: ignore[method-assign]
    asyncio.run(runtime.start())

    for args, kwargs in calls:
        no_mount_index = args.index("--no-mount")
        disabled_mounts = args[no_mount_index + 1].split(",")
        assert "cwd" in disabled_mounts
        assert "home" in disabled_mounts
        assert "bind-paths" in disabled_mounts
        assert kwargs["cwd"] == "/"
        assert kwargs["cwd"] != shared_data_root
    assert sentinel.read_text() == "safe"


def test_broker_start_gate_is_held_until_socket_readiness(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POLAR_APPTAINER_BROKER_START_CONCURRENCY", "2")
    monkeypatch.setenv(
        "POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY",
        "8",
    )
    active = 0
    peak_active = 0
    entered = 0
    first_wave_ready = asyncio.Event()
    release = asyncio.Event()

    async def exercise() -> None:
        nonlocal active, peak_active, entered
        runtimes = [
            _runtime(monkeypatch, tmp_path / str(index), direct=True) for index in range(6)
        ]

        def readiness_waiter():
            async def wait_until_ready() -> None:
                nonlocal active, peak_active, entered
                active += 1
                entered += 1
                peak_active = max(peak_active, active)
                if entered == 2:
                    first_wave_ready.set()
                try:
                    await release.wait()
                finally:
                    active -= 1

            return wait_until_ready

        for runtime in runtimes:
            runtime._start_direct_broker_once = readiness_waiter()  # type: ignore[method-assign]  # noqa: SLF001,E501

        tasks = [
            asyncio.create_task(runtime._start_direct_broker())  # noqa: SLF001
            for runtime in runtimes
        ]
        async with asyncio.timeout(1.0):
            await first_wave_ready.wait()
        await asyncio.sleep(0.02)
        assert entered == 2
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(exercise())

    assert peak_active == 2
    assert entered == 6


def test_broker_start_per_image_gate_limits_equivalent_image_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POLAR_APPTAINER_BROKER_START_CONCURRENCY", "8")
    monkeypatch.setenv(
        "POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY",
        "2",
    )
    active = 0
    peak_active = 0
    entered = 0
    first_wave_ready = asyncio.Event()
    release = asyncio.Event()

    async def exercise() -> None:
        nonlocal active, peak_active, entered
        image_dir = tmp_path / "images"
        runtimes = [
            _runtime(monkeypatch, tmp_path / f"session-{index}", direct=True) for index in range(6)
        ]
        for index, runtime in enumerate(runtimes):
            if index % 2:
                runtime.spec.image = str(image_dir / "unused" / ".." / "task.sif")
            else:
                runtime.spec.image = str(image_dir / "task.sif")

            async def wait_until_ready() -> None:
                nonlocal active, peak_active, entered
                active += 1
                entered += 1
                peak_active = max(peak_active, active)
                if entered == 2:
                    first_wave_ready.set()
                try:
                    await release.wait()
                finally:
                    active -= 1

            runtime._start_direct_broker_once = wait_until_ready  # type: ignore[method-assign]  # noqa: SLF001,E501

        tasks = [
            asyncio.create_task(runtime._start_direct_broker())  # noqa: SLF001
            for runtime in runtimes
        ]
        async with asyncio.timeout(1.0):
            await first_wave_ready.wait()
        await asyncio.sleep(0.02)
        assert entered == 2
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(exercise())

    assert peak_active == 2
    assert entered == 6


def test_same_image_waiter_does_not_consume_different_image_aggregate_permit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POLAR_APPTAINER_BROKER_START_CONCURRENCY", "2")
    monkeypatch.setenv(
        "POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY",
        "1",
    )
    entered: list[str] = []
    first_entered = asyncio.Event()
    different_images_entered = asyncio.Event()
    release = asyncio.Event()

    async def exercise() -> None:
        first = _runtime(monkeypatch, tmp_path / "first", direct=True)
        same_image_waiter = _runtime(monkeypatch, tmp_path / "second", direct=True)
        different_image = _runtime(monkeypatch, tmp_path / "third", direct=True)
        shared_image = str(tmp_path / "images" / "shared.sif")
        first.spec.image = shared_image
        same_image_waiter.spec.image = shared_image
        different_image.spec.image = str(tmp_path / "images" / "different.sif")

        def readiness_waiter(name: str):
            async def wait_until_ready() -> None:
                entered.append(name)
                if name == "first":
                    first_entered.set()
                if {"first", "different"}.issubset(entered):
                    different_images_entered.set()
                await release.wait()

            return wait_until_ready

        first._start_direct_broker_once = readiness_waiter("first")  # type: ignore[method-assign]  # noqa: SLF001,E501
        same_image_waiter._start_direct_broker_once = readiness_waiter("same")  # type: ignore[method-assign]  # noqa: SLF001,E501
        different_image._start_direct_broker_once = readiness_waiter("different")  # type: ignore[method-assign]  # noqa: SLF001,E501

        tasks = [asyncio.create_task(first._start_direct_broker())]  # noqa: SLF001
        await first_entered.wait()
        tasks.append(asyncio.create_task(same_image_waiter._start_direct_broker()))  # noqa: SLF001,E501
        await asyncio.sleep(0.02)
        tasks.append(asyncio.create_task(different_image._start_direct_broker()))  # noqa: SLF001,E501
        try:
            async with asyncio.timeout(1.0):
                await different_images_entered.wait()
            assert entered == ["first", "different"]
        finally:
            release.set()
            await asyncio.gather(*tasks)

    asyncio.run(exercise())

    assert entered == ["first", "different", "same"]


def test_broker_start_per_image_gate_rejects_loop_config_mutation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_name = "POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY"
    monkeypatch.setenv(env_name, "2")

    async def exercise() -> None:
        image = tmp_path / "images" / "task.sif"
        gate = apptainer._broker_start_per_image_gate(str(image))  # noqa: SLF001
        equivalent_gate = apptainer._broker_start_per_image_gate(  # noqa: SLF001
            str(image.parent / "." / image.name)
        )
        assert equivalent_gate is gate

        monkeypatch.setenv(env_name, "3")
        with pytest.raises(RuntimeError, match="changed from 2 to 3"):
            apptainer._broker_start_per_image_gate(  # noqa: SLF001
                str(tmp_path / "images" / "other.sif")
            )

    asyncio.run(exercise())


def test_broker_start_gate_cleans_failed_process_before_releasing_permit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POLAR_APPTAINER_BROKER_START_CONCURRENCY", "1")
    first = _runtime(monkeypatch, tmp_path / "first", direct=True)
    second = _runtime(monkeypatch, tmp_path / "second", direct=True)
    failure_seen = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    second_entered = asyncio.Event()

    async def fail_once() -> None:
        failure_seen.set()
        raise RuntimeError("synthetic mount failure")

    async def slow_cleanup() -> None:
        cleanup_started.set()
        await allow_cleanup.wait()

    async def enter_second() -> None:
        second_entered.set()

    first._start_direct_broker_once = fail_once  # type: ignore[method-assign]  # noqa: SLF001
    first._force_stop_broker = slow_cleanup  # type: ignore[method-assign]  # noqa: SLF001
    second._start_direct_broker_once = enter_second  # type: ignore[method-assign]  # noqa: SLF001

    async def exercise() -> None:
        first_task = asyncio.create_task(first._start_direct_broker())  # noqa: SLF001
        await failure_seen.wait()
        await cleanup_started.wait()
        second_task = asyncio.create_task(second._start_direct_broker())  # noqa: SLF001
        await asyncio.sleep(0.02)
        assert not second_entered.is_set()
        allow_cleanup.set()
        with pytest.raises(RuntimeError, match="synthetic mount failure"):
            await first_task
        await second_task

    asyncio.run(exercise())

    assert second_entered.is_set()


def test_broker_start_defaults_are_slow_mount_safe(monkeypatch, tmp_path: Path) -> None:
    for name in (
        "POLAR_APPTAINER_BROKER_START_TIMEOUT_SEC",
        "POLAR_APPTAINER_BROKER_RETRY_BACKOFF_SEC",
        "POLAR_APPTAINER_BROKER_RETRY_BACKOFF_MAX_SEC",
    ):
        monkeypatch.delenv(name, raising=False)
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime._overlay_dir = runtime.session_dir / "overlay"  # noqa: SLF001
    runtime._overlay_dir.mkdir(parents=True)  # noqa: SLF001
    observed_timeouts: list[float] = []

    async def fake_run(*_args, **_kwargs):
        await asyncio.Event().wait()

    async def fake_ready(timeout: float) -> None:
        observed_timeouts.append(timeout)

    runtime._run_local_command = fake_run  # type: ignore[method-assign]
    runtime._wait_for_broker_ready = fake_ready  # type: ignore[method-assign]
    monkeypatch.setattr(
        apptainer,
        "_direct_broker_process_snapshot",
        lambda *_args, **_kwargs: ({8_600_001}, {}),
    )
    runtime._cleanup_direct_broker_processes = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda *, known_sessions: None
    )

    async def exercise() -> None:
        await runtime._start_direct_broker_once()  # noqa: SLF001
        await runtime._force_stop_broker()  # noqa: SLF001

    asyncio.run(exercise())

    assert observed_timeouts == [120.0]
    assert 5.0 <= runtime._broker_retry_backoff(1) <= 6.25  # noqa: SLF001
    assert runtime._broker_retry_backoff(4) == 30.0  # noqa: SLF001


def test_broker_readiness_wait_survives_concurrent_task_clear(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    monkeypatch.setattr(
        apptainer,
        "_direct_broker_process_snapshot",
        lambda *_args, **_kwargs: ({8_600_002}, {}),
    )
    runtime._cleanup_direct_broker_processes = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda *, known_sessions: None
    )

    async def exercise() -> None:
        broker_task = asyncio.create_task(asyncio.Event().wait())
        runtime._broker_task = broker_task  # noqa: SLF001
        readiness = asyncio.create_task(
            runtime._wait_for_broker_ready(1.0)  # noqa: SLF001
        )
        await asyncio.sleep(0.02)
        await runtime._force_stop_broker()  # noqa: SLF001
        with pytest.raises(asyncio.CancelledError):
            await readiness

    asyncio.run(exercise())


def test_broker_start_concurrency_rejects_unsafe_fanout(monkeypatch) -> None:
    monkeypatch.setenv("POLAR_APPTAINER_BROKER_START_CONCURRENCY", "1000")

    async def read_gate_size() -> int:
        gate = apptainer._broker_start_gate()  # noqa: SLF001
        return gate._value  # noqa: SLF001

    assert asyncio.run(read_gate_size()) == 2


def test_broker_start_per_image_concurrency_rejects_unsafe_fanout(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY",
        "1000",
    )

    async def read_gate_size() -> int:
        gate = apptainer._broker_start_per_image_gate(  # noqa: SLF001
            "/images/task.sif"
        )
        return gate._value  # noqa: SLF001

    assert asyncio.run(read_gate_size()) == 2


def test_destroyed_direct_runtime_rejects_new_exec(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    asyncio.run(runtime.stop())

    with pytest.raises(RuntimeDestroyedError, match="already destroyed"):
        asyncio.run(runtime.exec("echo must-not-run"))


def test_dead_direct_runtime_owner_is_never_replaced_for_exec(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    replacement_attempted = False

    async def forbidden_replacement() -> None:
        nonlocal replacement_attempted
        replacement_attempted = True

    runtime._start_direct_broker = forbidden_replacement  # type: ignore[method-assign]  # noqa: SLF001,E501

    async def exercise() -> None:
        async def exited_owner() -> tuple[int, str | None, str | None]:
            return 137, None, "synthetic namespace-owner exit"

        runtime._broker_task = asyncio.create_task(exited_owner())  # noqa: SLF001
        await runtime._broker_task  # noqa: SLF001
        with pytest.raises(RuntimeError, match="refusing to start a replacement namespace"):
            await runtime.exec("echo must-not-run")

    asyncio.run(exercise())

    assert not replacement_attempted


def test_upload_dir_mkdir_failure_preserves_return_code_and_stderr(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)

    async def fake_exec(*_args, **_kwargs):
        return ExecResult(
            return_code=255,
            stderr="FATAL: task image parent directory disappeared",
        )

    runtime.exec = fake_exec  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as error:
        asyncio.run(runtime.upload_dir(str(tmp_path), "/tests"))

    message = str(error.value)
    assert "failed to create directory /tests in runtime" in message
    assert "exit code 255" in message
    assert "FATAL: task image parent directory disappeared" in message


def test_upload_file_mkdir_failure_preserves_stdout_when_stderr_is_empty(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)

    async def fake_exec(*_args, **_kwargs):
        return ExecResult(return_code=42, stdout="runtime mkdir diagnostic")

    runtime.exec = fake_exec  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as error:
        asyncio.run(runtime.upload_file(str(tmp_path / "source"), "/tests/file"))

    message = str(error.value)
    assert "failed to create directory /tests in runtime" in message
    assert "exit code 42" in message
    assert "runtime mkdir diagnostic" in message


def test_upload_dir_pipeline_preserves_tar_diagnostic(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    runtime._overlay_dir = tmp_path / "session" / "overlay"
    exec_results = iter(
        [
            ExecResult(return_code=0),
            ExecResult(
                return_code=1,
                stderr="FATAL: overlay mount was temporarily unavailable",
            ),
        ]
    )

    async def fake_exec(*_args, **_kwargs):
        return next(exec_results)

    runtime.exec = fake_exec  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as error:
        asyncio.run(runtime.upload_dir(str(tmp_path), "/tests"))

    assert "exit code 1" in str(error.value)
    assert "overlay mount was temporarily unavailable" in str(error.value)


def test_bind_mount_transfers_run_off_the_asyncio_event_loop(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def fake_copy(*_args) -> bool:
        worker_threads.append(threading.get_ident())
        return True

    runtime._copy_to_bind_mount = fake_copy  # type: ignore[method-assign]
    runtime._copy_from_bind_mount = fake_copy  # type: ignore[method-assign]

    async def transfer_all() -> None:
        await runtime.upload_file("source", "/polar/session/file")
        await runtime.upload_dir("source", "/polar/session/dir")
        await runtime.download_file("/polar/session/file", "target")
        await runtime.download_dir("/polar/session/dir", "target")

    asyncio.run(transfer_all())

    assert len(worker_threads) == 4
    assert all(thread_id != event_loop_thread for thread_id in worker_threads)
