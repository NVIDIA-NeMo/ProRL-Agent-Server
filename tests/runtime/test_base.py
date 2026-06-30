from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import pytest

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec


class StubRuntime(BaseRuntime):
    def __init__(self, session_dir: Path) -> None:
        super().__init__(RuntimeSpec(image="image"), "session", session_dir)
        self.stopped = False

    @property
    def runtime_id(self) -> str:
        return "stub"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self.stopped = True

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        return ExecResult(return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        pass

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        pass

    async def download_file(self, remote_path: str, local_path: str) -> None:
        pass

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        pass


class ProcessThatExitedBeforeKill:
    returncode = None
    pid = None
    stdout = None
    stderr = None

    def __init__(self) -> None:
        self.kill_calls = 0
        self.wait_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = 0
        raise ProcessLookupError

    def wait(self) -> int:
        self.wait_calls += 1
        self.returncode = 0
        return 0

    def poll(self) -> int | None:
        return self.returncode


class SuccessfulFakeProcess:
    returncode = 0
    pid = None
    stdout = None
    stderr = None

    def poll(self) -> int:
        return 0

    def wait(self) -> int:
        return 0


class CancellableFakeProcess:
    returncode = None
    pid = None
    stdout = None
    stderr = None

    def __init__(self) -> None:
        self.kill_calls = 0
        self.wait_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -signal.SIGKILL

    def wait(self) -> int:
        self.wait_calls += 1
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode


def test_cancel_ignores_process_exit_racing_kill(tmp_path) -> None:
    runtime = StubRuntime(tmp_path / "session")
    process = ProcessThatExitedBeforeKill()
    runtime._active_process = process  # type: ignore[assignment]

    asyncio.run(runtime.cancel())

    assert process.kill_calls == 1
    assert runtime.stopped is True


def test_local_command_timeout_ignores_process_exit_racing_kill(monkeypatch, tmp_path) -> None:
    runtime = StubRuntime(tmp_path / "session")
    process = ProcessThatExitedBeforeKill()

    def popen(*_args, **_kwargs):
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)

    result = asyncio.run(runtime._run_local_command("command", timeout=0.001, capture=True))

    assert result == (-1, None, None)
    assert process.kill_calls == 1
    assert runtime._active_process is None


def test_local_command_uses_explicit_working_directory(tmp_path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    runtime = StubRuntime(session_dir)

    rc, stdout, stderr = asyncio.run(
        runtime._run_local_command("/bin/pwd", capture=True, cwd=session_dir)
    )

    assert rc == 0
    assert stdout is not None
    assert Path(stdout.strip()) == session_dir
    assert stderr is None


def test_local_command_drains_stdout_and_stderr_concurrently(tmp_path) -> None:
    runtime = StubRuntime(tmp_path / "session")

    rc, stdout, stderr = asyncio.run(
        runtime._run_local_command(
            "/bin/bash",
            "-c",
            "head -c 1048576 /dev/zero; head -c 1048576 /dev/zero >&2",
            capture=True,
        )
    )

    assert rc == 0
    assert stdout is not None and len(stdout) == 1024 * 1024
    assert stderr is not None and len(stderr) == 1024 * 1024


def test_timed_out_local_command_kills_process_group(tmp_path) -> None:
    runtime = StubRuntime(tmp_path / "session")
    pid_path = tmp_path / "timeout-child.pid"

    rc, stdout, stderr = asyncio.run(
        runtime._run_local_command(
            "/bin/bash",
            "-c",
            f"echo $$ > {pid_path}; exec sleep 60",
            timeout=0.1,
            capture=True,
        )
    )

    assert (rc, stdout, stderr) == (-1, None, None)
    pid = int(pid_path.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, signal.SIGCONT)


def test_cancelled_local_command_kills_and_reaps_process_group(tmp_path) -> None:
    runtime = StubRuntime(tmp_path / "session")
    pid_path = tmp_path / "child.pid"

    async def exercise() -> int:
        task = asyncio.create_task(
            runtime._run_local_command(
                "/bin/bash",
                "-c",
                f"echo $$ > {pid_path}; exec sleep 60",
            )
        )
        async with asyncio.timeout(2.0):
            while not pid_path.exists():
                await asyncio.sleep(0.01)
        pid = int(pid_path.read_text().strip())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime._active_process is None
        return pid

    pid = asyncio.run(exercise())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, signal.SIGCONT)


def test_local_spawn_gate_is_shared_across_runtime_instances(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY", "4")

    async def exercise() -> None:
        entered = 0
        peak = 0
        create_calls = 0
        state_lock = threading.Lock()
        first_wave_entered = threading.Event()
        release = threading.Event()

        def popen(*_args, **_kwargs):
            nonlocal create_calls, entered, peak
            with state_lock:
                create_calls += 1
                entered += 1
                peak = max(peak, entered)
                if entered == 4:
                    first_wave_entered.set()
            assert release.wait(timeout=2.0)
            with state_lock:
                entered -= 1
            return SuccessfulFakeProcess()

        monkeypatch.setattr(subprocess, "Popen", popen)
        runtimes = [StubRuntime(tmp_path / f"session-{index}") for index in range(12)]
        tasks = [
            asyncio.create_task(runtime._run_local_command("command")) for runtime in runtimes
        ]
        async with asyncio.timeout(1.0):
            while not first_wave_entered.is_set():
                await asyncio.sleep(0.001)
        await asyncio.sleep(0.02)
        assert create_calls == 4
        assert entered == 4

        release.set()
        assert await asyncio.gather(*tasks) == [(0, None, None)] * 12
        assert peak == 4
        assert create_calls == 12

    asyncio.run(exercise())


def test_local_spawn_gate_keeps_event_loop_responsive_during_spawn_burst(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY", "4")

    async def exercise() -> tuple[list[float], float]:
        # Model the synchronous Popen clone/pipe portion seen in production.
        def popen(*_args, **_kwargs):
            time.sleep(0.005)
            return SuccessfulFakeProcess()

        monkeypatch.setattr(subprocess, "Popen", popen)
        stopped = False
        heartbeat_lags: list[float] = []

        async def heartbeat() -> None:
            interval = 0.005
            while not stopped:
                target = asyncio.get_running_loop().time() + interval
                await asyncio.sleep(interval)
                heartbeat_lags.append(asyncio.get_running_loop().time() - target)

        heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        runtimes = [StubRuntime(tmp_path / f"burst-{index}") for index in range(64)]
        started_at = time.perf_counter()
        await asyncio.gather(*(runtime._run_local_command("command") for runtime in runtimes))
        elapsed = time.perf_counter() - started_at
        stopped = True
        await heartbeat_task
        return heartbeat_lags, elapsed

    heartbeat_lags, elapsed = asyncio.run(exercise())

    # A gate-less 64x5ms ready burst blocks the heartbeat for roughly 320ms.
    # Four-at-a-time admission keeps making progress without reducing total
    # spawn throughput.
    assert elapsed < 1.0
    assert len(heartbeat_lags) >= 4
    assert max(heartbeat_lags) < 0.2


def test_real_local_spawn_burst_keeps_event_loop_responsive(tmp_path) -> None:
    async def exercise() -> tuple[list[float], list[tuple[int, str | None, str | None]]]:
        stopped = False
        heartbeat_lags: list[float] = []

        async def heartbeat() -> None:
            interval = 0.002
            while not stopped:
                target = asyncio.get_running_loop().time() + interval
                await asyncio.sleep(interval)
                heartbeat_lags.append(asyncio.get_running_loop().time() - target)

        heartbeat_task = asyncio.create_task(heartbeat())
        runtimes = [StubRuntime(tmp_path / f"real-{index}") for index in range(96)]
        results = await asyncio.gather(
            *(runtime._run_local_command("/bin/true") for runtime in runtimes)
        )
        stopped = True
        await heartbeat_task
        return heartbeat_lags, results

    heartbeat_lags, results = asyncio.run(exercise())

    assert results == [(0, None, None)] * 96
    assert heartbeat_lags
    assert max(heartbeat_lags) < 0.2


def test_process_wait_fallback_polls_at_bounded_interval(monkeypatch) -> None:
    process = SuccessfulFakeProcess()
    process.returncode = None
    poll_calls = 0
    sleep_delays: list[float] = []

    def poll() -> int | None:
        nonlocal poll_calls
        poll_calls += 1
        if poll_calls >= 2:
            process.returncode = 0
        return process.returncode

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    process.poll = poll  # type: ignore[method-assign]
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    assert asyncio.run(BaseRuntime._wait_for_process(process)) == 0  # type: ignore[arg-type]
    assert sleep_delays == [0.25]


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int", "1025"])
def test_local_spawn_gate_rejects_invalid_concurrency(
    monkeypatch,
    tmp_path,
    value: str,
) -> None:
    monkeypatch.setenv("POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY", value)
    runtime = StubRuntime(tmp_path / "session")

    with pytest.raises(ValueError, match="must be"):
        asyncio.run(runtime._run_local_command("command"))


def test_cancellation_during_spawn_fairness_yield_reaps_process(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = StubRuntime(tmp_path / "session")
    process = CancellableFakeProcess()

    spawn_entered = threading.Event()
    release_spawn = threading.Event()

    def popen(*_args, **_kwargs):
        spawn_entered.set()
        assert release_spawn.wait(timeout=2.0)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)

    async def exercise() -> None:
        task = asyncio.create_task(runtime._run_local_command("command"))
        async with asyncio.timeout(1.0):
            while not spawn_entered.is_set():
                await asyncio.sleep(0.001)
        task.cancel()
        release_spawn.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert process.kill_calls == 1
    assert process.returncode == -signal.SIGKILL
    assert runtime._active_process is None
