#!/usr/bin/env python3
"""Validate nested Apptainer and the shared mini-swe-agent runtime."""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import tempfile
from pathlib import Path

import polar.runtime.apptainer as apptainer_runtime
from polar.runtime.apptainer import ApptainerRuntime
from polar.runtime.models import RuntimeSpec


CONTAINER_RUNTIME_DIR = "/opt/polar-mini-swe-agent"
PRODUCTION_APPTAINER_FLAGS = (
    "POLAR_APPTAINER_NO_INSTANCE",
    "POLAR_APPTAINER_NO_MOUNT_HOSTFS",
    "POLAR_APPTAINER_NO_MOUNT_TMP",
    "POLAR_APPTAINER_ISOLATE_PID",
    "POLAR_APPTAINER_ISOLATE_IPC",
    "POLAR_APPTAINER_CLEANENV",
)


def _python_path_isolation_command() -> str:
    """Validate the CLI venv and task Python stay on disjoint import paths."""
    runtime_prefix = shlex.quote(CONTAINER_RUNTIME_DIR)
    venv_python = shlex.quote(f"{CONTAINER_RUNTIME_DIR}/venv/bin/python")
    return (
        f'{venv_python} -c "import minisweagent, polar_mini_swe_runner, '
        "polar_mini_swe_timing, polar_mini_swe_vanillux, sys; "
        "assert polar_mini_swe_timing.TIMING_SCHEMA_VERSION == 1; "
        f"assert sys.prefix == '{CONTAINER_RUNTIME_DIR}/venv', sys.prefix\" && "
        'python3 -c "import os, sys; '
        f"bad = [p for p in sys.path if p.startswith('{CONTAINER_RUNTIME_DIR}')]; "
        "python_path = os.environ.get('PYTHONPATH', ''); "
        f"assert '{CONTAINER_RUNTIME_DIR}' not in python_path, python_path; "
        'assert not bad, bad" && '
        f"test -x {runtime_prefix}/bin/mini-swe-agent && "
        f"test -f {runtime_prefix}/config/vanillux2.yaml"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mini-swe-runtime", required=True)
    parser.add_argument(
        "--stress-sessions",
        type=int,
        default=1,
        help="number of direct brokers to start concurrently in each wave",
    )
    parser.add_argument(
        "--cancel-sessions",
        type=int,
        default=0,
        help="brokers per wave to tear down through the early-stop cancel path",
    )
    parser.add_argument(
        "--stress-waves",
        type=int,
        default=1,
        help="number of start/teardown waves; later waves detect leaked FUSE mounts",
    )
    return parser.parse_args()


def _runtime_spec(args: argparse.Namespace, runtime_dir: Path) -> RuntimeSpec:
    return RuntimeSpec(
        backend="apptainer",
        image=str(Path(args.image).resolve()),
        network=os.environ.get("POLAR_SANDBOX_NETWORK", "none"),
        workdir="/root",
        env={
            "HOME": "/polar/session/home",
            "PATH": (
                f"{CONTAINER_RUNTIME_DIR}/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            ),
        },
        kwargs={"volumes": [f"{runtime_dir}:{CONTAINER_RUNTIME_DIR}:ro"]},
    )


async def _run_stress_wave(
    args: argparse.Namespace,
    runtime_dir: Path,
    root: Path,
    wave: int,
) -> None:
    runtimes = [
        ApptainerRuntime(
            _runtime_spec(args, runtime_dir),
            session_id=f"tmax-preflight-wave-{wave}-session-{index}",
            session_dir=root / f"wave-{wave}" / f"session-{index}",
        )
        for index in range(args.stress_sessions)
    ]
    captured_sessions: list[set[int]] = []
    try:
        await asyncio.gather(*(runtime.start() for runtime in runtimes))
        probes = await asyncio.gather(
            *(
                runtime.exec(
                    "mkdir -p /polar/session/home && printf ready",
                    timeout_sec=30,
                )
                for runtime in runtimes
            )
        )
        failed = [result for result in probes if result.return_code != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} stress broker probe(s) failed")

        for runtime in runtimes:
            sessions, _ = apptainer_runtime._direct_broker_process_snapshot(  # noqa: SLF001
                runtime.session_dir
            )
            if not sessions:
                raise RuntimeError(
                    f"could not identify Apptainer runtime SID for {runtime.session_id}"
                )
            captured_sessions.append(sessions)

        split = args.cancel_sessions
        await asyncio.gather(*(runtime.cancel() for runtime in runtimes[:split]))
        await asyncio.gather(*(runtime.stop() for runtime in runtimes[split:]))

        leaked: dict[str, list[int]] = {}
        for runtime, sessions in zip(runtimes, captured_sessions, strict=True):
            _, processes = apptainer_runtime._direct_broker_process_snapshot(  # noqa: SLF001
                runtime.session_dir,
                known_sessions=sessions,
            )
            if processes:
                leaked[runtime.session_id] = sorted(processes)
        if leaked:
            raise RuntimeError(f"escaped Apptainer processes after teardown: {leaked}")
    finally:
        await asyncio.gather(*(runtime.cancel() for runtime in runtimes), return_exceptions=True)


async def run(args: argparse.Namespace) -> None:
    # This script validates the exact nested-Apptainer mode used by TMax jobs,
    # independent of the submit shell's current environment.
    for name in PRODUCTION_APPTAINER_FLAGS:
        os.environ[name] = "1"

    runtime_dir = Path(args.mini_swe_runtime).resolve()
    mini_swe_bin = runtime_dir / "bin/mini-swe-agent"
    if not mini_swe_bin.is_file():
        raise FileNotFoundError(mini_swe_bin)
    if args.stress_sessions < 1 or args.stress_waves < 1:
        raise ValueError("--stress-sessions and --stress-waves must be positive")
    if not 0 <= args.cancel_sessions <= args.stress_sessions:
        raise ValueError("--cancel-sessions must be between zero and --stress-sessions")

    with tempfile.TemporaryDirectory(prefix="polar-tmax-preflight-") as session:
        session_dir = Path(session)
        runtime = ApptainerRuntime(
            _runtime_spec(args, runtime_dir),
            session_id="tmax-mini-swe-preflight",
            session_dir=session_dir,
        )
        await runtime.start()
        try:
            first = await runtime.exec(
                "mkdir -p /polar/session/home /polar/session/logs/agent && "
                "command -v mini-swe-agent && "
                "mini-swe-agent --help >/dev/null && "
                f"{_python_path_isolation_command()} && "
                "printf ready > /root/.polar-mini-swe-preflight",
                timeout_sec=120,
            )
            if first.return_code != 0:
                raise RuntimeError(first.stderr or first.stdout or "first exec failed")

            second = await runtime.exec(
                'test "$(cat /root/.polar-mini-swe-preflight)" = ready && '
                "mini-swe-agent --help >/dev/null && "
                f"{_python_path_isolation_command()}",
                timeout_sec=120,
            )
            if second.return_code != 0:
                raise RuntimeError(second.stderr or second.stdout or "second exec failed")
        finally:
            await runtime.stop()

    if args.stress_sessions > 1 or args.stress_waves > 1 or args.cancel_sessions:
        with tempfile.TemporaryDirectory(prefix="polar-tmax-stress-") as stress_root:
            for wave in range(args.stress_waves):
                await _run_stress_wave(
                    args,
                    runtime_dir,
                    Path(stress_root),
                    wave,
                )

    print(
        "nested Apptainer preflight passed: "
        f"image={args.image} sessions={args.stress_sessions} "
        f"cancelled={args.cancel_sessions} waves={args.stress_waves}"
    )


def main() -> int:
    asyncio.run(run(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
