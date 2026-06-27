#!/usr/bin/env python3
"""Validate nested Apptainer and the shared mini-swe-agent runtime."""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from polar.runtime.apptainer import ApptainerRuntime
from polar.runtime.models import RuntimeSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mini-swe-runtime", required=True)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    runtime_dir = Path(args.mini_swe_runtime).resolve()
    mini_swe_bin = runtime_dir / "venv/bin/mini-swe-agent"
    if not mini_swe_bin.is_file():
        raise FileNotFoundError(mini_swe_bin)

    with tempfile.TemporaryDirectory(prefix="polar-tmax-preflight-") as session:
        session_dir = Path(session)
        runtime = ApptainerRuntime(
            RuntimeSpec(
                backend="apptainer",
                image=str(Path(args.image).resolve()),
                workdir="/root",
                env={
                    "HOME": "/polar/session/home",
                    "PATH": (
                        f"{runtime_dir}/venv/bin:/usr/local/sbin:/usr/local/bin:"
                        "/usr/sbin:/usr/bin:/sbin:/bin"
                    ),
                },
                kwargs={"volumes": [f"{runtime_dir}:{runtime_dir}:ro"]},
            ),
            session_id="tmax-mini-swe-preflight",
            session_dir=session_dir,
        )
        await runtime.start()
        try:
            first = await runtime.exec(
                "mkdir -p /polar/session/home /polar/session/logs/agent && "
                "command -v mini-swe-agent && "
                "mini-swe-agent --help >/dev/null && "
                "printf ready > /root/.polar-mini-swe-preflight",
                timeout_sec=120,
            )
            if first.return_code != 0:
                raise RuntimeError(first.stderr or first.stdout or "first exec failed")

            second = await runtime.exec(
                "test \"$(cat /root/.polar-mini-swe-preflight)\" = ready && "
                "mini-swe-agent --help >/dev/null",
                timeout_sec=120,
            )
            if second.return_code != 0:
                raise RuntimeError(second.stderr or second.stdout or "second exec failed")
        finally:
            await runtime.stop()

    print(f"nested Apptainer preflight passed: image={args.image}")


def main() -> int:
    asyncio.run(run(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
