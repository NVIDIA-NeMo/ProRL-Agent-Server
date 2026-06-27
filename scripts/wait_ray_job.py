#!/usr/bin/env python3
"""Stream a Ray job's logs without making the WebSocket its lifecycle owner."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any


TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "STOPPED"}


def _status_name(status: Any) -> str:
    return str(getattr(status, "value", status))


def _emit_unseen(text: str, consumed: int, emitted: int) -> tuple[int, int]:
    """Emit only content not printed by an earlier WebSocket connection."""
    chunk_start = consumed
    consumed += len(text)
    if consumed <= emitted:
        return consumed, emitted

    unseen_start = max(0, emitted - chunk_start)
    sys.stdout.write(text[unseen_start:])
    sys.stdout.flush()
    return consumed, consumed


def _emit_final_logs(client: Any, submission_id: str, emitted: int) -> int:
    try:
        logs = client.get_job_logs(submission_id)
    except Exception as exc:  # noqa: BLE001 - logging must not mask job status
        print(f"WARNING: could not fetch final Ray job logs: {exc}", file=sys.stderr)
        return emitted

    if len(logs) > emitted:
        sys.stdout.write(logs[emitted:])
        sys.stdout.flush()
        return len(logs)
    return emitted


async def wait_for_job(
    address: str,
    submission_id: str,
    retry_interval: float,
    max_status_errors: int,
) -> int:
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(address)
    emitted = 0
    status_errors = 0

    while True:
        consumed = 0
        stream_error: Exception | None = None
        try:
            async for chunk in client.tail_job_logs(submission_id):
                consumed, emitted = _emit_unseen(chunk, consumed, emitted)
        except Exception as exc:  # noqa: BLE001 - reconnect after transport failures
            stream_error = exc

        try:
            status = _status_name(client.get_job_status(submission_id))
            status_errors = 0
        except Exception as exc:  # noqa: BLE001 - tolerate a transient dashboard outage
            status_errors += 1
            if status_errors >= max_status_errors:
                print(
                    f"ERROR: Ray job status unavailable for {status_errors} consecutive attempts: {exc}",
                    file=sys.stderr,
                )
                return 2
            print(
                f"WARNING: Ray job status unavailable ({status_errors}/{max_status_errors}): {exc}",
                file=sys.stderr,
            )
            await asyncio.sleep(retry_interval)
            continue

        if status in TERMINAL_STATUSES:
            emitted = _emit_final_logs(client, submission_id, emitted)
            try:
                info = client.get_job_info(submission_id)
                message = getattr(info, "message", None)
            except Exception as exc:  # noqa: BLE001 - terminal status is authoritative
                message = None
                print(f"WARNING: could not fetch final Ray job details: {exc}", file=sys.stderr)
            print(f"Ray job {submission_id} finished with status {status}", file=sys.stderr)
            if message:
                print(f"Ray job status message: {message}", file=sys.stderr)
            return 0 if status == "SUCCEEDED" else 1

        if stream_error is not None:
            print(
                f"WARNING: Ray log stream disconnected while job is {status}: {stream_error}; reconnecting",
                file=sys.stderr,
            )
        else:
            print(
                f"WARNING: Ray log stream ended while job is {status}; reconnecting",
                file=sys.stderr,
            )
        await asyncio.sleep(retry_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--retry-interval", type=float, default=5.0)
    parser.add_argument("--max-status-errors", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(
        wait_for_job(
            address=args.address,
            submission_id=args.submission_id,
            retry_interval=args.retry_interval,
            max_status_errors=args.max_status_errors,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
