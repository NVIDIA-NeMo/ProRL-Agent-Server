"""Forward filesystem Unix-domain sockets to TCP endpoints.

This module deliberately has no Polar service dependencies so launchers can
run it as a small, host-side helper process::

    python -m polar.runtime.uds_tunnel \
        --ready-file /tmp/polar-tunnel.ready \
        /tmp/gateway.sock=127.0.0.1:18100 \
        /tmp/proxy.sock=proxy.example.com:3128

The ready file is published atomically after every socket is listening and has
the requested permissions.  SIGINT and SIGTERM close active connections and
remove sockets owned by this process.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
import errno
import json
import logging
import os
from pathlib import Path
import signal
import stat
import sys
from typing import Final


_COPY_CHUNK_SIZE: Final[int] = 64 * 1024
_ACTIVE_SOCKET_PROBE_TIMEOUT_SEC: Final[float] = 0.25
_LINUX_UNIX_SOCKET_PATH_MAX_BYTES: Final[int] = 107
_DEFAULT_LISTEN_BACKLOG: Final[int] = 4096
_MAX_LISTEN_BACKLOG: Final[int] = 65535
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TunnelSpec:
    """One Unix-domain socket to TCP forwarding rule."""

    socket_path: Path
    target_host: str
    target_port: int

    @property
    def target(self) -> str:
        """Return an unambiguous, connectable display form of the target."""

        if ":" in self.target_host:
            return f"[{self.target_host}]:{self.target_port}"
        return f"{self.target_host}:{self.target_port}"


def _validate_socket_path(path: Path) -> None:
    path_value = str(path)
    if "\0" in path_value:
        raise ValueError("Unix socket path must not contain a null byte")
    if sys.platform.startswith("linux"):
        path_length = len(os.fsencode(path_value))
        if path_length > _LINUX_UNIX_SOCKET_PATH_MAX_BYTES:
            raise ValueError(
                f"Unix socket path is {path_length} bytes, but Linux filesystem "
                f"sockets support at most {_LINUX_UNIX_SOCKET_PATH_MAX_BYTES}: {path}"
            )


def _canonical_path(path: Path) -> Path:
    """Normalize a path for collision checks, resolving any existing parents."""

    return path.resolve(strict=False)


def parse_mapping(value: str) -> TunnelSpec:
    """Parse ``SOCKET=HOST:PORT`` into a validated forwarding rule."""

    try:
        socket_value, target_value = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"mapping must have the form SOCKET=HOST:PORT, got {value!r}"
        ) from exc

    if not socket_value:
        raise argparse.ArgumentTypeError("Unix socket path must not be empty")
    if not target_value:
        raise argparse.ArgumentTypeError("TCP target must not be empty")

    if target_value.startswith("["):
        closing_bracket = target_value.find("]")
        if closing_bracket < 0 or target_value[closing_bracket + 1 : closing_bracket + 2] != ":":
            raise argparse.ArgumentTypeError(
                f"bracketed TCP target must have the form [HOST]:PORT, got {target_value!r}"
            )
        host = target_value[1:closing_bracket]
        port_value = target_value[closing_bracket + 2 :]
    else:
        try:
            host, port_value = target_value.rsplit(":", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"TCP target must have the form HOST:PORT, got {target_value!r}"
            ) from exc

    if not host:
        raise argparse.ArgumentTypeError("TCP target host must not be empty")
    try:
        port = int(port_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"TCP target port must be an integer, got {port_value!r}"
        ) from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            f"TCP target port must be between 1 and 65535, got {port}"
        )

    socket_path = Path(socket_value)
    try:
        _validate_socket_path(socket_path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return TunnelSpec(socket_path, host, port)


def _parse_socket_mode(value: str) -> int:
    normalized = value.removeprefix("0o")
    try:
        mode = int(normalized, 8)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"socket mode must be an octal permission mode, got {value!r}"
        ) from exc
    if not 0 <= mode <= 0o777:
        raise argparse.ArgumentTypeError(
            f"socket mode must contain only permission bits, got {value!r}"
        )
    return mode


def _parse_listen_backlog(value: str) -> int:
    try:
        backlog = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"backlog must be an integer, got {value!r}"
        ) from exc
    if not 1 <= backlog <= _MAX_LISTEN_BACKLOG:
        raise argparse.ArgumentTypeError(
            f"backlog must be between 1 and {_MAX_LISTEN_BACKLOG}, got {backlog}"
        )
    return backlog


def _identity(path: Path) -> tuple[int, int] | None:
    """Return a path's device/inode identity without following symlinks."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def _unlink_if_owned(path: Path, identity: tuple[int, int] | None) -> None:
    """Remove a path only if it is still the object created by this process."""

    if identity is None or _identity(path) != identity:
        return
    with suppress(FileNotFoundError):
        path.unlink()


async def _prepare_socket_path(path: Path) -> None:
    """Reject active/non-socket paths and remove a confirmed stale socket."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return

    if not stat.S_ISSOCK(metadata.st_mode):
        raise FileExistsError(f"refusing to replace non-socket path: {path}")

    writer: asyncio.StreamWriter | None = None
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(path)),
            timeout=_ACTIVE_SOCKET_PROBE_TIMEOUT_SEC,
        )
    except (ConnectionRefusedError, FileNotFoundError):
        # No listener owns this filesystem socket, so it is safe to unlink.
        pass
    except OSError as exc:
        if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
            raise RuntimeError(f"refusing to replace unprobeable Unix socket: {path}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"refusing to replace busy Unix socket: {path}") from exc
    else:
        raise RuntimeError(f"refusing to replace active Unix socket: {path}")
    finally:
        if writer is not None:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    # Verify that the object did not change between lstat and the probe.
    current_identity = _identity(path)
    original_identity = (metadata.st_dev, metadata.st_ino)
    if current_identity == original_identity:
        path.unlink()
    elif current_identity is not None:
        raise RuntimeError(f"Unix socket path changed while checking whether it was stale: {path}")


async def _copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(_COPY_CHUNK_SIZE):
        writer.write(data)
        await writer.drain()

    if writer.can_write_eof():
        with suppress(ConnectionError, OSError):
            writer.write_eof()
            await writer.drain()


async def _close_writer(writer: asyncio.StreamWriter | None) -> None:
    if writer is None:
        return
    writer.close()
    with suppress(ConnectionError, OSError, TimeoutError):
        await writer.wait_closed()


class UdsTcpTunnel:
    """Manage one process's Unix-domain-socket TCP forwarders."""

    def __init__(
        self,
        specs: Sequence[TunnelSpec],
        *,
        socket_mode: int = 0o600,
        ready_file: Path | None = None,
        backlog: int = _DEFAULT_LISTEN_BACKLOG,
    ) -> None:
        if not specs:
            raise ValueError("at least one tunnel mapping is required")
        if not 0 <= socket_mode <= 0o777:
            raise ValueError("socket_mode must contain only permission bits")
        if not 1 <= backlog <= _MAX_LISTEN_BACKLOG:
            raise ValueError(
                f"backlog must be between 1 and {_MAX_LISTEN_BACKLOG}, got {backlog}"
            )

        socket_paths = [spec.socket_path for spec in specs]
        for path in socket_paths:
            _validate_socket_path(path)
        canonical_socket_paths = {_canonical_path(path) for path in socket_paths}
        if len(canonical_socket_paths) != len(socket_paths):
            raise ValueError("Unix socket paths must be unique")
        if ready_file is not None and _canonical_path(ready_file) in canonical_socket_paths:
            raise ValueError("ready_file must not refer to a tunnel's Unix socket path")

        self._specs = tuple(specs)
        self._socket_mode = socket_mode
        self._ready_file = ready_file
        self._backlog = backlog
        self._servers: list[asyncio.AbstractServer] = []
        self._socket_identities: dict[Path, tuple[int, int]] = {}
        self._ready_identity: tuple[int, int] | None = None
        self._connection_tasks: set[asyncio.Task[None]] = set()
        self._started = False
        self._closing = False

    async def start(self) -> None:
        """Bind all sockets and atomically publish readiness."""

        if self._started:
            raise RuntimeError("tunnel has already been started")
        self._started = True

        try:
            for spec in self._specs:
                spec.socket_path.parent.mkdir(parents=True, exist_ok=True)
                await _prepare_socket_path(spec.socket_path)
                server = await asyncio.start_unix_server(
                    lambda reader, writer, rule=spec: self._accept(rule, reader, writer),
                    path=str(spec.socket_path),
                    backlog=self._backlog,
                )
                self._servers.append(server)
                identity = _identity(spec.socket_path)
                if identity is None:
                    raise RuntimeError(
                        f"Unix socket disappeared during startup: {spec.socket_path}"
                    )
                self._socket_identities[spec.socket_path] = identity
                os.chmod(spec.socket_path, self._socket_mode)

            if self._ready_file is not None:
                self._publish_ready_file()
        except BaseException:
            await self.close()
            raise

    def _accept(
        self,
        spec: TunnelSpec,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._closing:
            writer.close()
            return
        task = asyncio.create_task(
            self._forward(spec, reader, writer),
            name=f"uds-tunnel:{spec.socket_path}",
        )
        self._connection_tasks.add(task)
        task.add_done_callback(self._connection_tasks.discard)

    async def _forward(
        self,
        spec: TunnelSpec,
        unix_reader: asyncio.StreamReader,
        unix_writer: asyncio.StreamWriter,
    ) -> None:
        tcp_writer: asyncio.StreamWriter | None = None
        pumps: list[asyncio.Task[None]] = []
        try:
            tcp_reader, tcp_writer = await asyncio.open_connection(
                spec.target_host,
                spec.target_port,
            )
            pumps = [
                asyncio.create_task(_copy_stream(unix_reader, tcp_writer)),
                asyncio.create_task(_copy_stream(tcp_reader, unix_writer)),
            ]
            await asyncio.gather(*pumps)
        except asyncio.CancelledError:
            raise
        except (ConnectionError, OSError) as exc:
            _LOGGER.warning(
                "Tunnel connection failed for %s -> %s: %s",
                spec.socket_path,
                spec.target,
                exc,
            )
        finally:
            for task in pumps:
                task.cancel()
            if pumps:
                await asyncio.gather(*pumps, return_exceptions=True)
            await asyncio.gather(
                _close_writer(tcp_writer),
                _close_writer(unix_writer),
            )

    def _publish_ready_file(self) -> None:
        assert self._ready_file is not None
        self._ready_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._ready_file.with_name(f".{self._ready_file.name}.{os.getpid()}.tmp")
        payload = {
            "pid": os.getpid(),
            "tunnels": [
                {"socket": str(spec.socket_path), "target": spec.target} for spec in self._specs
            ],
        }
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.chmod(temporary, self._socket_mode)
            os.replace(temporary, self._ready_file)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
        self._ready_identity = _identity(self._ready_file)

    async def close(self) -> None:
        """Stop accepting, close live connections, and remove owned paths."""

        self._closing = True
        for server in self._servers:
            server.close()
        if self._servers:
            await asyncio.gather(*(server.wait_closed() for server in self._servers))
        self._servers.clear()

        tasks = tuple(self._connection_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._connection_tasks.clear()

        if self._ready_file is not None:
            _unlink_if_owned(self._ready_file, self._ready_identity)
            self._ready_identity = None
        for path, identity in self._socket_identities.items():
            _unlink_if_owned(path, identity)
        self._socket_identities.clear()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward one or more filesystem Unix sockets to TCP endpoints.",
    )
    parser.add_argument(
        "mapping",
        nargs="+",
        type=parse_mapping,
        metavar="SOCKET=HOST:PORT",
        help="forwarding rule; repeat to expose multiple TCP targets",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="atomically create this JSON file once every socket is listening",
    )
    parser.add_argument(
        "--socket-mode",
        type=_parse_socket_mode,
        default=0o600,
        metavar="OCTAL",
        help="permissions for sockets and the ready file (default: 0600)",
    )
    parser.add_argument(
        "--backlog",
        type=_parse_listen_backlog,
        default=_DEFAULT_LISTEN_BACKLOG,
        metavar="N",
        help=f"pending connection backlog per socket (default: {_DEFAULT_LISTEN_BACKLOG})",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, stop_requested.set)
        except (NotImplementedError, RuntimeError):
            continue
        installed_signals.append(signal_number)

    tunnel = UdsTcpTunnel(
        args.mapping,
        socket_mode=args.socket_mode,
        ready_file=args.ready_file,
        backlog=args.backlog,
    )
    try:
        await tunnel.start()
        print(
            f"ready: {len(args.mapping)} Unix socket tunnel(s), pid={os.getpid()}",
            flush=True,
        )
        await stop_requested.wait()
    finally:
        await tunnel.close()
        for signal_number in installed_signals:
            loop.remove_signal_handler(signal_number)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone tunnel process."""

    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _LOGGER.error("Unix socket tunnel failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
