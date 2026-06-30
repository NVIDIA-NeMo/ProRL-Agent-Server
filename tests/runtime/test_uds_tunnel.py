from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import stat
import sys

import pytest

from polar.runtime.uds_tunnel import (
    TunnelSpec,
    UdsTcpTunnel,
    _parse_listen_backlog,
    parse_mapping,
)


def test_parse_mapping_supports_hostnames_and_ipv6() -> None:
    assert parse_mapping("/tmp/gateway.sock=127.0.0.1:18100") == TunnelSpec(
        Path("/tmp/gateway.sock"),
        "127.0.0.1",
        18100,
    )
    assert parse_mapping("proxy.sock=[::1]:3128") == TunnelSpec(
        Path("proxy.sock"),
        "::1",
        3128,
    )


@pytest.mark.parametrize(
    "mapping",
    [
        "missing-equals",
        "=host:80",
        "/tmp/a.sock=",
        "/tmp/a.sock=missing-port",
        "/tmp/a.sock=:80",
        "/tmp/a.sock=host:not-a-port",
        "/tmp/a.sock=host:0",
        "/tmp/a.sock=host:65536",
        "/tmp/a.sock=[::1:80",
    ],
)
def test_parse_mapping_rejects_invalid_values(mapping: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_mapping(mapping)


def test_ready_file_cannot_alias_a_socket_path(tmp_path: Path) -> None:
    socket_path = tmp_path / "nested" / "gateway.sock"
    aliased_ready_path = tmp_path / "nested" / ".." / "nested" / "gateway.sock"

    with pytest.raises(ValueError, match="ready_file"):
        UdsTcpTunnel(
            [TunnelSpec(socket_path, "127.0.0.1", 18100)],
            ready_file=aliased_ready_path,
        )


@pytest.mark.parametrize("value", ["not-an-int", "0", "65536"])
def test_invalid_listen_backlog_is_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="backlog"):
        _parse_listen_backlog(value)


@pytest.mark.asyncio
async def test_default_backlog_accepts_a_full_gateway_worker_burst(tmp_path: Path) -> None:
    concurrency = 512

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            payload = await reader.readexactly(1)
            writer.write(payload)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    tcp_server = await asyncio.start_server(
        echo,
        "127.0.0.1",
        0,
        backlog=concurrency * 2,
    )
    tcp_port = tcp_server.sockets[0].getsockname()[1]
    socket_path = tmp_path / "gateway.sock"
    tunnel = UdsTcpTunnel(
        [TunnelSpec(socket_path, "127.0.0.1", tcp_port)],
    )

    await tunnel.start()
    try:
        async def round_trip() -> bytes:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            try:
                writer.write(b"x")
                await writer.drain()
                return await reader.readexactly(1)
            finally:
                writer.close()
                await writer.wait_closed()

        results = await asyncio.gather(*(round_trip() for _ in range(concurrency)))
        assert results == [b"x"] * concurrency
    finally:
        await tunnel.close()
        tcp_server.close()
        await tcp_server.wait_closed()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux AF_UNIX path limit")
def test_overlong_linux_socket_path_is_rejected_before_bind() -> None:
    overlong_path = Path("/") / ("s" * 107)

    with pytest.raises(ValueError, match="at most 107"):
        UdsTcpTunnel([TunnelSpec(overlong_path, "127.0.0.1", 18100)])
    with pytest.raises(argparse.ArgumentTypeError, match="at most 107"):
        parse_mapping(f"{overlong_path}=127.0.0.1:18100")


@pytest.mark.asyncio
async def test_tunnel_round_trip_permissions_ready_and_cleanup(tmp_path: Path) -> None:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(4096):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    tcp_server = await asyncio.start_server(echo, "127.0.0.1", 0)
    tcp_address = tcp_server.sockets[0].getsockname()
    socket_path = tmp_path / "nested" / "gateway.sock"
    ready_file = tmp_path / "tunnel.ready"
    tunnel = UdsTcpTunnel(
        [TunnelSpec(socket_path, "127.0.0.1", tcp_address[1])],
        socket_mode=0o640,
        ready_file=ready_file,
    )

    await tunnel.start()
    try:
        assert stat.S_ISSOCK(socket_path.stat().st_mode)
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o640
        assert stat.S_IMODE(ready_file.stat().st_mode) == 0o640
        assert json.loads(ready_file.read_text()) == {
            "pid": os.getpid(),
            "tunnels": [{"socket": str(socket_path), "target": f"127.0.0.1:{tcp_address[1]}"}],
        }

        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(b"round trip")
        await writer.drain()
        assert await reader.readexactly(len(b"round trip")) == b"round trip"
        writer.close()
        await writer.wait_closed()
    finally:
        await tunnel.close()
        tcp_server.close()
        await tcp_server.wait_closed()

    assert not socket_path.exists()
    assert not ready_file.exists()


@pytest.mark.asyncio
async def test_start_refuses_to_replace_regular_file_or_active_socket(tmp_path: Path) -> None:
    regular_path = tmp_path / "regular"
    regular_path.write_text("do not remove")
    regular_tunnel = UdsTcpTunnel([TunnelSpec(regular_path, "127.0.0.1", 1)])
    with pytest.raises(FileExistsError, match="non-socket"):
        await regular_tunnel.start()
    assert regular_path.read_text() == "do not remove"

    active_path = tmp_path / "active.sock"

    async def close_probe(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.close()
        await writer.wait_closed()

    active_server = await asyncio.start_unix_server(close_probe, str(active_path))
    active_tunnel = UdsTcpTunnel([TunnelSpec(active_path, "127.0.0.1", 1)])
    try:
        with pytest.raises(RuntimeError, match="active Unix socket"):
            await active_tunnel.start()
        assert active_path.exists()
    finally:
        await active_tunnel.close()
        active_server.close()
        await active_server.wait_closed()
        active_path.unlink()


@pytest.mark.asyncio
async def test_cli_handles_multiple_mappings_and_sigterm_cleanup(tmp_path: Path) -> None:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        data = await reader.read(4096)
        writer.write(data)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    tcp_server = await asyncio.start_server(echo, "127.0.0.1", 0)
    tcp_port = tcp_server.sockets[0].getsockname()[1]
    socket_paths = [tmp_path / "one.sock", tmp_path / "two.sock"]
    ready_file = tmp_path / "ready.json"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "polar.runtime.uds_tunnel",
        "--ready-file",
        str(ready_file),
        *(f"{path}=127.0.0.1:{tcp_port}" for path in socket_paths),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        async with asyncio.timeout(5):
            while not ready_file.exists():
                if process.returncode is not None:
                    stdout, stderr = await process.communicate()
                    pytest.fail(
                        f"tunnel exited before ready: {process.returncode}; "
                        f"stdout={stdout!r}; stderr={stderr!r}"
                    )
                await asyncio.sleep(0.01)

        assert [item["socket"] for item in json.loads(ready_file.read_text())["tunnels"]] == [
            str(path) for path in socket_paths
        ]
        for path in socket_paths:
            reader, writer = await asyncio.open_unix_connection(str(path))
            writer.write(path.name.encode())
            await writer.drain()
            assert await reader.read() == path.name.encode()
            writer.close()
            await writer.wait_closed()

        process.send_signal(signal.SIGTERM)
        assert await asyncio.wait_for(process.wait(), timeout=5) == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        tcp_server.close()
        await tcp_server.wait_closed()

    assert not ready_file.exists()
    assert all(not path.exists() for path in socket_paths)


@pytest.mark.asyncio
async def test_stale_socket_is_removed_but_replacement_is_preserved(tmp_path: Path) -> None:
    socket_path = tmp_path / "stale.sock"
    stale_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_socket.bind(str(socket_path))
    stale_socket.close()

    tunnel = UdsTcpTunnel([TunnelSpec(socket_path, "127.0.0.1", 1)])
    await tunnel.start()
    socket_path.unlink()
    socket_path.write_text("replacement")
    await tunnel.close()

    assert socket_path.read_text() == "replacement"
