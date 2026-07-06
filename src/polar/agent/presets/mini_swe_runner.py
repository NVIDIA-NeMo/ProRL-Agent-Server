"""Network-isolated entry point for the shared mini-SWE-agent runtime.

Apptainer's rootless ``network=none`` mode gives every invocation a private
loopback namespace, but it cannot reach the Polar gateway or the site's HTTP
proxy over TCP.  The launcher exposes the gateway at ``/polar/gateway`` and,
only for internet-enabled runtimes, the proxy at ``/polar/proxy`` through
separate Unix-domain-socket binds.  This module wires LiteLLM directly to the
gateway socket and, when configured, provides a loopback HTTP-proxy port for
agent-created subprocesses such as ``pip``.

The module is copied into the portable mini-SWE environment as
``polar_mini_swe_runner.py`` by ``prepare_mini_swe_agent.sh``.  Imports of
mini-SWE-agent/LiteLLM stay inside ``main`` so the forwarding primitives remain
unit-testable without those optional packages in the Polar control venv.
"""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path
import re
import select
import socket
import socketserver
import sys
import threading
from types import TracebackType
from typing import Any


_GATEWAY_SOCKET_ENV = "POLAR_GATEWAY_UDS"
_PROXY_SOCKET_ENV = "POLAR_HTTP_PROXY_UDS"
_PROXY_PORT_ENV = "POLAR_HTTP_PROXY_PORT"
_PROXY_BROKER_READY_ENV = "POLAR_HTTP_PROXY_BROKER_READY"
_ALLOW_INTERNET_ENV = "POLAR_ALLOW_INTERNET"
_TASK_PYTHONPATH_ENV = "POLAR_TASK_PYTHONPATH"
_TASK_B64_ENV = "POLAR_MINI_SWE_TASK_B64"
_APT_HTTP_SOURCE_POLICY_ENV = "POLAR_APT_HTTP_SOURCE_POLICY"
_DEFAULT_PROXY_PORT = 28100
_PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "ftp_proxy",
    "FTP_PROXY",
)
_NO_PROXY_ENV_NAMES = ("no_proxy", "NO_PROXY")
_LOOPBACK_NO_PROXY = ("localhost", "127.0.0.1", "::1")

_APT_LIST_SOURCE_RE = re.compile(
    r"^(?P<prefix>\s*deb(?:-src)?\s+(?:\[[^\]\n]*\]\s+)?)"
    r"http://(?P<remainder>[^\s#]+)",
    flags=re.MULTILINE,
)
_APT_DEB822_FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*\s*:")
_APT_HTTP_TOKEN_RE = re.compile(r"(?<!\S)http://")


def _inject_task_from_env() -> None:
    """Move the private task payload into Python's argv, never the OS argv.

    Keeping arbitrary task text out of ``/proc/*/cmdline`` prevents task-owned
    process-management commands such as ``pkill -f`` from matching and killing
    the mini-SWE parent process.  The variable is popped before validation so
    it cannot leak into agent-created action subprocesses, including on an
    invalid launch.
    """

    encoded_task = os.environ.pop(_TASK_B64_ENV, None)
    if encoded_task is None:
        return
    if any(arg == "--task" or arg.startswith("--task=") for arg in sys.argv[1:]):
        raise ValueError(
            f"{_TASK_B64_ENV} cannot be combined with a --task command-line argument"
        )
    try:
        task_bytes = base64.b64decode(encoded_task, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{_TASK_B64_ENV} must be strict base64") from exc
    try:
        task = task_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{_TASK_B64_ENV} must decode to UTF-8 text") from exc
    # Python argument parsing sees the task, but mutating sys.argv does not
    # change the kernel-owned process command line exposed through /proc.
    sys.argv.append(f"--task={task}")


def _configure_model_retry_policy() -> None:
    """Fail fast on permanent HTTP 400 model errors.

    mini-SWE-agent already treats several specific LiteLLM 4xx exceptions as
    non-retryable, but some OpenAI-compatible servers (including the SGLang
    version used here) surface context overflow as ``BadRequestError`` even
    when the response contains ``code=context_length_exceeded``.  Retrying an
    identical over-length request can otherwise occupy one rollout slot for
    several minutes.  A generic HTTP 400 is a request defect, not a transient
    transport failure, so it is safe to add the base bad-request class to the
    abort set while leaving 429/5xx/transport retries intact.
    """

    import litellm
    from minisweagent.models.litellm_model import LitellmModel

    bad_request = litellm.exceptions.BadRequestError
    if bad_request not in LitellmModel.abort_exceptions:
        LitellmModel.abort_exceptions.insert(0, bad_request)


def _restore_task_pythonpath() -> None:
    """Restore OCI PYTHONPATH after the portable runner has safely imported.

    The injected shell wrapper hides the task image's Python path while the
    portable interpreter resolves mini-SWE-agent. Updating ``os.environ`` here
    does not mutate this interpreter's already-initialized ``sys.path``, but it
    does restore the official image environment for every action subprocess.
    """

    task_pythonpath = os.environ.pop(_TASK_PYTHONPATH_ENV, None)
    if task_pythonpath is not None:
        os.environ["PYTHONPATH"] = task_pythonpath


class _UnixForwardHandler(socketserver.BaseRequestHandler):
    """Relay one loopback TCP connection to a configured Unix socket."""

    def handle(self) -> None:
        unix_path = str(getattr(self.server, "unix_path"))
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect(unix_path)
            _relay_bidirectional(self.request, upstream)
        finally:
            upstream.close()


class _ThreadingUnixForwardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    # Package managers can open many parallel HTTP connections. The inherited
    # socketserver default is only five and resets otherwise-valid connections
    # during a short burst inside one sandbox.
    request_queue_size = 256

    def __init__(self, port: int, unix_path: str) -> None:
        self.unix_path = unix_path
        super().__init__(("127.0.0.1", port), _UnixForwardHandler)


def _relay_bidirectional(left: socket.socket, right: socket.socket) -> None:
    """Copy bytes in both directions until either peer reaches EOF."""

    peers = {left: right, right: left}
    while peers:
        readable, _, _ = select.select(list(peers), [], [])
        for source in readable:
            destination = peers.get(source)
            if destination is None:
                continue
            try:
                chunk = source.recv(64 * 1024)
            except (ConnectionError, OSError):
                peers.clear()
                break
            if not chunk:
                peers.clear()
                break
            try:
                destination.sendall(chunk)
            except (ConnectionError, OSError):
                peers.clear()
                break


class LoopbackProxy:
    """Lifecycle wrapper for the local TCP-to-UDS HTTP proxy forwarder."""

    def __init__(self, unix_path: str, port: int) -> None:
        if not 0 <= port <= 65535:
            raise ValueError(f"proxy port must be between 0 and 65535, got {port}")
        self._server = _ThreadingUnixForwardServer(port, unix_path)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="polar-mini-swe-http-proxy",
            daemon=True,
        )
        self.port = int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> LoopbackProxy:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _proxy_port_from_env() -> int:
    raw = os.environ.get(_PROXY_PORT_ENV, str(_DEFAULT_PROXY_PORT))
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_PROXY_PORT_ENV} must be an integer, got {raw!r}") from exc
    if not 0 <= port <= 65535:
        raise ValueError(f"{_PROXY_PORT_ENV} must be between 0 and 65535, got {port}")
    return port


def _merge_loopback_no_proxy(*values: str | None) -> str:
    """Preserve configured bypasses and always bypass proxying for loopback."""

    entries: list[str] = []
    seen: set[str] = set()
    for value in (*values, *_LOOPBACK_NO_PROXY):
        if value is None:
            continue
        for raw_entry in value.split(","):
            entry = raw_entry.strip()
            normalized = entry.lower()
            if entry and normalized not in seen:
                entries.append(entry)
                seen.add(normalized)
    return ",".join(entries)


def _configure_http_proxy() -> LoopbackProxy | None:
    allow_internet = os.environ.get(_ALLOW_INTERNET_ENV)
    if allow_internet is not None and allow_internet.strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        # ``network=none`` already blocks direct TCP.  Remove inherited proxy
        # variables as well so a task marked offline fails immediately instead
        # of repeatedly trying an unreachable corporate-proxy hostname.
        for name in _PROXY_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.pop(_PROXY_SOCKET_ENV, None)
        os.environ.pop(_PROXY_PORT_ENV, None)
        return None
    socket_path = os.environ.get(_PROXY_SOCKET_ENV, "").strip()
    if not socket_path:
        return None
    proxy = LoopbackProxy(socket_path, _proxy_port_from_env())
    proxy.start()
    proxy_url = f"http://127.0.0.1:{proxy.port}"
    for name in _PROXY_ENV_NAMES:
        os.environ[name] = proxy_url
    no_proxy = _merge_loopback_no_proxy(
        os.environ.get("no_proxy"), os.environ.get("NO_PROXY")
    )
    for name in _NO_PROXY_ENV_NAMES:
        os.environ[name] = no_proxy
    return proxy


def _rewrite_apt_list(content: str) -> str:
    """Upgrade active one-line apt repository entries without touching comments."""

    return _APT_LIST_SOURCE_RE.sub(
        lambda match: f"{match.group('prefix')}https://{match.group('remainder')}",
        content,
    )


def _rewrite_apt_deb822(content: str) -> str:
    """Upgrade URI tokens in deb822 ``URIs`` fields and their continuations."""

    rewritten: list[str] = []
    in_uris_field = False
    for line in content.splitlines(keepends=True):
        stripped = line.lstrip()
        field_match = _APT_DEB822_FIELD_RE.match(line)
        if field_match is not None:
            field_name = line[: field_match.end()].split(":", 1)[0].strip().lower()
            in_uris_field = field_name == "uris"
        elif not line[:1].isspace() or not stripped or stripped.startswith("#"):
            in_uris_field = False
        if in_uris_field:
            line = _APT_HTTP_TOKEN_RE.sub("https://", line)
        rewritten.append(line)
    return "".join(rewritten)


def _apt_source_paths(root: Path) -> list[Path]:
    apt_dir = root / "etc/apt"
    paths = [apt_dir / "sources.list"]
    source_dir = apt_dir / "sources.list.d"
    try:
        entries = sorted(source_dir.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        entries = []
    paths.extend(path for path in entries if path.suffix in {".list", ".sources"})
    return paths


def _upgrade_apt_sources_to_https(root: Path = Path("/")) -> tuple[Path, ...]:
    """Best-effort upgrade of apt HTTP sources in the session's writable overlay.

    Some HPC sites expose a Docker registry cache on the inherited HTTP-proxy
    address.  Such an endpoint can correctly tunnel HTTPS with CONNECT while
    answering every plain HTTP request with its registry health body.  Apt then
    reports that tiny response as ``NOSPLIT``.  Switching repository URIs to
    HTTPS makes apt use CONNECT, without granting the sandbox direct network
    access or changing the immutable SIF.

    This is deliberately opt-in because a general-purpose proxy may support
    HTTP and a private apt mirror may be HTTP-only.  Failures are non-fatal: an
    unusual read-only image should still be able to run non-package tasks.
    """

    policy = os.environ.get(_APT_HTTP_SOURCE_POLICY_ENV, "preserve").strip().lower()
    if policy != "https":
        return ()

    # Modern apt has native HTTPS support (often a symlink to the HTTP method).
    # Do not create unusable sources in an old/minimal image without it.
    if not (root / "usr/lib/apt/methods/https").exists():
        return ()

    changed: list[Path] = []
    for path in _apt_source_paths(root):
        # Avoid following image-provided links outside the apt configuration.
        if path.is_symlink() or not path.is_file():
            continue
        try:
            original = path.read_text(encoding="utf-8")
            if path.suffix == ".sources":
                updated = _rewrite_apt_deb822(original)
            else:
                updated = _rewrite_apt_list(original)
            if updated == original:
                continue
            # A direct write preserves the source file's ownership and mode;
            # Apptainer's per-session overlay handles copy-up from the SIF.
            path.write_text(updated, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            print(
                f"warning: could not upgrade apt source {path} to HTTPS: {exc}",
                file=sys.stderr,
            )
            continue
        changed.append(path)
    return tuple(changed)


def _configure_litellm_gateway() -> Any | None:
    socket_path = os.environ.get(_GATEWAY_SOCKET_ENV, "").strip()
    if not socket_path:
        return None

    import httpx
    import litellm

    # LiteLLM's OpenAI provider consults this global client first.  A UDS
    # transport ignores the URL host while preserving paths, headers, streaming
    # semantics, and the session-id bearer token used by Polar attribution.
    client = httpx.Client(
        transport=httpx.HTTPTransport(uds=socket_path),
        trust_env=False,
    )
    litellm.client_session = client
    return client


def main() -> int | None:
    """Configure isolated transports, then invoke mini-SWE-agent's CLI."""

    _inject_task_from_env()
    proxy: LoopbackProxy | None = None
    gateway_client: Any | None = None
    try:
        _restore_task_pythonpath()
        proxy = _configure_http_proxy()
        if proxy is not None or os.environ.get(
            _PROXY_BROKER_READY_ENV, ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            _upgrade_apt_sources_to_https()
        gateway_client = _configure_litellm_gateway()
        _configure_model_retry_policy()
        from minisweagent.run.mini import app

        return app()
    finally:
        if gateway_client is not None:
            gateway_client.close()
        if proxy is not None:
            proxy.close()


if __name__ == "__main__":
    raise SystemExit(main())
