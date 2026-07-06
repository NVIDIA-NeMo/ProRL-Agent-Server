#!/usr/bin/env python3
"""Serve Qwen chat-template token counts for services-only pool evaluation.

The production training topology obtains ``/v1/tokenize`` from the local
Qwen3.5 actor.  Forced-route evaluation intentionally has no actor, but the
Vanillux2 cumulative response budget still needs the same chat template.  This
loopback-only service loads tokenizer assets, never model weights, and exposes
only health, empty worker discovery, and scalar tokenization.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_MODEL_LEN = 262_144


class TokenizerRequestError(ValueError):
    """A caller supplied a malformed tokenization request."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("the tokenizer service must bind to 127.0.0.1")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    args.tokenizer_path = args.tokenizer_path.expanduser().resolve()
    if not (args.tokenizer_path / "tokenizer_config.json").is_file():
        parser.error(f"tokenizer assets are missing: {args.tokenizer_path}")
    return args


def _normalize_token_ids(encoded: object) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise RuntimeError("tokenizer did not return an input_ids sequence")
    values: object = list(encoded)
    if (
        isinstance(values, list)
        and len(values) == 1
        and isinstance(values[0], Sequence)
        and not isinstance(values[0], (str, bytes, bytearray))
    ):
        values = list(values[0])
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise RuntimeError("tokenizer returned invalid token ids")
    return values


def count_chat_tokens(tokenizer: Any, payload: object) -> int:
    if not isinstance(payload, dict):
        raise TokenizerRequestError("JSON body must be an object")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise TokenizerRequestError("messages must be a list")
    tools = payload.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise TokenizerRequestError("tools must be a list when provided")
    chat_template_kwargs = payload.get("chat_template_kwargs", {})
    if not isinstance(chat_template_kwargs, dict):
        raise TokenizerRequestError("chat_template_kwargs must be an object")

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        **chat_template_kwargs,
    }
    if tools is not None:
        template_kwargs["tools"] = tools
    encoded = tokenizer.apply_chat_template(messages, **template_kwargs)
    return len(_normalize_token_ids(encoded))


def tokenizer_max_model_len(tokenizer: Any) -> int:
    value = getattr(tokenizer, "model_max_length", DEFAULT_MAX_MODEL_LEN)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_MAX_MODEL_LEN
    if not math.isfinite(float(value)) or value <= 0 or value > 10_000_000:
        return DEFAULT_MAX_MODEL_LEN
    return int(value)


class TokenizerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], tokenizer: Any):
        super().__init__(address, TokenizerRequestHandler)
        self.tokenizer = tokenizer
        self.max_model_len = tokenizer_max_model_len(tokenizer)


class TokenizerRequestHandler(BaseHTTPRequestHandler):
    server: TokenizerHTTPServer

    def _json_response(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "tokenizer-only",
                    "max_model_len": self.server.max_model_len,
                },
            )
            return
        if self.path == "/workers":
            # Gateway worker discovery may query this SGLang-compatible path.
            # An empty set makes it fall back to this service's /v1/tokenize.
            self._json_response(HTTPStatus.OK, {"workers": []})
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/tokenize":
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        raw_length = self.headers.get("Content-Length", "")
        try:
            content_length = int(raw_length)
        except ValueError:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
            return
        if not 0 <= content_length <= MAX_REQUEST_BYTES:
            self._json_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            count = count_chat_tokens(self.server.tokenizer, payload)
        except (json.JSONDecodeError, UnicodeDecodeError, TokenizerRequestError) as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            # Do not echo request content or tokenizer internals into logs.
            self._json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"tokenization failed: {type(exc).__name__}"},
            )
            return
        self._json_response(
            HTTPStatus.OK,
            {"count": count, "max_model_len": self.server.max_model_len},
        )

    def log_message(self, _format: str, *_args: object) -> None:
        # Prompts are sensitive evaluation inputs; suppress access-log echoes.
        return


def load_tokenizer(path: Path) -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer_path)
    server = TokenizerHTTPServer((args.host, args.port), tokenizer)
    print(
        f"tokenizer-only service ready at http://{args.host}:{args.port} "
        f"max_model_len={server.max_model_len}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
