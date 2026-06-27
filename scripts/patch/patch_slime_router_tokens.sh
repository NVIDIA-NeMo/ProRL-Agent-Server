#!/usr/bin/env bash
# Preserve exact SGLang token metadata in Slime's OpenAI-compatible agent
# adapter responses. Polar consumes these fields to build training
# trajectories without local retokenization.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/../slime}"

if ! git -C "${SLIME_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: Slime git checkout not found at ${SLIME_DIR}" >&2
    exit 1
fi

LEGACY_PATCH_FILE="$(mktemp)"
REFACTORED_PATCH_FILE="$(mktemp)"
cleanup() {
    rm -f "${LEGACY_PATCH_FILE}" "${REFACTORED_PATCH_FILE}"
}
trap cleanup EXIT

cat > "${LEGACY_PATCH_FILE}" <<'PATCH'
diff --git a/slime/agent/adapters/common.py b/slime/agent/adapters/common.py
index ed5d2e06..cb915f2b 100644
--- a/slime/agent/adapters/common.py
+++ b/slime/agent/adapters/common.py
@@ -237,6 +237,7 @@ async def call_sglang_generate(
         output_ids=output_ids,
         finish_reason=finish,
         output_log_probs=output_log_probs,
+        meta_info=dict(meta),
     )


diff --git a/slime/agent/adapters/openai.py b/slime/agent/adapters/openai.py
index f9d91d8f..e11a7c06 100644
--- a/slime/agent/adapters/openai.py
+++ b/slime/agent/adapters/openai.py
@@ -351,6 +351,93 @@ def _usage(in_tok: int, out_tok: int) -> dict[str, int]:
     }


+def _decode_token(tok, token_id: int) -> str:
+    try:
+        return tok.decode([token_id], skip_special_tokens=False)
+    except TypeError:
+        return tok.decode([token_id])
+
+
+def _output_token_logprobs(turn: TurnRecord, tok) -> list[Any]:
+    raw = turn.meta_info.get("output_token_logprobs")
+    if isinstance(raw, list) and len(raw) == len(turn.output_ids):
+        return [list(item) if isinstance(item, (list, tuple)) else item for item in raw]
+
+    if len(turn.output_log_probs) != len(turn.output_ids):
+        return []
+
+    return [
+        [float(logprob), int(token_id), _decode_token(tok, int(token_id))]
+        for logprob, token_id in zip(turn.output_log_probs, turn.output_ids, strict=True)
+    ]
+
+
+def _token_text_from_logprob_item(item: Any, tok, token_id: int) -> str:
+    if isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[2], str):
+        return item[2]
+    if isinstance(item, dict):
+        token = item.get("token")
+        if isinstance(token, str):
+            return token
+        text = item.get("text")
+        if isinstance(text, str):
+            return text
+    return _decode_token(tok, token_id)
+
+
+def _logprob_from_item(item: Any, fallback: float) -> float:
+    if isinstance(item, (list, tuple)) and item:
+        return float(item[0])
+    if isinstance(item, dict) and item.get("logprob") is not None:
+        return float(item["logprob"])
+    return float(fallback)
+
+
+def _chat_logprobs(turn: TurnRecord, tok) -> dict[str, list[dict[str, Any]]] | None:
+    output_token_logprobs = _output_token_logprobs(turn, tok)
+    if len(output_token_logprobs) != len(turn.output_ids):
+        return None
+
+    content: list[dict[str, Any]] = []
+    for i, (token_id, item) in enumerate(zip(turn.output_ids, output_token_logprobs, strict=True)):
+        token = _token_text_from_logprob_item(item, tok, int(token_id))
+        logprob = _logprob_from_item(
+            item,
+            turn.output_log_probs[i] if i < len(turn.output_log_probs) else 0.0,
+        )
+        content.append(
+            {
+                "token": token,
+                "bytes": list(token.encode("utf-8")),
+                "logprob": logprob,
+                "token_id": int(token_id),
+                "top_logprobs": [],
+            }
+        )
+    return {"content": content}
+
+
+def _chat_meta_info(turn: TurnRecord, tok) -> dict[str, Any]:
+    meta = dict(turn.meta_info)
+    output_token_logprobs = _output_token_logprobs(turn, tok)
+    if output_token_logprobs:
+        meta["output_token_logprobs"] = output_token_logprobs
+    meta.setdefault("finish_reason", {"type": turn.finish_reason})
+    return meta
+
+
+def _attach_training_token_fields(choice: dict[str, Any], turn: TurnRecord, tok) -> None:
+    prompt_ids = [int(token_id) for token_id in turn.prompt_ids]
+    output_ids = [int(token_id) for token_id in turn.output_ids]
+    choice["input_token_ids"] = prompt_ids
+    choice["prompt_token_ids"] = list(prompt_ids)
+    choice["token_ids"] = output_ids
+    logprobs = _chat_logprobs(turn, tok)
+    if logprobs is not None:
+        choice["logprobs"] = logprobs
+    choice["meta_info"] = _chat_meta_info(turn, tok)
+
+
 def _responses_usage(in_tok: int, out_tok: int) -> dict[str, int]:
     return {
         "input_tokens": in_tok,
@@ -396,28 +483,29 @@ async def _handle_chat_completions(request: web.Request) -> web.StreamResponse:
     turn, parsed, in_tok, out_tok = await _run_turn(request, body, messages)
     if body.get("stream"):
         return await _stream_chat_completion(request, body, parsed, turn.finish_reason, in_tok, out_tok)
-    return web.json_response(_chat_completion_response(body, parsed, turn.finish_reason, in_tok, out_tok))
+    return web.json_response(_chat_completion_response(body, parsed, turn, request.app[TOKENIZER_KEY], in_tok, out_tok))


 def _chat_completion_response(
     body: dict,
     parsed: ParsedModelOutput,
-    finish: str,
+    turn: TurnRecord,
+    tok,
     in_tok: int,
     out_tok: int,
 ) -> dict[str, Any]:
+    choice = {
+        "index": 0,
+        "message": _chat_message(parsed),
+        "finish_reason": _finish_reason(parsed, turn.finish_reason),
+    }
+    _attach_training_token_fields(choice, turn, tok)
     return {
         "id": f"chatcmpl_{secrets.token_hex(12)}",
         "object": "chat.completion",
         "created": int(time.time()),
         "model": body.get("model", "slime-actor"),
-        "choices": [
-            {
-                "index": 0,
-                "message": _chat_message(parsed),
-                "finish_reason": _finish_reason(parsed, finish),
-            }
-        ],
+        "choices": [choice],
         "usage": _usage(in_tok, out_tok),
     }

diff --git a/slime/agent/trajectory.py b/slime/agent/trajectory.py
index 51db112f..8d102e90 100644
--- a/slime/agent/trajectory.py
+++ b/slime/agent/trajectory.py
@@ -27,6 +27,7 @@ class TurnRecord:
     output_ids: list[int]
     finish_reason: str
     output_log_probs: list[float] = dataclasses.field(default_factory=list)
+    meta_info: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
PATCH

cat > "${REFACTORED_PATCH_FILE}" <<'PATCH'
diff --git a/slime/agent/adapters/anthropic.py b/slime/agent/adapters/anthropic.py
index 0a48b09..a7ec4d0 100644
--- a/slime/agent/adapters/anthropic.py
+++ b/slime/agent/adapters/anthropic.py
@@ -68,7 +68,7 @@ class AnthropicAdapter(BaseAdapter):
             wire=(blocks, stop_reason),
         )

-    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
+    async def _respond(self, request, body, reply, _turn, in_tok, out_tok, stream) -> web.StreamResponse:
         blocks, stop_reason = reply.wire
         if stream:
             return await _render_stream(request, blocks, stop_reason, in_tok, out_tok)
diff --git a/slime/agent/adapters/common.py b/slime/agent/adapters/common.py
index fa31c2b..ca3ceab 100644
--- a/slime/agent/adapters/common.py
+++ b/slime/agent/adapters/common.py
@@ -198,6 +198,7 @@ class BaseAdapter:
         request: web.Request,
         body: dict,
         reply: Reply,
+        turn: TurnRecord,
         in_tok: int,
         out_tok: int,
         stream: bool,
@@ -359,7 +360,7 @@ class BaseAdapter:
             in_tok, out_tok = len(prompt_ids), len(turn.output_ids)

             stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")
-            return await self._respond(request, body, reply, in_tok, out_tok, stream)
+            return await self._respond(request, body, reply, turn, in_tok, out_tok, stream)
         finally:
             self.inflight.get(sid, set()).discard(task)

@@ -486,6 +487,7 @@ async def call_sglang_generate(
         output_ids=output_ids,
         finish_reason=finish,
         output_log_probs=output_log_probs,
+        meta_info=dict(meta),
     )


diff --git a/slime/agent/adapters/openai.py b/slime/agent/adapters/openai.py
index ad3d2e4..b9f1d75 100644
--- a/slime/agent/adapters/openai.py
+++ b/slime/agent/adapters/openai.py
@@ -25,6 +25,7 @@ from aiohttp import web
 from slime.agent.adapters.common import (
     BaseAdapter,
     Reply,
+    TurnRecord,
     flatten_content,
     manager_finish_reason,
     sid_from_bearer,
@@ -66,11 +67,11 @@ class OpenAIAdapter(BaseAdapter):
             wire=(wire_message, wire_finish),
         )

-    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
+    async def _respond(self, request, body, reply, turn, in_tok, out_tok, stream) -> web.StreamResponse:
         wire_message, wire_finish = reply.wire
         if stream:
             return await _render_stream(request, body, wire_message, wire_finish, in_tok, out_tok)
-        return web.json_response(_render_response(body, wire_message, wire_finish, in_tok, out_tok))
+        return web.json_response(_render_response(body, wire_message, wire_finish, turn, self.tokenizer, in_tok, out_tok))


 # --- Translation (OpenAI wire -> chat-template messages) ---
@@ -298,25 +299,114 @@ def _usage(in_tok: int, out_tok: int) -> dict[str, int]:
     }


+def _decode_token(tok, token_id: int) -> str:
+    try:
+        return tok.decode([token_id], skip_special_tokens=False)
+    except TypeError:
+        return tok.decode([token_id])
+
+
+def _output_token_logprobs(turn: TurnRecord, tok) -> list[Any]:
+    raw = turn.meta_info.get("output_token_logprobs")
+    if isinstance(raw, list) and len(raw) == len(turn.output_ids):
+        return [list(item) if isinstance(item, (list, tuple)) else item for item in raw]
+
+    if len(turn.output_log_probs) != len(turn.output_ids):
+        return []
+
+    return [
+        [float(logprob), int(token_id), _decode_token(tok, int(token_id))]
+        for logprob, token_id in zip(turn.output_log_probs, turn.output_ids, strict=True)
+    ]
+
+
+def _token_text_from_logprob_item(item: Any, tok, token_id: int) -> str:
+    if isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[2], str):
+        return item[2]
+    if isinstance(item, dict):
+        token = item.get("token")
+        if isinstance(token, str):
+            return token
+        text = item.get("text")
+        if isinstance(text, str):
+            return text
+    return _decode_token(tok, token_id)
+
+
+def _logprob_from_item(item: Any, fallback: float) -> float:
+    if isinstance(item, (list, tuple)) and item:
+        return float(item[0])
+    if isinstance(item, dict) and item.get("logprob") is not None:
+        return float(item["logprob"])
+    return float(fallback)
+
+
+def _chat_logprobs(turn: TurnRecord, tok) -> dict[str, list[dict[str, Any]]] | None:
+    output_token_logprobs = _output_token_logprobs(turn, tok)
+    if len(output_token_logprobs) != len(turn.output_ids):
+        return None
+
+    content: list[dict[str, Any]] = []
+    for i, (token_id, item) in enumerate(zip(turn.output_ids, output_token_logprobs, strict=True)):
+        token = _token_text_from_logprob_item(item, tok, int(token_id))
+        logprob = _logprob_from_item(
+            item,
+            turn.output_log_probs[i] if i < len(turn.output_log_probs) else 0.0,
+        )
+        content.append(
+            {
+                "token": token,
+                "bytes": list(token.encode("utf-8")),
+                "logprob": logprob,
+                "token_id": int(token_id),
+                "top_logprobs": [],
+            }
+        )
+    return {"content": content}
+
+
+def _chat_meta_info(turn: TurnRecord, tok) -> dict[str, Any]:
+    meta = dict(turn.meta_info)
+    output_token_logprobs = _output_token_logprobs(turn, tok)
+    if output_token_logprobs:
+        meta["output_token_logprobs"] = output_token_logprobs
+    meta.setdefault("finish_reason", {"type": turn.finish_reason})
+    return meta
+
+
+def _attach_training_token_fields(choice: dict[str, Any], turn: TurnRecord, tok) -> None:
+    prompt_ids = [int(token_id) for token_id in turn.prompt_ids]
+    output_ids = [int(token_id) for token_id in turn.output_ids]
+    choice["input_token_ids"] = prompt_ids
+    choice["prompt_token_ids"] = list(prompt_ids)
+    choice["token_ids"] = output_ids
+    logprobs = _chat_logprobs(turn, tok)
+    if logprobs is not None:
+        choice["logprobs"] = logprobs
+    choice["meta_info"] = _chat_meta_info(turn, tok)
+
+
 def _render_response(
     body: dict,
     wire_message: dict[str, Any],
     wire_finish: str,
+    turn: TurnRecord,
+    tok,
     in_tok: int,
     out_tok: int,
 ) -> dict[str, Any]:
+    choice = {
+        "index": 0,
+        "message": wire_message,
+        "finish_reason": wire_finish,
+    }
+    _attach_training_token_fields(choice, turn, tok)
     return {
         "id": f"chatcmpl_{secrets.token_hex(12)}",
         "object": "chat.completion",
         "created": int(time.time()),
         "model": body.get("model", "slime-actor"),
-        "choices": [
-            {
-                "index": 0,
-                "message": wire_message,
-                "finish_reason": wire_finish,
-            }
-        ],
+        "choices": [choice],
         "usage": _usage(in_tok, out_tok),
     }

diff --git a/slime/agent/trajectory.py b/slime/agent/trajectory.py
index 33ec036..c4f08f0 100644
--- a/slime/agent/trajectory.py
+++ b/slime/agent/trajectory.py
@@ -35,6 +35,7 @@ class TurnRecord:
     output_ids: list[int]
     finish_reason: str
     output_log_probs: list[float] = dataclasses.field(default_factory=list)
+    meta_info: dict[str, Any] = dataclasses.field(default_factory=dict)


 # ===========================================================================
PATCH

try_patch() {
    local label="$1"
    local patch_file="$2"

    if git -C "${SLIME_DIR}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
        echo "Slime router token patch already applied (${label})."
        exit 0
    fi

    if git -C "${SLIME_DIR}" apply --check "${patch_file}" >/dev/null 2>&1; then
        git -C "${SLIME_DIR}" apply "${patch_file}"
        echo "Applied Slime router token patch to ${SLIME_DIR} (${label})."
        exit 0
    fi
}

try_patch "legacy adapter layout" "${LEGACY_PATCH_FILE}"
try_patch "refactored adapter layout" "${REFACTORED_PATCH_FILE}"

echo "ERROR: Slime router token patch does not apply to ${SLIME_DIR}." >&2
echo "legacy adapter layout check:" >&2
git -C "${SLIME_DIR}" apply --check "${LEGACY_PATCH_FILE}" >&2 || true
echo "refactored adapter layout check:" >&2
git -C "${SLIME_DIR}" apply --check "${REFACTORED_PATCH_FILE}" >&2 || true
exit 1
