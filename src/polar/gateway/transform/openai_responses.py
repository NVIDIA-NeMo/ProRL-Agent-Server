"""OpenAI Responses API transformer.

Transforms between OpenAI Responses API (Codex CLI) and OpenAI Chat Completions.
Aligned with agent-harness-proxy/src/harness_proxy/transform/openai_responses.py.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from polar.gateway.transform.base import BaseTransformer


@dataclass
class _ResponsesToolCallState:
    name: str = ""
    call_id: str = ""
    arguments: str = ""
    started: bool = False
    fc_id: str = ""


class ResponsesStreamState:
    """Per-request Responses API streaming state."""

    def __init__(self, model: str):
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.model = model
        self.text_started = False
        self.text_content = ""
        self.output_index_offset = 0
        self.tool_calls: dict[int, _ResponsesToolCallState] = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.completed = False

    def process_chunk(self, chunk: dict[str, Any], is_first: bool = False) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        if is_first:
            events.append({
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": self.model,
                    "output": [],
                    "usage": self.usage.copy(),
                },
            })

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage["input_tokens"] = usage.get("prompt_tokens", self.usage["input_tokens"])
            self.usage["output_tokens"] = usage.get("completion_tokens", self.usage["output_tokens"])
            self.usage["total_tokens"] = usage.get("total_tokens", self.usage["total_tokens"])

        choices = chunk.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {}) or {}

        content = delta.get("content")
        if content:
            if not self.text_started:
                self.text_started = True
                self.output_index_offset = 1
                message_id = f"msg_{uuid.uuid4().hex[:24]}"
                events.append({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": message_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                })
                events.append({
                    "type": "response.content_part.added",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                })

            self.text_content += content
            events.append({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": content,
            })

        tool_calls_delta = delta.get("tool_calls") or []
        if not isinstance(tool_calls_delta, list):
            tool_calls_delta = [tool_calls_delta]

        for tool_call in tool_calls_delta:
            if not isinstance(tool_call, dict):
                continue

            tool_index = tool_call.get("index", 0)
            if not isinstance(tool_index, int):
                tool_index = 0

            tool_state = self.tool_calls.get(tool_index)
            if tool_state is None:
                tool_state = _ResponsesToolCallState(
                    call_id=tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                )
                self.tool_calls[tool_index] = tool_state
            elif tool_call.get("id"):
                tool_state.call_id = tool_call["id"]

            function = tool_call.get("function", {})
            name = function.get("name")
            if isinstance(name, str) and name:
                tool_state.name += name

            arguments = function.get("arguments")
            arguments_str = ""
            if isinstance(arguments, str) and arguments:
                arguments_str = arguments
            elif arguments not in (None, ""):
                arguments_str = json.dumps(arguments)
            if arguments_str:
                tool_state.arguments += arguments_str

            output_index = self.output_index_offset + tool_index
            if tool_state.name and not tool_state.started:
                tool_state.started = True
                tool_state.fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                events.append({
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": tool_state.fc_id,
                        "call_id": tool_state.call_id,
                        "name": tool_state.name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                })

                if tool_state.arguments:
                    events.append({
                        "type": "response.function_call_arguments.delta",
                        "output_index": output_index,
                        "delta": tool_state.arguments,
                    })
            elif tool_state.started and arguments_str:
                events.append({
                    "type": "response.function_call_arguments.delta",
                    "output_index": output_index,
                    "delta": arguments_str,
                })

        return events

    def finalize(self) -> list[dict[str, Any]]:
        if self.completed:
            return []

        events: list[dict[str, Any]] = []

        if self.text_started:
            events.append({
                "type": "response.content_part.done",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": self.text_content},
            })
            events.append({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": self.text_content}],
                },
            })

        for tool_index in sorted(self.tool_calls):
            tool_state = self.tool_calls[tool_index]
            if not tool_state.started:
                continue

            output_index = self.output_index_offset + tool_index
            events.append({
                "type": "response.function_call_arguments.done",
                "output_index": output_index,
                "arguments": tool_state.arguments,
            })
            events.append({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "type": "function_call",
                    "id": tool_state.fc_id or f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": tool_state.call_id,
                    "name": tool_state.name,
                    "arguments": tool_state.arguments,
                    "status": "completed",
                },
            })

        output: list[dict[str, Any]] = []
        if self.text_started:
            output.append({
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": self.text_content}],
            })
        for tool_index in sorted(self.tool_calls):
            tool_state = self.tool_calls[tool_index]
            if tool_state.started:
                output.append({
                    "type": "function_call",
                    "id": tool_state.fc_id or f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": tool_state.call_id,
                    "name": tool_state.name,
                    "arguments": tool_state.arguments,
                    "status": "completed",
                })

        events.append({
            "type": "response.completed",
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "completed",
                "model": self.model,
                "output": output,
                "usage": self.usage.copy(),
            },
        })
        self.completed = True
        return events


class OpenAIResponsesTransformer(BaseTransformer):
    """Transform OpenAI Responses API to/from SGLang chat completions."""

    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []

        instructions = body.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        input_data = body.get("input", "")
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            messages.extend(self._convert_input_items_to_messages(input_data))

        # vLLM / Qwen chat templates require exactly ONE system message at
        # position 0.  Responses API clients (e.g. codex CLI) may produce
        # multiple system messages (from "instructions" + "developer" items).
        # Merge them into a single system message.
        system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        if system_parts:
            messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + other_msgs
        else:
            messages = other_msgs
        result: dict[str, Any] = {"messages": messages}

        if "max_tokens" in body:
            result["max_tokens"] = body["max_tokens"]
        if "max_output_tokens" in body:
            result["max_tokens"] = body["max_output_tokens"]
        if "temperature" in body:
            result["temperature"] = body["temperature"]
        if "top_p" in body:
            result["top_p"] = body["top_p"]
        if "stream" in body:
            result["stream"] = body["stream"]

        tools = body.get("tools", [])
        if tools:
            result["tools"] = self._convert_tools(tools)
        if "tool_choice" in body:
            result["tool_choice"] = body["tool_choice"]

        return self._enhance_for_training(
            result,
            body.get("_polar_model_served"),
        )

    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        choices = response.get("choices", [])
        if not choices:
            return self._make_error_response("No choices in response")

        choice = choices[0]
        message = choice.get("message", {})

        output_items: list[dict[str, Any]] = []

        content = message.get("content")
        if content:
            output_items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            })

        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            name = func.get("name", "")
            if name in ("shell", "execute", "run_command"):
                output_items.append({
                    "type": "local_shell_call",
                    "call_id": tc.get("id", ""),
                    "status": "completed",
                    "action": {"type": "execute", "command": func.get("arguments", "{}")},
                })
            else:
                output_items.append({
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": tc.get("id", ""),
                    "name": name,
                    "arguments": func.get("arguments", "{}"),
                    "status": "completed",
                })

        usage = response.get("usage", {})
        return {
            "id": response.get("id", f"resp_{uuid.uuid4().hex}"),
            "object": "response",
            "status": "completed",
            "model": original_request.get("model", response.get("model", "unknown")),
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    def create_stream_state(self, original_request: dict[str, Any]) -> ResponsesStreamState:
        return ResponsesStreamState(
            model=original_request.get("model", "unknown"),
        )

    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> list[dict[str, Any]]:
        """Best-effort single-chunk Responses transform."""
        state = self.create_stream_state(original_request)
        events = state.process_chunk(chunk, is_first=is_first)
        choices = chunk.get("choices", [])
        if choices and choices[0].get("finish_reason"):
            events.extend(state.finalize())
        return events

    def _convert_input_items_to_messages(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        pending_tool_calls: list[dict[str, Any]] = []
        pending_tool_outputs: list[dict[str, Any]] = []

        for item in items:
            item_type = item.get("type")

            if item_type == "message":
                if pending_tool_calls or pending_tool_outputs:
                    messages.extend(self._flush_tool_block(pending_tool_calls, pending_tool_outputs))
                    pending_tool_calls = []
                    pending_tool_outputs = []

                role = item.get("role", "user")
                # Responses API uses "developer" for system-level instructions;
                # map to "system" for Chat Completions compatibility.
                if role == "developer":
                    role = "system"
                content = self._extract_text_from_content(item.get("content", ""))
                messages.append({"role": role, "content": content})

            elif item_type == "function_call":
                if pending_tool_outputs:
                    messages.extend(
                        self._flush_tool_block(
                            pending_tool_calls,
                            pending_tool_outputs,
                        )
                    )
                    pending_tool_calls = []
                    pending_tool_outputs = []
                pending_tool_calls.append({
                    "id": item.get("call_id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                })

            elif item_type == "function_call_output":
                pending_tool_outputs.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                })

            else:
                if pending_tool_calls or pending_tool_outputs:
                    messages.extend(
                        self._flush_tool_block(
                            pending_tool_calls,
                            pending_tool_outputs,
                        )
                    )
                    pending_tool_calls = []
                    pending_tool_outputs = []
                message = self._convert_response_item_to_message(item)
                if message:
                    messages.append(message)

        if pending_tool_calls or pending_tool_outputs:
            messages.extend(self._flush_tool_block(pending_tool_calls, pending_tool_outputs))

        return messages

    def _flush_tool_block(
        self,
        tool_calls: list[dict[str, Any]],
        tool_outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": list(tool_calls),
            })
        messages.extend(tool_outputs)
        return messages

    def _convert_response_item_to_message(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        item_type = item.get("type", "")

        if item_type == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            content_items = item.get("content", [])
            text_parts = []
            for c in content_items if isinstance(content_items, list) else []:
                if isinstance(c, dict) and c.get("type") in ("input_text", "output_text", "text"):
                    text_parts.append(c.get("text", ""))
            if text_parts:
                return {"role": role, "content": "\n".join(text_parts)}

        elif item_type == "function_call_output":
            output = item.get("output", {})
            output_str = output.get("output", "") if isinstance(output, dict) else str(output)
            return {
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": output_str,
            }

        # Fallback: plain {role, content} dict
        if not item_type and "role" in item and "content" in item:
            role = item["role"]
            if role == "developer":
                role = "system"
            content = item["content"]
            if isinstance(content, str):
                return {"role": role, "content": content}
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("input_text", "output_text", "text"):
                        text_parts.append(c.get("text", ""))
                    elif isinstance(c, str):
                        text_parts.append(c)
                if text_parts:
                    return {"role": role, "content": "\n".join(text_parts)}

        return None

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                converted.append({"type": "function", "function": tool["function"]})
                continue

            name = tool.get("name") or tool.get("id", "")
            if not name:
                continue

            parameters = tool.get("parameters")
            if parameters is None:
                input_schema = tool.get("inputSchema")
                if isinstance(input_schema, dict):
                    parameters = input_schema.get("jsonSchema", {})
                else:
                    parameters = {}

            func_def: dict[str, Any] = {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": parameters,
            }
            if "strict" in tool:
                func_def["strict"] = tool["strict"]
            converted.append({"type": "function", "function": func_def})

        return converted

    def _extract_text_from_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("input_text", "output_text", "text"):
                    parts.append(block.get("text", ""))
            return "\n".join(parts) if parts else ""
        return str(content) if content else ""

    def _make_error_response(self, message: str) -> dict[str, Any]:
        return {
            "type": "response.failed",
            "response": {
                "id": "resp_error",
                "object": "response",
                "status": "failed",
                "error": {"code": "internal_error", "message": message},
            },
        }
