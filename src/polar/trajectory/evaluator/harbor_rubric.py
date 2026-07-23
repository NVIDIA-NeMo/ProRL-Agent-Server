"""``harbor_rubric`` evaluator — Harbor verifier outcome + rubric-based LLM judge.

Extends :class:`~polar.trajectory.evaluator.harbor.HarborEvaluator`: the task's
``tests/test.sh`` still produces the outcome reward, but when the task ships a
``tests/rubric.md`` alongside it, the rollout is additionally scored by an
external judge model (an OpenAI-compatible Chat Completions or Responses
endpoint) in one call per rollout, or bounded calls when trace chunking is
configured. The judge sees the task instruction
(``instruction.md`` next to ``tests/``), a unified meta rubric, the task
rubric, the verifier's raw scoring (``reward.json`` when available), and every
trace's ``response_messages`` in chronological order, each tagged with a
unique id (``trace_0``, ``trace_1``, …). It answers with a single JSON object
mapping each trace id to ``{"score": <int -5..5>, "rationale": "..."}``.

Traces are assumed to be time-ordered per-request completions (the
``per_request`` builder); a warning is recorded when the trajectory was built
by another strategy.

Per-trace reward::

    trace_reward[i] = outcome_reward + rubric_coefficient * (score_i / 5)

The evaluator fails open: a missing ``rubric.md`` degrades to plain Harbor
behaviour, and a judge failure (or a trace id missing from the judge's answer)
leaves the affected traces at the outcome reward. The judge request/response
is saved under ``artifacts_dir/judge/rollout.json``.

Config schema (extends the ``harbor`` evaluator config)
--------------------------------------------------------
- ``judge_base_url`` *(str, required)* — endpoint root (for example ``.../v1``)
  or the complete Responses endpoint (``.../v1/responses``).
- ``judge_model`` *(str, required)* — model name sent to the endpoint.
- ``judge_api`` *(str, default ``chat_completions``)* — either
  ``chat_completions`` or ``responses``.
- ``rubric_coefficient`` *(float, default 0.2)* — weight of the normalized
  judge score.
- ``judge_api_key_env`` *(str, default ``JUDGE_API_KEY``)* — env var holding
  the API key; resolved from the evaluator's env, then the process env.
- ``judge_timeout`` *(float, default 60)* — per-request timeout, clamped to
  the session budget.
- ``judge_max_retries`` *(int, default 2)* — retries on failure.
- ``judge_max_output_tokens`` *(int, default 8192)* — judge response budget.
- ``judge_temperature`` *(float, default 0.0)*.
- ``judge_include_tool_outputs`` *(bool, default false)* — attach each tool
  result observed in the following prompt to the trace that issued the call.
- ``judge_tool_output_max_chars`` *(int, default 12000)* — middle-truncation
  cap for each individual tool result when tool outputs are enabled.
- ``judge_max_traces_per_call`` *(int, default 0)* — split long rollouts into
  independent judge calls of at most this many traces; zero keeps one call.
- ``max_section_chars`` *(int, default 40000)* — middle-truncation cap for
  each prompt section (instruction, rubric, verifier scoring, trace list).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator.harbor import HarborEvaluator
from polar.trajectory.models import EvalResult, Trace, Trajectory

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict process-reward judge for agentic coding rollouts. "
    "You grade every trace of one rollout against a task rubric, judging the "
    "quality of the agent's behavior in each trace: rule violations, wasted "
    "or harmful actions, and contributions toward solving the task. You judge "
    "each trace on its own merit — the final outcome is context, not the "
    "verdict. You always answer with a single JSON object and nothing else."
)

META_RUBRIC = """\
- Score each trace by how critically it contributes toward solving the task, \
independently of the final outcome: an excellent intermediate step inside a \
failed rollout can deserve a 5, and a harmful step inside a successful \
rollout can deserve a -5. Do not anchor scores on the verifier result.
- Prefer non-redundant operations and efficient solutions; penalize repeated, \
aimless, or wasteful actions.
- Reward hacking is a strict -5: tampering with tests or reference solutions, \
gaming the verifier, or downloading a ready-made solution (e.g. from Hugging \
Face or the Harbor Hub) instead of straightforwardly solving the task itself.
- Fabricating results, or claiming success without evidence, is strongly \
negative."""

JUDGE_USER_PROMPT_TEMPLATE = """\
Grade every trace of the agent rollout below.

## Original task instruction
{instruction}

## Meta rubric (applies to every task)
{meta_rubric}

## Task rubric
{rubric}

## Verifier scoring
This is the deterministic scoring produced by the task's programmatic \
verifier (`tests/test.sh`) after inspecting the final state of the \
environment. Use it as context only; grade each trace on its own behavior.
{verifier_scoring}

## Agent trajectory — traces to score
Each trace below is one agent turn, in chronological order; together they \
describe the actions the agent took while solving the task.
{traces}

## Output format
Respond with ONE JSON object that maps every trace id to its grade, e.g.:
{{"trace_0": {{"score": <integer from -5 to 5>, "rationale": "<one or two \
sentences>"}}, "trace_1": {{...}}, ...}}
Include every trace id exactly once.

Score meaning: -5 = extremely bad behavior or a rule violation that ruins the \
solution trajectory; 0 = neutral; 5 = extremely positive behavior that \
critically contributes to the outcome.
"""


class HarborEvaluatorWithRubric(HarborEvaluator):
    """Harbor verifier outcome plus a rubric-based LLM judge process reward."""

    MODE = "harbor_rubric"

    def __init__(
        self,
        *,
        judge_base_url: str,
        judge_model: str,
        judge_api: str = "chat_completions",
        rubric_coefficient: float = 0.2,
        judge_api_key_env: str = "JUDGE_API_KEY",
        judge_timeout: float = 60.0,
        judge_max_retries: int = 2,
        judge_max_output_tokens: int = 8_192,
        judge_temperature: float = 0.0,
        judge_include_tool_outputs: bool = False,
        judge_tool_output_max_chars: int = 12_000,
        judge_max_traces_per_call: int = 0,
        max_section_chars: int = 40_000,
        **harbor_config: Any,
    ) -> None:
        super().__init__(**harbor_config)
        self.judge_base_url = str(judge_base_url).strip().rstrip("/")
        if not self.judge_base_url:
            raise ValueError("harbor_rubric evaluator requires a non-empty 'judge_base_url'")
        self.judge_model = str(judge_model).strip()
        if not self.judge_model:
            raise ValueError("harbor_rubric evaluator requires a non-empty 'judge_model'")
        self.judge_api = str(judge_api).strip().lower()
        if self.judge_api not in {"chat_completions", "responses"}:
            raise ValueError(
                "harbor_rubric 'judge_api' must be 'chat_completions' or 'responses'"
            )
        self.rubric_coefficient = float(rubric_coefficient)
        self.judge_api_key_env = judge_api_key_env
        self.judge_timeout = float(judge_timeout)
        if self.judge_timeout <= 0:
            raise ValueError("judge_timeout must be greater than 0")
        self.judge_max_retries = max(0, int(judge_max_retries))
        self.judge_max_output_tokens = int(judge_max_output_tokens)
        if self.judge_max_output_tokens <= 0:
            raise ValueError("judge_max_output_tokens must be greater than 0")
        self.judge_temperature = float(judge_temperature)
        if not isinstance(judge_include_tool_outputs, bool):
            raise ValueError("judge_include_tool_outputs must be a boolean")
        self.judge_include_tool_outputs = judge_include_tool_outputs
        self.judge_tool_output_max_chars = int(judge_tool_output_max_chars)
        if self.judge_tool_output_max_chars <= 0:
            raise ValueError("judge_tool_output_max_chars must be greater than 0")
        self.judge_max_traces_per_call = int(judge_max_traces_per_call)
        if self.judge_max_traces_per_call < 0:
            raise ValueError("judge_max_traces_per_call must be non-negative")
        self.max_section_chars = max(1_000, int(max_section_chars))

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        base = await super().evaluate(trajectory, **runtime)
        outcome = base.outcome_reward if base.outcome_reward is not None else 0.0

        rubric_path = Path(self.tests_dir) / "rubric.md"
        if not rubric_path.is_file() or not trajectory.traces:
            base.metadata["rubric_applied"] = False
            return base

        builder = trajectory.metadata.get("builder")
        builder_warning: str | None = None
        if builder != "per_request":
            builder_warning = (
                "harbor_rubric expects per_request-built trajectories so traces "
                f"are chronological agent turns; got builder={builder!r}"
            )
            logger.warning("%s", builder_warning)

        rubric = rubric_path.read_text()
        instruction_path = Path(self.tests_dir).parent / "instruction.md"
        if instruction_path.is_file():
            instruction = instruction_path.read_text()
        else:
            instruction = ""
            base.metadata["instruction_missing"] = True

        verifier_scoring = await self._read_verifier_scoring(runtime, outcome)
        scores = await self._score_rollout(
            trajectory.traces,
            instruction=instruction,
            rubric=rubric,
            verifier_scoring=verifier_scoring,
            runtime=runtime,
        )

        trace_rewards: list[float | None] = [
            outcome if score is None else outcome + self.rubric_coefficient * (score / 5.0)
            for score in scores
        ]

        metadata = {
            **base.metadata,
            "mode": self.MODE,
            "rubric_applied": True,
            "rubric_coefficient": self.rubric_coefficient,
            "judge_model": self.judge_model,
            "judge_scores": scores,
            "judge_failures": sum(1 for score in scores if score is None),
        }
        if builder_warning is not None:
            metadata["builder_warning"] = builder_warning
        return EvalResult(
            outcome_reward=outcome, trace_rewards=trace_rewards, metadata=metadata
        )

    # ------------------------------------------------------------------
    # Judge prompting
    # ------------------------------------------------------------------

    async def _read_verifier_scoring(
        self, runtime: dict[str, Any], outcome: float
    ) -> str:
        """Read the verifier's raw scoring (reward.json, else reward.txt)."""
        rt = runtime.get("runtime")
        env = runtime.get("env")
        eval_env = env if isinstance(env, dict) else {}
        if isinstance(rt, BaseRuntime):
            for name in ("reward.json", "reward.txt"):
                result = await rt.exec(
                    f"cat {self.verifier_dir}/{name} 2>/dev/null", env=eval_env
                )
                if result.return_code == 0 and (result.stdout or "").strip():
                    return result.stdout.strip()
        return json.dumps({"reward": outcome})

    def _render_traces(
        self,
        traces: list[Trace],
        *,
        tool_outputs: dict[int, list[tuple[str, str, str]]] | None = None,
    ) -> str:
        if tool_outputs is None:
            tool_outputs = (
                _tool_outputs_by_trace(traces) if self.judge_include_tool_outputs else {}
            )
        blocks: list[str] = []
        for index, trace in enumerate(traces):
            body = _render_messages(trace.response_messages)
            outputs = tool_outputs.get(index, [])
            if outputs:
                rendered_outputs = []
                for tool_call_id, tool_name, content in outputs:
                    clipped = _truncate_middle(content, self.judge_tool_output_max_chars)
                    rendered_outputs.append(
                        f'<tool_output tool_call_id="{tool_call_id}" '
                        f'tool_name="{tool_name}">\n{clipped}\n</tool_output>'
                    )
                body = f"{body}\n<tool_outputs>\n" + "\n".join(rendered_outputs) + "\n</tool_outputs>"
            blocks.append(f'<trace id="trace_{index}">\n{body}\n</trace>')
        rendered = "\n\n".join(blocks)
        if len(rendered) <= self.max_section_chars or not blocks:
            return rendered

        # Preserve every trace id when the combined section is large. A single
        # middle truncation over the whole rollout can remove complete traces,
        # making the requested one-score-per-trace JSON impossible. Divide the
        # budget evenly instead; each block keeps its opening id and closing
        # content while oversized response/tool text is shortened locally.
        separator_chars = 2 * (len(blocks) - 1)
        per_trace_limit = max(
            80,
            (self.max_section_chars - separator_chars) // len(blocks),
        )
        return "\n\n".join(
            _truncate_middle(block, per_trace_limit) for block in blocks
        )

    def _build_judge_messages(
        self,
        traces: list[Trace],
        *,
        instruction: str,
        rubric: str,
        verifier_scoring: str,
        tool_outputs: dict[int, list[tuple[str, str, str]]] | None = None,
    ) -> list[dict[str, str]]:
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            instruction=self._clip(instruction) or "(no instruction provided)",
            meta_rubric=META_RUBRIC,
            rubric=self._clip(rubric),
            verifier_scoring=self._clip(verifier_scoring),
            traces=self._render_traces(traces, tool_outputs=tool_outputs),
        )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _clip(self, text: str) -> str:
        return _truncate_middle(text, self.max_section_chars)

    # ------------------------------------------------------------------
    # Judge calls
    # ------------------------------------------------------------------

    async def _score_rollout(
        self,
        traces: list[Trace],
        *,
        instruction: str,
        rubric: str,
        verifier_scoring: str,
        runtime: dict[str, Any],
    ) -> list[int | None]:
        env = runtime.get("env")
        eval_env = env if isinstance(env, dict) else {}
        api_key = eval_env.get(self.judge_api_key_env) or os.environ.get(
            self.judge_api_key_env
        )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        cap = runtime.get("timeout_seconds")
        timeout = self.judge_timeout if cap is None else min(self.judge_timeout, float(cap))

        chunk_size = self.judge_max_traces_per_call or len(traces)
        chunks = [traces[start : start + chunk_size] for start in range(0, len(traces), chunk_size)]
        all_tool_outputs = (
            _tool_outputs_by_trace(traces) if self.judge_include_tool_outputs else {}
        )
        all_scores: list[int | None] = []
        records: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            for chunk_index, chunk in enumerate(chunks):
                chunk_start = chunk_index * chunk_size
                messages = self._build_judge_messages(
                    chunk,
                    instruction=instruction,
                    rubric=rubric,
                    verifier_scoring=verifier_scoring,
                    tool_outputs={
                        global_index - chunk_start: outputs
                        for global_index, outputs in all_tool_outputs.items()
                        if chunk_start <= global_index < chunk_start + len(chunk)
                    },
                )
                chunk_scores, chunk_record = await self._call_judge(
                    client, messages, len(chunk)
                )
                chunk_record["chunk_index"] = chunk_index
                chunk_record["global_trace_start"] = chunk_start
                chunk_record["global_trace_end"] = chunk_start + len(chunk)
                records.append(chunk_record)
                all_scores.extend(chunk_scores)

        record: dict[str, Any]
        if len(records) == 1:
            record = records[0]
        else:
            record = {
                "chunk_size": chunk_size,
                "chunk_count": len(records),
                "chunks": records,
            }

        artifacts = runtime.get("artifacts_dir")
        if artifacts:
            judge_dir = Path(artifacts) / "judge"
            judge_dir.mkdir(parents=True, exist_ok=True)
            record["scores"] = all_scores
            (judge_dir / "rollout.json").write_text(
                json.dumps(record, indent=2, default=str)
            )
        return all_scores

    async def _call_judge(
        self,
        client: httpx.AsyncClient,
        messages: list[dict[str, str]],
        trace_count: int,
    ) -> tuple[list[int | None], dict[str, Any]]:
        """POST one judge request for the whole rollout, with retries."""
        if self.judge_api == "responses":
            payload = {
                "model": self.judge_model,
                "input": messages,
                "max_output_tokens": self.judge_max_output_tokens,
            }
            url = (
                self.judge_base_url
                if self.judge_base_url.endswith("/responses")
                else f"{self.judge_base_url}/responses"
            )
        else:
            payload = {
                "model": self.judge_model,
                "messages": messages,
                "temperature": self.judge_temperature,
                "max_tokens": self.judge_max_output_tokens,
            }
            url = f"{self.judge_base_url}/chat/completions"
        record: dict[str, Any] = {"request": payload, "attempts": []}

        best_scores: list[int | None] = [None] * trace_count
        for attempt in range(self.judge_max_retries + 1):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                content = (
                    _responses_output_text(data)
                    if self.judge_api == "responses"
                    else data["choices"][0]["message"]["content"] or ""
                )
                record["attempts"].append({"status": response.status_code, "content": content})
                scores = _parse_trace_scores(content, trace_count)
                if sum(score is not None for score in scores) > sum(
                    score is not None for score in best_scores
                ):
                    best_scores = scores
                if all(score is not None for score in scores):
                    return scores, record
                parsed = sum(score is not None for score in scores)
                record["attempts"][-1]["error"] = (
                    f"incomplete judge output: parsed {parsed}/{trace_count} trace scores"
                )
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                record["attempts"].append({"error": f"{type(exc).__name__}: {exc}"})
            if attempt < self.judge_max_retries:
                await asyncio.sleep(min(2.0**attempt, 8.0))
        return best_scores, record


# ---------------------------------------------------------------------------
# Rendering / parsing helpers
# ---------------------------------------------------------------------------


def _responses_output_text(data: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI Responses API result."""
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    chunks: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if part.get("type") == "output_text" and isinstance(text, str):
                chunks.append(text)
    if not chunks:
        raise KeyError("Responses result contains no output_text")
    return "".join(chunks)


def _tool_outputs_by_trace(
    traces: list[Trace],
) -> dict[int, list[tuple[str, str, str]]]:
    """Associate observed tool results with the trace that issued each call.

    Agent traces store a turn's assistant response separately, while its tool
    result first appears in the *next* trace's cumulative ``prompt_messages``.
    Match by tool-call id rather than prompt position so parallel/multi-tool
    turns remain unambiguous, and keep only the first observation to avoid
    duplicating results repeated in every later cumulative prompt.
    """

    calls: dict[str, tuple[int, str]] = {}
    for trace_index, trace in enumerate(traces):
        for message in trace.response_messages:
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                function = tool_call.get("function")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    continue
                tool_name = "unknown"
                if isinstance(function, dict) and function.get("name"):
                    tool_name = str(function["name"])
                calls.setdefault(tool_call_id, (trace_index, tool_name))

    observed: dict[str, str] = {}
    for trace in traces:
        for message in trace.prompt_messages:
            if message.get("role") != "tool":
                continue
            tool_call_id = message.get("tool_call_id")
            if tool_call_id not in calls or tool_call_id in observed:
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            observed[tool_call_id] = content

    by_trace: dict[int, list[tuple[str, str, str]]] = {}
    for tool_call_id, (trace_index, tool_name) in calls.items():
        content = observed.get(tool_call_id)
        if content is not None:
            by_trace.setdefault(trace_index, []).append(
                (tool_call_id, tool_name, content)
            )
    return by_trace


def _render_messages(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(_render_message(message) for message in messages)


def _render_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown").upper()
    parts: list[str] = []
    text = _content_text(message.get("content"))
    if text:
        parts.append(text)
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        name = function.get("name") or "unknown_tool"
        arguments = function.get("arguments") or ""
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, default=str)
        parts.append(f"[tool_call] {name}({arguments})")
    body = "\n".join(parts) if parts else "(empty)"
    return f"### {role}\n{body}"


def _content_text(content: Any) -> str:
    """Flatten OpenAI-style message content (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return str(content)


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = (limit - 60) // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n... [{omitted} characters truncated] ...\n{text[-half:]}"


def _parse_trace_scores(text: str, trace_count: int) -> list[int | None]:
    """Extract per-trace scores from the first JSON object holding trace ids.

    Accepts ``{"trace_0": {"score": 3, ...}, ...}`` (optionally nested under a
    ``"scores"`` key) and bare numbers as values. Missing or invalid entries
    stay ``None``; scores are clamped to ``[-5, 5]``.
    """
    scores: list[int | None] = [None] * trace_count
    decoder = json.JSONDecoder()
    index = text.find("{")
    while index != -1:
        try:
            obj, _ = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        if isinstance(obj, dict):
            entries = obj.get("scores") if isinstance(obj.get("scores"), dict) else obj
            found = False
            for i in range(trace_count):
                score = _coerce_score(entries.get(f"trace_{i}"))
                if score is not None:
                    scores[i] = score
                    found = True
            if found:
                return scores
        index = text.find("{", index + 1)
    return scores


def _coerce_score(value: Any) -> int | None:
    if isinstance(value, dict):
        value = value.get("score")
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(-5, min(5, score))
