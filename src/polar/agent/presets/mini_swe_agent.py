"""mini-SWE-agent harness — https://github.com/swe-agent/mini-swe-agent"""

from __future__ import annotations

import base64
import json
import math
import shlex
from pathlib import Path

from polar.agent.models import AgentRunResult
from polar.agent.base import BaseHarness
from polar.runtime.base import RUNTIME_AGENT_LOG_DIR
from polar.runtime.base import BaseRuntime
from polar.runtime.command_timing import RUNTIME_EXEC_CATEGORIES
from polar.runtime.models import ExecInput


MINI_SWE_TIMING_PATH = f"{RUNTIME_AGENT_LOG_DIR}/mini-swe-command-timing.jsonl"
MINI_SWE_TASK_B64_ENV = "POLAR_MINI_SWE_TASK_B64"


class MiniSweAgentHarness(BaseHarness):
    """Run mini-SWE-agent non-interactively (``mini-swe-agent --yolo``).

    mini-SWE-agent talks to models only through LiteLLM. The gateway exposes an
    OpenAI-compatible endpoint and injects ``OPENAI_BASE_URL`` + ``OPENAI_API_KEY``
    into the run step, but LiteLLM's OpenAI provider reads the base URL from
    ``OPENAI_API_BASE`` — so we alias it. The model is forced onto the ``openai/``
    provider so LiteLLM uses that transport; the gateway rewrites the model id to
    the served model regardless, so the id itself is cosmetic.

    Mirrors Harbor's mini-swe-agent setup: ``--yolo`` (no confirmations),
    ``--cost-limit 0`` (disabled) plus ``MSWEA_COST_TRACKING=ignore_errors`` so the
    served model's missing price table doesn't error, ``MSWEA_CONFIGURED=true`` to
    skip the first-run interactive setup, and ``--exit-immediately`` to finish
    without prompting.

    The shared CLI runtime uses Python 3.12, while many task images expose
    Python 3.10.  Explicitly clear ``PYTHONPATH`` in mini-SWE-agent's local
    environment so agent-issued commands cannot import packages from the CLI
    runtime.  This override applies to action subprocesses only; it does not
    alter the Python process running mini-SWE-agent itself.
    """

    def run_steps(self, instruction: str) -> list[ExecInput]:
        return self._run_mini_swe(
            instruction,
            config_spec="mini",
            environment_class="polar_mini_swe_timing.TimedLocalEnvironment",
            default_model_retry_attempts=3,
        )

    def _run_mini_swe(
        self,
        instruction: str,
        *,
        config_spec: str,
        environment_class: str,
        model_class: str | None = None,
        default_step_limit: int | None = None,
        default_model_kwargs: dict[str, object] | None = None,
        default_model_retry_attempts: int = 3,
        extra_config_specs: tuple[str, ...] = (),
    ) -> list[ExecInput]:
        """Build one portable mini-SWE invocation for a named protocol.

        Protocol-specific harnesses reuse the transport, retry, logging, and
        artifact behavior here while selecting their own mini-SWE config and
        model/environment adapters. The stock path calls this helper with its
        historical defaults.
        """

        model_id = (self.model_name or "gpt-5.4").rsplit("/", 1)[-1]
        cost_limit = self.settings.get("cost_limit", 0)
        model_retry_attempts = self.settings.get(
            "model_retry_attempts", default_model_retry_attempts
        )
        if isinstance(model_retry_attempts, bool) or not isinstance(
            model_retry_attempts, int
        ):
            raise ValueError("mini-swe-agent model_retry_attempts must be an integer")
        if model_retry_attempts <= 0:
            raise ValueError("mini-swe-agent model_retry_attempts must be positive")

        flags = [
            "--yolo",
            f"--environment-class {shlex.quote(environment_class)}",
            f"--model={shlex.quote(f'openai/{model_id}')}",
            f"--cost-limit {shlex.quote(str(cost_limit))}",
            "--exit-immediately",
        ]
        if model_class is not None:
            flags.append(f"--model-class {shlex.quote(model_class)}")
        # -c is a spec *list*, not a merge over the defaults: any -c suppresses
        # the packaged config, so prepend "-c mini" before layering overrides.
        config_flags = [f"-c {shlex.quote(config_spec)}"]
        step_limit = self.settings.get("step_limit", default_step_limit)
        if step_limit is not None:
            config_flags.append(f"-c agent.step_limit={int(step_limit)}")
        raw_model_kwargs = self.settings.get("model_kwargs") or {}
        if not isinstance(raw_model_kwargs, dict):
            raise ValueError("mini-swe-agent settings.model_kwargs must be a mapping")
        model_kwargs = dict(default_model_kwargs or {})
        model_kwargs.update(raw_model_kwargs)
        sampling_seed = self.settings.get("sampling_seed")
        if sampling_seed is not None:
            if isinstance(sampling_seed, bool):
                raise ValueError("mini-swe-agent sampling_seed must be an integer")
            model_kwargs["seed"] = int(sampling_seed)
        if model_kwargs:
            encoded_model_kwargs = json.dumps(
                model_kwargs,
                separators=(",", ":"),
                sort_keys=True,
            )
            config_flags.append("-c model.model_kwargs=" + shlex.quote(encoded_model_kwargs))
        # LocalEnvironment merges its config over os.environ for every action.
        # An empty value therefore removes the CLI runtime's Python path from
        # task commands without preventing the parent CLI from importing.
        config_flags.append("-c environment.env.PYTHONPATH=")
        config_flags.append(f"-c environment.timing_path={MINI_SWE_TIMING_PATH}")
        config_flags.extend(f"-c {spec}" for spec in extra_config_specs)
        flags.extend(config_flags)
        flags_str = " ".join(flags)

        return [
            ExecInput(
                command=(
                    # Preserve the agent's real exit code through ``tee`` so
                    # failed/timeout runs cannot be mislabeled completed.
                    "set -o pipefail; "
                    # uv tool drops the entry point in $HOME/.local/bin.
                    'export PATH="$HOME/.local/bin:$PATH" && '
                    # LiteLLM reads OPENAI_API_BASE; the gateway only sets OPENAI_BASE_URL.
                    'export OPENAI_API_BASE="$OPENAI_BASE_URL" && '
                    # MANDATORY fail-closed task-protocol preflight. The task
                    # travels only in POLAR_MINI_SWE_TASK_B64 (never argv), so a
                    # runtime that does not inject it runs every session with an
                    # EMPTY task (zero-trace training, observed on cont300
                    # 2026-07-20). Resolve the interpreter mini-swe-agent runs
                    # under — the bundled venv beside the portable wrapper, else
                    # the entry-point shebang for a uv-tool/pip console script —
                    # and require polar_mini_swe_runner._inject_task_from_env.
                    # Any layout that cannot prove injection (stale portable
                    # runtime OR upstream-only install) fails closed here rather
                    # than launching a taskless rollout.
                    '_mswea_bin="$(command -v mini-swe-agent)" || '
                    '{ echo "FATAL: mini-swe-agent is not on PATH" >&2; exit 64; }; '
                    '_mswea_py="$(dirname "${_mswea_bin}")/../venv/bin/python"; '
                    '[ -x "${_mswea_py}" ] || '
                    "_mswea_py=\"$(sed -n '1{s/^#! *//;s/ .*//;p}' \"${_mswea_bin}\")\"; "
                    'if [ ! -x "${_mswea_py}" ] || ! "${_mswea_py}" -c '
                    "'import sys, polar_mini_swe_runner as r; "
                    "sys.exit(0 if hasattr(r, \"_inject_task_from_env\") else 64)'; then "
                    'echo "FATAL: mini-SWE runtime cannot inject POLAR_MINI_SWE_TASK_B64 '
                    "(stale or non-polar install); rebuild it with prepare_mini_swe_agent.sh "
                    '(otherwise it trains on empty tasks)" >&2; exit 64; fi && '
                    f"mini-swe-agent {flags_str} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/mini-swe-agent.txt"
                ),
                env={
                    **self.env,
                    MINI_SWE_TASK_B64_ENV: base64.b64encode(
                        instruction.encode("utf-8")
                    ).decode("ascii"),
                    "MSWEA_CONFIGURED": "true",
                    "MSWEA_COST_TRACKING": "ignore_errors",
                    "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": str(model_retry_attempts),
                    # LiteLLM otherwise fetches its pricing map from GitHub at
                    # import time. Hundreds of fresh agent processes all wait
                    # on that unnecessary network request and can stampede the
                    # single gateway before their first model turn. The wheel
                    # already ships the same map as a local fallback, and cost
                    # accounting is disabled for these rollouts.
                    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                },
            )
        ]

    async def postprocess(self, runtime: BaseRuntime, result: AgentRunResult) -> None:
        """Attach a sanitized aggregate of inner mini-SWE action timings."""

        summary = load_mini_swe_command_timing(runtime)
        if summary is not None:
            result.metadata["mini_swe_command_timing"] = summary


def load_mini_swe_command_timing(runtime: BaseRuntime) -> dict[str, object] | None:
    """Read inner timings directly from the session bind, even after timeout."""

    host_path = runtime.resolve_host_path(MINI_SWE_TIMING_PATH)
    if host_path is None or not host_path.is_file():
        return None
    return _summarize_timing_records(host_path)


def _summarize_timing_records(
    path: Path,
    *,
    max_records: int = 10_000,
) -> dict[str, object]:
    """Read fixed-schema records while dropping unknown fields and categories."""

    ms_by_category = {category: 0.0 for category in RUNTIME_EXEC_CATEGORIES}
    count_by_category = {category: 0 for category in RUNTIME_EXEC_CATEGORIES}
    timeout_count = 0
    failure_count = 0
    try:
        with path.open() as stream:
            for index, line in enumerate(stream):
                if index >= max_records:
                    break
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                category = record.get("category")
                if category not in RUNTIME_EXEC_CATEGORIES:
                    continue
                try:
                    duration_ms = float(record.get("duration_ms"))
                    return_code = int(record.get("return_code"))
                except (TypeError, ValueError):
                    continue
                if duration_ms < 0.0 or not math.isfinite(duration_ms):
                    continue
                ms_by_category[category] += duration_ms
                count_by_category[category] += 1
                if bool(record.get("timed_out")):
                    timeout_count += 1
                elif return_code != 0:
                    failure_count += 1
    except (OSError, UnicodeError):
        pass
    return {
        "total_ms": sum(ms_by_category.values()),
        "count": sum(count_by_category.values()),
        "timeout_count": timeout_count,
        "failure_count": failure_count,
        "ms_by_category": ms_by_category,
        "count_by_category": count_by_category,
    }
