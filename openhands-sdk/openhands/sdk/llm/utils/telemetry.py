import json
import os
import time
import uuid
from collections.abc import Callable
from typing import Any, ClassVar

from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import ModelResponse, Usage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from openhands.sdk.llm.utils.metrics import Metrics
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class Telemetry(BaseModel):
    """
    Handles token accounting and optional logging.
    All runtime state (like start times) lives in private attrs.
    """

    # --- Config fields ---
    model_name: str = Field(default="unknown", description="Name of the LLM model")
    log_enabled: bool = Field(default=False, description="Whether to log completions")
    log_dir: str | None = Field(
        default=None, description="Directory to write logs if enabled"
    )

    metrics: Metrics = Field(..., description="Metrics collector instance")

    # --- Runtime fields (not serialized) ---
    _req_start: float = PrivateAttr(default=0.0)
    _req_ctx: dict[str, Any] = PrivateAttr(default_factory=dict)
    _last_latency: float = PrivateAttr(default=0.0)
    _log_completions_callback: Callable[[str, str], None] | None = PrivateAttr(
        default=None
    )
    _stats_update_callback: Callable[[], None] | None = PrivateAttr(default=None)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True
    )

    # ---------- Lifecycle ----------
    def set_log_completions_callback(
        self, callback: Callable[[str, str], None] | None
    ) -> None:
        """Set a callback function for logging instead of writing to file.

        Args:
            callback: A function that takes (filename, log_data) and handles the log.
                     Used for streaming logs in remote execution contexts.
        """
        self._log_completions_callback = callback

    def set_stats_update_callback(self, callback: Callable[[], None] | None) -> None:
        """Set a callback function to be notified when stats are updated.

        Args:
            callback: A function called whenever metrics are updated.
                     Used for streaming stats updates in remote execution contexts.
        """
        self._stats_update_callback = callback

    def on_request(self, log_ctx: dict | None) -> None:
        self._req_start = time.time()
        self._req_ctx = log_ctx or {}

    def on_response(
        self,
        resp: ModelResponse | ResponsesAPIResponse,
        raw_resp: ModelResponse | None = None,
    ) -> Metrics:
        """
        Side-effects:
          - records tokens into Metrics
          - optionally writes a JSON log file
        """
        # Track latency for logging purposes
        self._last_latency = time.time() - (self._req_start or time.time())

        # Record token usage
        usage = getattr(resp, "usage", None)
        if usage and self._has_meaningful_usage(usage):
            self._record_usage(usage)

        # Optional logging
        if self.log_enabled:
            self.log_llm_call(resp, raw_resp=raw_resp)

        # Notify about stats update
        if self._stats_update_callback is not None:
            try:
                self._stats_update_callback()
            except Exception:
                logger.exception("Stats update callback failed", exc_info=True)

        return self.metrics.deep_copy()

    def on_error(self, _err: BaseException) -> None:
        # Stub for error tracking / counters
        return

    # ---------- Helpers ----------
    def _has_meaningful_usage(self, usage: Usage | ResponseAPIUsage | None) -> bool:
        """Check if usage has meaningful (non-zero) token counts."""
        if usage is None:
            return False
        try:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = getattr(usage, "output_tokens", 0)

            pt = int(prompt_tokens or 0)
            ct = int(completion_tokens or 0)
            return pt > 0 or ct > 0
        except Exception:
            return False

    def _record_usage(self, usage: Usage | ResponseAPIUsage) -> None:
        """Record token usage."""
        prompt_tokens = int(
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", 0)
            or 0
        )
        completion_tokens = int(
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", 0)
            or 0
        )

        self.metrics.add_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def log_llm_call(
        self,
        resp: ModelResponse | ResponsesAPIResponse,
        raw_resp: ModelResponse | ResponsesAPIResponse | None = None,
    ) -> None:
        # Skip if neither file logging nor callback is configured
        if not self.log_dir and not self._log_completions_callback:
            return
        try:
            # Prepare filename and log data
            filename = (
                f"{self.model_name.replace('/', '__')}-"
                f"{time.time():.3f}-"
                f"{uuid.uuid4().hex[:4]}.json"
            )

            data = self._req_ctx.copy()
            data["response"] = resp
            data["timestamp"] = time.time()
            data["latency_sec"] = self._last_latency

            # Usage summary for quick inspection
            try:
                usage = getattr(resp, "usage", None)
                if usage:
                    prompt_tokens = int(
                        getattr(usage, "prompt_tokens", None)
                        or getattr(usage, "input_tokens", 0)
                        or 0
                    )
                    completion_tokens = int(
                        getattr(usage, "completion_tokens", None)
                        or getattr(usage, "output_tokens", 0)
                        or 0
                    )

                    data["usage_summary"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
            except Exception:
                pass

            if raw_resp:
                data["raw_response"] = raw_resp

            # Pop duplicated tools to avoid logging twice
            if (
                "tools" in data
                and isinstance(data.get("kwargs"), dict)
                and "tools" in data["kwargs"]
            ):
                data["kwargs"].pop("tools")

            log_data = json.dumps(data, default=_safe_json, ensure_ascii=False)

            # Use callback if set (for remote execution), otherwise write to file
            if self._log_completions_callback:
                self._log_completions_callback(filename, log_data)
            elif self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                if not os.access(self.log_dir, os.W_OK):
                    raise PermissionError(f"log_dir is not writable: {self.log_dir}")

                fname = os.path.join(self.log_dir, filename)
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(log_data)
        except Exception as e:
            logger.warning(f"Telemetry logging failed: {e}")


def _safe_json(obj: Any) -> Any:
    # Centralized serializer for telemetry logs.
    if isinstance(obj, ModelResponse) or isinstance(obj, ResponsesAPIResponse):
        return obj.model_dump(mode="json", exclude_none=True)

    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", exclude_none=True)

    try:
        return obj.__dict__
    except Exception:
        return str(obj)
