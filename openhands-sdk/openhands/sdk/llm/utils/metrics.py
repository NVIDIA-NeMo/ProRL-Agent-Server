import copy
from typing import final

from pydantic import BaseModel, Field, model_validator


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    model: str = Field(default="")
    prompt_tokens: int = Field(
        default=0, ge=0, description="Prompt tokens for this call"
    )
    completion_tokens: int = Field(
        default=0, ge=0, description="Completion tokens for this call"
    )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances together."""
        return TokenUsage(
            model=self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class MetricsSnapshot(BaseModel):
    """A snapshot of metrics at a point in time."""

    model_name: str = Field(default="default", description="Name of the model")
    token_usages: list[TokenUsage] = Field(
        default_factory=list, description="Per-call token usage history"
    )
    accumulated_token_usage: TokenUsage | None = Field(
        default=None, description="Accumulated token usage across all calls"
    )


@final
class Metrics(MetricsSnapshot):
    """Metrics class tracking per-call and accumulated token usage."""

    @model_validator(mode="after")
    def initialize_accumulated_token_usage(self) -> "Metrics":
        if self.accumulated_token_usage is None:
            self.accumulated_token_usage = TokenUsage(
                model=self.model_name,
                prompt_tokens=0,
                completion_tokens=0,
            )
        return self

    def get_snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of the current metrics."""
        return MetricsSnapshot(
            model_name=self.model_name,
            token_usages=copy.deepcopy(self.token_usages),
            accumulated_token_usage=copy.deepcopy(self.accumulated_token_usage)
            if self.accumulated_token_usage
            else None,
        )

    def add_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        **_kwargs,  # Accept but ignore extra kwargs for compatibility
    ) -> None:
        """Add token usage for a single call and update accumulated totals."""
        new_usage = TokenUsage(
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        # Track per-call usage
        self.token_usages.append(new_usage)
        # Update accumulated totals
        if self.accumulated_token_usage is None:
            self.accumulated_token_usage = new_usage
        else:
            self.accumulated_token_usage = self.accumulated_token_usage + new_usage

    def merge(self, other: "Metrics") -> None:
        """Merge 'other' metrics into this one."""
        # Merge per-call token usages
        self.token_usages.extend(other.token_usages)
        # Merge accumulated totals
        if self.accumulated_token_usage is None:
            self.accumulated_token_usage = other.accumulated_token_usage
        elif other.accumulated_token_usage is not None:
            self.accumulated_token_usage = (
                self.accumulated_token_usage + other.accumulated_token_usage
            )

    def get(self) -> dict:
        """Return the metrics in a dictionary."""
        return {
            "token_usages": [u.model_dump() for u in self.token_usages],
            "accumulated_token_usage": self.accumulated_token_usage.model_dump()
            if self.accumulated_token_usage
            else None,
        }

    def log(self) -> str:
        """Log the metrics."""
        metrics = self.get()
        logs = ""
        for key, value in metrics.items():
            logs += f"{key}: {value}\n"
        return logs

    def deep_copy(self) -> "Metrics":
        """Create a deep copy of the Metrics object."""
        return copy.deepcopy(self)

    def diff(self, baseline: "Metrics") -> "Metrics":
        """Calculate the difference between current metrics and a baseline.

        Args:
            baseline: A metrics object representing the baseline state

        Returns:
            A new Metrics object containing only the differences since the baseline
        """
        result = Metrics(model_name=self.model_name)

        # Include only the new token usages since baseline
        baseline_count = len(baseline.token_usages)
        result.token_usages = copy.deepcopy(self.token_usages[baseline_count:])

        base_usage = baseline.accumulated_token_usage
        current_usage = self.accumulated_token_usage

        if current_usage is not None and base_usage is not None:
            result.accumulated_token_usage = TokenUsage(
                model=self.model_name,
                prompt_tokens=current_usage.prompt_tokens - base_usage.prompt_tokens,
                completion_tokens=current_usage.completion_tokens
                - base_usage.completion_tokens,
            )
        elif current_usage is not None:
            result.accumulated_token_usage = current_usage
        else:
            result.accumulated_token_usage = None

        return result

    def __repr__(self) -> str:
        return f"Metrics({self.get()}"
