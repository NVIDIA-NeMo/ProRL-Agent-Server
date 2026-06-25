"""Generic strategy registry for builders and evaluators."""

from __future__ import annotations

from typing import Generic, TypeVar

from polar._imports import import_subclass
from polar.trajectory.models import StrategySpec

T = TypeVar("T")


class StrategyRegistry(Generic[T]):
    """Factory-based registry that instantiates a fresh strategy per request.

    Stores *classes* (or import-path strings), not instances.  Each call to
    :meth:`create` constructs a new object with ``cls(**spec.config)``.
    """

    def __init__(self, base_type: type[T]) -> None:
        self._base_type = base_type
        self._factories: dict[str, type[T]] = {}

    def register(self, name: str, factory: type[T] | str) -> None:
        """Register a strategy class or ``'module:ClassName'`` import path."""
        if isinstance(factory, str):
            factory = self._import_class(factory)
        if not (isinstance(factory, type) and issubclass(factory, self._base_type)):
            raise TypeError(
                f"factory must be a subclass of {self._base_type.__name__}, "
                f"got {factory!r}"
            )
        self._factories[name] = factory

    def create(self, spec: StrategySpec) -> T:
        """Instantiate a strategy from a :class:`StrategySpec`."""
        cls = self._resolve(spec.strategy)
        return cls(**spec.config)

    def list_strategies(self) -> list[str]:
        return sorted(self._factories)

    # ------------------------------------------------------------------

    def _resolve(self, strategy: str) -> type[T]:
        if strategy in self._factories:
            return self._factories[strategy]
        if ":" in strategy:
            cls = self._import_class(strategy)
            self._factories[strategy] = cls
            return cls
        raise KeyError(f"Unknown strategy: {strategy!r}")

    def _import_class(self, import_path: str) -> type[T]:
        return import_subclass(import_path, self._base_type)


def default_builder_registry() -> StrategyRegistry:
    """Pre-populated registry with built-in trajectory builders."""
    from polar.trajectory.builder import (
        BaseTrajectoryBuilder,
        PerRequestBuilder,
        PrefixMergingBuilder,
    )

    registry: StrategyRegistry[BaseTrajectoryBuilder] = StrategyRegistry(BaseTrajectoryBuilder)
    registry.register("per_request", PerRequestBuilder)
    registry.register("prefix_merging", PrefixMergingBuilder)
    return registry


def default_evaluator_registry() -> StrategyRegistry:
    """Pre-populated registry with built-in trajectory evaluators."""
    from polar.trajectory.evaluator import (
        BaseTrajectoryEvaluator,
        HarborEvaluator,
        SessionCompletedEvaluator,
        SwebenchHarnessEvaluator,
        TestOnOutputEvaluator,
    )

    registry: StrategyRegistry[BaseTrajectoryEvaluator] = StrategyRegistry(BaseTrajectoryEvaluator)
    registry.register("session_completed", SessionCompletedEvaluator)
    registry.register("swebench_harness", SwebenchHarnessEvaluator)
    registry.register("test_on_output", TestOnOutputEvaluator)
    registry.register("harbor", HarborEvaluator)
    return registry


