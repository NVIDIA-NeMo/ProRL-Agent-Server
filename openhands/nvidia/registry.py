import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from evaluation.utils.shared import EvalMetadata  # type: ignore
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.agent_controller import AgentController
from openhands.core.config import OpenHandsConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia.reward import Reward
from openhands.nvidia.timer import PausableTimer
from openhands.runtime.base import Runtime

_DEFAULT_AGENT_CONFIG = {
    'max_iterations': 2,
    'ensure_thinking_end_properly': False,
    'strict_loop_detector': False,
    'enable_vision': False,
    'enable_a11y_tree': False,
}


@dataclass
class JobDetails:
    job_id: str | None = None
    instance: dict | None = None
    agent_config: dict = field(default_factory=lambda: _DEFAULT_AGENT_CONFIG.copy())
    llm_config: LLMConfig | None = None
    runtime: Runtime | None = None
    metadata: EvalMetadata | None = None
    config: OpenHandsConfig | None = None
    agent: CodeActAgent | None = None  # for intermediate results
    controller: AgentController | None = None  # for intermediate results
    run_results: dict | None = None
    eval_results: dict | None = None
    results: dict | None = None
    event: threading.Event | None = None
    timeout_error: bool = False
    timer: Optional['PausableTimer'] = None  # Import handled in async_server.py
    # Task reference for cancellation support
    current_task: Optional[asyncio.Task] = None
    is_reasoning_task: bool = False


class AgentHandler(ABC):
    """Abstract base class defining the interface for agent handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name identifier for this agent handler."""
        pass

    @abstractmethod
    async def init(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
        """Initialize the agent with instance and config."""
        pass

    @abstractmethod
    async def run(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ) -> dict[str, object]:
        """Run the agent with runtime and instance."""
        pass

    @abstractmethod
    async def eval(
        self,
        job_details: JobDetails,
        sid: str | None = None,
        allow_skip: bool = True,
        reward: Optional[Reward] = None,
    ) -> dict[str, Any]:
        """Evaluate the agent results."""
        pass

    @abstractmethod
    def init_exception(
        self, job_details: JobDetails, exception: Exception
    ) -> dict[str, Any]:
        """Handle exceptions during initialization."""
        pass

    @abstractmethod
    def run_exception(
        self, job_details: JobDetails, exception: Exception
    ) -> dict[str, Any]:
        """Handle exceptions during run."""
        pass

    @abstractmethod
    def eval_exception(
        self, job_details: JobDetails, exception: Exception
    ) -> dict[str, Any]:
        """Handle exceptions during evaluation."""
        pass

    @abstractmethod
    def final_result(self, job_details: JobDetails) -> dict[str, Any]:
        """Process final results."""
        pass


# Registry for different types of agent functions
_registries: dict[str, dict[str, Callable[..., Any]]] = {
    'init': {},
    'run': {},
    'eval': {},
    'init_exception': {},
    'run_exception': {},
    'eval_exception': {},
    'final_result': {},
}

_registered_handlers: set[str] = set()

_name_mapping: dict[str, str] = {}

# Registry for different types of reasoning functions
_registries_reasoning: dict[str, dict[str, Callable[..., Any]]] = {
    'init': {},
    'run': {},
    'eval': {},
    'init_exception': {},
    'run_exception': {},
    'eval_exception': {},
    'final_result': {},
}

_registered_handlers_reasoning: set[str] = set()

_name_mapping_reasoning: dict[str, str] = {}


class FunctionNotRegisteredError(Exception):
    """Raised when a requested function is not found in the registry."""

    pass


# Utility functions for registry management
def get_registered_functions(registry_type, name, reasoning=False):
    if not reasoning:
        name_mapping = _name_mapping
        registries = _registries
    else:
        name_mapping = _name_mapping_reasoning
        registries = _registries_reasoning
    mapped_name = name_mapping.get(name)
    if name[:9] == 'deepcoder':
        mapped_name = 'deepcoder'
    if mapped_name is None:
        return None
    return registries.get(registry_type, {}).get(mapped_name)


def is_registered_handler(name, reasoning=False):
    """Check if a handler is registered correctly."""
    if not reasoning:
        name_mapping = _name_mapping
    else:
        name_mapping = _name_mapping_reasoning
    if name[:9] == 'deepcoder':
        return True
    return name in name_mapping


def register_agent_handler(handler: AgentHandler, reasoning=False):
    """Register all methods of an AgentHandler instance to their corresponding registries."""
    if not reasoning:
        registries = _registries
        name_mapping = _name_mapping
    else:
        registries = _registries_reasoning
        name_mapping = _name_mapping_reasoning
    registries['init'][handler.name] = handler.init
    registries['run'][handler.name] = handler.run
    registries['eval'][handler.name] = handler.eval
    registries['init_exception'][handler.name] = handler.init_exception
    registries['run_exception'][handler.name] = handler.run_exception
    registries['eval_exception'][handler.name] = handler.eval_exception
    registries['final_result'][handler.name] = handler.final_result
    name_mapping[handler.name] = handler.name


def add_name_mapping(name: str, mapped_name: str, reasoning=False):
    if not reasoning:
        name_mapping = _name_mapping
    else:
        name_mapping = _name_mapping_reasoning
    name_mapping[name] = mapped_name
