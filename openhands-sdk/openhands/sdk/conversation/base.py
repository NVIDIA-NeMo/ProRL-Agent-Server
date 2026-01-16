from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from openhands.sdk.conversation.conversation_stats import ConversationStats
from openhands.sdk.conversation.events_list_base import EventsListBase
from openhands.sdk.conversation.types import (
    ConversationCallbackType,
    ConversationID,
    ConversationTokenCallbackType,
)
from openhands.sdk.llm.message import Message
from openhands.sdk.workspace.base import BaseWorkspace


if TYPE_CHECKING:
    from openhands.sdk.agent.base import AgentBase
    from openhands.sdk.conversation.state import ConversationExecutionStatus


CallbackType = TypeVar(
    "CallbackType",
    ConversationCallbackType,
    ConversationTokenCallbackType,
)


class ConversationStateProtocol(Protocol):
    """Protocol defining the interface for conversation state objects."""

    @property
    def id(self) -> ConversationID:
        """The conversation ID."""
        ...

    @property
    def events(self) -> EventsListBase:
        """Access to the events list."""
        ...

    @property
    def execution_status(self) -> "ConversationExecutionStatus":
        """The current conversation execution status."""
        ...

    @property
    def activated_knowledge_skills(self) -> list[str]:
        """List of activated knowledge skills."""
        ...

    @property
    def workspace(self) -> BaseWorkspace:
        """The workspace for agent operations and tool execution."""
        ...

    @property
    def persistence_dir(self) -> str | None:
        """The persistence directory from the FileStore.

        If None, it means the conversation is not being persisted.
        """
        ...

    @property
    def agent(self) -> "AgentBase":
        """The agent running in the conversation."""
        ...

    @property
    def stats(self) -> ConversationStats:
        """The conversation statistics."""
        ...


class BaseConversation(ABC):
    """Abstract base class for conversation implementations.

    This class defines the interface that all conversation implementations must follow.
    Conversations manage the interaction between users and agents, handling message
    exchange, execution control, and state management.
    """

    @property
    @abstractmethod
    def id(self) -> ConversationID: ...

    @property
    @abstractmethod
    def state(self) -> ConversationStateProtocol: ...

    @property
    @abstractmethod
    def conversation_stats(self) -> ConversationStats: ...

    @abstractmethod
    def send_message(self, message: str | Message, sender: str | None = None) -> None:
        """Send a message to the agent.

        Args:
            message: Either a string (which will be converted to a user message)
                    or a Message object
            sender: Optional identifier of the sender. Can be used to track
                   message origin in multi-agent scenarios. For example, when
                   one agent delegates to another, the sender can be set to
                   identify which agent is sending the message.
        """
        ...

    @abstractmethod
    def run(self) -> None:
        """Execute the agent to process messages and perform actions.

        This method runs the agent until it finishes processing the current
        message or reaches the maximum iteration limit.
        """
        ...

    @abstractmethod
    def reject_pending_actions(
        self, reason: str = "User rejected the action"
    ) -> None: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @staticmethod
    def get_persistence_dir(
        persistence_base_dir: str | Path, conversation_id: ConversationID
    ) -> str:
        """Get the persistence directory for the conversation.

        Args:
            persistence_base_dir: Base directory for persistence. Can be a string
                path or Path object.
            conversation_id: Unique conversation ID.

        Returns:
            String path to the conversation-specific persistence directory.
            Always returns a normalized string path even if a Path was provided.
        """
        return str(Path(persistence_base_dir) / conversation_id.hex)

    @abstractmethod
    def condense(self) -> None:
        """Force condensation of the conversation history.

        This method uses the existing condensation request pattern to trigger
        condensation. It adds a CondensationRequest event to the conversation
        and forces the agent to take a single step to process it.

        The condensation will be applied immediately and will modify the conversation
        state by adding a condensation event to the history.

        Raises:
            ValueError: If no condenser is configured or the condenser doesn't
                       handle condensation requests.
        """
        ...

    @staticmethod
    def compose_callbacks(callbacks: Iterable[CallbackType]) -> CallbackType:
        """Compose multiple callbacks into a single callback function.

        Args:
            callbacks: An iterable of callback functions

        Returns:
            A single callback function that calls all provided callbacks
        """

        def composed(event) -> None:
            for cb in callbacks:
                if cb:
                    cb(event)

        return cast(CallbackType, composed)
