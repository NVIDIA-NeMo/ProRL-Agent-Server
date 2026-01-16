from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from openhands.agent_server.utils import OpenHandsUUID, utc_now
from openhands.sdk import AgentBase, Event, ImageContent, Message, TextContent
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
    ConversationState,
)
from openhands.sdk.llm.utils.metrics import MetricsSnapshot
from openhands.sdk.utils.models import DiscriminatedUnionMixin, OpenHandsModel
from openhands.sdk.workspace import LocalWorkspace


class ConversationSortOrder(str, Enum):
    """Enum for conversation sorting options."""

    CREATED_AT = "CREATED_AT"
    UPDATED_AT = "UPDATED_AT"
    CREATED_AT_DESC = "CREATED_AT_DESC"
    UPDATED_AT_DESC = "UPDATED_AT_DESC"


class EventSortOrder(str, Enum):
    """Enum for event sorting options."""

    TIMESTAMP = "TIMESTAMP"
    TIMESTAMP_DESC = "TIMESTAMP_DESC"


class SendMessageRequest(BaseModel):
    """Payload to send a message to the agent.

    This is a simplified version of openhands.sdk.Message.
    """

    role: Literal["user", "system", "assistant", "tool"] = "user"
    content: list[TextContent | ImageContent] = Field(default_factory=list)
    run: bool = Field(
        default=False,
        description=("Whether the agent loop should automatically run if not running"),
    )

    def create_message(self) -> Message:
        message = Message(role=self.role, content=self.content)
        return message


class StartConversationRequest(BaseModel):
    """Payload to create a new conversation.

    Contains an Agent configuration along with conversation-specific options.
    """

    agent: AgentBase
    workspace: LocalWorkspace = Field(
        ...,
        description="Working directory for agent operations and tool execution",
    )
    conversation_id: OpenHandsUUID | None = Field(
        default=None,
        description=(
            "Optional conversation ID. If not provided, a random UUID will be "
            "generated."
        ),
    )
    initial_message: SendMessageRequest | None = Field(
        default=None, description="Initial message to pass to the LLM"
    )
    max_iterations: int = Field(
        default=500,
        ge=1,
        description="If set, the max number of iterations the agent will run "
        "before stopping. This is useful to prevent infinite loops.",
    )
    tool_module_qualnames: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of tool names to their module qualnames from the client's "
            "registry. These modules will be dynamically imported on the server "
            "to register the tools for this conversation."
        ),
    )


class StoredConversation(StartConversationRequest):
    """Stored details about a conversation"""

    id: OpenHandsUUID
    title: str | None = Field(
        default=None, description="User-defined title for the conversation"
    )
    metrics: MetricsSnapshot | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ConversationInfo(ConversationState):
    """Information about a conversation running locally without a Runtime sandbox."""

    # ConversationState already includes id and agent
    # Add additional metadata fields

    title: str | None = Field(
        default=None, description="User-defined title for the conversation"
    )
    metrics: MetricsSnapshot | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ConversationPage(BaseModel):
    items: list[ConversationInfo]
    next_page_id: str | None = None


class ConversationResponse(BaseModel):
    conversation_id: str
    state: ConversationExecutionStatus


class ConfirmationResponseRequest(BaseModel):
    """Payload to accept or reject a pending action."""

    accept: bool
    reason: str = "User rejected the action."


class Success(BaseModel):
    success: bool = True


class EventPage(OpenHandsModel):
    items: list[Event]
    next_page_id: str | None = None


class UpdateConversationRequest(BaseModel):
    """Payload to update conversation metadata."""

    title: str = Field(
        ..., min_length=1, max_length=200, description="New conversation title"
    )


class BashEventBase(DiscriminatedUnionMixin, ABC):
    """Base class for all bash event types"""

    id: OpenHandsUUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=utc_now)


class ExecuteBashRequest(BaseModel):
    command: str = Field(description="The bash command to execute")
    cwd: str | None = Field(default=None, description="The current working directory")
    timeout: int = Field(
        default=300,
        description="The max number of seconds a command may be permitted to run.",
    )


class BashCommand(BashEventBase, ExecuteBashRequest):
    pass


class BashOutput(BashEventBase):
    """
    Output of a bash command. A single command may have multiple pieces of output
    depending on how large the output is.
    """

    command_id: OpenHandsUUID
    order: int = Field(
        default=0, description="The order for this output, sequentially starting with 0"
    )
    exit_code: int | None = Field(
        default=None, description="Exit code None implies the command is still running."
    )
    stdout: str | None = Field(
        default=None, description="The standard output from the command"
    )
    stderr: str | None = Field(
        default=None, description="The error output from the command"
    )


class BashEventSortOrder(Enum):
    TIMESTAMP = "TIMESTAMP"
    TIMESTAMP_DESC = "TIMESTAMP_DESC"


class BashEventPage(OpenHandsModel):
    items: list[BashEventBase]
    next_page_id: str | None = None
