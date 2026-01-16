import atexit
import uuid
from pathlib import Path

from openhands.sdk.agent.base import AgentBase
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
    ConversationState,
)
from openhands.sdk.conversation.types import (
    ConversationCallbackType,
    ConversationID,
    ConversationTokenCallbackType,
)
from openhands.sdk.conversation.visualizer import (
    ConversationVisualizerBase,
    DefaultConversationVisualizer,
)
from openhands.sdk.event import (
    CondensationRequest,
    MessageEvent,
    PauseEvent,
    UserRejectObservation,
)
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.llm.llm_registry import LLMRegistry
from openhands.sdk.logger import get_logger
from openhands.sdk.workspace import LocalWorkspace


logger = get_logger(__name__)


class LocalConversation(BaseConversation):
    agent: AgentBase
    workspace: LocalWorkspace
    _state: ConversationState
    _visualizer: ConversationVisualizerBase | None
    _on_event: ConversationCallbackType
    _on_token: ConversationTokenCallbackType | None
    max_iteration_per_run: int
    llm_registry: LLMRegistry
    _cleanup_initiated: bool

    def __init__(
        self,
        agent: AgentBase,
        workspace: str | Path | LocalWorkspace | None = None,
        persistence_dir: str | Path | None = None,
        conversation_id: ConversationID | None = None,
        callbacks: list[ConversationCallbackType] | None = None,
        token_callbacks: list[ConversationTokenCallbackType] | None = None,
        max_iteration_per_run: int = 500,
        visualizer: (
            type[ConversationVisualizerBase] | ConversationVisualizerBase | None
        ) = DefaultConversationVisualizer,
        **_: object,
    ):
        """Initialize the conversation.

        Args:
            agent: The agent to use for the conversation
            workspace: Working directory for agent operations and tool execution.
                Can be a string path, Path object, LocalWorkspace instance, or None.
                When None, a LocalWorkspace without working_dir is created,
                which disables filesystem-based tools but allows MCP tools.
            persistence_dir: Directory for persisting conversation state and events.
                Can be a string path or Path object.
            conversation_id: Optional ID for the conversation. If provided, will
                      be used to identify the conversation. The user might want to
                      suffix their persistent filestore with this ID.
            callbacks: Optional list of callback functions to handle events
            token_callbacks: Optional list of callbacks invoked for streaming deltas
            max_iteration_per_run: Maximum number of iterations per run
            visualizer: Visualization configuration. Can be:
                       - ConversationVisualizerBase subclass: Class to instantiate
                         (default: ConversationVisualizer)
                       - ConversationVisualizerBase instance: Use custom visualizer
                       - None: No visualization
        """
        # Mark cleanup as initiated as early as possible to avoid races or partially
        # initialized instances during interpreter shutdown.
        self._cleanup_initiated = False

        self.agent = agent
        if workspace is None:
            # Create MCP-only workspace without filesystem access
            workspace = LocalWorkspace()
        elif isinstance(workspace, (str, Path)):
            # LocalWorkspace accepts both str and Path via BeforeValidator
            workspace = LocalWorkspace(working_dir=workspace)
        assert isinstance(workspace, LocalWorkspace), (
            "workspace must be a LocalWorkspace instance"
        )
        self.workspace = workspace

        # Create working directory only if specified
        if self.workspace.working_dir is not None:
            ws_path = Path(self.workspace.working_dir)
            if not ws_path.exists():
                ws_path.mkdir(parents=True, exist_ok=True)

        # Create-or-resume: factory inspects BASE_STATE to decide
        desired_id = conversation_id or uuid.uuid4()
        self._state = ConversationState.create(
            id=desired_id,
            agent=agent,
            workspace=self.workspace,
            persistence_dir=self.get_persistence_dir(persistence_dir, desired_id)
            if persistence_dir
            else None,
            max_iterations=max_iteration_per_run,
        )

        # Default callback: persist every event to state
        def _default_callback(e):
            self._state.events.append(e)

        callback_list = list(callbacks) if callbacks else []
        composed_list = callback_list + [_default_callback]
        # Handle visualization configuration
        if isinstance(visualizer, ConversationVisualizerBase):
            # Use custom visualizer instance
            self._visualizer = visualizer
            # Initialize the visualizer with conversation state
            self._visualizer.initialize(self._state)
            composed_list = [self._visualizer.on_event] + composed_list
            # visualizer should happen first for visibility
        elif isinstance(visualizer, type) and issubclass(
            visualizer, ConversationVisualizerBase
        ):
            # Instantiate the visualizer class with appropriate parameters
            self._visualizer = visualizer()
            # Initialize with state
            self._visualizer.initialize(self._state)
            composed_list = [self._visualizer.on_event] + composed_list
            # visualizer should happen first for visibility
        else:
            # No visualization (visualizer is None)
            self._visualizer = None

        # Compose the base callback chain (visualizer -> user callbacks -> default)
        self._on_event = BaseConversation.compose_callbacks(composed_list)
        self._on_token = (
            BaseConversation.compose_callbacks(token_callbacks)
            if token_callbacks
            else None
        )

        self.max_iteration_per_run = max_iteration_per_run

        with self._state:
            self.agent.init_state(self._state, on_event=self._on_event)

        # Register existing llms in agent
        self.llm_registry = LLMRegistry()
        self.llm_registry.subscribe(self._state.stats.register_llm)
        for llm in list(self.agent.get_all_llms()):
            self.llm_registry.add(llm)

        atexit.register(self.close)

    @property
    def id(self) -> ConversationID:
        """Get the unique ID of the conversation."""
        return self._state.id

    @property
    def state(self) -> ConversationState:
        """Get the conversation state.

        It returns a protocol that has a subset of ConversationState methods
        and properties. We will have the ability to access the same properties
        of ConversationState on a remote conversation object.
        But we won't be able to access methods that mutate the state.
        """
        return self._state

    @property
    def conversation_stats(self):
        return self._state.stats

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
        # Convert string to Message if needed
        if isinstance(message, str):
            message = Message(role="user", content=[TextContent(text=message)])

        assert message.role == "user", (
            "Only user messages are allowed to be sent to the agent."
        )
        with self._state:
            if self._state.execution_status == ConversationExecutionStatus.FINISHED:
                self._state.execution_status = (
                    ConversationExecutionStatus.IDLE
                )  # now we have a new message

            user_msg_event = MessageEvent(
                source="user",
                llm_message=message,
                activated_skills=[],
                extended_content=[],
                sender=sender,
            )
            self._on_event(user_msg_event)

    def run(self) -> None:
        """Runs the conversation until the agent finishes.

        In confirmation mode:
        - First call: creates actions but doesn't execute them, stops and waits
        - Second call: executes pending actions (implicit confirmation)

        In normal mode:
        - Creates and executes actions immediately

        Can be paused between steps
        """

        with self._state:
            if self._state.execution_status in [
                ConversationExecutionStatus.IDLE,
                ConversationExecutionStatus.PAUSED,
                ConversationExecutionStatus.ERROR,
            ]:
                self._state.execution_status = ConversationExecutionStatus.RUNNING

        iteration = 0
        try:
            while True:
                logger.debug(f"Conversation run iteration {iteration}")
                with self._state:
                    # Pause attempts to acquire the state lock
                    # Before value can be modified step can be taken
                    # Ensure step conditions are checked when lock is already acquired
                    if (
                        self._state.execution_status
                        == ConversationExecutionStatus.PAUSED
                    ):
                        break

                    # Check if conversation finished
                    if (
                        self._state.execution_status
                        == ConversationExecutionStatus.FINISHED
                    ):
                        break

                    self.agent.step(
                        self, on_event=self._on_event, on_token=self._on_token
                    )
                    iteration += 1

                    if iteration >= self.max_iteration_per_run:
                        error_msg = (
                            f"Agent reached maximum iterations limit "
                            f"({self.max_iteration_per_run})."
                        )
                        logger.error(error_msg)
                        self._state.execution_status = ConversationExecutionStatus.ERROR
                        self._on_event(
                            ConversationErrorEvent(
                                source="environment",
                                code="MaxIterationsReached",
                                detail=error_msg,
                            )
                        )
                        break
        except Exception as e:
            self._state.execution_status = ConversationExecutionStatus.ERROR

            # Add an error event
            self._on_event(
                ConversationErrorEvent(
                    source="environment",
                    code=e.__class__.__name__,
                    detail=str(e),
                )
            )

                    # Re-raise with conversation id and persistence dir for better UX
            raise ConversationRunError(
                self._state.id, e, persistence_dir=self._state.persistence_dir
            ) from e

    def reject_pending_actions(self, reason: str = "User rejected the action") -> None:
        """Reject all pending actions from the agent.

        This is a non-invasive method to reject actions between run() calls.
        """
        pending_actions = ConversationState.get_unmatched_actions(self._state.events)

        with self._state:
            if not pending_actions:
                logger.warning("No pending actions to reject")
                return

            for action_event in pending_actions:
                # Create rejection observation
                rejection_event = UserRejectObservation(
                    action_id=action_event.id,
                    tool_name=action_event.tool_name,
                    tool_call_id=action_event.tool_call_id,
                    rejection_reason=reason,
                )
                self._on_event(rejection_event)
                logger.info(f"Rejected pending action: {action_event} - {reason}")

    def pause(self) -> None:
        """Pause agent execution.

        This method can be called from any thread to request that the agent
        pause execution. The pause will take effect at the next iteration
        of the run loop (between agent steps).

        Note: If called during an LLM completion, the pause will not take
        effect until the current LLM call completes.
        """

        if self._state.execution_status == ConversationExecutionStatus.PAUSED:
            return

        with self._state:
            # Only pause when running or idle
            if (
                self._state.execution_status == ConversationExecutionStatus.IDLE
                or self._state.execution_status == ConversationExecutionStatus.RUNNING
            ):
                self._state.execution_status = ConversationExecutionStatus.PAUSED
                pause_event = PauseEvent()
                self._on_event(pause_event)
                logger.info("Agent execution pause requested")

    def close(self) -> None:
        """Close the conversation and clean up all tool executors."""
        # Use getattr for safety - object may be partially constructed
        if getattr(self, "_cleanup_initiated", False):
            return
        self._cleanup_initiated = True
        logger.debug("Closing conversation and cleaning up tool executors")
        try:
            tools_map = self.agent.tools_map
        except (AttributeError, RuntimeError):
            # Agent not initialized or partially constructed
            return
        for tool in tools_map.values():
            try:
                executable_tool = tool.as_executable()
                executable_tool.executor.close()
            except NotImplementedError:
                # Tool has no executor, skip it without erroring
                continue
            except Exception as e:
                logger.warning(f"Error closing executor for tool '{tool.name}': {e}")

    def condense(self) -> None:
        """Synchronously force condense the conversation history.

        If the agent is currently running, `condense()` will wait for the
        ongoing step to finish before proceeding.

        Raises ValueError if no compatible condenser exists.
        """

        # Check if condenser is configured and handles condensation requests
        if (
            self.agent.condenser is None
            or not self.agent.condenser.handles_condensation_requests()
        ):
            condenser_info = (
                "No condenser configured"
                if self.agent.condenser is None
                else (
                    f"Condenser {type(self.agent.condenser).__name__} does not handle "
                    "condensation requests"
                )
            )
            raise ValueError(
                f"Cannot condense conversation: {condenser_info}. "
                "To enable manual condensation, configure an "
                "LLMSummarizingCondenser:\n\n"
                "from openhands.sdk.condenser import LLMSummarizingCondenser\n"
                "agent = Agent(\n"
                "    llm=your_llm,\n"
                "    condenser=LLMSummarizingCondenser(\n"
                "        llm=your_llm,\n"
                "        max_size=120,\n"
                "        keep_first=4\n"
                "    )\n"
                ")"
            )

        # Add a condensation request event
        condensation_request = CondensationRequest()
        self._on_event(condensation_request)

        # Force the agent to take a single step to process the condensation request
        # This will trigger the condenser if it handles condensation requests
        with self._state:
            # Take a single step to process the condensation request
            self.agent.step(self, on_event=self._on_event, on_token=self._on_token)

        logger.info("Condensation request processed")

    def __del__(self) -> None:
        """Ensure cleanup happens when conversation is destroyed."""
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Error during conversation cleanup: {e}", exc_info=True)
