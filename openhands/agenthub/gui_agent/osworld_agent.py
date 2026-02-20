import os
from jinja2 import Template
from collections import deque

from openhands.agenthub.gui_agent.tools import OSWORLD_TOOLS

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    Action,
    AgentFinishAction,
    MessageAction,
)
import openhands.agenthub.gui_agent.function_calling as codeact_function_calling
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.events.event import EventSource, Event
from openhands.events.observation import OSWorldOutputObservation
from openhands.events.observation import ErrorObservation
from openhands.llm.llm import LLM
from openhands.runtime.plugins import (
    PluginRequirement,
)
from openhands.core.exceptions import (
    AgentFormatError,
    AgentEndThinkError,
    AgentToolCallError,
    AgentLengthError,
)
from openhands.nvidia.os_world.accessibility_tree_wrap.heuristic_retrieve import linearize_accessibility_tree

from openhands.agenthub.gui_agent.prompts.osworld import OSWORLD_OBSERVATION_FEEDBACK_PROMPT, ERROR_OBSERVATION_FEEDBACK_PROMPT

def get_instruction(action: MessageAction) -> str:
    if 'Instruction:' in action.content:
        return action.content.split('Instruction:')[1].strip()
    return action.content

def convert_action_to_message(action: OSWorldInteractiveAction) -> Message:
    # These are action from the LLM
    # Only have text format
    tool_metadata = action.tool_call_metadata
    llm_response = tool_metadata.model_response
    assistant_msg = getattr(llm_response.choices[0], 'message')
    input_ids = getattr(llm_response.choices[0], 'input_ids', None)
    output_ids = getattr(llm_response.choices[0], 'output_ids', None)
    logprobs = getattr(llm_response.choices[0], 'logprobs', None)

    text_content = assistant_msg.content
    if text_content is None:
        text_content = ''

    return Message(
        role=getattr(assistant_msg, 'role', 'assistant'),
        content=[TextContent(text=text_content)],
        tool_calls=assistant_msg.tool_calls,
        input_ids=input_ids,
        output_ids=output_ids,
        logprobs=logprobs,
    )

def convert_message_action_to_message(
    action: MessageAction,
    include_a11y_tree: bool = True,
    include_screenshot: bool = True,
    ) -> Message:
    text_content = action.content
    if include_a11y_tree:
        accessibility_tree = action.accessibility_tree
        if accessibility_tree is None or len(accessibility_tree) < 1:
            logger.error('Accessibility tree is None or empty, skipping')
        else:
            accessibility_tree = linearize_accessibility_tree(accessibility_tree)
            text_content += f"\n\nAccessibility Tree:\n{accessibility_tree}"
    content = [TextContent(text=text_content)]
    if include_screenshot:
        image_urls = action.image_urls
        if image_urls is None or len(image_urls) < 1:
            logger.error('Image urls is None or empty, skipping')
        else:
            content.append(ImageContent(image_urls=image_urls))
    return Message(
        role='user',
        content=content,
    )

def convert_observation_to_message(
    observation: OSWorldOutputObservation | ErrorObservation,
    instruction: str,
    include_a11y_tree: bool = True,
    include_screenshot: bool = True,
    ) -> Message:
    if isinstance(observation, OSWorldOutputObservation):
        prompt_text = OSWORLD_OBSERVATION_FEEDBACK_PROMPT.format(instruction=instruction)
        if include_a11y_tree:
            accessibility_tree = observation.accessibility_tree
            if accessibility_tree and len(accessibility_tree) >= 1:
                logger.error('Accessibility tree is None or empty, skipping')
            else:
                accessibility_tree = linearize_accessibility_tree(accessibility_tree)
                prompt_text += f"\n\nAccessibility Tree:\n{accessibility_tree}"
        content = [TextContent(text=prompt_text)]
        if include_screenshot:
            image_url = observation.image_urls
            if image_url is None or len(image_url) < 1:
                logger.error('Image urls is None or empty, skipping')
            else:
                content.append(ImageContent(image_urls=image_url))
        return Message(
            role='tool', # or user?
            content=content,
            tool_call_id=observation.tool_call_id,
            name=observation.name,
        )
    else:
        prompt_text = ERROR_OBSERVATION_FEEDBACK_PROMPT.format(instruction=instruction, error_message=observation.content)
        return Message(
            role='tool', # or user?
            content=[TextContent(text=prompt_text)],
            tool_call_id=observation.error_id,
            name=observation.name,
        )

def convert_message_action_to_message_full_state(
    action: MessageAction,
    include_a11y_tree: bool = True,
    ) -> Message:
    text_content = action.content
    if include_a11y_tree:
        accessibility_tree = action.accessibility_tree
        if accessibility_tree and len(accessibility_tree) > 0:
            accessibility_tree = linearize_accessibility_tree(action.accessibility_tree)
            text_content += f"\n\nAccessibility Tree:\n{accessibility_tree}"
    if isinstance(text_content, str):
        content = [TextContent(text=text_content)]
    if action.image_urls is not None and len(action.image_urls) > 0:
        content.append(ImageContent(image_urls=action.image_urls))
    if isinstance(action.accessibility_tree, str):
        content.append(TextContent(text=action.accessibility_tree))
    return Message(
        role='user',
        content=content,
    )

def convert_observation_to_message_full_state(
    observation: OSWorldOutputObservation | ErrorObservation,
    instruction: str,
    include_a11y_tree: bool = True,
    ) -> Message:
    if isinstance(observation, OSWorldOutputObservation):
        prompt_text = OSWORLD_OBSERVATION_FEEDBACK_PROMPT.format(instruction=instruction)
        if include_a11y_tree:
            accessibility_tree = observation.accessibility_tree
            if accessibility_tree and len(accessibility_tree) > 0:
                accessibility_tree = linearize_accessibility_tree(accessibility_tree)
                prompt_text += f"\n\nAccessibility Tree:\n{accessibility_tree}"
        content = [TextContent(text=prompt_text)]
        
        # We always add screenshot and accessibility tree to the message
        if observation.image_urls is not None and len(observation.image_urls) > 0:
            content.append(ImageContent(image_urls=observation.image_urls))
        if isinstance(observation.accessibility_tree, str):
            content.append(TextContent(text=observation.accessibility_tree))
        return Message(
            role='tool', # or user?
            content=content,
            tool_call_id=observation.tool_call_id,
            name=observation.name,
        )
    else:
        prompt_text = ERROR_OBSERVATION_FEEDBACK_PROMPT.format(instruction=instruction, error_message=observation.content)
        return Message(
            role='tool', # or user?
            content=[TextContent(text=prompt_text)],
            tool_call_id=observation.error_id,
            name=observation.name,
        )

class OSWorldAgent(Agent):
    VERSION = '1.0'
    """
    OSWorldAgent that operates on the OSWorld virtual machine.
    """

    sandbox_plugins: list[PluginRequirement] = []

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        """Initializes a new instance of the GuiAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm, config)

        self.pause_time = 0.0
        self.system_prompt = os.path.join(os.path.dirname(__file__), 'prompts', 'system_prompt_osworld.j2')
        with open(self.system_prompt, 'r') as file:
            self.system_prompt = file.read()
        self.system_prompt = Template(self.system_prompt).render(CLIENT_PASSWORD='password').format(CLIENT_PASSWORD='password')

        self.tools = OSWORLD_TOOLS

        # enable vision for all models
        if isinstance(self.llm.model_info, dict):
            self.llm.model_info['supports_vision'] = True
        else:
            self.llm.model_info = {'supports_vision': True}
        
        self.pending_actions: deque['Action'] = deque()

        self.reset()

    def reset(self) -> None:
        """Resets the GuiAgent."""
        super().reset()
        self.cost_accumulator = 0
        self.error_accumulator = 0
        self.pending_actions.clear()

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        """Get the initial user message from the conversation history.

        Args:
            history: List of events from the conversation

        Returns:
            MessageAction: The initial user message

        Raises:
            ValueError: If no initial user message is found
        """
        initial_user_message = None

        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                initial_user_message = event
                break

        if initial_user_message is None:
            # This should not happen in a valid conversation
            logger.error(
                f'CRITICAL: Could not find the initial user MessageAction in the full {len(history)} events history.'
            )
            raise ValueError(
                'Could not find the initial user MessageAction in the conversation history.'
            )
        return initial_user_message

    def _get_messages(
        self, events: list[Event], initial_user_message: MessageAction,
    ) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand, similar to how the step
        method constructs messages but for the full conversation history.

        Args:
            events: The list of events to convert to messages
            initial_user_message: The initial user message action

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption
        """
        messages: list[Message] = []

        # Get instruction from initial user message
        # User message is a MessageAction with content and image_urls, will be processed in events
        instruction = get_instruction(initial_user_message)
        include_a11y_tree = self.config.enable_a11y_tree
        total_screenshot_count = 0

        llm_response_ids_action = set()
        llm_response_ids_observation = set()

        # Build history prompts (alternating assistant/user messages) in reverse order
        for event in reversed(events):
            include_screenshot = self.config.enable_vision
            if self.config.max_image_history is not None and total_screenshot_count >= self.config.max_image_history:
                include_screenshot = False

            if isinstance(event, OSWorldInteractiveAction):
                llm_response_id = event.tool_call_metadata.model_response.id
                if llm_response_id in llm_response_ids_action:
                    continue
                llm_response_ids_action.add(llm_response_id)
                messages.append(convert_action_to_message(event))
            elif isinstance(event, MessageAction):
                messages.append(convert_message_action_to_message(
                    event, include_a11y_tree=include_a11y_tree, include_screenshot=include_screenshot))
                total_screenshot_count += 1
            elif isinstance(event, OSWorldOutputObservation) or isinstance(event, ErrorObservation):
                llm_response_id = event.tool_call_metadata.model_response.id
                if llm_response_id in llm_response_ids_observation:
                    continue
                msg = convert_observation_to_message(
                    event, instruction, include_a11y_tree=include_a11y_tree, include_screenshot=include_screenshot)
                messages.append(msg)
                total_screenshot_count += 1
                llm_response_ids_observation.add(llm_response_id)

        # System message
        messages.append(Message(role='system', content=[TextContent(text=self.system_prompt)]))

        return messages[::-1]

    def step(self, state: State) -> Action:
        """Performs one step using the GuiAgent.

        This includes gathering information on previous steps and prompting the model to make a browsing command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - OSWorldInteractiveAction(browsergym_command) - BrowserGym commands to run
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        format_error = state.get_last_agent_format_error()
        if format_error and isinstance(format_error, str):
            if format_error == 'length':
                raise AgentLengthError("LLM did not format the response properly")
            elif format_error == 'tool_call':
                raise AgentToolCallError("LLM did not format the tool call properly")
            elif format_error != '':
                logger.error(f"Unknown format error: {format_error}, continue")

        # check if last thought is properly ended
        # reuse stuck in loop error to exit agent loop
        if self.config.ensure_thinking_end_properly:
            latest_agent_thought = state.get_last_agent_thought()
            if latest_agent_thought and '<think>' in latest_agent_thought and '</think>' not in latest_agent_thought:
                raise AgentEndThinkError("LLM does not end properly reasoning properly")

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(state.history, initial_user_message)
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
        }
        params['tools'] = self.tools
        params['extra_body'] = {'metadata': state.to_llm_metadata(agent_name=self.name)}
        for msg in params['messages']:
            if msg.get('role') == 'tool':
                msg['role'] = 'user'
        response = self.llm.completion(**params)
        logger.debug(f'Response from LLM: {response}')
        actions = codeact_function_calling.response_to_actions(response, timeout=self.config.action_timeout)
        logger.debug(f'Actions after response_to_actions: {actions}')
        for action in actions:
            if self.pause_time > 0.5:
                logger.info(f'Setting pause time to {self.pause_time} seconds for agentic action')
                action.pause_time = self.pause_time
            self.pending_actions.append(action)
        
        return self.pending_actions.popleft()

    def _get_messages_from_agent_state(
        self, events: list[Event], initial_user_message: MessageAction,
    ) -> list[dict]:
        """This message although similar to _get_messages, is used to process the messages from the agent state.
        Key difference is to preserve all the content items in the message, including image and accessibility tree.
        Also will add AgentFinishAction to the messages.
        Used in process_messages_from_agent_state.

        Args:
            events: The list of events to convert to messages
            initial_user_message: The initial user message action

        Returns:
            list[dict]: A list of formatted messages ready for LLM consumption
        """
        """
        messages: list[Message] = []

        # System message
        messages.append(Message(role='system', content=[TextContent(text=self.system_prompt)]))

        # Get instruction from initial user message
        # User message is a MessageAction with content and image_urls, will be processed in events
        instruction = get_instruction(initial_user_message)
        include_a11y_tree = self.config.enable_a11y_tree

        # Build history prompts (alternating assistant/user messages)
        for event in events:
            if isinstance(event, OSWorldInteractiveAction) or isinstance(event, AgentFinishAction):
                messages.append(convert_action_to_message(event))
            elif isinstance(event, MessageAction):
                messages.append(convert_message_action_to_message_full_state(event, include_a11y_tree=include_a11y_tree))
            elif isinstance(event, OSWorldOutputObservation) or isinstance(event, ErrorObservation):
                msg = convert_observation_to_message_full_state(
                    event, instruction, include_a11y_tree=include_a11y_tree)
                messages.append(msg)
        """
        messages: list[Message] = []

        # Get instruction from initial user message
        # User message is a MessageAction with content and image_urls, will be processed in events
        instruction = get_instruction(initial_user_message)
        include_a11y_tree = self.config.enable_a11y_tree

        llm_response_ids_action = set()
        llm_response_ids_observation = set()

        # Build history prompts (alternating assistant/user messages) in reverse order
        for event in reversed(events):
            if isinstance(event, AgentFinishAction):
                messages.append(convert_action_to_message(event))
            elif isinstance(event, OSWorldInteractiveAction):
                llm_response_id = event.tool_call_metadata.model_response.id
                if llm_response_id in llm_response_ids_action:
                    continue
                llm_response_ids_action.add(llm_response_id)
                messages.append(convert_action_to_message(event))
            elif isinstance(event, MessageAction):
                messages.append(convert_message_action_to_message_full_state(event, include_a11y_tree=include_a11y_tree))
            elif isinstance(event, OSWorldOutputObservation) or isinstance(event, ErrorObservation):
                llm_response_id = event.tool_call_metadata.model_response.id
                if llm_response_id in llm_response_ids_observation:
                    continue
                msg = convert_observation_to_message_full_state(
                    event, instruction, include_a11y_tree=include_a11y_tree)
                messages.append(msg)
                llm_response_ids_observation.add(llm_response_id)

        # System message
        messages.append(Message(role='system', content=[TextContent(text=self.system_prompt)]))

        messages = messages[::-1]
        # set flags to know how to serialize the messages
        for message in messages:
            message.cache_enabled = False
            message.vision_enabled = True
            message.function_calling_enabled = True

        # let pydantic handle the serialization
        return [message.model_dump() for message in messages]