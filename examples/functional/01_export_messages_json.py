"""Example: Export conversation events as ChatML-style messages JSON.

This example demonstrates how to:
1. Run an agent in a Docker sandbox
2. Collect conversation events during execution
3. Convert events to ChatML-style message format
4. Save the messages to a JSON file

The output format is compatible with OpenAI Chat Completions API and can be used for:
- Fine-tuning datasets
- Conversation replay/analysis
- Integration with other LLM tools

Requirements:
- Docker installed and running
- LLM_API_KEY environment variable set
"""

import json
import os
import platform
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    Tool,
    get_logger,
)
from openhands.sdk.event import View
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


def detect_platform() -> str:
    """Detect the correct Docker platform for the current machine."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def events_to_chatml(events: list[Event]) -> list[dict]:
    """Convert conversation events to ChatML-style message dictionaries.

    This utility function:
    1. Filters events to only LLM-convertible ones using View
    2. Converts events to Message objects (handling action batching)
    3. Serializes each Message to ChatML dict format

    Args:
        events: List of conversation events

    Returns:
        List of ChatML-style message dicts with 'role' and 'content' keys
    """
    # Filter to LLM-convertible events using View
    view = View.from_events(events)
    llm_convertible_events = view.events

    if not llm_convertible_events:
        return []

    # Convert events to Message objects
    messages = LLMConvertibleEvent.events_to_messages(llm_convertible_events)

    # Convert each Message to ChatML dict format
    chatml_messages = []
    for msg in messages:
        # Enable function calling for proper tool_calls serialization
        msg.function_calling_enabled = True
        chatml_messages.append(msg.to_chat_dict())

    return chatml_messages


# =============================================================================
# 1. Configure LLM
# =============================================================================
llm = LLM(
    usage_id="agent",
    model=os.getenv("LLM_MODEL"),
    api_key=SecretStr(os.getenv("LLM_API_KEY")),
    base_url=os.getenv("LLM_BASE_URL"),
)

# =============================================================================
# 2. Configure Tools
# =============================================================================
tools = [
    Tool(name=TerminalTool.name),
    Tool(name=FileEditorTool.name),
]

# =============================================================================
# 3. Create Agent
# =============================================================================
agent = Agent(
    llm=llm,
    tools=tools,
)

# =============================================================================
# 4. Run with Docker Workspace
# =============================================================================
print("\n" + "=" * 80)
print("Export Conversation as ChatML Messages Demo")
print("=" * 80)

# Output file for messages
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_FILE = SCRIPT_DIR / "messages.json"

# Collect events during execution
received_events: list[Event] = []


def event_callback(event: Event) -> None:
    """Collect events for later conversion to ChatML format."""
    received_events.append(event)


with DockerWorkspace(
    server_image="openhands-agent-server:local",
    host_port=None,
    platform=detect_platform(),
    enable_gpu=False,
) as workspace:
    # Create conversation with event callback
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        callbacks=[event_callback],
    )

    # Run a simple multi-step task
    print("\n" + "-" * 40)
    print("Task: Create a Python script and run it")
    print("-" * 40)

    conversation.send_message(
        """Create a simple Python script called hello.py that:
1. Prints "Hello from OpenHands!"
2. Lists the current directory contents
3. Prints the current date and time

Then run the script and show me the output."""
    )
    conversation.run()

    # Convert and save events as ChatML messages
    print("\n" + "-" * 40)
    print("Exporting conversation to ChatML format")
    print("-" * 40)

    chatml_messages = events_to_chatml(received_events)

    # Save to JSON file with pretty formatting
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chatml_messages, f, indent=2, ensure_ascii=False)

    # Print summary of message roles
    role_counts: dict[str, int] = {}
    for msg in chatml_messages:
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    print(f"\nMessage breakdown by role:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")

    print(f"\nTotal events collected: {len(received_events)}")
    print(f"ChatML messages exported: {len(chatml_messages)}")

print(f"\n✅ Messages saved to: {OUTPUT_FILE}")
