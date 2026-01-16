"""Example: Workspace initialization hooks (init_directories and init_commands).

This example demonstrates using workspace initialization hooks to:
1. Pre-create directories when the workspace starts
2. Run setup commands before the agent begins work

Use cases:
- Create memory/cache directories for persistent agent state
- Install additional packages the agent needs
- Set up git configuration or other environment settings
- Initialize databases or other services

Requirements:
- Docker installed and running
- LLM_API_KEY environment variable set
"""

import os
import platform

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Tool,
    get_logger,
)
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
# 4. Configure Workspace Init Hooks
# =============================================================================
# Define directories to create on workspace init
init_directories = [
    "/workspace/memories",
    "/workspace/cache",
    "/workspace/output/reports",
]

# Define commands to run on workspace init
init_commands = [
    # Create an initialization marker file
    "echo 'Workspace initialized at $(date)' > /workspace/.init_marker",
    # Set up a simple config file
    "echo '{\"version\": 1, \"initialized\": true}' > /workspace/config.json",
]

# =============================================================================
# 5. Run with Docker Workspace + Init Hooks
# =============================================================================
print("\n" + "=" * 80)
print("Workspace Init Hooks Demo")
print("=" * 80)
print(f"\nInit directories: {init_directories}")
print(f"Init commands: {init_commands}")

with DockerWorkspace(
    server_image="openhands-agent-server:local",
    host_port=None,
    platform=detect_platform(),
    enable_gpu=False,
    init_directories=init_directories,
    init_commands=init_commands,
) as workspace:
    # Create conversation with remote workspace
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
    )

    # Task: Verify init hooks ran successfully
    print("\n" + "-" * 40)
    print("Task: Verify workspace initialization")
    print("-" * 40)

    conversation.send_message(
        """Please verify that the workspace was initialized correctly by checking:

1. Check if the following directories exist:
   - /workspace/memories
   - /workspace/cache
   - /workspace/output/reports

2. Check if the init marker file exists and show its contents:
   - /workspace/.init_marker

3. Check if the config file exists and show its contents:
   - /workspace/config.json

4. Create a test file in /workspace/memories/test.txt with the content "Init hooks work!"

Report what you find for each item."""
    )
    conversation.run()

print("\n✅ Workspace init hooks example completed!")
