"""Example: Agent with tools, MCP, and skills in a Docker sandbox.

This example demonstrates setting up an agent in an isolated Docker container with:
1. Built-in tools (Terminal, FileEditor, Glob, Grep, Browser)
2. MCP (Model Context Protocol) servers running inside the container
3. Skills - uploaded to the sandbox for specialized knowledge
4. DockerWorkspace for sandboxed, secure execution

Docker workspaces provide:
- Isolated execution environment (no host system access)
- Security through containerization
- Consistent environment across different hosts
- Automatic cleanup on exit
- Optional GPU support

Requirements:
- Docker installed and running
- LLM_API_KEY environment variable set
"""

import os
import platform
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    SkillConfig,
    Tool,
    get_logger,
    load_skills,
    mount_skills,
)
from openhands.tools.browser_use import BrowserToolSet
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from openhands.tools.terminal import TerminalTool
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


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
# Full set of file system, terminal, and browser tools
tools = [
    Tool(name=TerminalTool.name),  # Execute shell commands
    Tool(name=FileEditorTool.name),  # Read/write/edit files
    Tool(name=GlobTool.name),  # Find files by pattern
    Tool(name=GrepTool.name),  # Search file contents
    Tool(name=BrowserToolSet.name),  # Web browser automation
]

# =============================================================================
# 3. Configure MCP Servers
# =============================================================================
# Note: MCP servers run within the Docker container, so ensure the
# required packages (uvx, npx) are available in the container image.
mcp_config = {
    "mcpServers": {
        # Fetch tool for retrieving web content
        "fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
    }
}

# =============================================================================
# 4. Configure Skills
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_FACTORY_DIR = SCRIPT_DIR.parent.parent / "skill_factory"
skills = load_skills([SkillConfig(source=SKILL_FACTORY_DIR / "pdf")])

# =============================================================================
# 5. Create Agent
# =============================================================================
agent = Agent(
    llm=llm,
    tools=tools,
    mcp_config=mcp_config,
    skills=skills,
)

# =============================================================================
# 6. Run with Docker Workspace
# =============================================================================
# build docker image locally:
# docker buildx build \
#     --file openhands-agent-server/openhands/agent_server/docker/Dockerfile \
#     --target source \
#     --tag openhands-agent-server:local \
#     --load \
#     .

with DockerWorkspace(
    server_image="openhands-agent-server:local",
    platform="linux/arm64",  # Match the platform of your built image
    host_port=None,
    enable_gpu=False,
) as workspace:
    # Mount skills to the sandbox (uploads skill files to /workspace/skills/)
    mount_skills(agent.skills, workspace)

    # Create conversation with remote workspace
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
    )

    # Task 1: Explore the sandbox environment
    print("\n📝 Task 1: Exploring the sandbox environment...")
    conversation.send_message(
        "Show me the current directory structure and system information "
        "(uname -a, python --version, df -h)."
    )
    conversation.run()

    # Task 2: Test MCP and file operations
    print("\n📝 Task 2: Testing MCP fetch and file operations...")
    conversation.send_message(
        "Use the fetch tool to get https://httpbin.org/json and save the response "
        "to test_response.json. Then use grep to find 'slideshow' in the saved file."
    )
    conversation.run()

    # Task 3: Test browser tool
    print("\n📝 Task 3: Testing browser tool...")
    conversation.send_message(
        "Use the browser to navigate to https://example.com, then get the page content "
        "and tell me what the page title is."
    )
    conversation.run()

    # Task 4: Test glob and grep tools
    print("\n📝 Task 4: Testing glob and grep tools...")
    conversation.send_message(
        "Use glob to list all JSON files in /workspace, then use grep to search "
        "for any 'title' fields in those files."
    )
    conversation.run()

    # Task 5: Test skill knowledge
    print("\n📝 Task 5: Testing skill knowledge...")
    conversation.send_message(
        "Based on your PDF skill, list the steps to create a PDF with Python. "
        "You can reference files in /workspace/skills/pdf/ for guidance."
    )
    conversation.run()

    # Cleanup
    print("\n📝 Cleanup: Removing created files...")
    conversation.send_message("Delete test_response.json if it exists.")
    conversation.run()

print("\n✅ Docker workspace example completed!")
