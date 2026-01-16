"""Example: Full local agent with tools, MCP, and skills.

This example demonstrates setting up a complete local agent with:
1. Built-in tools (Terminal, FileEditor, Glob, Grep)
2. MCP (Model Context Protocol) servers for extended capabilities
3. Skills - loaded from SKILL.md files for specialized knowledge
4. LocalWorkspace for direct filesystem access

The agent runs entirely locally, with direct access to the host filesystem.
This is ideal for development and scenarios where you trust the environment.
"""

import os
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    LocalWorkspace,
    SkillConfig,
    Tool,
    get_logger,
    load_skills,
    mount_skills,
)
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from openhands.tools.terminal import TerminalTool


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
# Full set of file system and terminal tools
tools = [
    Tool(name=TerminalTool.name),  # Execute shell commands
    Tool(name=FileEditorTool.name),  # Read/write/edit files
    Tool(name=GlobTool.name),  # Find files by pattern
    Tool(name=GrepTool.name),  # Search file contents
]

# =============================================================================
# 3. Configure MCP Servers
# =============================================================================
# MCP (Model Context Protocol) servers provide additional tool capabilities.
# Each server runs as a subprocess and exposes tools via the MCP protocol.
#
# Common MCP servers:
#   - mcp-server-fetch: Fetch web content
#   - repomix: Analyze and pack codebases
#   - mcp-server-filesystem: Enhanced file operations
#   - mcp-server-github: GitHub API integration
mcp_config = {
    "mcpServers": {
        # Fetch tool for retrieving web content
        "fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
        # Repomix for code analysis
        "repomix": {
            "command": "npx",
            "args": ["-y", "repomix@1.4.2", "--mcp"],
        },
    }
}

logger.info(f"Configured MCP servers: {list(mcp_config['mcpServers'].keys())}")

# =============================================================================
# 4. Configure Skills
# =============================================================================

# Load Skills (e.g., skill_factory)
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
    # Optional: Filter which tools are exposed to the LLM
    # This regex allows all non-repomix tools + only repomix's pack_codebase
    filter_tools_regex="^(?!repomix)(.*)|^repomix.*pack_codebase.*$",
)


# =============================================================================
# 6. Set up Local Workspace
# =============================================================================
# Use current working directory as the workspace
workspace_dir = Path(os.getcwd())
workspace = LocalWorkspace(working_dir=workspace_dir)

# Upload skills to workspace (LocalWorkspace uses source paths)
mount_skills(skills, workspace)

# =============================================================================
# 7. Set up Conversation
# =============================================================================
conversation = Conversation(
    agent=agent,
    workspace=workspace,
)

# =============================================================================
# 8. Run the Agent
# =============================================================================
print("\n" + "=" * 80)
print("Local Agent with Tools, MCP, and Skills Demo")
print("=" * 80)

# Task 1: Test file operations with tools
print("\n📝 Task 1: Create and explore files using tools...")
conversation.send_message(
    "Create a Python file called 'demo_calculator.py' with a simple Calculator class "
    "that has add, subtract, multiply, and divide methods. Include proper docstrings "
    "and type hints."
)
conversation.run()

# Task 2: Test MCP fetch tool
print("\n📝 Task 2: Using MCP fetch tool to get web content...")
conversation.send_message(
    "Use the fetch tool to get https://httpbin.org/json and save the 'slideshow.title' "
    "value into a file called 'fetched_title.txt'."
)
conversation.run()

# Task 3: Test grep and glob tools
print("\n📝 Task 3: Using grep and glob tools to search the workspace...")
conversation.send_message(
    "Use the glob tool to find all .py files in the current directory, "
    "then use grep to find any function definitions in those files."
)
conversation.run()

# Task 4: Test skill knowledge (if file skills available)
print("\n📝 Task 4: Testing file-based skill knowledge...")
conversation.send_message(
    "Based on your PDF skill knowledge, what Python library should I use "
    "to create a PDF document programmatically? Give me a brief code example."
)
conversation.run()

# Cleanup
print("\n📝 Cleanup: Removing created files...")
conversation.send_message(
    "Delete demo_calculator.py and fetched_title.txt if they exist."
)
conversation.run()