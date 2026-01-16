"""Example: MCP-only agent without filesystem access.

This example demonstrates setting up a lightweight agent that uses only
MCP (Model Context Protocol) servers for tool capabilities - no filesystem
access is needed or provided.

Use cases:
- RAG-based conversational agents using MCP-based retrieval tools
- Agents that only need to fetch external data (web, APIs)
- Chatbots with extended capabilities via MCP servers
- Scenarios where you want to explicitly disable filesystem access for safety

The agent runs locally but cannot read/write files or execute terminal commands.
"""

import os

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    LocalWorkspace,
    get_logger,
)


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
# 2. Configure MCP Servers (Our Only Tool Source)
# =============================================================================
# MCP (Model Context Protocol) servers provide tool capabilities without
# requiring filesystem access. Each server runs as a subprocess.
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
# 3. Create Agent (No Tools - MCP Only)
# =============================================================================
# Note: We're NOT providing any tools - only MCP servers.
# This means no Terminal, FileEditor, Glob, or Grep access.
agent = Agent(
    llm=llm,
    mcp_config=mcp_config,
)

# =============================================================================
# 4. Create Workspace Without Filesystem Access
# =============================================================================
workspace = LocalWorkspace()  # No working_dir = no filesystem access

# =============================================================================
# 5. Set up Conversation
# =============================================================================
conversation = Conversation(
    agent=agent,
    workspace=workspace,
)

# =============================================================================
# 6. Run the Agent
# =============================================================================
print("\n" + "=" * 80)
print("MCP-Only Agent Demo (No Filesystem Access)")
print("=" * 80)

# Task 1: Use MCP fetch to get web content
print("\n📝 Task 1: Using MCP fetch tool to get JSON data...")
conversation.send_message(
    "Use the fetch tool to get https://httpbin.org/json and tell me what "
    "the title of the slideshow is."
)
conversation.run()

# Task 2: Web research using MCP fetch
print("\n📝 Task 2: Research task using MCP fetch...")
conversation.send_message(
    "What is the current HTTP status code for 'OK' according to httpbin? "
    "Fetch https://httpbin.org/status/200 and explain what happens."
)
conversation.run()

# Task 3: Demonstrate that filesystem is not available
print("\n📝 Task 3: Ask about limitations...")
conversation.send_message(
    "What tools do you have available? Can you create or read files?"
)
conversation.run()
