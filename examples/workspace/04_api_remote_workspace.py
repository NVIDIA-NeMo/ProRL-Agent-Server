"""Example: Agent with APIRemoteWorkspace for Kubernetes-based execution.

This example demonstrates using the Runtime API to provision and run agents
in Kubernetes pods. This is ideal for:
1. Scalable, cloud-native agent execution
2. Parallel batch rollouts across multiple pods
3. Production deployments with resource management
4. Isolated execution environments managed by K8s

APIRemoteWorkspace provides:
- Automatic K8s pod provisioning via Runtime API
- Resource scaling (1x, 2x, 4x, 8x)
- Session persistence (pause/resume)
- Health checking and automatic retries
- Clean teardown on exit

How it works:
1. APIRemoteWorkspace calls the Runtime API to create a new K8s pod
2. The pod runs the OpenHands agent-server container image
3. Runtime API returns a URL to the running agent server
4. All agent operations are proxied through this URL
5. On cleanup, the pod is stopped (or paused if configured)

Architecture:
    ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
    │   Your Script   │──────▶│   Runtime API   │──────▶│   K8s Cluster   │
    │ (this example)  │       │  (provisions)   │       │   (runs pods)   │
    └─────────────────┘       └─────────────────┘       └─────────────────┘
            │                                                    │
            │                    HTTP/WebSocket                  │
            └────────────────────────────────────────────────────┘
                           (agent communication)

Requirements:
- LLM_API_KEY environment variable set
- RUNTIME_API_KEY environment variable set (get from runtime.all-hands.dev)
- RUNTIME_API_URL (optional, defaults to https://runtime.eval.all-hands.dev)

Usage:
    # Set required environment variables
    export LLM_API_KEY="your-llm-api-key"
    export RUNTIME_API_KEY="your-runtime-api-key"

    # Run with defaults
    python 04_api_remote_workspace.py

    # Run with custom settings
    python 04_api_remote_workspace.py --resource-factor 2 --keep-alive

    # Run with session persistence (can reconnect later)
    python 04_api_remote_workspace.py --session-id my-session --pause-on-close
"""

import argparse
import os
import time

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Tool,
    get_logger,
)
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from openhands.tools.terminal import TerminalTool
from openhands.workspace import APIRemoteWorkspace


logger = get_logger(__name__)


def main():
    # ==========================================================================
    # Parse Arguments
    # ==========================================================================
    parser = argparse.ArgumentParser(
        description="Run an agent with APIRemoteWorkspace (Kubernetes-based)"
    )
    parser.add_argument(
        "--runtime-api-url",
        type=str,
        default=os.getenv("RUNTIME_API_URL", "https://runtime.eval.all-hands.dev"),
        help="Runtime API URL (default: from RUNTIME_API_URL env or eval endpoint)",
    )
    parser.add_argument(
        "--server-image",
        type=str,
        default=os.getenv(
            "SERVER_IMAGE", "ghcr.io/openhands/agent-server:main-python-amd64"
        ),
        help="Container image for the agent server",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for the workspace (auto-generated if not specified)",
    )
    parser.add_argument(
        "--resource-factor",
        type=int,
        choices=[1, 2, 4, 8],
        default=1,
        help="Resource scaling factor (1=base, 2=2x, 4=4x, 8=8x resources)",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the runtime alive after script exits (for debugging)",
    )
    parser.add_argument(
        "--pause-on-close",
        action="store_true",
        help="Pause (instead of stop) the runtime on exit (can resume later)",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=60.0,
        help="Timeout for API requests in seconds",
    )
    args = parser.parse_args()

    # ==========================================================================
    # 1. Validate Environment
    # ==========================================================================
    runtime_api_key = os.getenv("RUNTIME_API_KEY")
    if not runtime_api_key:
        logger.error(
            "RUNTIME_API_KEY environment variable is required.\n"
            "Get your API key from: https://runtime.all-hands.dev"
        )
        exit(1)

    # ==========================================================================
    # 2. Configure LLM
    # ==========================================================================
    llm = LLM(
        usage_id="api-remote-agent",
        model=os.getenv("LLM_MODEL"),
        api_key=SecretStr(os.getenv("LLM_API_KEY")),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    # ==========================================================================
    # 3. Configure Tools
    # ==========================================================================
    tools = [
        Tool(name=TerminalTool.name),  # Execute shell commands
        Tool(name=FileEditorTool.name),  # Read/write/edit files
        Tool(name=GlobTool.name),  # Find files by pattern
        Tool(name=GrepTool.name),  # Search file contents
    ]

    # ==========================================================================
    # 4. Create Agent
    # ==========================================================================
    agent = Agent(
        llm=llm,
        tools=tools,
    )

    # ==========================================================================
    # 5. Run with APIRemoteWorkspace
    # ==========================================================================
    print("\n" + "=" * 80)
    print("APIRemoteWorkspace Demo (Kubernetes-Based Agent Execution)")
    print("=" * 80)

    # Configure workspace
    workspace_kwargs = {
        "runtime_api_url": args.runtime_api_url,
        "runtime_api_key": runtime_api_key,
        "server_image": args.server_image,
        "resource_factor": args.resource_factor,
        "keep_alive": args.keep_alive,
        "pause_on_close": args.pause_on_close,
        "api_timeout": args.api_timeout,
        "image_pull_policy": "IfNotPresent",
    }

    if args.session_id:
        workspace_kwargs["session_id"] = args.session_id
    else:
        workspace_kwargs["session_id"] = f"demo-{int(time.time())}"

    print("\n🚀 Provisioning Kubernetes pod via Runtime API...")
    print("   (This may take 30-60 seconds on first run)")

    with APIRemoteWorkspace(**workspace_kwargs) as workspace:
        # Create conversation with remote workspace
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
        )

        # Task 1: Explore the K8s pod environment
        print("\n📝 Task 1: Exploring the Kubernetes pod environment...")
        conversation.send_message(
            "Show me the pod environment:\n"
            "1. Run 'uname -a' to show system info\n"
            "2. Run 'whoami' to show current user\n"
            "3. Run 'cat /proc/meminfo | head -5' to show memory info\n"
            "4. Run 'nproc' to show CPU count\n"
            "5. List files in /workspace"
        )
        conversation.run()

        # Task 2: Create and run a Python script
        print("\n📝 Task 2: Testing file operations and code execution...")
        conversation.send_message(
            "Create a file called 'k8s_test.py' with a Python script that:\n"
            "1. Prints 'Hello from Kubernetes!'\n"
            "2. Reads and prints the hostname from /etc/hostname\n"
            "3. Prints the current working directory\n"
            "4. Lists all files in /workspace\n"
            "\nThen run the script."
        )
        conversation.run()

        # Task 3: Test network connectivity (typical in K8s)
        print("\n📝 Task 3: Testing network connectivity...")
        conversation.send_message(
            "Test network connectivity:\n"
            "1. Run 'curl -s https://httpbin.org/ip' to show external IP\n"
            "2. Save the response to 'network_test.json'\n"
            "3. Use grep to find 'origin' in the saved file"
        )
        conversation.run()

        # Cleanup
        print("\n📝 Cleanup: Removing created files...")
        conversation.send_message(
            "Delete k8s_test.py and network_test.json if they exist. "
            "List remaining files in /workspace."
        )
        conversation.run()

    # Post-cleanup info
    if args.keep_alive:
        print("\n⚠️  Runtime kept alive. Remember to stop it manually!")
    elif args.pause_on_close:
        print(
            f"\n💤 Runtime paused. Resume later with: --session-id {workspace_kwargs['session_id']}"
        )
    else:
        print("\n✅ Runtime stopped and cleaned up.")

    print("\n💡 Tips for production usage:")
    print("   1. Use --session-id for persistent sessions")
    print("   2. Use --pause-on-close to save costs between runs")
    print("   3. Use --resource-factor for resource-intensive tasks")
    print("   4. Set RUNTIME_API_URL for production vs eval environments")


if __name__ == "__main__":
    main()
