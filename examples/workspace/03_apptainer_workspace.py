"""Example: Agent with Apptainer workspace for HPC environments.

This example demonstrates setting up an agent in an Apptainer (formerly Singularity)
container, which is ideal for:
1. HPC (High-Performance Computing) clusters where Docker is not available
2. Slurm-managed environments
3. Multi-user systems where root access is not available
4. Secure, isolated execution without Docker daemon

Apptainer workspaces provide:
- No root/sudo required (unlike Docker)
- Compatible with HPC job schedulers (Slurm, PBS, etc.)
- Security model designed for multi-tenant systems
- Can use pre-built Docker images (converts to SIF format)
- Automatic caching of SIF files for faster subsequent runs

How it works:
1. ApptainerWorkspace pulls a Docker image and converts it to Apptainer SIF format
2. OR uses an existing SIF file directly (faster for repeated runs)
3. Starts the agent server inside the Apptainer container
4. Exposes the server on a local port for HTTP/WebSocket communication
5. Cleans up the container process on exit

Requirements:
- Apptainer installed (https://apptainer.org/docs/user/main/quick_start.html)
- LLM_API_KEY environment variable set
- Optional: Pre-built SIF file for faster startup

Installation (Ubuntu/Debian):
    sudo apt-get update && sudo apt-get install -y apptainer

Installation (CentOS/RHEL):
    sudo yum install -y apptainer

Usage:
    # Basic usage with Docker image (will convert to SIF on first run)
    python 03_apptainer_workspace.py

    # With pre-built SIF file (faster)
    python 03_apptainer_workspace.py --sif-file /path/to/agent-server.sif

    # Build SIF file for later use:
    apptainer pull agent-server.sif docker://ghcr.io/openhands/agent-server:main-python
"""

import argparse
import os
import subprocess

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
from openhands.workspace import ApptainerWorkspace


logger = get_logger(__name__)


def check_apptainer_installed() -> bool:
    """Check if Apptainer is installed and accessible."""
    try:
        result = subprocess.run(
            ["apptainer", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info(f"Apptainer version: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


def main():
    # ==========================================================================
    # Parse Arguments
    # ==========================================================================
    parser = argparse.ArgumentParser(
        description="Run an agent with Apptainer workspace (HPC-friendly)"
    )
    parser.add_argument(
        "--sif-file",
        type=str,
        default=None,
        help="Path to pre-built Apptainer SIF file (faster startup)",
    )
    parser.add_argument(
        "--server-image",
        type=str,
        default="ghcr.io/openhands/agent-server:main-python",
        help="Docker image to use (converted to SIF if --sif-file not provided)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the agent server (auto-assigned if not specified)",
    )
    parser.add_argument(
        "--mount-dir",
        type=str,
        default=None,
        help="Host directory to mount into the container at /workspace",
    )
    parser.add_argument(
        "--no-fakeroot",
        action="store_true",
        help="Disable --fakeroot (use if fakeroot is not supported)",
    )
    args = parser.parse_args()

    # ==========================================================================
    # 1. Check Prerequisites
    # ==========================================================================
    if not check_apptainer_installed():
        logger.error(
            "Apptainer is not installed. Please install it from:\n"
            "  https://apptainer.org/docs/user/main/quick_start.html\n"
            "\n"
            "Ubuntu/Debian: sudo apt-get install -y apptainer\n"
            "CentOS/RHEL:   sudo yum install -y apptainer"
        )
        exit(1)

    # ==========================================================================
    # 2. Configure LLM
    # ==========================================================================
    llm = LLM(
        usage_id="apptainer-agent",
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
    # 5. Run with Apptainer Workspace
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Apptainer Workspace Demo (HPC-Friendly Agent Execution)")
    print("=" * 80)

    # Determine workspace configuration
    workspace_kwargs = {
        "host_port": args.port,
        "use_fakeroot": not args.no_fakeroot,
        "detach_logs": True,
    }

    if args.mount_dir:
        workspace_kwargs["mount_dir"] = args.mount_dir

    # Use either SIF file or Docker image
    if args.sif_file:
        workspace_kwargs["sif_file"] = args.sif_file
    else:
        workspace_kwargs["server_image"] = args.server_image

    with ApptainerWorkspace(**workspace_kwargs) as workspace:
        # Create conversation with remote workspace
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
        )

        # Task 1: Explore the container environment
        print("\n📝 Task 1: Exploring the Apptainer container environment...")
        conversation.send_message(
            "Show me the container environment:\n"
            "1. Run 'uname -a' to show system info\n"
            "2. Run 'whoami' to show current user\n"
            "3. Run 'python --version' to show Python version\n"
            "4. List files in /workspace"
        )
        conversation.run()

        # Task 2: Create and manipulate files
        print("\n📝 Task 2: Testing file operations...")
        conversation.send_message(
            "Create a file called 'hello.py' with a simple Python script that:\n"
            "1. Prints 'Hello from Apptainer!'\n"
            "2. Prints the current working directory\n"
            "3. Lists environment variables starting with 'HOME' or 'USER'\n"
            "\nThen run the script."
        )
        conversation.run()

        # Task 3: Test glob and grep tools
        print("\n📝 Task 3: Testing glob and grep tools...")
        conversation.send_message(
            "Use glob to find all .py files in /workspace, "
            "then use grep to search for 'print' in those files."
        )
        conversation.run()

        # Cleanup
        print("\n📝 Cleanup: Removing created files...")
        conversation.send_message(
            "Delete hello.py if it exists. List remaining files in /workspace."
        )
        conversation.run()

    print("\n✅ Apptainer workspace example completed!")
    print("\n💡 Tips for HPC/Slurm usage:")
    print("   1. Pre-build your SIF file on a login node:")
    print(f"      apptainer pull agent-server.sif docker://{args.server_image}")
    print("   2. Use --sif-file in your Slurm job scripts for faster startup")
    print("   3. Mount shared storage with --mount-dir for data persistence")


if __name__ == "__main__":
    main()
