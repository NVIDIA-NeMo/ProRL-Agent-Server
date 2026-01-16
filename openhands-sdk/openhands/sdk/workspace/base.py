from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.models import DiscriminatedUnionMixin
from openhands.sdk.workspace.models import CommandResult, FileOperationResult


logger = get_logger(__name__)


def _convert_path_to_str(v: str | Path | None) -> str | None:
    """Convert Path objects to string for working_dir."""
    if v is None:
        return None
    if isinstance(v, Path):
        return str(v)
    return v


class BaseWorkspace(DiscriminatedUnionMixin, ABC):
    """Abstract base class for workspace implementations.

    Workspaces provide a sandboxed environment where agents can execute commands,
    read/write files, and perform other operations. All workspace implementations
    support the context manager protocol for safe resource management.

    When working_dir is None, filesystem-based tools (Terminal, FileEditor, Glob,
    Grep) will not be available. MCP tools are still available.

    Example:
        >>> with workspace:
        ...     result = workspace.execute_command("echo 'hello'")
        ...     content = workspace.read_file("example.txt")
    """

    working_dir: Annotated[
        str | None,
        BeforeValidator(_convert_path_to_str),
        Field(
            default=None,
            description=(
                "The working directory for agent operations and tool execution. "
                "Accepts both string paths and Path objects. "
                "Path objects are automatically converted to strings. "
                "When None, filesystem-based tools are disabled but MCP still works."
            ),
        ),
    ]

    def __enter__(self) -> "BaseWorkspace":
        """Enter the workspace context.

        Returns:
            Self for use in with statements
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the workspace context and cleanup resources.

        Default implementation performs no cleanup. Subclasses should override
        to add cleanup logic (e.g., stopping containers, closing connections).

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        pass

    @abstractmethod
    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        """Execute a bash command on the system.

        Args:
            command: The bash command to execute
            cwd: Working directory for the command (optional)
            timeout: Timeout in seconds (defaults to 30.0)

        Returns:
            CommandResult: Result containing stdout, stderr, exit_code, and other
                metadata

        Raises:
            Exception: If command execution fails
        """
        ...

    @abstractmethod
    def file_upload(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        """Upload a file to the system.

        Args:
            source_path: Path to the source file
            destination_path: Path where the file should be uploaded

        Returns:
            FileOperationResult: Result containing success status and metadata

        Raises:
            Exception: If file upload fails
        """
        ...

    @abstractmethod
    def file_download(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        """Download a file from the system.

        Args:
            source_path: Path to the source file on the system
            destination_path: Path where the file should be downloaded

        Returns:
            FileOperationResult: Result containing success status and metadata

        Raises:
            Exception: If file download fails
        """
        ...

    def upload_directory(
        self,
        source_dir: str | Path,
        destination_path: str | Path,
    ) -> str:
        """Upload a directory and all its contents to the workspace.

        Recursively uploads all files from a local directory to the workspace,
        preserving the directory structure.

        Args:
            source_dir: Local path to the directory to upload
            destination_path: Destination path in the workspace

        Returns:
            The destination path where files were uploaded

        Raises:
            ValueError: If source_dir is not a directory
            RuntimeError: If any file upload fails

        Example:
            >>> workspace.upload_directory("./my-project", "/workspace/my-project")
            '/workspace/my-project'
        """
        source = Path(source_dir).resolve()
        dest = str(destination_path)

        if not source.is_dir():
            raise ValueError(f"Source must be a directory: {source}")

        file_count = 0
        for file_path in source.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(source)
                dest_file = f"{dest}/{relative}"
                result = self.file_upload(str(file_path), dest_file)
                if not result.success:
                    raise RuntimeError(
                        f"Failed to upload {file_path} to {dest_file}: {result.error}"
                    )
                file_count += 1

        logger.debug(f"Uploaded {file_count} files to {dest}")
        return dest

    def download_directory(
        self,
        source_path: str | Path,
        destination_dir: str | Path,
    ) -> str:
        """Download a directory and all its contents from the workspace.

        Recursively downloads all files from a workspace directory to the local
        filesystem, preserving the directory structure.

        Args:
            source_path: Path to the directory in the workspace
            destination_dir: Local path where the directory should be downloaded

        Returns:
            The local destination path where files were downloaded

        Raises:
            RuntimeError: If listing files or downloading fails

        Example:
            >>> workspace.download_directory("/workspace/output", "./local-output")
            '/path/to/local-output'
        """
        source = str(source_path)
        dest = Path(destination_dir).resolve()

        # Create destination directory
        dest.mkdir(parents=True, exist_ok=True)

        # List all files in the source directory using find
        result = self.execute_command(
            f"find {source} -type f",
            timeout=60.0,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to list files in {source}: {result.stderr}"
            )

        # Parse file list from stdout
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

        file_count = 0
        for file_path in files:
            # Calculate relative path from source directory
            if file_path.startswith(source):
                relative = file_path[len(source):].lstrip("/")
            else:
                relative = Path(file_path).name

            dest_file = dest / relative
            download_result = self.file_download(file_path, str(dest_file))
            if not download_result.success:
                raise RuntimeError(
                    f"Failed to download {file_path} to {dest_file}: {download_result.error}"
                )
            file_count += 1

        logger.debug(f"Downloaded {file_count} files to {dest}")
        return str(dest)

    def upload_skill(self, source_dir: str | Path, mount_path: str) -> str:
        """Upload a skill directory into the workspace.

        Subclasses may override this (e.g., LocalWorkspace can avoid copying).

        Args:
            source_dir: Local path to the skill directory.
            mount_path: Destination path in the workspace.

        Returns:
            The sandbox path where the skill was uploaded.
        """
        return self.upload_directory(source_dir, mount_path)

    def pause(self) -> None:
        """Pause the workspace to conserve resources.

        For local workspaces, this is a no-op.
        For container-based workspaces, this pauses the container.

        Raises:
            NotImplementedError: If the workspace type does not support pausing.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support pause()")

    def resume(self) -> None:
        """Resume a paused workspace.

        For local workspaces, this is a no-op.
        For container-based workspaces, this resumes the container.

        Raises:
            NotImplementedError: If the workspace type does not support resuming.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support resume()")
