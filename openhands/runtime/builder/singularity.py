import datetime
import os
import subprocess
import time
from pathlib import Path

from openhands import __version__ as oh_version
from openhands.core.exceptions import AgentRuntimeBuildError
from openhands.core.logger import RollingLogger
from openhands.core.logger import openhands_logger as logger
from openhands.runtime.builder.base import RuntimeBuilder
from openhands.utils.term_color import TermColor, colorize


class SingularityRuntimeBuilder(RuntimeBuilder):
    def __init__(self, singularity_cmd: str = 'apptainer'):
        """Initialize the Singularity runtime builder.

        Args:
            singularity_cmd: Command to use ('apptainer' or 'singularity')
        """
        self.singularity_cmd = singularity_cmd
        self._check_singularity_availability()
        self.rolling_logger = RollingLogger(max_lines=10)

    def _check_singularity_availability(self):
        """Check if Singularity/Apptainer is available on the system."""
        try:
            result = subprocess.run(
                [self.singularity_cmd, '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f'{self.singularity_cmd} version: {result.stdout.strip()}')
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise AgentRuntimeBuildError(
                f'{self.singularity_cmd} is not available on this system. '
                f'Please install {self.singularity_cmd} to use SingularityRuntimeBuilder.'
            ) from e

    def build(
        self,
        path: str,
        tags: list[str],
        platform: str | None = None,
        extra_build_args: list[str] | None = None,
        use_local_cache: bool = False,
    ) -> str:
        """Build a Singularity image.

        Args:
            path: Path to the build directory containing the definition file
            tags: List of image names/paths (only first one is used for Singularity)
            platform: Target platform (ignored for Singularity)
            extra_build_args: Additional build arguments
            use_local_cache: Whether to use local cache (ignored for Singularity)

        Returns:
            str: Path to the built SIF file

        Raises:
            AgentRuntimeBuildError: If the build fails
        """
        if not tags:
            raise AgentRuntimeBuildError("At least one tag must be provided")

        # Use the first tag as the primary image name
        primary_tag = tags[0]
        if not primary_tag.endswith('.sif'):
            primary_tag += '.sif'

        def_file = Path(path) / 'singularity.def'
        if not def_file.exists():
            raise AgentRuntimeBuildError(f"Definition file not found: {def_file}")

        # Build the image
        cmd = [
            self.singularity_cmd,
            'build',
#            f'--build-arg=OPENHANDS_RUNTIME_VERSION={oh_version}',
#            f'--build-arg=OPENHANDS_RUNTIME_BUILD_TIME={datetime.datetime.now().isoformat()}',
        ]

        if extra_build_args:
            cmd.extend(extra_build_args)

        cmd.extend([primary_tag, str(def_file)])

        self.rolling_logger.start(
            f'================ {self.singularity_cmd.upper()} BUILD STARTED ================'
        )

        logger.info(f"Building Singularity image with command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=path
            )

            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        self._output_logs(line)

            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code,
                    process.args,
                    output=process.stdout.read() if process.stdout else None,
                    stderr=process.stderr.read() if process.stderr else None,
                )

        except subprocess.CalledProcessError as e:
            logger.error(f'Singularity build failed: {e}')
            logger.error(f'Command output: {e.output}')
            if self.rolling_logger.is_enabled():
                logger.error(
                    f'{self.singularity_cmd} build output:\n' + self.rolling_logger.all_lines
                )
            raise AgentRuntimeBuildError(f"Singularity build failed: {e}")

        except subprocess.TimeoutExpired:
            logger.error('Singularity build timed out')
            raise AgentRuntimeBuildError('Singularity build timed out')

        except FileNotFoundError as e:
            logger.error(f'Singularity executable not found: {e}')
            raise AgentRuntimeBuildError(f'Singularity executable not found: {e}')

        except PermissionError as e:
            logger.error(f'Permission denied when trying to execute the build command: {e}')
            raise AgentRuntimeBuildError(f'Permission denied: {e}')

        except Exception as e:
            logger.error(f'An unexpected error occurred during the build process: {e}')
            raise AgentRuntimeBuildError(f'Unexpected error during build: {e}')

        logger.info(f'Image [{primary_tag}] build finished.')

        # Create symbolic links or copies for additional tags
        primary_path = Path(primary_tag)
        for tag in tags[1:]:
            if not tag.endswith('.sif'):
                tag += '.sif'
            tag_path = Path(tag)
            if tag_path != primary_path:
                try:
                    if tag_path.exists():
                        tag_path.unlink()
                    # Create hard link if possible, otherwise copy
                    try:
                        tag_path.hardlink_to(primary_path)
                    except OSError:
                        import shutil
                        shutil.copy2(primary_path, tag_path)
                    logger.debug(f"Created additional tag: {tag}")
                except Exception as e:
                    logger.warning(f"Failed to create additional tag {tag}: {e}")

        # Verify the image was built successfully
        if not primary_path.exists():
            raise AgentRuntimeBuildError(
                f'Build failed: Image {primary_tag} not found after build'
            )

        logger.info(f'Image {primary_tag} built successfully')
        return str(primary_path)

    def image_exists(self, image_name: str, pull_from_repo: bool = True) -> bool:
        """Check if a SIF image exists locally.

        Args:
            image_name: Path to the SIF file or image name
            pull_from_repo: Whether to pull from remote (not applicable for Singularity local builds)

        Returns:
            bool: True if the SIF file exists locally
        """
        if not image_name:
            logger.error(f'Invalid image name: `{image_name}`')
            return False

        # Handle both .sif files and paths without extension
        if image_name.endswith('.sif'):
            image_path = Path(image_name)
        else:
            image_path = Path(f"{image_name}.sif")

        exists = image_path.exists()

        if exists:
            logger.debug(f'Singularity image found locally: {image_path}')
        else:
            logger.debug(f'Singularity image {colorize("not found", TermColor.WARNING)} locally: {image_path}')

            # For Singularity, we could potentially pull from a registry if the image_name
            # looks like a Docker image reference and pull_from_repo is True
            if pull_from_repo and self._is_docker_image_reference(image_name):
                return self._try_pull_docker_image(image_name)

        return exists

    def _is_docker_image_reference(self, image_name: str) -> bool:
        """Check if the image name looks like a Docker image reference."""
        # Simple heuristic: contains '/' or ':' and doesn't end with .sif
        return ('/' in image_name or ':' in image_name) and not image_name.endswith('.sif')

    def _try_pull_docker_image(self, image_name: str) -> bool:
        """Try to pull a Docker image and convert it to SIF format."""
        try:
            # Generate SIF filename from Docker image name
            sif_name = image_name.replace(':', '_').replace('/', '_') + '.sif'

            logger.debug(f'Attempting to pull Docker image {image_name} and convert to {sif_name}')

            pull_cmd = [
                self.singularity_cmd,
                'pull',
                '--disable-cache',
                sif_name,
                f'docker://{image_name}'
            ]

            result = subprocess.run(
                pull_cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.debug(f'Successfully pulled and converted {image_name} to {sif_name}')
            return True

        except subprocess.CalledProcessError as e:
            logger.debug(f'Failed to pull Docker image {image_name}: {e.stderr}')
            return False
        except Exception as e:
            logger.debug(f'Unexpected error pulling Docker image {image_name}: {e}')
            return False

    def _output_logs(self, new_line: str) -> None:
        """Output build logs either to rolling logger or debug logger."""
        if not self.rolling_logger.is_enabled():
            logger.debug(new_line)
        else:
            self.rolling_logger.add_line(new_line)

    def _prune_old_cache_files(self, cache_dir: str, max_age_days: int = 7) -> None:
        """Prune cache files older than the specified number of days.

        Note: Singularity doesn't use the same caching mechanism as Docker,
        but this method is provided for interface compatibility.
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60

            for root, _, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith('.sif'):
                        file_path = os.path.join(root, file)
                        try:
                            file_age = current_time - os.path.getmtime(file_path)
                            if file_age > max_age_seconds:
                                os.remove(file_path)
                                logger.debug(f'Removed old SIF cache file: {file_path}')
                        except Exception as e:
                            logger.warning(f'Error processing cache file {file_path}: {e}')
        except Exception as e:
            logger.warning(f'Error during SIF cache pruning: {e}')

    def _is_cache_usable(self, cache_dir: str) -> bool:
        """Check if the cache directory is usable.

        Note: Provided for interface compatibility with DockerRuntimeBuilder.
        Singularity has its own caching mechanisms.
        """
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                logger.debug(f'Created cache directory: {cache_dir}')
            except OSError as e:
                logger.debug(f'Failed to create cache directory {cache_dir}: {e}')
                return False

        if not os.access(cache_dir, os.W_OK):
            logger.warning(
                f'Cache directory {cache_dir} is not writable. Caches will not be used for Singularity builds.'
            )
            return False

        self._prune_old_cache_files(cache_dir)

        logger.debug(f'Cache directory {cache_dir} is usable')
        return True
