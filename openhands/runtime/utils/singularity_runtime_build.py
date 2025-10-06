import argparse
import hashlib
import os
import shutil
import string
import subprocess
import tempfile
from enum import Enum
from pathlib import Path

from dirhash import dirhash
from jinja2 import Environment, FileSystemLoader

import openhands
from openhands import __version__ as oh_version
from openhands.core.exceptions import AgentRuntimeBuildError
from openhands.core.logger import openhands_logger as logger


class BuildFromImageType(Enum):
    SCRATCH = 'scratch'  # Slowest: Build from base image (no dependencies are reused)
    VERSIONED = 'versioned'  # Medium speed: Reuse the most recent image with the same base image & OH version
    LOCK = 'lock'  # Fastest: Reuse the most recent image with the exact SAME dependencies


class SingularityRuntimeBuilder:
    """Runtime builder for Singularity/Apptainer containers."""

    def __init__(self, singularity_cmd: str = 'apptainer'):
        """Initialize the Singularity runtime builder.

        Args:
            singularity_cmd: Command to use ('apptainer' or 'singularity')
        """
        self.singularity_cmd = singularity_cmd

    def image_exists(self, image_path: str, check_remote: bool = True) -> bool:
        """Check if a SIF image exists locally.

        Args:
            image_path: Path to the SIF file
            check_remote: Ignored for Singularity (no remote registry concept like Docker)

        Returns:
            bool: True if the SIF file exists locally
        """
        if image_path.endswith('.sif'):
            return Path(image_path).exists()
        else:
            # Assume it's a path without extension
            sif_path = f"{image_path}.sif"
            return Path(sif_path).exists()

    def build(
        self,
        path: str,
        tags: list[str],
        platform: str | None = None,
        extra_build_args: list[str] | None = None,
    ) -> str:
        """Build a Singularity image.

        Args:
            path: Path to the build directory containing the definition file
            tags: List of image names/paths (only first one is used for Singularity)
            platform: Target platform (ignored for Singularity)
            extra_build_args: Additional build arguments

        Returns:
            str: Path to the built SIF file
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
        cmd = [self.singularity_cmd, 'build']

        if extra_build_args:
            cmd.extend(extra_build_args)

        cmd.extend([primary_tag, str(def_file)])

        logger.info(f"Building Singularity image with command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=path
            )
            logger.debug(f"Build output: {result.stdout}")
            if result.stderr:
                logger.debug(f"Build stderr: {result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Build failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise AgentRuntimeBuildError(f"Singularity build failed: {e}")

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
                        shutil.copy2(primary_path, tag_path)
                    logger.debug(f"Created additional tag: {tag}")
                except Exception as e:
                    logger.warning(f"Failed to create additional tag {tag}: {e}")

        return str(primary_path)


def get_runtime_image_repo() -> str:
    return os.getenv('OH_RUNTIME_SINGULARITY_IMAGE_REPO', '/root/singularity_images')


def _generate_singularity_def(
    base_image: str,
    build_from: BuildFromImageType = BuildFromImageType.SCRATCH,
    extra_deps: str | None = None,
) -> str:
    """Generate the Singularity definition file content based on the base image.

    Parameters:
    - base_image (str): The base image provided for the runtime image
    - build_from (BuildFromImageType): The build method for the runtime image.
    - extra_deps (str): Extra dependencies to install

    Returns:
    - str: The resulting Singularity definition file content
    """
    env = Environment(
        loader=FileSystemLoader(
            searchpath=os.path.join(os.path.dirname(__file__), 'runtime_templates')
        )
    )
    template = env.get_template('singularity.j2')

    def_content = template.render(
        base_image=base_image,
        build_from_scratch=build_from == BuildFromImageType.SCRATCH,
        build_from_versioned=build_from == BuildFromImageType.VERSIONED,
        extra_deps=extra_deps if extra_deps is not None else '',
        oh_version=oh_version,
    )
    return def_content


def get_runtime_image_path_and_tag(base_image: str) -> tuple[str, str]:
    """Retrieves the Singularity image path and tag associated with the base image.

    Parameters:
    - base_image (str): The name of the base Docker image

    Returns:
    - tuple[str, str]: The image directory path and filename tag
    """
    repo_dir = get_runtime_image_repo()

    if repo_dir in base_image and base_image.endswith('.sif'):
        logger.debug(
            f'The provided image [{base_image}] is already a valid runtime SIF file.\n'
            f'Will try to reuse it as is.'
        )
        path_obj = Path(base_image)
        return str(path_obj.parent), path_obj.stem
    else:
        if base_image.endswith('.sif'):
            repo = Path(base_image).stem
            tag = 'latest'
        else:
            if ':' not in base_image:
                base_image = base_image + ':latest'
            [repo, tag] = base_image.split(':')
        # Hash the repo if it's too long
        if len(repo) > 32:
            repo_hash = hashlib.md5(repo[:-24].encode()).hexdigest()[:8]
            repo = f'{repo_hash}_{repo[-24:]}'  # Use 8 char hash + last 24 chars
        else:
            repo = repo.replace('/', '_s_')

        new_tag = f'oh_v{oh_version}_image_{repo}_tag_{tag}'

        # if it's still too long, hash the entire image name
        if len(new_tag) > 128:
            new_tag = f'oh_v{oh_version}_image_{hashlib.md5(new_tag.encode()).hexdigest()[:64]}'
            logger.warning(
                f'The new tag [{new_tag}] is still too long, so we use a hash of the entire image name: {new_tag}'
            )

        return repo_dir, new_tag


def build_runtime_image(
    base_image: str,
    runtime_builder: SingularityRuntimeBuilder,
    platform: str | None = None,
    extra_deps: str | None = None,
    build_folder: str | None = None,
    dry_run: bool = False,
    force_rebuild: bool = False,
    extra_build_args: list[str] | None = None,
) -> str:
    """Prepares the final build folder and builds the OpenHands runtime Singularity image.

    Parameters:
    - base_image (str): The name of the base Docker image to use
    - runtime_builder (SingularityRuntimeBuilder): The runtime builder to use
    - platform (str): The target platform for the build (ignored for Singularity)
    - extra_deps (str): Extra dependencies to install
    - build_folder (str): The directory to use for the build. If not provided a temporary directory will be used
    - dry_run (bool): if True, it will only ready the build folder. It will not actually build the image
    - force_rebuild (bool): if True, it will create the definition file from scratch
    - extra_build_args (List[str]): Additional build arguments to pass to the builder

    Returns:
    - str: Path to the built SIF file
    """
    if build_folder is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = build_runtime_image_in_folder(
                base_image=base_image,
                runtime_builder=runtime_builder,
                build_folder=Path(temp_dir),
                extra_deps=extra_deps,
                dry_run=dry_run,
                force_rebuild=force_rebuild,
                platform=platform,
                extra_build_args=extra_build_args,
            )
            return result

    result = build_runtime_image_in_folder(
        base_image=base_image,
        runtime_builder=runtime_builder,
        build_folder=Path(build_folder),
        extra_deps=extra_deps,
        dry_run=dry_run,
        force_rebuild=force_rebuild,
        platform=platform,
        extra_build_args=extra_build_args,
    )
    return result


def build_runtime_image_in_folder(
    base_image: str,
    runtime_builder: SingularityRuntimeBuilder,
    build_folder: Path,
    extra_deps: str | None,
    dry_run: bool,
    force_rebuild: bool,
    platform: str | None = None,
    extra_build_args: list[str] | None = None,
) -> str:
    runtime_image_dir, _ = get_runtime_image_path_and_tag(base_image)
    lock_tag = f'oh_v{oh_version}_{get_hash_for_lock_files(base_image)}'
    versioned_tag = f'oh_v{oh_version}_{get_tag_for_versioned_image(base_image)}'
    source_tag = f'{lock_tag}_{get_hash_for_source_files()}'

    # Ensure the image directory exists
    os.makedirs(runtime_image_dir, exist_ok=True)

    versioned_image_path = f'{runtime_image_dir}/{versioned_tag}.sif'
    hash_image_path = f'{runtime_image_dir}/{source_tag}.sif'
    lock_image_path = f'{runtime_image_dir}/{lock_tag}.sif'

    logger.info(f'Building image: {hash_image_path}')

    if force_rebuild:
        logger.debug(f'Force rebuild: [{hash_image_path}] from scratch.')
        prep_build_folder(
            build_folder,
            base_image,
            build_from=BuildFromImageType.SCRATCH,
            extra_deps=extra_deps,
        )
        if not dry_run:
            _build_singularity_image(
                build_folder,
                runtime_builder,
                hash_image_path,
                lock_image_path,
                versioned_image_path,
                platform,
                extra_build_args=extra_build_args,
            )
        return hash_image_path

    build_from = BuildFromImageType.SCRATCH

    # If the exact image already exists, we do not need to build it
    if runtime_builder.image_exists(hash_image_path, False):
        logger.debug(f'Reusing Image [{hash_image_path}]')
        return hash_image_path

    # We look for an existing image that shares the same lock_tag
    if runtime_builder.image_exists(lock_image_path):
        logger.debug(f'Build [{hash_image_path}] from lock image [{lock_image_path}]')
        build_from = BuildFromImageType.LOCK
        base_image = lock_image_path  # lock_image_path is already a complete path
    elif runtime_builder.image_exists(versioned_image_path):
        logger.info(f'Build [{hash_image_path}] from versioned image [{versioned_image_path}]')
        build_from = BuildFromImageType.VERSIONED
        base_image = versioned_image_path  # versioned_image_path is already a complete path
    else:
        logger.debug(f'Build [{hash_image_path}] from scratch')

    prep_build_folder(build_folder, base_image, build_from, extra_deps)
    if not dry_run:
        _build_singularity_image(
            build_folder,
            runtime_builder,
            hash_image_path,
            lock_image_path,
            # Only tag the versioned image if we are building from scratch
            versioned_image_path if build_from == BuildFromImageType.SCRATCH else None,
            platform,
            extra_build_args=extra_build_args,
        )

    return hash_image_path


def prep_build_folder(
    build_folder: Path,
    base_image: str,
    build_from: BuildFromImageType,
    extra_deps: str | None,
) -> None:
    # Copy the source code to directory. It will end up in build_folder/code
    openhands_source_dir = Path(openhands.__file__).parent
    project_root = openhands_source_dir.parent
    logger.debug(f'Building source distribution using project root: {project_root}')

    # Copy the 'openhands' directory (Source code)
    shutil.copytree(
        openhands_source_dir,
        Path(build_folder, 'code', 'openhands'),
        ignore=shutil.ignore_patterns(
            '.*/',
            '__pycache__/',
            '*.pyc',
            '*.md',
        ),
    )

    # Copy pyproject.toml and poetry.lock files
    for file in ['pyproject.toml', 'poetry.lock']:
        src = Path(openhands_source_dir, file)
        if not src.exists():
            src = Path(project_root, file)
        shutil.copy2(src, Path(build_folder, 'code', file))

    # Create a Singularity definition file and write it to build_folder
    def_content = _generate_singularity_def(
        base_image,
        build_from=build_from,
        extra_deps=extra_deps,
    )
    def_path = Path(build_folder, 'singularity.def')
    with open(str(def_path), 'w') as f:
        f.write(def_content)


_ALPHABET = string.digits + string.ascii_lowercase


def truncate_hash(hash: str) -> str:
    """Convert the base16 hash to base36 and truncate at 16 characters."""
    value = int(hash, 16)
    result: list[str] = []
    while value > 0 and len(result) < 16:
        value, remainder = divmod(value, len(_ALPHABET))
        result.append(_ALPHABET[remainder])
    return ''.join(result)


def get_hash_for_lock_files(base_image: str) -> str:
    openhands_source_dir = Path(openhands.__file__).parent
    md5 = hashlib.md5()
    md5.update(base_image.encode())
    for file in ['pyproject.toml', 'poetry.lock']:
        src = Path(openhands_source_dir, file)
        if not src.exists():
            src = Path(openhands_source_dir.parent, file)
        with open(src, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
    result = truncate_hash(md5.hexdigest())
    return result


def get_tag_for_versioned_image(base_image: str) -> str:
    # remove '.sif' if it exists
    if base_image.endswith('.sif'):
        base_image = base_image[:-4]
    return base_image.replace('/', '_s_').replace(':', '_t_').lower()[-96:]


def get_hash_for_source_files() -> str:
    openhands_source_dir = Path(openhands.__file__).parent
    dir_hash = dirhash(
        openhands_source_dir,
        'md5',
        ignore=[
            '.*/',  # hidden directories
            '__pycache__/',
            '*.pyc',
        ],
    )
    result = truncate_hash(dir_hash)
    return result


def _build_singularity_image(
    build_folder: Path,
    runtime_builder: SingularityRuntimeBuilder,
    source_image_path: str,
    lock_image_path: str,
    versioned_image_path: str | None,
    platform: str | None = None,
    extra_build_args: list[str] | None = None,
) -> str:
    """Build and tag the Singularity image."""
    paths = [source_image_path, lock_image_path]
    if versioned_image_path is not None:
        paths.append(versioned_image_path)

    # Filter out paths that already exist
    paths = [path for path in paths if not runtime_builder.image_exists(path, False)]

    if not paths:
        logger.warning("All target images already exist, skipping build")
        return source_image_path

    image_path = runtime_builder.build(
        path=str(build_folder),
        tags=paths,
        platform=platform,
        extra_build_args=extra_build_args,
    )
    if not image_path:
        raise AgentRuntimeBuildError(f'Build failed for image {paths}')

    return image_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_image', type=str, default='ubuntu:22.04'
    )
    parser.add_argument('--build_folder', type=str, default=None)
    parser.add_argument('--force_rebuild', action='store_true', default=False)
    parser.add_argument('--platform', type=str, default=None)
    parser.add_argument('--singularity_cmd', type=str, default='apptainer',
                       help='Singularity command to use (apptainer or singularity)')
    args = parser.parse_args()

    if args.build_folder is not None:
        # If a build_folder is provided, we prepare the build folder but don't build
        build_folder = args.build_folder
        assert os.path.exists(build_folder), f'Build folder {build_folder} does not exist'
        logger.debug(f'Preparing build folder: {build_folder}')

        runtime_image_dir, runtime_image_tag = get_runtime_image_path_and_tag(args.base_image)
        logger.debug(f'Runtime image dir: {runtime_image_dir} and runtime image tag: {runtime_image_tag}')

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_image_path = build_runtime_image(
                args.base_image,
                runtime_builder=SingularityRuntimeBuilder(args.singularity_cmd),
                build_folder=temp_dir,
                dry_run=True,
                force_rebuild=args.force_rebuild,
                platform=args.platform,
            )

            # Move contents of temp_dir to build_folder
            shutil.copytree(temp_dir, build_folder, dirs_exist_ok=True)

        logger.debug(f'Build folder [{build_folder}] is ready: {os.listdir(build_folder)}')

        # Write config file with image information
        with open(os.path.join(build_folder, 'config.sh'), 'w') as file:
            file.write(f'SINGULARITY_IMAGE_PATH={runtime_image_path}\n')
            file.write(f'SINGULARITY_IMAGE_TAG={runtime_image_tag}\n')

        logger.debug(f'Singularity definition file and source code are ready in {build_folder}')
    else:
        # Build the image directly
        logger.debug('Building Singularity image in a temporary folder')
        singularity_builder = SingularityRuntimeBuilder(args.singularity_cmd)
        image_path = build_runtime_image(
            args.base_image, singularity_builder, platform=args.platform
        )
        logger.debug(f'\nBuilt Singularity image: {image_path}\n')
