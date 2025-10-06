import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytest import TempPathFactory

from openhands.core.exceptions import AgentRuntimeBuildError
from openhands.runtime.builder.singularity import SingularityRuntimeBuilder


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_singularity_builder'))


@pytest.fixture
def singularity_builder():
    with patch('openhands.runtime.builder.singularity.subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(stdout='apptainer version 1.2.0')
        return SingularityRuntimeBuilder('apptainer')


def test_singularity_builder_init():
    """Test SingularityRuntimeBuilder initialization."""
    with patch('openhands.runtime.builder.singularity.subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(stdout='apptainer version 1.2.0')
        builder = SingularityRuntimeBuilder('apptainer')
        assert builder.singularity_cmd == 'apptainer'
        mock_run.assert_called_once()


def test_singularity_builder_init_not_available():
    """Test SingularityRuntimeBuilder initialization when singularity is not available."""
    with patch('openhands.runtime.builder.singularity.subprocess.run') as mock_run:
        mock_run.side_effect = FileNotFoundError('Command not found')
        with pytest.raises(AgentRuntimeBuildError):
            SingularityRuntimeBuilder('singularity')


def test_build_success(singularity_builder, temp_dir):
    """Test successful image build."""
    # Create a mock definition file
    def_file = Path(temp_dir) / 'singularity.def'
    def_file.write_text('Bootstrap: docker\nFrom: ubuntu:22.04\n')

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    mock_process.stdout.readline.side_effect = ['Building...', 'Complete!', '']

    with (
        patch(
            'openhands.runtime.builder.singularity.subprocess.Popen',
            return_value=mock_process,
        ),
        patch('pathlib.Path.exists', return_value=True),
    ):
        result = singularity_builder.build(
            path=temp_dir,
            tags=['test_image.sif'],
        )
        assert result == 'test_image.sif'


def test_build_no_tags(singularity_builder, temp_dir):
    """Test build with no tags provided."""
    with pytest.raises(AgentRuntimeBuildError):
        singularity_builder.build(path=temp_dir, tags=[])


def test_build_no_def_file(singularity_builder, temp_dir):
    """Test build when definition file is missing."""
    with pytest.raises(AgentRuntimeBuildError):
        singularity_builder.build(path=temp_dir, tags=['test.sif'])


def test_build_failed_process(singularity_builder, temp_dir):
    """Test build when subprocess fails."""
    # Create a mock definition file
    def_file = Path(temp_dir) / 'singularity.def'
    def_file.write_text('Bootstrap: docker\nFrom: ubuntu:22.04\n')

    mock_process = MagicMock()
    mock_process.wait.return_value = 1  # Non-zero return code
    mock_process.stdout.readline.side_effect = ['Error building...', '']

    with (
        patch(
            'openhands.runtime.builder.singularity.subprocess.Popen',
            return_value=mock_process,
        ),
        pytest.raises(AgentRuntimeBuildError),
    ):
        singularity_builder.build(
            path=temp_dir,
            tags=['test_image.sif'],
        )


def test_build_multiple_tags(singularity_builder, temp_dir):
    """Test build with multiple tags."""
    # Create a mock definition file
    def_file = Path(temp_dir) / 'singularity.def'
    def_file.write_text('Bootstrap: docker\nFrom: ubuntu:22.04\n')

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    mock_process.stdout.readline.side_effect = ['Building...', 'Complete!', '']

    with (
        patch(
            'openhands.runtime.builder.singularity.subprocess.Popen',
            return_value=mock_process,
        ),
        patch('pathlib.Path.exists', return_value=True),
        patch('pathlib.Path.hardlink_to') as mock_hardlink,
        patch('pathlib.Path.unlink'),
    ):
        result = singularity_builder.build(
            path=temp_dir,
            tags=['primary.sif', 'secondary.sif', 'tertiary.sif'],
        )
        assert result == 'primary.sif'
        # Should create hard links for additional tags
        assert mock_hardlink.call_count == 2


def test_image_exists_local_sif(singularity_builder):
    """Test image_exists for local SIF file."""
    with patch('pathlib.Path.exists', return_value=True):
        assert singularity_builder.image_exists('/path/to/image.sif')


def test_image_exists_local_without_extension(singularity_builder):
    """Test image_exists for local file without .sif extension."""
    with patch('pathlib.Path.exists', return_value=True):
        assert singularity_builder.image_exists('/path/to/image')


def test_image_not_exists(singularity_builder):
    """Test image_exists when image doesn't exist."""
    with patch('pathlib.Path.exists', return_value=False):
        assert not singularity_builder.image_exists('/path/to/nonexistent.sif')


def test_image_exists_docker_image_pull_success(singularity_builder):
    """Test image_exists with Docker image that can be pulled."""
    with (
        patch('pathlib.Path.exists', return_value=False),
        patch('openhands.runtime.builder.singularity.subprocess.run') as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = singularity_builder.image_exists('ubuntu:22.04', pull_from_repo=True)
        assert result is True


def test_image_exists_docker_image_pull_failure(singularity_builder):
    """Test image_exists with Docker image that fails to pull."""
    with (
        patch('pathlib.Path.exists', return_value=False),
        patch('openhands.runtime.builder.singularity.subprocess.run') as mock_run,
    ):
        mock_run.side_effect = subprocess.CalledProcessError(1, 'cmd')
        result = singularity_builder.image_exists('ubuntu:22.04', pull_from_repo=True)
        assert result is False


def test_image_exists_invalid_name(singularity_builder):
    """Test image_exists with invalid image name."""
    assert not singularity_builder.image_exists('')
    assert not singularity_builder.image_exists(None)


def test_is_docker_image_reference(singularity_builder):
    """Test _is_docker_image_reference method."""
    assert singularity_builder._is_docker_image_reference('ubuntu:22.04')
    assert singularity_builder._is_docker_image_reference('nginx/nginx')
    assert not singularity_builder._is_docker_image_reference('image.sif')
    assert not singularity_builder._is_docker_image_reference('localimage')


def test_cache_functionality(singularity_builder, temp_dir):
    """Test cache-related functionality."""
    # Test cache directory creation
    cache_dir = os.path.join(temp_dir, 'cache')
    assert singularity_builder._is_cache_usable(cache_dir)
    assert os.path.exists(cache_dir)

    # Test cache pruning with old files
    old_file = Path(cache_dir) / 'old_image.sif'
    old_file.touch()
    # Set old modification time
    old_time = os.path.getmtime(old_file) - (8 * 24 * 60 * 60)  # 8 days ago
    os.utime(old_file, (old_time, old_time))

    singularity_builder._prune_old_cache_files(cache_dir, max_age_days=7)
    # File should be removed
    assert not old_file.exists()


def test_output_logs(singularity_builder):
    """Test log output functionality."""
    # Test with rolling logger disabled
    singularity_builder.rolling_logger.is_enabled = MagicMock(return_value=False)
    with patch('openhands.runtime.builder.singularity.logger.debug') as mock_debug:
        singularity_builder._output_logs('test log line')
        mock_debug.assert_called_once_with('test log line')

    # Test with rolling logger enabled
    singularity_builder.rolling_logger.is_enabled = MagicMock(return_value=True)
    singularity_builder.rolling_logger.add_line = MagicMock()
    singularity_builder._output_logs('test log line')
    singularity_builder.rolling_logger.add_line.assert_called_once_with('test log line')


def test_build_with_extra_args(singularity_builder, temp_dir):
    """Test build with extra build arguments."""
    # Create a mock definition file
    def_file = Path(temp_dir) / 'singularity.def'
    def_file.write_text('Bootstrap: docker\nFrom: ubuntu:22.04\n')

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    mock_process.stdout.readline.side_effect = ['Building...', 'Complete!', '']

    with (
        patch('openhands.runtime.builder.singularity.subprocess.Popen') as mock_popen,
        patch('pathlib.Path.exists', return_value=True),
    ):
        mock_popen.return_value = mock_process

        singularity_builder.build(
            path=temp_dir,
            tags=['test_image.sif'],
            extra_build_args=['--fakeroot', '--sandbox'],
        )

        # Check that extra args were included in the command
        call_args = mock_popen.call_args[0][0]
        assert '--fakeroot' in call_args
        assert '--sandbox' in call_args


def test_build_timeout_exception(singularity_builder, temp_dir):
    """Test build when subprocess times out."""
    # Create a mock definition file
    def_file = Path(temp_dir) / 'singularity.def'
    def_file.write_text('Bootstrap: docker\nFrom: ubuntu:22.04\n')

    with (
        patch('openhands.runtime.builder.singularity.subprocess.Popen') as mock_popen,
        pytest.raises(AgentRuntimeBuildError),
    ):
        mock_popen.side_effect = subprocess.TimeoutExpired('cmd', 30)
        singularity_builder.build(path=temp_dir, tags=['test_image.sif'])


def test_build_permission_error(singularity_builder, temp_dir):
    """Test build when permission is denied."""
    # Create a mock definition file
    def_file = Path(temp_dir) / 'singularity.def'
    def_file.write_text('Bootstrap: docker\nFrom: ubuntu:22.04\n')

    with (
        patch('openhands.runtime.builder.singularity.subprocess.Popen') as mock_popen,
        pytest.raises(AgentRuntimeBuildError),
    ):
        mock_popen.side_effect = PermissionError('Permission denied')
        singularity_builder.build(path=temp_dir, tags=['test_image.sif'])
