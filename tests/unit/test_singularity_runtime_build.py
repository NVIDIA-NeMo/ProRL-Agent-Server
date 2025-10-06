import hashlib
import os
from pathlib import Path
from unittest.mock import ANY, MagicMock, mock_open, patch

import pytest
import toml
from pytest import TempPathFactory

import openhands
from openhands import __version__ as oh_version
from openhands.core.exceptions import AgentRuntimeBuildError
from openhands.runtime.utils.singularity_runtime_build import (
    BuildFromImageType,
    SingularityRuntimeBuilder,
    _generate_singularity_def,
    build_runtime_image,
    get_hash_for_lock_files,
    get_hash_for_source_files,
    get_runtime_image_path_and_tag,
    get_runtime_image_repo,
    get_tag_for_versioned_image,
    prep_build_folder,
    truncate_hash,
)

OH_VERSION = f'oh_v{oh_version}'
DEFAULT_BASE_IMAGE = 'ubuntu:22.04'


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_singularity_runtime_build'))


@pytest.fixture
def singularity_runtime_builder():
    return SingularityRuntimeBuilder('apptainer')


def _check_source_code_in_dir(temp_dir):
    # assert there is a folder called 'code' in the temp_dir
    code_dir = os.path.join(temp_dir, 'code')
    assert os.path.exists(code_dir)
    assert os.path.isdir(code_dir)

    # check the source file is the same as the current code base
    assert os.path.exists(os.path.join(code_dir, 'pyproject.toml'))

    # The source code should only include the `openhands` folder,
    # and pyproject.toml & poetry.lock that are needed to build the runtime image
    assert set(os.listdir(code_dir)) == {
        'openhands',
        'pyproject.toml',
        'poetry.lock',
    }
    assert os.path.exists(os.path.join(code_dir, 'openhands'))
    assert os.path.isdir(os.path.join(code_dir, 'openhands'))

    # make sure the version from the pyproject.toml is the same as the current version
    with open(os.path.join(code_dir, 'pyproject.toml'), 'r') as f:
        pyproject = toml.load(f)

    _pyproject_version = pyproject['tool']['poetry']['version']
    assert _pyproject_version == oh_version


def test_prep_build_folder(temp_dir):
    shutil_mock = MagicMock()
    with patch(f'{prep_build_folder.__module__}.shutil', shutil_mock):
        prep_build_folder(
            Path(temp_dir),
            base_image=DEFAULT_BASE_IMAGE,
            build_from=BuildFromImageType.SCRATCH,
            extra_deps=None,
        )

    # make sure that the code was copied
    shutil_mock.copytree.assert_called_once()
    assert shutil_mock.copy2.call_count == 2

    # Now check singularity definition file is in the folder
    def_file_path = os.path.join(temp_dir, 'singularity.def')
    assert os.path.exists(def_file_path)
    assert os.path.isfile(def_file_path)


def test_get_hash_for_lock_files():
    with patch('builtins.open', mock_open(read_data='mock-data'.encode())):
        hash = get_hash_for_lock_files('some_base_image')
        # Since we mocked open to always return "mock_data", the hash is the result
        # of hashing the name of the base image followed by "mock-data" twice
        md5 = hashlib.md5()
        md5.update('some_base_image'.encode())
        for _ in range(2):
            md5.update('mock-data'.encode())
        assert hash == truncate_hash(md5.hexdigest())


def test_get_hash_for_source_files():
    dirhash_mock = MagicMock()
    dirhash_mock.return_value = '1f69bd20d68d9e3874d5bf7f7459709b'
    with patch(f'{get_hash_for_source_files.__module__}.dirhash', dirhash_mock):
        result = get_hash_for_source_files()
        assert result == truncate_hash(dirhash_mock.return_value)
        dirhash_mock.assert_called_once_with(
            Path(openhands.__file__).parent,
            'md5',
            ignore=[
                '.*/',  # hidden directories
                '__pycache__/',
                '*.pyc',
            ],
        )


def test_generate_singularity_def_build_from_scratch():
    base_image = 'ubuntu:22.04'
    def_content = _generate_singularity_def(
        base_image,
        build_from=BuildFromImageType.SCRATCH,
    )
    assert base_image in def_content
    assert 'apt-get update' in def_content
    assert 'wget curl' in def_content
    assert 'micromamba' in def_content
    assert 'python=3.12' in def_content

    # Check the file copy commands in Singularity format
    assert './code/openhands /openhands/code/openhands' in def_content
    assert 'poetry install' in def_content


def test_generate_singularity_def_build_from_lock():
    base_image = 'ubuntu:22.04'
    def_content = _generate_singularity_def(
        base_image,
        build_from=BuildFromImageType.LOCK,
    )

    # These commands SHOULD NOT include in the definition file if build_from_scratch is False
    assert 'wget curl sudo apt-utils git' not in def_content
    assert 'conda-forge poetry python=3.12' not in def_content
    assert 'poetry install' not in def_content

    # These file copy commands SHOULD still be in the definition file
    assert './code/openhands /openhands/code/openhands' in def_content
    # MAMBA_ROOT_PREFIX should still be set in environment
    assert 'MAMBA_ROOT_PREFIX=/openhands/micromamba' in def_content


def test_generate_singularity_def_build_from_versioned():
    base_image = 'ubuntu:22.04'
    def_content = _generate_singularity_def(
        base_image,
        build_from=BuildFromImageType.VERSIONED,
    )

    # these commands should not exist when build from versioned
    assert 'wget curl sudo apt-utils git' not in def_content
    assert 'conda-forge poetry python=3.12' not in def_content

    # this SHOULD exist when build from versioned
    assert 'poetry install' in def_content
    assert './code/openhands /openhands/code/openhands' in def_content


def test_get_runtime_image_path_and_tag():
    base_image = 'ubuntu:22.04'
    img_dir, img_tag = get_runtime_image_path_and_tag(base_image)
    assert img_dir == get_runtime_image_repo()
    assert img_tag == f'{OH_VERSION}_image_ubuntu_tag_22.04'

    base_image = 'nikolaik/python-nodejs:python3.12-nodejs22'
    img_dir, img_tag = get_runtime_image_path_and_tag(base_image)
    assert img_dir == get_runtime_image_repo()
    assert (
        img_tag
        == f'{OH_VERSION}_image_nikolaik_s_python-nodejs_tag_python3.12-nodejs22'
    )

    base_image = 'ubuntu'
    img_dir, img_tag = get_runtime_image_path_and_tag(base_image)
    assert img_dir == get_runtime_image_repo()
    assert img_tag == f'{OH_VERSION}_image_ubuntu_tag_latest'


def test_get_tag_for_versioned_image():
    base_image = 'ubuntu:22.04'
    tag = get_tag_for_versioned_image(base_image)
    assert tag == 'ubuntu_t_22.04'

    base_image = 'nikolaik/python-nodejs:python3.12-nodejs22'
    tag = get_tag_for_versioned_image(base_image)
    assert tag == 'nikolaik_s_python-nodejs_t_python3.12-nodejs22'


def test_truncate_hash():
    truncated = truncate_hash('b08f254d76b1c6a7ad924708c0032251')
    assert truncated == 'pma2wc71uq3c9a85'
    truncated = truncate_hash('102aecc0cea025253c0278f54ebef078')
    assert truncated == '4titk6gquia3taj5'


# Tests for SingularityRuntimeBuilder class
def test_singularity_runtime_builder_init():
    builder = SingularityRuntimeBuilder('apptainer')
    assert builder.singularity_cmd == 'apptainer'

    builder = SingularityRuntimeBuilder('singularity')
    assert builder.singularity_cmd == 'singularity'


def test_image_exists_with_sif_extension(singularity_runtime_builder):
    with patch('pathlib.Path.exists') as mock_exists:
        mock_exists.return_value = True
        assert singularity_runtime_builder.image_exists('/path/to/image.sif')
        mock_exists.assert_called_once()


def test_image_exists_without_sif_extension(singularity_runtime_builder):
    with patch('pathlib.Path.exists') as mock_exists:
        mock_exists.return_value = True
        assert singularity_runtime_builder.image_exists('/path/to/image')
        mock_exists.assert_called_once()


def test_image_not_exists(singularity_runtime_builder):
    with patch('pathlib.Path.exists') as mock_exists:
        mock_exists.return_value = False
        assert not singularity_runtime_builder.image_exists('/path/to/nonexistent.sif')


def test_build_success(singularity_runtime_builder, temp_dir):
    # Create a mock definition file
    def_file = Path(temp_dir) / 'singularity.def'
    def_file.write_text('Bootstrap: docker\nFrom: ubuntu:22.04\n')

    mock_result = MagicMock()
    mock_result.stdout = 'Build successful'
    mock_result.stderr = ''

    with patch('subprocess.run', return_value=mock_result) as mock_run:
        result = singularity_runtime_builder.build(
            path=temp_dir,
            tags=['test_image.sif'],
        )
        assert result == 'test_image.sif'
        mock_run.assert_called_once()


def test_build_no_tags(singularity_runtime_builder, temp_dir):
    with pytest.raises(AgentRuntimeBuildError):
        singularity_runtime_builder.build(path=temp_dir, tags=[])


def test_build_no_def_file(singularity_runtime_builder, temp_dir):
    with pytest.raises(AgentRuntimeBuildError):
        singularity_runtime_builder.build(path=temp_dir, tags=['test.sif'])


def test_build_multiple_tags(singularity_runtime_builder, temp_dir):
    # Create a mock definition file
    def_file = Path(temp_dir) / 'singularity.def'
    def_file.write_text('Bootstrap: docker\nFrom: ubuntu:22.04\n')

    mock_result = MagicMock()
    mock_result.stdout = 'Build successful'
    mock_result.stderr = ''

    with (
        patch('subprocess.run', return_value=mock_result),
        patch('pathlib.Path.hardlink_to') as mock_hardlink,
        patch('pathlib.Path.unlink'),
    ):
        result = singularity_runtime_builder.build(
            path=temp_dir,
            tags=['primary.sif', 'secondary.sif', 'tertiary.sif'],
        )
        assert result == 'primary.sif'
        # Should create hard links for additional tags
        assert mock_hardlink.call_count == 2


def test_build_runtime_image_from_scratch():
    base_image = 'ubuntu:22.04'
    mock_lock_hash = MagicMock()
    mock_lock_hash.return_value = 'mock-lock-tag'
    mock_versioned_tag = MagicMock()
    mock_versioned_tag.return_value = 'mock-versioned-tag'
    mock_source_hash = MagicMock()
    mock_source_hash.return_value = 'mock-source-tag'
    mock_runtime_builder = MagicMock()
    mock_runtime_builder.image_exists.return_value = False
    mock_runtime_builder.build.return_value = (
        f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag_mock-source-tag.sif'
    )
    mock_prep_build_folder = MagicMock()
    mod = build_runtime_image.__module__

    with (
        patch(f'{mod}.get_hash_for_lock_files', mock_lock_hash),
        patch(f'{mod}.get_hash_for_source_files', mock_source_hash),
        patch(f'{mod}.get_tag_for_versioned_image', mock_versioned_tag),
        patch(f'{mod}.prep_build_folder', mock_prep_build_folder),
        patch(f'{mod}.os.makedirs'),
    ):
        image_path = build_runtime_image(base_image, mock_runtime_builder)
        expected_hash_path = (
            f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag_mock-source-tag.sif'
        )
        expected_lock_path = (
            f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag.sif'
        )
        expected_versioned_path = (
            f'{get_runtime_image_repo()}/{OH_VERSION}_mock-versioned-tag.sif'
        )

        mock_runtime_builder.build.assert_called_once_with(
            path=ANY,
            tags=[expected_hash_path, expected_lock_path, expected_versioned_path],
            platform=None,
            extra_build_args=None,
        )
        assert image_path == expected_hash_path
        mock_prep_build_folder.assert_called_once_with(
            ANY, base_image, BuildFromImageType.SCRATCH, None
        )


def test_build_runtime_image_exact_hash_exist():
    base_image = 'ubuntu:22.04'
    mock_lock_hash = MagicMock()
    mock_lock_hash.return_value = 'mock-lock-tag'
    mock_source_hash = MagicMock()
    mock_source_hash.return_value = 'mock-source-tag'
    mock_versioned_tag = MagicMock()
    mock_versioned_tag.return_value = 'mock-versioned-tag'
    mock_runtime_builder = MagicMock()
    mock_runtime_builder.image_exists.return_value = True
    mock_prep_build_folder = MagicMock()
    mod = build_runtime_image.__module__

    with (
        patch(f'{mod}.get_hash_for_lock_files', mock_lock_hash),
        patch(f'{mod}.get_hash_for_source_files', mock_source_hash),
        patch(f'{mod}.get_tag_for_versioned_image', mock_versioned_tag),
        patch(f'{mod}.prep_build_folder', mock_prep_build_folder),
        patch(f'{mod}.os.makedirs'),
    ):
        image_path = build_runtime_image(base_image, mock_runtime_builder)
        expected_hash_path = (
            f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag_mock-source-tag.sif'
        )
        assert image_path == expected_hash_path
        mock_runtime_builder.build.assert_not_called()
        mock_prep_build_folder.assert_not_called()


def test_build_runtime_image_exact_hash_not_exist_and_lock_exist():
    base_image = 'ubuntu:22.04'
    mock_lock_hash = MagicMock()
    mock_lock_hash.return_value = 'mock-lock-tag'
    mock_source_hash = MagicMock()
    mock_source_hash.return_value = 'mock-source-tag'
    mock_versioned_tag = MagicMock()
    mock_versioned_tag.return_value = 'mock-versioned-tag'
    mock_runtime_builder = MagicMock()

    def image_exists_side_effect(image_path, *args):
        if 'mock-lock-tag_mock-source-tag' in image_path:
            return False
        elif 'mock-lock-tag.sif' in image_path:
            return True
        elif 'mock-versioned-tag' in image_path:
            return False
        else:
            return False

    mock_runtime_builder.image_exists.side_effect = image_exists_side_effect
    mock_runtime_builder.build.return_value = (
        f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag_mock-source-tag.sif'
    )

    mock_prep_build_folder = MagicMock()
    mod = build_runtime_image.__module__

    with (
        patch(f'{mod}.get_hash_for_lock_files', mock_lock_hash),
        patch(f'{mod}.get_hash_for_source_files', mock_source_hash),
        patch(f'{mod}.get_tag_for_versioned_image', mock_versioned_tag),
        patch(f'{mod}.prep_build_folder', mock_prep_build_folder),
        patch(f'{mod}.os.makedirs'),
    ):
        image_path = build_runtime_image(base_image, mock_runtime_builder)
        expected_hash_path = (
            f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag_mock-source-tag.sif'
        )
        expected_lock_path = (
            f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag.sif'
        )

        assert image_path == expected_hash_path
        mock_runtime_builder.build.assert_called_once_with(
            path=ANY,
            tags=[expected_hash_path],  # Only hash tag since lock already exists
            platform=None,
            extra_build_args=None,
        )
        mock_prep_build_folder.assert_called_once_with(
            ANY,
            f'library://localcontainer/{expected_lock_path}',
            BuildFromImageType.LOCK,
            None,
        )


def test_build_runtime_image_exact_hash_not_exist_and_lock_not_exist_and_versioned_exist():
    base_image = 'ubuntu:22.04'
    mock_lock_hash = MagicMock()
    mock_lock_hash.return_value = 'mock-lock-tag'
    mock_source_hash = MagicMock()
    mock_source_hash.return_value = 'mock-source-tag'
    mock_versioned_tag = MagicMock()
    mock_versioned_tag.return_value = 'mock-versioned-tag'
    mock_runtime_builder = MagicMock()

    def image_exists_side_effect(image_path, *args):
        if 'mock-lock-tag_mock-source-tag' in image_path:
            return False
        elif 'mock-lock-tag.sif' in image_path:
            return False
        elif 'mock-versioned-tag.sif' in image_path:
            return True
        else:
            return False

    mock_runtime_builder.image_exists.side_effect = image_exists_side_effect
    mock_runtime_builder.build.return_value = (
        f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag_mock-source-tag.sif'
    )

    mock_prep_build_folder = MagicMock()
    mod = build_runtime_image.__module__

    with (
        patch(f'{mod}.get_hash_for_lock_files', mock_lock_hash),
        patch(f'{mod}.get_hash_for_source_files', mock_source_hash),
        patch(f'{mod}.get_tag_for_versioned_image', mock_versioned_tag),
        patch(f'{mod}.prep_build_folder', mock_prep_build_folder),
        patch(f'{mod}.os.makedirs'),
    ):
        image_path = build_runtime_image(base_image, mock_runtime_builder)
        expected_hash_path = (
            f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag_mock-source-tag.sif'
        )
        expected_lock_path = (
            f'{get_runtime_image_repo()}/{OH_VERSION}_mock-lock-tag.sif'
        )
        expected_versioned_path = (
            f'{get_runtime_image_repo()}/{OH_VERSION}_mock-versioned-tag.sif'
        )

        assert image_path == expected_hash_path
        mock_runtime_builder.build.assert_called_once_with(
            path=ANY,
            tags=[expected_hash_path, expected_lock_path],
            platform=None,
            extra_build_args=None,
        )
        mock_prep_build_folder.assert_called_once_with(
            ANY,
            f'library://localcontainer/{expected_versioned_path}',
            BuildFromImageType.VERSIONED,
            None,
        )


def test_get_runtime_image_repo():
    # Test default value
    with patch.dict(os.environ, {}, clear=True):
        repo = get_runtime_image_repo()
        assert repo == '/tmp/openhands/singularity'

    # Test custom value
    custom_repo = '/custom/path/to/singularity'
    with patch.dict(os.environ, {'OH_RUNTIME_SINGULARITY_IMAGE_REPO': custom_repo}):
        repo = get_runtime_image_repo()
        assert repo == custom_repo


def test_build_runtime_image_with_extra_deps():
    base_image = 'ubuntu:22.04'
    extra_deps = 'numpy pandas'
    mock_runtime_builder = MagicMock()
    mock_runtime_builder.image_exists.return_value = True

    with (
        patch(f'{build_runtime_image.__module__}.get_hash_for_lock_files'),
        patch(f'{build_runtime_image.__module__}.get_hash_for_source_files'),
        patch(f'{build_runtime_image.__module__}.get_tag_for_versioned_image'),
        patch(f'{build_runtime_image.__module__}.os.makedirs'),
    ):
        image_path = build_runtime_image(
            base_image, mock_runtime_builder, extra_deps=extra_deps
        )
        # Should return existing image path since image_exists returns True
        assert '.sif' in image_path


def test_build_runtime_image_force_rebuild():
    base_image = 'ubuntu:22.04'
    mock_runtime_builder = MagicMock()
    # Mock the hash functions to return string values instead of MagicMock objects
    mock_lock_hash = 'mock-lock-tag'
    mock_source_hash = 'mock-source-tag'
    mock_versioned_tag = 'mock-versioned-tag'

    mock_runtime_builder.image_exists.return_value = (
        False  # Force build by saying images don't exist
    )
    mock_runtime_builder.build.return_value = 'rebuilt_image.sif'
    mock_prep_build_folder = MagicMock()

    with (
        patch(
            f'{build_runtime_image.__module__}.get_hash_for_lock_files',
            return_value=mock_lock_hash,
        ),
        patch(
            f'{build_runtime_image.__module__}.get_hash_for_source_files',
            return_value=mock_source_hash,
        ),
        patch(
            f'{build_runtime_image.__module__}.get_tag_for_versioned_image',
            return_value=mock_versioned_tag,
        ),
        patch(
            f'{build_runtime_image.__module__}.prep_build_folder',
            mock_prep_build_folder,
        ),
        patch(f'{build_runtime_image.__module__}.os.makedirs'),
    ):
        build_runtime_image(base_image, mock_runtime_builder, force_rebuild=True)
        # Should build from scratch even if image exists
        mock_prep_build_folder.assert_called_once()
        mock_runtime_builder.build.assert_called_once()
