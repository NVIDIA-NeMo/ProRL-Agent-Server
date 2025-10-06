# Suppress warnings at the very beginning before any imports
import os  # noqa: E402
import warnings  # noqa: E402

warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Monkey-patch warnings to make them completely silent

# Import everything else after warning suppression
from unittest.mock import AsyncMock, Mock  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# More specific suppression for categories that might still show up
warnings.filterwarnings('ignore', category=UserWarning)  # noqa: E402
warnings.filterwarnings('ignore', category=DeprecationWarning)  # noqa: E402
warnings.filterwarnings('ignore', category=FutureWarning)  # noqa: E402
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)  # noqa: E402

# Suppress specific warning messages with broader patterns
warning_patterns = [
    '.*class-based.*config.*deprecated.*',
    '.*model_fields.*deprecated.*',
    '.*deprecated.*',
    '.*utcfromtimestamp.*deprecated.*',
    '.*audioop.*deprecated.*',
    '.*aifc.*deprecated.*',
    '.*PyPDF2.*deprecated.*',
    '.*pkg_resources.*deprecated.*',
    '.*__version_info__.*deprecated.*',
    '.*Jupyter.*migrating.*paths.*',
    '.*ffmpeg.*',
    '.*google.*PyType_Spec.*',
    '.*Support for class-based.*',
    '.*Accessing the.*model_fields.*',
    '.*coroutine.*never awaited.*',
]

for pattern in warning_patterns:
    warnings.filterwarnings('ignore', message=pattern)
    warnings.filterwarnings('ignore', message=pattern, category=DeprecationWarning)
    warnings.filterwarnings('ignore', message=pattern, category=UserWarning)
    warnings.filterwarnings('ignore', message=pattern, category=RuntimeWarning)

# Configuration for real data tests
PARQUET_FILE = '/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet'
HAS_REAL_DATA = os.path.exists(PARQUET_FILE)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Auto-use fixture to suppress warnings for all tests."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', ResourceWarning)
        yield


@pytest.fixture(scope='session')
def real_dataset():
    """Load real dataset if available, otherwise skip"""
    if not HAS_REAL_DATA:
        pytest.skip('Real dataset not available')
    return pd.read_parquet(PARQUET_FILE)


@pytest.fixture(scope='session')
def real_instance(real_dataset):
    """Get a processed real instance from the dataset"""
    instance = real_dataset.iloc[0]['instance']
    instance = pd.Series(instance)
    # Convert numpy arrays to lists like in __main__
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return instance


@pytest.fixture
def mock_patch():
    """The mock patch used in __main__ section"""
    return """
    diff --git a/openhands_patch_test.txt b/openhands_patch_test.txt
    new file mode 100644
    index 0000000..e69de29
    --- /dev/null
    +++ b/openhands_patch_test.txt
    @@
    +This is an OpenHands test patch.
    """


@pytest.fixture
def minimal_llm_config():
    """Minimal LLM config for testing"""
    from openhands.core.config.llm_config import LLMConfig

    return LLMConfig(
        model='gpt-4o-mini',
        base_url='https://api.openai.com/v1',
        api_key=os.environ.get('OPENAI_API_KEY', 'dummy-key'),
        modify_params=False,
        log_completions=False,
    )


@pytest.fixture
def mock_runtime():
    """Create a mock runtime for testing"""
    runtime = Mock()
    runtime.connect = AsyncMock()
    runtime.event_stream = Mock()
    runtime.event_stream.close = Mock()
    runtime.close = Mock()
    runtime.sid = 'test_sid'
    return runtime


@pytest.fixture
def sample_evaluation_result():
    """Sample evaluation result structure"""
    return {
        'report': {
            'resolved': True,
            'passed': 3,
            'failed': 0,
            'empty_generation': False,
            'failed_apply_patch': False,
            'error_eval': False,
            'test_timeout': False,
        },
        'apply_patch_output': 'APPLY_PATCH_PASS\nPatch applied successfully',
        'test_output': 'All tests passed successfully',
    }


# Pytest markers for different test categories - handled in pytest_configure below


# Skip conditions
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers"""
    if not HAS_REAL_DATA:
        skip_real_data = pytest.mark.skip(reason='Real data files not available')
        for item in items:
            if 'real_data' in item.keywords:
                item.add_marker(skip_real_data)


def pytest_configure(config):
    """Configure pytest with custom warning filters."""
    # Suppress all deprecation warnings from external libraries
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='google.*')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pydub.*')
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning, message='.*google._upb.*'
    )
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning, message='.*utcfromtimestamp.*'
    )
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning, message='.*audioop.*'
    )

    # More general filters
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning, module='<frozen importlib._bootstrap>'
    )
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=ResourceWarning)

    # Configure custom pytest markers
    config.addinivalue_line('markers', 'integration: marks tests as integration tests')
    config.addinivalue_line('markers', 'slow: marks tests as slow running')
    config.addinivalue_line(
        'markers', 'real_data: marks tests that require real data files'
    )

    # Completely disable warnings at the warnings module level
    warnings.simplefilter('ignore')

    # Override warnings.warn to be a no-op
    warnings.warn = lambda *args, **kwargs: None

    # Additional programmatic warning filters that are harder to catch with pytest.ini
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='pkg_resources')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*ffmpeg.*')

    # Pydantic specific warnings
    try:
        import pydantic.warnings

        warnings.filterwarnings(
            'ignore', category=pydantic.warnings.PydanticDeprecatedSince20
        )
        warnings.filterwarnings(
            'ignore', category=pydantic.warnings.PydanticDeprecatedSince211
        )
    except (ImportError, AttributeError):
        pass
