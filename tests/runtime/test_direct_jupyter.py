"""Test the DirectJupyterPlugin functionality including matplotlib plots and magic commands."""

import base64

import pytest

from openhands.events.action import IPythonRunCellAction
from openhands.events.observation import IPythonRunCellObservation
from openhands.runtime.plugins.jupyter.direct_jupyter import (
    DirectIPythonExecutor,
    DirectJupyterPlugin,
)


async def create_test_plugin():
    """Helper function to create and initialize a test plugin."""
    plugin = DirectJupyterPlugin()
    plugin.name = 'direct_jupyter'
    await plugin.initialize(username='test_user', kernel_id='test_kernel')
    return plugin


@pytest.mark.asyncio
async def test_direct_jupyter_initialization():
    """Test that DirectJupyterPlugin initializes correctly."""
    plugin = DirectJupyterPlugin()
    plugin.name = 'direct_jupyter'

    await plugin.initialize(username='test_user', kernel_id='test_kernel')

    assert hasattr(plugin, 'executor')
    assert plugin.executor.initialized
    assert plugin.kernel_id == 'test_kernel'

    # Test that Python interpreter path is set
    assert hasattr(plugin, 'python_interpreter_path')
    assert plugin.python_interpreter_path.endswith('python')

    await plugin.cleanup()


@pytest.mark.asyncio
async def test_simple_python_execution():
    """Test basic Python code execution."""
    plugin = await create_test_plugin()

    try:
        action = IPythonRunCellAction(code='print("Hello, World!")')

        obs = await plugin.run(action)

        assert isinstance(obs, IPythonRunCellObservation)
        assert 'Hello, World!' in obs.content
        assert obs.code == 'print("Hello, World!")'
    finally:
        await plugin.cleanup()


@pytest.mark.asyncio
async def test_variable_persistence():
    """Test that variables persist between code executions."""
    plugin = await create_test_plugin()

    try:
        # Set a variable
        action1 = IPythonRunCellAction(code='x = 42')
        obs1 = await plugin.run(action1)
        assert isinstance(obs1, IPythonRunCellObservation)

        # Use the variable in next execution
        action2 = IPythonRunCellAction(code='print(f"x is {x}")')
        obs2 = await plugin.run(action2)
        assert isinstance(obs2, IPythonRunCellObservation)
        assert 'x is 42' in obs2.content
    finally:
        await plugin.cleanup()


@pytest.mark.asyncio
async def test_matplotlib_plot_generation():
    """Test that matplotlib plots are captured as base64 images."""
    plugin = await create_test_plugin()

    try:
        plot_code = """
import matplotlib.pyplot as plt
import numpy as np

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Sine Wave')
plt.legend()
plt.grid(True)
plt.show()
"""

        action = IPythonRunCellAction(code=plot_code)
        obs = await plugin.run(action)

        assert isinstance(obs, IPythonRunCellObservation)
        assert obs.image_urls is not None
        assert len(obs.image_urls) > 0

        # Check that the image is a valid base64 PNG
        image_url = obs.image_urls[0]
        assert image_url.startswith('data:image/png;base64,')

        # Extract and validate base64 data
        base64_data = image_url.split(',')[1]
        try:
            decoded_data = base64.b64decode(base64_data)
            # Check PNG header
            assert decoded_data.startswith(b'\x89PNG')
        except Exception as e:
            pytest.fail(f'Invalid base64 PNG data: {e}')
    finally:
        await plugin.cleanup()


@pytest.mark.asyncio
async def test_multiple_plots():
    """Test that multiple plots in one execution are all captured."""
    plugin = await create_test_plugin()

    try:
        plot_code = """
import matplotlib.pyplot as plt
import numpy as np

# Create first plot
plt.figure(figsize=(6, 4))
x = np.linspace(0, 5, 50)
plt.plot(x, x**2, 'r-', label='x²')
plt.title('Plot 1: Quadratic')
plt.legend()

# Create second plot
plt.figure(figsize=(6, 4))
plt.plot(x, np.exp(x), 'g-', label='e^x')
plt.title('Plot 2: Exponential')
plt.legend()

plt.show()
"""

        action = IPythonRunCellAction(code=plot_code)
        obs = await plugin.run(action)

        assert isinstance(obs, IPythonRunCellObservation)
        assert obs.image_urls is not None
        assert len(obs.image_urls) == 2  # Should capture both plots

        for image_url in obs.image_urls:
            assert image_url.startswith('data:image/png;base64,')
    finally:
        await plugin.cleanup()


@pytest.mark.asyncio
async def test_magic_command_pip_install():
    """Test that IPython magic commands work, specifically %pip install."""
    plugin = await create_test_plugin()

    try:
        # Test %pip install magic command
        action = IPythonRunCellAction(code='%pip install requests==2.31.0')
        obs = await plugin.run(action)

        assert isinstance(obs, IPythonRunCellObservation)
        # The output should contain installation information OR externally-managed environment error
        # (which is expected in newer Python distributions)
        content_lower = obs.content.lower()
        assert (
            'requests' in content_lower
            or 'requirement already satisfied' in content_lower
            or 'externally-managed-environment' in content_lower
            or 'externally managed' in content_lower
        )
    finally:
        await plugin.cleanup()


@pytest.mark.asyncio
async def test_magic_command_who():
    """Test %who magic command."""
    plugin = await create_test_plugin()

    try:
        # First set some variables
        setup_action = IPythonRunCellAction(code='a = 1\nb = "hello"\nc = [1, 2, 3]')
        await plugin.run(setup_action)

        # Test %who magic command
        action = IPythonRunCellAction(code='%who')
        obs = await plugin.run(action)

        assert isinstance(obs, IPythonRunCellObservation)
        # Should list the variables we created
        content_lower = obs.content.lower()
        assert 'a' in content_lower
        assert 'b' in content_lower
        assert 'c' in content_lower
    finally:
        await plugin.cleanup()


@pytest.mark.asyncio
async def test_error_handling():
    """Test that errors are properly captured and returned."""
    plugin = await create_test_plugin()

    try:
        action = IPythonRunCellAction(code='print(undefined_variable)')
        obs = await plugin.run(action)

        assert isinstance(obs, IPythonRunCellObservation)
        assert 'NameError' in obs.content or 'undefined_variable' in obs.content
    finally:
        await plugin.cleanup()


@pytest.mark.asyncio
async def test_empty_code_execution():
    """Test execution of empty or whitespace-only code."""
    plugin = await create_test_plugin()

    try:
        action = IPythonRunCellAction(code='   \n  \t  ')
        obs = await plugin.run(action)

        assert isinstance(obs, IPythonRunCellObservation)
        assert obs.content == ''
        assert obs.image_urls is None
    finally:
        await plugin.cleanup()


@pytest.mark.asyncio
async def test_expression_evaluation():
    """Test that expressions are properly evaluated and their results shown."""
    plugin = await create_test_plugin()

    try:
        action = IPythonRunCellAction(code='2 + 3 * 4')
        obs = await plugin.run(action)

        assert isinstance(obs, IPythonRunCellObservation)
        assert '14' in obs.content
    finally:
        await plugin.cleanup()


@pytest.mark.asyncio
async def test_combined_code_and_plot():
    """Test code that produces both text output and plots."""
    plugin = await create_test_plugin()

    try:
        combined_code = """
import matplotlib.pyplot as plt
import numpy as np

print("Generating a plot...")

x = np.linspace(0, 2*np.pi, 100)
y = np.cos(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b-', label='cos(x)')
plt.title('Cosine Wave')
plt.xlabel('x (radians)')
plt.ylabel('cos(x)')
plt.grid(True)
plt.legend()

print(f"Plot completed with {len(x)} data points")
"""

        action = IPythonRunCellAction(code=combined_code)
        obs = await plugin.run(action)

        assert isinstance(obs, IPythonRunCellObservation)
        # Should have both text output and image
        assert 'Generating a plot...' in obs.content
        assert 'Plot completed with 100 data points' in obs.content
        assert obs.image_urls is not None
        assert len(obs.image_urls) == 1
    finally:
        await plugin.cleanup()


# Test the DirectIPythonExecutor directly
@pytest.mark.asyncio
async def test_direct_ipython_executor():
    """Test the DirectIPythonExecutor class directly."""
    executor = DirectIPythonExecutor('test_kernel')

    await executor.initialize()
    assert executor.initialized

    # Test basic execution
    result = await executor.execute('print("Direct executor test")')
    assert result['text'].strip() == 'Direct executor test'
    assert result['images'] == []

    # Test plot generation
    plot_result = await executor.execute("""
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Test Plot')
""")

    assert len(plot_result['images']) == 1
    assert plot_result['images'][0].startswith('data:image/png;base64,')

    await executor.shutdown_async()


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])
