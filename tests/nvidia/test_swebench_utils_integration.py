# Suppress warnings at the very beginning before any imports
import os  # noqa: E402
import warnings  # noqa: E402

warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import asyncio  # noqa: E402
from unittest.mock import AsyncMock, Mock, patch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from openhands.core.config.llm_config import LLMConfig  # noqa: E402
from openhands.nvidia.registry import JobDetails  # noqa: E402
from openhands.nvidia.swe_agent.utils import (  # noqa: E402
    evaluate_agent,
    initialize_agents,
    run_agent,
)

# Skip these tests if the required data files don't exist
PARQUET_FILE = '/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet'
SKIP_INTEGRATION = not os.path.exists(PARQUET_FILE)


@pytest.mark.integration
@pytest.mark.skipif(SKIP_INTEGRATION, reason='Integration data file not available')
class TestSWEBenchUtilsIntegration:
    """Integration tests using real data from the __main__ section"""

    @classmethod
    def setup_class(cls):
        """Load real dataset for integration tests"""
        cls.dataset = pd.read_parquet(PARQUET_FILE)
        cls.instance = cls.dataset.iloc[0]['instance']
        cls.instance = pd.Series(cls.instance)
        cls.instance = cls.instance.apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

    def test_real_data_structure(self):
        """Test that real data has expected structure"""
        assert 'instance_id' in self.instance
        assert 'patch' in self.instance
        assert isinstance(self.instance['instance_id'], str)
        assert isinstance(self.instance['patch'], str)

    @pytest.mark.asyncio
    async def test_evaluate_agent_with_real_patch(self):
        """Test evaluation with real patch from dataset"""
        # Use the actual patch from the dataset
        gold_patch = self.instance['patch']

        # Create a test instance with modified ID to avoid conflicts
        test_instance = self.instance.copy()
        test_instance['instance_id'] = f'{self.instance["instance_id"]}_test'

        # Mock expensive operations but use real data structure
        with patch('openhands.nvidia.swe_agent.utils.initialize_agents') as mock_init:
            with patch(
                'openhands.nvidia.swe_agent.utils._apply_patch_and_evaluate'
            ) as mock_eval:
                mock_runtime = Mock()
                mock_runtime.event_stream = Mock()
                mock_runtime.event_stream.close = Mock()
                mock_runtime.close = Mock()
                mock_init.return_value = (mock_runtime, Mock(), Mock())

                mock_eval.return_value = {
                    'report': {'resolved': True, 'passed': 5, 'failed': 0},
                    'apply_patch_output': 'APPLY_PATCH_PASS',
                    'test_output': 'All tests passed',
                }

                result = await evaluate_agent(gold_patch, test_instance)

                assert 'report' in result
                assert result['report']['resolved'] is True
                mock_init.assert_called_once()
                mock_eval.assert_called_once_with(
                    mock_runtime, gold_patch, test_instance
                )

    @pytest.mark.asyncio
    async def test_evaluate_agent_with_mock_patch(self):
        """Test evaluation with mock patch similar to __main__ example"""
        mock_patch = """
        diff --git a/openhands_patch_test.txt b/openhands_patch_test.txt
        new file mode 100644
        index 0000000..e69de29
        --- /dev/null
        +++ b/openhands_patch_test.txt
        @@
        +This is an OpenHands test patch.
        """

        test_instance = self.instance.copy()
        test_instance['instance_id'] = f'{self.instance["instance_id"]}_mock_test'

        with patch('openhands.nvidia.swe_agent.utils.initialize_agents') as mock_init:
            with patch(
                'openhands.nvidia.swe_agent.utils._apply_patch_and_evaluate'
            ) as mock_eval:
                mock_runtime = Mock()
                mock_runtime.event_stream = Mock()
                mock_runtime.event_stream.close = Mock()
                mock_runtime.close = Mock()
                mock_init.return_value = (mock_runtime, Mock(), Mock())

                mock_eval.return_value = {
                    'report': {'resolved': False, 'passed': 0, 'failed': 1},
                    'apply_patch_output': 'APPLY_PATCH_PASS',
                    'test_output': 'Tests failed',
                }

                result = await evaluate_agent(mock_patch, test_instance)

                assert 'report' in result
                assert result['report']['resolved'] is False

    @pytest.mark.asyncio
    async def test_parallel_evaluation_like_main(self):
        """Test parallel evaluation similar to __main__ section"""
        # Create multiple test instances
        test_instances = []
        for idx in range(2):
            inst_clone = self.instance.copy()
            inst_clone['instance_id'] = f'{self.instance["instance_id"]}_{idx}'
            test_instances.append(inst_clone)

        with patch('openhands.nvidia.swe_agent.utils.initialize_agents') as mock_init:
            with patch(
                'openhands.nvidia.swe_agent.utils._apply_patch_and_evaluate'
            ) as mock_eval:
                mock_runtime = Mock()
                mock_runtime.event_stream = Mock()
                mock_runtime.event_stream.close = Mock()
                mock_runtime.close = Mock()
                mock_runtime.sid = 'test_sid'
                mock_init.return_value = (mock_runtime, Mock(), Mock())

                mock_eval.return_value = {
                    'report': {'resolved': True, 'passed': 3, 'failed': 0},
                    'apply_patch_output': 'APPLY_PATCH_PASS',
                    'test_output': 'All tests passed',
                }

                # Run parallel evaluation like in __main__
                tasks = []
                for inst in test_instances:
                    gold_patch = inst['patch']
                    tasks.append(evaluate_agent(gold_patch, inst))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                assert len(results) == 2
                for result in results:
                    assert not isinstance(result, Exception)
                    assert 'report' in result
                    assert result['report']['resolved'] is True


@pytest.mark.integration
@pytest.mark.skipif(SKIP_INTEGRATION, reason='Integration data file not available')
class TestRealDataProcessing:
    """Test data processing steps from __main__ section"""

    def test_dataset_loading_and_processing(self):
        """Test the exact data loading and processing from __main__"""
        dataset = pd.read_parquet(PARQUET_FILE)

        # Test dataset structure
        assert len(dataset) > 0
        assert 'instance' in dataset.columns

        # Test instance extraction and processing
        instance = dataset.iloc[0]['instance']
        instance = pd.Series(instance)

        # Test numpy array conversion like in __main__
        processed_instance = instance.apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

        # Verify that numpy arrays were converted
        for key, value in processed_instance.items():
            if isinstance(value, np.ndarray):
                pytest.fail(f'Found unconverted numpy array in {key}')

        assert 'instance_id' in processed_instance
        assert isinstance(processed_instance['instance_id'], str)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(SKIP_INTEGRATION, reason='Integration data file not available')
class TestFullIntegrationFlow:
    """Full integration tests that exercise the complete flow"""

    @pytest.mark.asyncio
    async def test_initialize_agents_with_real_instance(self):
        """Test agent initialization with real instance data"""
        dataset = pd.read_parquet(PARQUET_FILE)
        instance = pd.Series(dataset.iloc[0]['instance'])
        instance = instance.apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

        # Use a minimal LLM config for testing
        llm_config = LLMConfig(
            model='gpt-4o-mini',
            base_url='https://api.openai.com/v1',
            api_key=os.environ.get('OPENAI_API_KEY', 'dummy-key'),
            modify_params=False,
            log_completions=False,
        )

        # Mock expensive runtime operations
        with patch(
            'openhands.nvidia.swe_agent.utils.create_runtime'
        ) as mock_create_runtime:
            with patch(
                'openhands.nvidia.swe_agent.utils.initialize_runtime'
            ) as mock_init_runtime:
                mock_runtime = AsyncMock()
                mock_runtime.sid = 'test_sid'
                mock_create_runtime.return_value = mock_runtime
                mock_init_runtime.return_value = None

                runtime, metadata, config = await initialize_agents(
                    instance, llm_config=llm_config, sid='test_integration'
                )

                assert runtime == mock_runtime
                assert metadata.agent_class == 'CodeActAgent'
                assert config.default_agent == 'CodeActAgent'
                mock_runtime.connect.assert_called_once()


@pytest.mark.integration
@pytest.mark.skipif(SKIP_INTEGRATION, reason='Integration data file not available')
@pytest.mark.slow
class TestRealRuntimeIntegration:
    """Integration tests using real runtime instead of mocked runtime"""

    @classmethod
    def setup_class(cls):
        """Load real dataset for integration tests"""
        cls.dataset = pd.read_parquet(PARQUET_FILE)
        cls.instance = cls.dataset.iloc[0]['instance']
        cls.instance = pd.Series(cls.instance)
        cls.instance = cls.instance.apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

    @pytest.mark.asyncio
    async def test_initialize_agents_with_real_runtime(self):
        """Test initialization with real runtime instead of mocked one"""
        # Use a minimal LLM config for testing
        llm_config = LLMConfig(
            model='gpt-4o-mini',
            base_url='https://api.openai.com/v1',
            api_key=os.environ.get('OPENAI_API_KEY', 'dummy-key'),
            modify_params=False,
            log_completions=False,
        )

        runtime = None
        try:
            # Initialize with real runtime - this will create actual DockerRuntime by default
            runtime, metadata, config = await initialize_agents(
                self.instance, llm_config=llm_config, sid='real_runtime_test'
            )

            # Verify we got a real runtime instance, not a mock
            assert runtime is not None
            assert hasattr(runtime, 'event_stream')
            assert hasattr(runtime, 'connect')
            assert hasattr(runtime, 'close')
            assert runtime.sid == 'real_runtime_test'

            # Verify the runtime is connected and functional
            assert runtime.event_stream is not None

            # Test basic runtime functionality
            from openhands.events.action import CmdRunAction

            action = CmdRunAction(command='echo "Hello from real runtime"')
            obs = runtime.run_action(action)
            assert obs.exit_code == 0
            assert 'Hello from real runtime' in obs.content

        finally:
            # Clean up real runtime resources
            if runtime:
                try:
                    runtime.event_stream.close()
                    runtime.close()
                except Exception as e:
                    print(f'Error cleaning up runtime: {e}')

    @pytest.mark.asyncio
    async def test_evaluate_agent_with_real_golden_patch(self):
        """Test full end-to-end evaluation with real runtime and real golden patch execution"""
        gold_patch = self.instance['patch']
        test_instance = self.instance.copy()
        test_instance['instance_id'] = f'{self.instance["instance_id"]}'

        LLMConfig(
            model='gpt-4o-mini',
            base_url='https://api.openai.com/v1',
            api_key=os.environ.get('OPENAI_API_KEY', 'dummy-key'),
            modify_params=False,
            log_completions=False,
        )

        try:
            # Run full evaluation with real runtime and real patch application
            # This will:
            # 1. Create real runtime (Singularity by default)
            # 2. Initialize the runtime with the instance
            # 3. Apply the golden patch
            # 4. Run the actual test suite
            # 5. Return real evaluation results
            result = await evaluate_agent(
                gold_patch,
                test_instance,
                sid=f'real_eval_{test_instance["instance_id"]}',
            )

            # Verify we got real evaluation results
            assert 'report' in result
            assert isinstance(result['report'], dict)

            # The results should contain actual test execution data
            if 'resolved' in result['report']:
                assert isinstance(result['report']['resolved'], bool)
                assert result['report']['resolved'] is True

            if 'passed' in result['report']:
                assert isinstance(result['report']['passed'], int)
                assert result['report']['passed'] >= 0

            if 'failed' in result['report']:
                assert isinstance(result['report']['failed'], int)
                assert result['report']['failed'] >= 0

            # Should have real patch application output
            if 'apply_patch_output' in result:
                assert isinstance(result['apply_patch_output'], str)
                # Should contain actual patch application results
                assert len(result['apply_patch_output']) > 0

            # Should have real test output
            if 'test_output' in result:
                assert isinstance(result['test_output'], str)
                assert len(result['test_output']) > 0

            print(f'Real evaluation completed for {test_instance["instance_id"]}')
            print(f'Results: {result}')

        except Exception as e:
            # Log the error for debugging but don't fail the test necessarily
            # since some instances might have issues with the golden patch
            print(f'Evaluation failed (this might be expected): {str(e)}')

            # Re-raise if it's a critical error we should catch
            if 'Runtime' in str(e) or 'connection' in str(e).lower():
                raise
            else:
                print(f'Non-critical evaluation error: {e}')
                raise

    @pytest.mark.asyncio
    async def test_run_agent_with_real_runtime_mock_llm(self):
        """Test run_agent with real runtime but mocked LLM responses"""
        from unittest.mock import AsyncMock, patch

        test_instance = self.instance.copy()

        # Set up mock LLM config that points to a mock server
        llm_config = LLMConfig(
            model='gpt-4o-mini',
            base_url='http://localhost:11111/v1',  # Mock server URL
            api_key='mock-key',
            modify_params=False,
            log_completions=False,
        )

        # Define realistic mock LLM responses for the agent
        mock_responses = [
            {
                'role': 'assistant',
                'content': """I'll help you resolve this issue. Let me start by exploring the codebase to understand the problem.

<execute_bash>
find . -type f -name "*.py" | head -10
</execute_bash>""",
                'finish_reason': 'stop',
            },
            {
                'role': 'assistant',
                'content': """Let me examine the specific files mentioned in the issue.

<execute_bash>
ls -la
</execute_bash>""",
                'finish_reason': 'stop',
            },
            {
                'role': 'assistant',
                'content': """Based on my exploration, I understand the issue. Let me create a fix:

<execute_bash>
echo "# Creating a test fix" > fix.py
</execute_bash>

I have implemented the necessary changes to resolve this issue.""",
                'finish_reason': 'stop',
            },
        ]

        runtime = None
        response_index = 0

        def mock_llm_response(*args, **kwargs):
            nonlocal response_index
            # Import litellm types for proper response structure
            from litellm.types.utils import ModelResponse

            # Return the next mock response
            if response_index < len(mock_responses):
                response = mock_responses[response_index]
                response_index += 1

                # Create a proper ModelResponse object
                mock_response = ModelResponse(
                    id='mock_response_id',
                    object='chat.completion',
                    created=1234567890,
                    model='gpt-4o-mini',
                    choices=[
                        {
                            'index': 0,
                            'message': {
                                'role': response['role'],
                                'content': response['content'],
                            },
                            'finish_reason': response['finish_reason'],
                        }
                    ],
                    usage={
                        'prompt_tokens': 100,
                        'completion_tokens': 50,
                        'total_tokens': 150,
                    },
                )

                return mock_response
            else:
                # If we run out of responses, return a finish message
                mock_response = ModelResponse(
                    id='mock_response_id_final',
                    object='chat.completion',
                    created=1234567890,
                    model='gpt-4o-mini',
                    choices=[
                        {
                            'index': 0,
                            'message': {
                                'role': 'assistant',
                                'content': 'I have completed the task.',
                            },
                            'finish_reason': 'stop',
                        }
                    ],
                    usage={
                        'prompt_tokens': 50,
                        'completion_tokens': 25,
                        'total_tokens': 75,
                    },
                )

                return mock_response

        try:
            # Initialize real runtime and config
            runtime, metadata, config = await initialize_agents(
                test_instance,
                llm_config=llm_config,
                sid=f'run_agent_test_{test_instance["instance_id"]}',
            )

            # Verify we have real runtime
            assert runtime is not None
            assert hasattr(runtime, 'run_action')

            # Create JobDetails object with all necessary fields
            job_details = JobDetails(
                job_id=f'test_job_{test_instance["instance_id"]}',
                instance=test_instance,
                runtime=runtime,
                metadata=metadata,
                config=config,
                llm_config=llm_config,
            )

            # Mock the LLM calls while keeping everything else real
            with patch('openai.AsyncOpenAI') as mock_openai:
                # Set up the mock OpenAI client
                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(
                    side_effect=mock_llm_response
                )
                mock_openai.return_value = mock_client

                # Patch the specific litellm_completion import used by the LLM class
                with patch(
                    'openhands.llm.llm.litellm_completion',
                    side_effect=mock_llm_response,
                ):
                    # Run the agent with JobDetails object
                    result = await run_agent(
                        job_details=job_details,
                        sid=f'run_agent_test_{test_instance["instance_id"]}',
                    )

                    # Verify we got real results
                    assert isinstance(result, dict)
                    assert 'git_patch' in result
                    assert 'success' in result
                    assert 'messages' in result
                    assert 'tools' in result

                    # Check that messages contain our mock responses
                    assert isinstance(result['messages'], list)
                    assert len(result['messages']) > 0

                    # Verify the git patch is a string (could be empty)
                    assert isinstance(result['git_patch'], str)

                    # Check success/error handling
                    assert isinstance(result['success'], bool)

                    if result.get('error'):
                        assert isinstance(result['error'], str)

                    # Verify tools were properly extracted
                    assert isinstance(result['tools'], list)

                    print('Agent run completed successfully!')
                    print(f'Generated {len(result["messages"])} messages')
                    print(f'Git patch length: {len(result["git_patch"])}')
                    print(f'Success: {result["success"]}')

        finally:
            # Clean up runtime
            if runtime:
                try:
                    runtime.event_stream.close()
                    runtime.close()
                except Exception as e:
                    print(f'Error cleaning up runtime: {e}')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
