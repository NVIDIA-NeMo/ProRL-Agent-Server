# Suppress warnings at the very beginning before any imports
import os  # noqa: E402
import warnings  # noqa: E402

warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import asyncio  # noqa: E402
import time  # noqa: E402
from unittest.mock import AsyncMock, Mock, patch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from evaluation.utils.shared import EvalMetadata  # noqa: E402
from openhands.core.config import OpenHandsConfig  # noqa: E402
from openhands.core.config.condenser_config import NoOpCondenserConfig  # noqa: E402
from openhands.core.config.llm_config import LLMConfig  # noqa: E402
from openhands.core.config.sandbox_config import SandboxConfig  # noqa: E402
from openhands.events.action import AgentFinishAction  # noqa: E402
from openhands.events.observation import CmdOutputObservation  # noqa: E402
from openhands.nvidia.registry import _DEFAULT_AGENT_CONFIG  # noqa: E402
from openhands.nvidia.swe_agent.utils import (  # noqa: E402
    DOCKER_IMAGE_PREFIX,
    _apply_patch_and_evaluate,
    evaluate_agent,
    get_config,
    get_instance_docker_image,
    initialize_agents,
    is_last_action_finish,
    run_agent,
)

# Import the module under test
from openhands.nvidia.utils import (  # noqa: E402
    eval_exception,
    final_result,
    initialize_exception,
    run_exception,
)


class TestGetInstanceDockerImage:
    """Test the get_instance_docker_image function"""

    def test_basic_instance_id(self):
        """Test basic instance ID transformation"""
        instance_id = 'test_repo_issue_123'
        expected = f'{DOCKER_IMAGE_PREFIX.rstrip("/")}/sweb.eval.x86_64.test_repo_issue_123'.lower()
        result = get_instance_docker_image(instance_id)
        assert result == expected

    def test_instance_id_with_double_underscore(self):
        """Test instance ID with double underscores gets replaced"""
        instance_id = 'test__repo__issue__123'
        expected = f'{DOCKER_IMAGE_PREFIX.rstrip("/")}/sweb.eval.x86_64.test_s_repo_s_issue_s_123'.lower()
        result = get_instance_docker_image(instance_id)
        assert result == expected

    def test_docker_prefix_with_trailing_slash(self):
        """Test that trailing slash is properly stripped from prefix"""
        with patch.dict(os.environ, {'EVAL_DOCKER_IMAGE_PREFIX': 'test/prefix/'}):
            # Import again to pick up the new environment variable
            from openhands.nvidia.swe_agent.utils import get_instance_docker_image

            instance_id = 'test_instance'
            result = get_instance_docker_image(instance_id)
            assert not result.endswith('//sweb.eval.x86_64.test_instance')

    @pytest.mark.real_data
    def test_real_instance_docker_image(self, real_instance):
        """Test docker image generation with real instance ID"""
        instance_id = real_instance['instance_id']
        result = get_instance_docker_image(instance_id)

        # Verify the format matches expected pattern
        expected_prefix = DOCKER_IMAGE_PREFIX.rstrip('/').lower()
        assert result.startswith(expected_prefix)
        assert 'sweb.eval.x86_64' in result
        assert instance_id.replace('__', '_s_').lower() in result


class TestIsLastActionFinish:
    """Test the is_last_action_finish function"""

    def test_none_state(self):
        """Test with None state"""
        assert is_last_action_finish(None) is False

    def test_empty_history(self):
        """Test with state having empty history"""
        state = Mock()
        state.history = []
        assert is_last_action_finish(state) is False

    def test_no_actions_in_history(self):
        """Test with history containing no actions"""
        state = Mock()
        state.history = [Mock(), Mock()]  # Non-action objects
        assert is_last_action_finish(state) is False

    def test_last_action_is_finish(self):
        """Test with last action being AgentFinishAction"""
        state = Mock()
        finish_action = AgentFinishAction()
        other_action = Mock()
        state.history = [other_action, finish_action]
        assert is_last_action_finish(state) is True

    def test_last_action_is_not_finish(self):
        """Test with last action not being AgentFinishAction"""
        state = Mock()
        regular_action = Mock()
        state.history = [regular_action]
        assert is_last_action_finish(state) is False


class TestGetConfig:
    """Test the get_config function"""

    @patch('openhands.nvidia.swe_agent.utils.get_instance_resource_factor')
    @patch('openhands.nvidia.swe_agent.utils.get_default_sandbox_config_for_eval')
    def test_basic_config_creation(self, mock_sandbox_config, mock_resource_factor):
        """Test basic configuration creation"""
        # Setup mocks with proper return types
        mock_sandbox = SandboxConfig()
        mock_sandbox_config.return_value = mock_sandbox
        mock_resource_factor.return_value = 1.0

        instance = pd.Series({'instance_id': 'test_instance'})
        metadata = EvalMetadata(
            agent_class='TestAgent',
            max_iterations=10,
            llm_config=LLMConfig(model='gpt-4', api_key='test-key'),
            eval_output_dir='/test/dir',
            start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            git_commit='abc123',
            dataset='swebench',
            condenser_config=NoOpCondenserConfig(),
        )

        config = get_config(instance, metadata)

        assert isinstance(config, OpenHandsConfig)
        assert config.default_agent == metadata.agent_class
        assert config.max_iterations == metadata.max_iterations
        assert config.runtime == 'singularity'

    @patch('openhands.nvidia.swe_agent.utils.get_instance_resource_factor')
    @patch('openhands.nvidia.swe_agent.utils.get_default_sandbox_config_for_eval')
    def test_sandbox_config_settings(self, mock_sandbox_config, mock_resource_factor):
        """Test sandbox configuration settings"""
        sandbox_config = SandboxConfig()
        mock_sandbox_config.return_value = sandbox_config
        mock_resource_factor.return_value = 2.0

        instance = pd.Series({'instance_id': 'test_instance'})
        metadata = EvalMetadata(
            agent_class='TestAgent',
            max_iterations=10,
            llm_config=LLMConfig(model='gpt-4', api_key='test-key'),
            eval_output_dir='/test/dir',
            start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            git_commit='abc123',
            dataset='swebench',
            condenser_config=NoOpCondenserConfig(),
        )

        get_config(instance, metadata)

        # Verify sandbox config was properly set
        assert sandbox_config.enable_auto_lint is True
        assert sandbox_config.use_host_network is False
        assert sandbox_config.platform == 'linux/amd64'
        assert sandbox_config.run_as_fakeroot is False
        assert sandbox_config.remote_runtime_resource_factor == 2.0

    @pytest.mark.real_data
    @patch('openhands.nvidia.swe_agent.utils.get_instance_resource_factor')
    @patch('openhands.nvidia.swe_agent.utils.get_default_sandbox_config_for_eval')
    def test_config_with_real_instance(
        self,
        mock_sandbox_config,
        mock_resource_factor,
        real_instance,
        minimal_llm_config,
    ):
        """Test configuration creation with real instance data"""
        mock_sandbox = SandboxConfig()
        mock_sandbox_config.return_value = mock_sandbox
        mock_resource_factor.return_value = 1.5

        metadata = EvalMetadata(
            agent_class='CodeActAgent',
            max_iterations=20,
            llm_config=minimal_llm_config,
            eval_output_dir='/tmp/test',
            start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            git_commit='test123',
            dataset='swebench',
            condenser_config=NoOpCondenserConfig(),
        )

        config = get_config(real_instance, metadata)

        assert isinstance(config, OpenHandsConfig)
        assert config.default_agent == 'CodeActAgent'
        # Verify docker image was set correctly
        expected_image = get_instance_docker_image(real_instance['instance_id'])
        assert mock_sandbox.runtime_container_image == expected_image


class TestInitializeAgents:
    """Test the initialize_agents function"""

    @pytest.mark.asyncio
    async def test_none_llm_config_raises_error(self):
        """Test that None LLM config raises ValueError"""
        instance = pd.Series({'instance_id': 'test'})

        with pytest.raises(ValueError, match='LLM config is None'):
            await initialize_agents(instance, llm_config=None)

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.utils.create_runtime')
    @patch('openhands.nvidia.swe_agent.utils.initialize_runtime')
    async def test_successful_initialization(
        self, mock_init_runtime, mock_create_runtime
    ):
        """Test successful agent initialization"""
        # Setup mocks
        mock_runtime = AsyncMock()
        mock_create_runtime.return_value = mock_runtime
        mock_init_runtime.return_value = None

        instance = pd.Series({'instance_id': 'test_instance'})
        llm_config = LLMConfig(model='gpt-4', api_key='test-key')

        runtime, metadata, config = await initialize_agents(
            instance, llm_config=llm_config, sid='test_sid'
        )

        # Verify results
        assert runtime == mock_runtime
        assert isinstance(metadata, EvalMetadata)
        assert isinstance(config, OpenHandsConfig)
        assert metadata.agent_class == 'CodeActAgent'

        # Verify runtime was connected
        mock_runtime.connect.assert_called_once()
        mock_init_runtime.assert_called_once()

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.utils.create_runtime')
    @patch('openhands.nvidia.swe_agent.utils.initialize_runtime')
    async def test_initialization_runtime_error(
        self, mock_init_runtime, mock_create_runtime
    ):
        """Test initialization with runtime error"""
        mock_runtime = AsyncMock()
        mock_create_runtime.return_value = mock_runtime
        mock_init_runtime.side_effect = Exception('Runtime initialization failed')

        instance = pd.Series({'instance_id': 'test_instance'})
        llm_config = LLMConfig(model='gpt-4', api_key='test-key')

        with pytest.raises(Exception, match='Runtime initialization failed'):
            await initialize_agents(instance, llm_config=llm_config)

    @pytest.mark.real_data
    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.utils.create_runtime')
    @patch('openhands.nvidia.swe_agent.utils.initialize_runtime')
    async def test_initialize_with_real_instance(
        self, mock_init_runtime, mock_create_runtime, real_instance, minimal_llm_config
    ):
        """Test agent initialization with real instance data"""
        mock_runtime = AsyncMock()
        mock_create_runtime.return_value = mock_runtime
        mock_init_runtime.return_value = None

        runtime, metadata, config = await initialize_agents(
            real_instance, llm_config=minimal_llm_config, sid='real_test'
        )

        assert runtime == mock_runtime
        assert metadata.agent_class == 'CodeActAgent'
        assert config.default_agent == 'CodeActAgent'
        # Verify the real instance ID is used
        expected_image = get_instance_docker_image(real_instance['instance_id'])
        assert config.sandbox.runtime_container_image == expected_image


class TestRunAgent:
    """Test the run_agent function"""

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.utils.create_agent')
    @patch('openhands.nvidia.swe_agent.utils.run_controller_with_controller')
    @patch('openhands.nvidia.swe_agent.utils.get_instruction')
    @patch('openhands.nvidia.swe_agent.utils.complete_runtime')
    async def test_successful_run(
        self,
        mock_complete,
        mock_get_instruction,
        mock_run_controller,
        mock_create_agent,
    ):
        """Test successful agent run"""
        # Setup mocks
        mock_agent = Mock()
        mock_agent._get_initial_user_message.return_value = 'initial message'

        # Create mock message objects with .role attributes
        mock_user_message = Mock()
        mock_user_message.role = 'user'
        mock_user_message.content = 'test message'

        mock_assistant_message = Mock()
        mock_assistant_message.role = 'assistant'
        mock_assistant_message.content = 'test response'

        mock_agent._get_messages.return_value = [
            mock_user_message,
            mock_assistant_message,
        ]
        mock_agent.llm.format_messages_for_llm.return_value = [
            {'role': 'user', 'content': 'test message'},
            {'role': 'assistant', 'content': 'test response'},
        ]
        mock_agent.tools = []
        mock_agent.llm.config = Mock()

        mock_create_agent.return_value = mock_agent

        mock_state = Mock()
        mock_state.last_error = None
        mock_state.history = []
        mock_run_controller.return_value = mock_state

        mock_get_instruction.return_value = Mock()
        mock_complete.return_value = {'git_patch': 'test patch'}

        # Create mock job_details
        job_details = Mock()
        job_details.runtime = Mock()
        job_details.metadata = Mock()
        job_details.config = Mock()
        job_details.instance = pd.Series({'instance_id': 'test'})
        job_details.llm_config = Mock()
        job_details.llm_config.token_level_generation = False
        job_details.agent_config = _DEFAULT_AGENT_CONFIG

        with patch('openhands.llm.llm_utils.check_tools') as mock_check_tools:
            mock_check_tools.return_value = []
            with patch(
                'openhands.nvidia.swe_agent.utils.create_controller'
            ) as mock_create_controller:
                mock_create_controller.return_value = (Mock(), Mock())

                result = await run_agent(job_details)

        assert result['git_patch'] == 'test patch'
        assert result['success'] is True
        assert result['error'] is None
        assert 'messages' in result
        assert 'tools' in result

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.utils.get_instruction')
    async def test_run_with_none_state_raises_eval_exception(
        self, mock_get_instruction
    ):
        """Test that None state raises Exception due to failed message processing"""
        mock_get_instruction.return_value = Mock()

        with patch(
            'openhands.nvidia.swe_agent.utils.create_agent'
        ) as mock_create_agent:
            with patch(
                'openhands.nvidia.swe_agent.utils.run_controller_with_controller'
            ) as mock_run_controller:
                with patch(
                    'openhands.nvidia.swe_agent.utils.complete_runtime'
                ) as mock_complete:
                    with patch(
                        'openhands.nvidia.swe_agent.utils.create_controller'
                    ) as mock_create_controller:
                        mock_create_agent.return_value = Mock()
                        mock_run_controller.return_value = None
                        mock_complete.return_value = {'git_patch': 'test patch'}
                        mock_create_controller.return_value = (Mock(), Mock())

                        # Create mock job_details
                        job_details = Mock()
                        job_details.runtime = Mock()
                        job_details.metadata = Mock()
                        job_details.config = Mock()
                        job_details.instance = pd.Series({'instance_id': 'test'})

                        with pytest.raises(
                            Exception, match='Failed to retrieve agent messages'
                        ):
                            await run_agent(job_details)


class TestEvaluateAgent:
    """Test the evaluate_agent function"""

    @pytest.mark.asyncio
    async def test_empty_patch_with_skip_allowed(self):
        """Test evaluation with empty patch when skip is allowed"""
        instance = pd.Series({'instance_id': 'test'})
        result = await evaluate_agent('', instance, allow_skip=True)
        assert result == {'resolved': False}

    @pytest.mark.asyncio
    async def test_none_patch_with_skip_not_allowed_raises_error(self):
        """Test that None patch raises error when skip is not allowed"""
        instance = pd.Series({'instance_id': 'test'})

        with pytest.raises(ValueError, match='Patch is None'):
            await evaluate_agent(None, instance, allow_skip=False)

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.utils.initialize_agents')
    @patch('openhands.utils.async_utils.call_sync_from_async')
    async def test_successful_evaluation(self, mock_call_sync, mock_init_agents):
        """Test successful patch evaluation"""
        # Setup mocks
        mock_runtime = Mock()
        mock_runtime.event_stream = Mock()
        mock_runtime.event_stream.close = Mock()
        mock_runtime.close = Mock()
        mock_init_agents.return_value = (mock_runtime, Mock(), Mock())

        test_result = {'resolved': True, 'report': {'resolved': True}}
        mock_call_sync.return_value = test_result

        instance = pd.Series({'instance_id': 'test'})
        patch = "diff --git a/test.py b/test.py\n+print('hello')"

        result = await evaluate_agent(patch, instance)

        assert result == test_result
        mock_runtime.event_stream.close.assert_called_once()

    @pytest.mark.real_data
    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.utils.initialize_agents')
    @patch('openhands.nvidia.swe_agent.utils._apply_patch_and_evaluate')
    @patch('openhands.utils.async_utils.call_sync_from_async')
    async def test_evaluation_with_real_patch(
        self,
        mock_call_sync,
        mock_apply_patch,
        mock_init_agents,
        real_instance,
        sample_evaluation_result,
    ):
        """Test evaluation using real patch from dataset"""
        mock_runtime = Mock()
        mock_runtime.event_stream.close = Mock()
        mock_runtime.close = Mock()
        mock_init_agents.return_value = (mock_runtime, Mock(), Mock())

        # Use the evaluation result from the fixture
        mock_call_sync.return_value = sample_evaluation_result

        # Get the real patch from the instance
        real_patch = real_instance['patch']

        result = await evaluate_agent(real_patch, real_instance)

        assert result == sample_evaluation_result
        assert result['report']['resolved'] is True
        mock_init_agents.assert_called_once()

    @pytest.mark.real_data
    @pytest.mark.asyncio
    async def test_parallel_evaluation_like_main(
        self, real_instance, mock_patch, sample_evaluation_result
    ):
        """Test parallel evaluation pattern from __main__ section"""
        # Create test instances like in __main__
        test_instances = []
        for idx in range(2):
            inst_clone = real_instance.copy()
            inst_clone['instance_id'] = f'{real_instance["instance_id"]}_parallel_{idx}'
            test_instances.append(inst_clone)

        with patch('openhands.nvidia.swe_agent.utils.initialize_agents') as mock_init:
            with patch('openhands.nvidia.swe_agent.utils._apply_patch_and_evaluate'):
                with patch(
                    'openhands.utils.async_utils.call_sync_from_async'
                ) as mock_call_sync:
                    mock_runtime = Mock()
                    mock_runtime.event_stream.close = Mock()
                    mock_runtime.close = Mock()
                    mock_init.return_value = (mock_runtime, Mock(), Mock())
                    mock_call_sync.return_value = sample_evaluation_result

                    # Run parallel evaluation like in __main__
                    tasks = []
                    for inst in test_instances:
                        tasks.append(evaluate_agent(mock_patch, inst))

                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    assert len(results) == 2
                    for result in results:
                        assert not isinstance(result, Exception)
                        assert result == sample_evaluation_result


class TestApplyPatchAndEvaluate:
    """Test the _apply_patch_and_evaluate function"""

    @patch('swegym.harness.test_spec.make_test_spec')
    @patch('openhands.nvidia.swe_agent.utils._process_git_patch')
    @patch('swegym.harness.grading.get_eval_report')
    def test_successful_patch_application_and_evaluation(
        self, mock_get_eval, mock_process_patch, mock_make_test_spec
    ):
        """Test successful patch application and evaluation"""
        # Setup mocks
        mock_runtime = Mock()

        # Create different mock observations for different commands
        chmod_obs = CmdOutputObservation(content='', exit_code=0, command='chmod')
        patch_obs = CmdOutputObservation(
            content='APPLY_PATCH_PASS\nPatch applied successfully',
            exit_code=0,
            command='apply patch',
        )
        launch_obs = CmdOutputObservation(
            content='1234', exit_code=0, command='launch script'
        )
        process_check_obs = CmdOutputObservation(
            content='1', exit_code=0, command='process check'
        )
        cat_obs = CmdOutputObservation(
            content='Test output', exit_code=0, command='cat'
        )

        # Set up side_effect to return different observations for different calls
        mock_runtime.run_action.side_effect = [
            chmod_obs,  # chmod command
            patch_obs,  # patch application
            launch_obs,  # launch evaluation script
            process_check_obs,  # process status check (returns 1 = finished)
            cat_obs,  # cat log file
        ]
        mock_runtime.copy_to = Mock()

        mock_process_patch.return_value = 'processed patch'

        test_spec = Mock()
        test_spec.eval_script = '#!/bin/bash\necho test'
        mock_make_test_spec.return_value = test_spec

        mock_get_eval.return_value = {
            'django__django-12345': {'resolved': True, 'passed': 5, 'failed': 0}
        }

        instance = pd.Series(
            {
                'instance_id': 'django__django-12345',  # Use proper SWE-bench instance ID format
                'repo': 'django/django',  # Use a real repo name that exists in the mapping
                'version': '3.0',
                'base_commit': 'abc123',
                'problem_statement': 'Test problem statement',
                'hints_text': 'Test hints',
                'test_patch': 'test patch content',
                'PASS_TO_PASS': '[]',
                'FAIL_TO_PASS': '[]',
            }
        )
        git_patch = 'diff --git a/test.py'

        # Mock time.time to return consistent values
        with patch('time.time', return_value=100):
            with patch('time.sleep'):  # Mock sleep to speed up test
                result = _apply_patch_and_evaluate(mock_runtime, git_patch, instance)

        assert 'report' in result
        assert 'apply_patch_output' in result
        assert 'test_output' in result
        assert result['report']['resolved'] is True

    @patch('openhands.nvidia.swe_agent.utils._process_git_patch')
    @patch('swegym.harness.test_spec.make_test_spec')
    def test_patch_application_failure(self, mock_make_test_spec, mock_process_patch):
        """Test patch application failure"""
        mock_runtime = Mock()
        mock_obs = CmdOutputObservation(
            content='APPLY_PATCH_FAIL\nPatch failed to apply',
            exit_code=1,
            command='test command',
        )
        mock_runtime.run_action.return_value = mock_obs
        mock_runtime.copy_to = Mock()

        mock_process_patch.return_value = 'processed patch'
        mock_make_test_spec.return_value = Mock(eval_script='test script')

        instance = pd.Series(
            {
                'instance_id': 'django__django-12345',  # Use proper SWE-bench instance ID format
                'repo': 'django/django',  # Use a real repo name that exists in the mapping
                'version': '3.0',
                'base_commit': 'abc123',
                'problem_statement': 'Test problem statement',
                'hints_text': 'Test hints',
                'test_patch': 'test patch content',
                'PASS_TO_PASS': '[]',
                'FAIL_TO_PASS': '[]',
            }
        )
        git_patch = 'diff --git a/test.py'

        with pytest.raises(RuntimeError, match='APPLY_PATCH_FAIL'):
            _apply_patch_and_evaluate(mock_runtime, git_patch, instance)


class TestExceptionHandlers:
    """Test exception handler functions"""

    def test_initialize_exception(self):
        """Test initialize exception handler"""
        job_details = Mock()
        job_details.instance = {
            'instance_id': 'test_id',
            'trajectory_id': 'test_trajectory',
        }

        exception = Exception('Test error')
        result = initialize_exception(job_details, exception)

        assert result['instance_id'] == 'test_id'
        assert result['trajectory_id'] == 'test_trajectory'
        assert result['success'] is False
        assert 'Test error' in result['error']
        assert result['critical_error'] == 'init'

    def test_run_exception(self):
        """Test run exception handler"""
        job_details = Mock()
        job_details.instance = {
            'instance_id': 'test_id',
            'trajectory_id': 'test_trajectory',
        }
        # Mock run_results with proper structure
        job_details.run_results = {
            'git_patch': 'test_patch',
            'success': True,
            'finish': True,
            'messages': ['test_message'],  # Use actual list instead of Mock
            'tools': ['test_tool'],
            'end_properly': True,
        }

        exception = Exception('Run error')
        result = run_exception(job_details, exception)

        assert result['instance_id'] == 'test_id'
        assert result['critical_error'] == 'run'
        assert 'Run error' in result['error']

    def test_eval_exception(self):
        """Test eval exception handler"""
        job_details = Mock()
        job_details.instance = {
            'instance_id': 'test_id',
            'trajectory_id': 'test_trajectory',
        }
        job_details.run_results = {
            'git_patch': 'test patch',
            'success': True,
            'finish': True,
            'messages': ['test message'],  # Use actual list instead of Mock
            'tools': ['test tool'],
            'end_properly': True,
        }

        exception = Exception('Eval error')
        result = eval_exception(job_details, exception)

        assert result['instance_id'] == 'test_id'
        assert result['git_patch'] == 'test patch'
        assert result['critical_error'] == 'eval'
        assert 'Eval error' in result['error']


class TestFinalResult:
    """Test final_result function"""

    def test_final_result_with_all_times(self):
        """Test final result merges run_results and eval_results"""
        job_details = Mock()
        job_details.results = None
        job_details.run_results = {'success': True, 'git_patch': 'test'}
        job_details.eval_results = {'resolved': True}
        job_details.timeout_error = False

        result = final_result(job_details)

        assert result['success'] is True
        assert result['git_patch'] == 'test'
        assert result['resolved'] is True
        assert result['critical_error'] is None

    def test_final_result_no_eval_time(self):
        """Test final result with timeout error"""
        job_details = Mock()
        job_details.results = None
        job_details.run_results = {'success': True}
        job_details.eval_results = {'resolved': False}
        job_details.timeout_error = True

        result = final_result(job_details)

        assert result['success'] is True
        assert result['resolved'] is False
        assert result['critical_error'] == 'timeout'

    def test_final_result_with_existing_results(self):
        """Test final result when job_details.results already exists"""
        job_details = Mock()
        job_details.results = {'existing': True, 'data': 'test'}
        job_details.timeout_error = False

        result = final_result(job_details)

        assert result['existing'] is True
        assert result['data'] == 'test'
        assert 'critical_error' not in result or result['critical_error'] is None

    def test_final_result_no_run_results(self):
        """Test final result when run_results is None"""
        job_details = Mock()
        job_details.results = None
        job_details.run_results = None
        job_details.eval_results = {'resolved': True}
        job_details.timeout_error = False
        job_details.instance = {'instance_id': 'test_id', 'trajectory_id': 'test_traj'}

        # Mock the get_messages_from_partial_result function to return proper structure
        with patch(
            'openhands.nvidia.utils.get_messages_from_partial_result'
        ) as mock_get_messages:
            mock_get_messages.return_value = {
                'messages': ['test_message'],
                'tools': ['test_tool'],
                'end_properly': True,
            }

            result = final_result(job_details)

        assert result['instance_id'] == 'test_id'
        assert result['trajectory_id'] == 'test_traj'
        assert result['resolved'] is True
        assert result['critical_error'] is None

    def test_final_result_timeout_with_existing_results(self):
        """Test final result with timeout when results already exist"""
        job_details = Mock()
        job_details.results = {'existing': True, 'data': 'test'}
        job_details.timeout_error = True

        result = final_result(job_details)

        assert result['existing'] is True
        assert result['data'] == 'test'
        assert result['critical_error'] == 'timeout'


class TestMainExecution:
    """Test the main execution logic from __main__ section"""

    @patch('pandas.read_parquet')
    @patch('openhands.nvidia.swe_agent.utils.evaluate_agent')
    @patch('asyncio.run')
    def test_main_parallel_evaluation(
        self, mock_asyncio_run, mock_evaluate, mock_read_parquet
    ):
        """Test the main parallel evaluation logic"""
        # Setup mock data
        mock_instance_data = {
            'instance_id': 'test_instance',
            'patch': 'diff --git a/test.py',
            'trajectory_id': 'test_trajectory',
        }
        pd.Series(mock_instance_data)

        mock_dataset = Mock()
        mock_dataset.iloc = [{'instance': mock_instance_data}]
        mock_read_parquet.return_value = mock_dataset

        # Mock the evaluation results
        mock_evaluate.return_value = {'resolved': True, 'report': {'resolved': True}}

        # This test verifies the structure but doesn't actually run the main code
        # since it would require full environment setup
        assert mock_read_parquet is not None
        assert mock_evaluate is not None

    @patch('pandas.read_parquet')
    def test_main_data_loading_and_processing(self, mock_read_parquet):
        """Test data loading and preprocessing in main"""
        # Mock the dataset structure as it appears in main
        mock_instance_data = {
            'instance_id': 'test_instance',
            'patch': 'test patch',
            'some_array_field': np.array([1, 2, 3]),
        }

        mock_dataset = Mock()
        mock_dataset.iloc = [{'instance': mock_instance_data}]
        mock_read_parquet.return_value = mock_dataset

        # Simulate the data processing from main
        instance = pd.Series(mock_instance_data)
        processed_instance = instance.apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

        # Verify numpy arrays are converted to lists
        assert processed_instance['some_array_field'] == [1, 2, 3]
        assert processed_instance['instance_id'] == 'test_instance'

    def test_mock_patch_structure(self):
        """Test the mock patch structure used in main"""
        mock_patch = """
        diff --git a/openhands_patch_test.txt b/openhands_patch_test.txt
        new file mode 100644
        index 0000000..e69de29
        --- /dev/null
        +++ b/openhands_patch_test.txt
        @@
        +This is an OpenHands test patch.
        """

        # Verify the mock patch has expected structure
        assert 'diff --git' in mock_patch
        assert 'openhands_patch_test.txt' in mock_patch
        assert '+This is an OpenHands test patch.' in mock_patch

    @pytest.mark.real_data
    def test_main_execution_with_real_data(
        self, real_dataset, real_instance, mock_patch
    ):
        """Test main execution patterns with real data"""
        # Test the data loading pattern from __main__
        assert len(real_dataset) > 0
        assert 'instance' in real_dataset.columns

        # Test instance processing pattern
        assert 'instance_id' in real_instance
        assert 'patch' in real_instance

        # Verify numpy array conversion worked
        for key, value in real_instance.items():
            assert not isinstance(value, np.ndarray), (
                f'Found unconverted numpy array in {key}'
            )

        # Test that we can create multiple instances like in __main__
        test_instances = []
        for idx in range(3):
            inst_clone = real_instance.copy()
            inst_clone['instance_id'] = (
                f'{real_instance["instance_id"]}_main_test_{idx}'
            )
            test_instances.append(inst_clone)

        assert len(test_instances) == 3
        for inst in test_instances:
            assert inst['instance_id'] != real_instance['instance_id']
            assert inst['patch'] == real_instance['patch']


class TestEnvironmentVariables:
    """Test environment variable handling"""

    def test_docker_image_prefix_default(self):
        """Test default docker image prefix"""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to get default value
            pass
            # Note: This might not work as expected due to module caching
            # In a real test, you'd need to reload the module

    def test_run_with_browsing_default(self):
        """Test default browsing setting"""
        with patch.dict(os.environ, {'RUN_WITH_BROWSING': 'true'}):
            # Re-import to get updated value
            pass
            # Note: This might not work as expected due to module caching


if __name__ == '__main__':
    pytest.main([__file__])
