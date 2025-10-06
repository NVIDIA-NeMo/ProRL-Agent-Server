# Suppress warnings at the very beginning before any imports
import os  # noqa: E402
import warnings  # noqa: E402

warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import asyncio  # noqa: E402
from unittest.mock import Mock, patch  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from openhands.nvidia.registry import _DEFAULT_AGENT_CONFIG, JobDetails  # noqa: E402
from openhands.nvidia.swe_agent.swe_agent_handler import SweAgentHandler  # noqa: E402


class TestSweAgentHandler:
    """Test suite for SweAgentHandler class"""

    def test_name_property(self):
        """Test that the name property returns the correct identifier"""
        handler = SweAgentHandler()
        assert handler.name == 'swebench'

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.initialize_agents')
    async def test_init_success(self, mock_initialize_agents, minimal_llm_config):
        """Test successful initialization"""
        # Setup mock return values
        mock_runtime = Mock()
        mock_metadata = Mock()
        mock_config = Mock()
        mock_initialize_agents.return_value = (mock_runtime, mock_metadata, mock_config)

        # Create test instance and job details
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(
            job_id='test_job', instance=instance, llm_config=minimal_llm_config
        )
        handler = SweAgentHandler()

        # Call init method
        result = await handler.init(
            job_details=job_details,
            sid='test_sid',
        )

        # Verify the result
        assert result == (mock_runtime, mock_metadata, mock_config)

        # Verify the mock was called with correct parameters
        mock_initialize_agents.assert_called_once_with(
            instance=instance,
            llm_config=minimal_llm_config,
            sid='test_sid',
            agent_config=_DEFAULT_AGENT_CONFIG,
        )

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.initialize_agents')
    async def test_init_with_defaults(self, mock_initialize_agents):
        """Test initialization with default parameters"""
        # Setup mock return values
        mock_runtime = Mock()
        mock_metadata = Mock()
        mock_config = Mock()
        mock_initialize_agents.return_value = (mock_runtime, mock_metadata, mock_config)

        # Create test instance and job details
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(job_id='test_job', instance=instance)
        handler = SweAgentHandler()

        # Call init method with defaults
        result = await handler.init(job_details=job_details)

        # Verify the result
        assert result == (mock_runtime, mock_metadata, mock_config)

        # Verify the mock was called with default parameters
        mock_initialize_agents.assert_called_once_with(
            instance=instance,
            llm_config=None,
            sid=None,
            agent_config=_DEFAULT_AGENT_CONFIG,
        )

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.initialize_agents')
    async def test_init_exception_propagation(self, mock_initialize_agents):
        """Test that initialization exceptions are properly propagated"""
        # Setup mock to raise exception
        mock_initialize_agents.side_effect = RuntimeError('Initialization failed')

        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(job_id='test_job', instance=instance)
        handler = SweAgentHandler()

        # Verify exception is propagated
        with pytest.raises(RuntimeError, match='Initialization failed'):
            await handler.init(job_details=job_details)

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.run_agent')
    async def test_run_success(self, mock_run_agent, mock_runtime):
        """Test successful run"""
        # Setup mock return values
        expected_result = {'git_patch': 'test_patch', 'status': 'completed'}
        mock_run_agent.return_value = expected_result

        # Create test data
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(job_id='test_job', instance=instance)
        handler = SweAgentHandler()

        # Call run method
        result = await handler.run(
            job_details=job_details,
            sid='test_sid',
        )

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with correct parameters
        mock_run_agent.assert_called_once_with(
            job_details=job_details,
            sid='test_sid',
        )

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.run_agent')
    async def test_run_exception_propagation(self, mock_run_agent, mock_runtime):
        """Test that run exceptions are properly propagated"""
        # Setup mock to raise exception
        mock_run_agent.side_effect = RuntimeError('Run failed')

        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(job_id='test_job', instance=instance)
        handler = SweAgentHandler()

        # Verify exception is propagated
        with pytest.raises(RuntimeError, match='Run failed'):
            await handler.run(
                job_details=job_details,
                sid='test_sid',
            )

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.evaluate_agent')
    async def test_eval_success_with_patch(self, mock_evaluate_agent):
        """Test successful evaluation with git patch"""
        # Setup mock return values
        expected_result = {'resolved': True, 'passed': 3, 'failed': 0}
        mock_evaluate_agent.return_value = expected_result

        # Create test job details with run results containing git_patch
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(
            job_id='test_job',
            instance=instance,
            run_results={'git_patch': 'test_patch', 'status': 'completed'},
        )

        handler = SweAgentHandler()

        # Call eval method
        result = await handler.eval(
            job_details=job_details, sid='test_sid', allow_skip=False
        )

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with correct parameters
        mock_evaluate_agent.assert_called_once_with(
            git_patch='test_patch', instance=instance, sid='test_sid', allow_skip=False
        )

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.evaluate_agent')
    async def test_eval_success_no_patch(self, mock_evaluate_agent):
        """Test successful evaluation with no git patch"""
        # Setup mock return values
        expected_result = {'resolved': False, 'empty_generation': True}
        mock_evaluate_agent.return_value = expected_result

        # Create test job details with no run results
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(job_id='test_job', instance=instance, run_results=None)

        handler = SweAgentHandler()

        # Call eval method
        result = await handler.eval(job_details=job_details)

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with empty git_patch
        mock_evaluate_agent.assert_called_once_with(
            git_patch='', instance=instance, sid=None, allow_skip=True
        )

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.evaluate_agent')
    async def test_eval_success_run_results_no_patch(self, mock_evaluate_agent):
        """Test evaluation when run_results has no git_patch key (should raise KeyError)"""
        # Create test job details with run results but no git_patch key
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(
            job_id='test_job',
            instance=instance,
            run_results={'status': 'completed', 'other_data': 'value'},
        )

        handler = SweAgentHandler()

        # Call eval method - should raise KeyError due to missing 'git_patch' key
        with pytest.raises(KeyError):
            await handler.eval(job_details=job_details)

        # Verify the mock was not called due to the exception
        mock_evaluate_agent.assert_not_called()

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.evaluate_agent')
    async def test_eval_success_with_empty_git_patch(self, mock_evaluate_agent):
        """Test successful evaluation with run_results containing empty git_patch"""
        # Setup mock return values
        expected_result = {'resolved': False, 'empty_generation': True}
        mock_evaluate_agent.return_value = expected_result

        # Create test job details with empty git_patch in run_results
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(
            job_id='test_job',
            instance=instance,
            run_results={'git_patch': '', 'status': 'completed'},
        )

        handler = SweAgentHandler()

        # Call eval method
        result = await handler.eval(job_details=job_details)

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with empty git_patch
        mock_evaluate_agent.assert_called_once_with(
            git_patch='', instance=instance, sid=None, allow_skip=True
        )

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.evaluate_agent')
    async def test_eval_exception_propagation(self, mock_evaluate_agent):
        """Test that evaluation exceptions are properly propagated"""
        # Setup mock to raise exception
        mock_evaluate_agent.side_effect = RuntimeError('Evaluation failed')

        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(
            job_id='test_job',
            instance=instance,
            run_results={'git_patch': 'test_patch'},
        )

        handler = SweAgentHandler()

        # Verify exception is propagated
        with pytest.raises(RuntimeError, match='Evaluation failed'):
            await handler.eval(job_details=job_details)

    @patch('openhands.nvidia.swe_agent.swe_agent_handler.initialize_exception')
    def test_init_exception_handler(self, mock_initialize_exception):
        """Test initialization exception handler"""
        # Setup mock return values
        expected_result = {'error': 'initialization_error', 'exception': 'test error'}
        mock_initialize_exception.return_value = expected_result

        # Create test data
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(job_id='test_job', instance=instance)
        exception = RuntimeError('Test initialization error')
        handler = SweAgentHandler()

        # Call init_exception method
        result = handler.init_exception(job_details=job_details, exception=exception)

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with correct parameters
        mock_initialize_exception.assert_called_once_with(job_details, exception)

    @patch('openhands.nvidia.swe_agent.swe_agent_handler.run_exception')
    def test_run_exception_handler(self, mock_run_exception):
        """Test run exception handler"""
        # Setup mock return values
        expected_result = {'error': 'run_error', 'exception': 'test error'}
        mock_run_exception.return_value = expected_result

        # Create test data
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(job_id='test_job', instance=instance)
        exception = RuntimeError('Test run error')
        handler = SweAgentHandler()

        # Call run_exception method
        result = handler.run_exception(job_details=job_details, exception=exception)

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with correct parameters
        mock_run_exception.assert_called_once_with(job_details, exception)

    @patch('openhands.nvidia.swe_agent.swe_agent_handler.eval_exception')
    def test_eval_exception_handler(self, mock_eval_exception):
        """Test evaluation exception handler"""
        # Setup mock return values
        expected_result = {'error': 'eval_error', 'exception': 'test error'}
        mock_eval_exception.return_value = expected_result

        # Create test data
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(job_id='test_job', instance=instance)
        exception = RuntimeError('Test eval error')
        handler = SweAgentHandler()

        # Call eval_exception method
        result = handler.eval_exception(job_details=job_details, exception=exception)

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with correct parameters
        mock_eval_exception.assert_called_once_with(job_details, exception)

    @patch('openhands.nvidia.swe_agent.swe_agent_handler.utils_final_result')
    def test_final_result(self, mock_utils_final_result):
        """Test final result processing"""
        # Setup mock return values
        expected_result = {
            'init_time': 10.5,
            'run_time': 30.2,
            'eval_time': 5.1,
            'total_time': 45.8,
            'git_patch': 'test_patch',
        }
        mock_utils_final_result.return_value = expected_result

        # Create test data
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(job_id='test_job', instance=instance)
        handler = SweAgentHandler()

        # Call final_result method
        result = handler.final_result(job_details=job_details)

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with correct parameters
        mock_utils_final_result.assert_called_once_with(job_details)


class TestSweAgentHandlerIntegration:
    """Integration tests for SweAgentHandler using real data"""

    @pytest.mark.real_data
    def test_name_with_real_data(self, real_instance):
        """Test name property works with real data context"""
        handler = SweAgentHandler()
        assert handler.name == 'swebench'
        # Verify this works even with real data loaded
        assert isinstance(real_instance, pd.Series)
        assert 'instance_id' in real_instance

    @pytest.mark.real_data
    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.initialize_agents')
    async def test_init_with_real_instance(
        self, mock_initialize_agents, real_instance, minimal_llm_config
    ):
        """Test initialization with real instance data"""
        # Setup mock return values
        mock_runtime = Mock()
        mock_metadata = Mock()
        mock_config = Mock()
        mock_initialize_agents.return_value = (mock_runtime, mock_metadata, mock_config)

        # Create job details with real instance
        job_details = JobDetails(
            job_id='real_test_job',
            instance=real_instance,
            llm_config=minimal_llm_config,
        )
        handler = SweAgentHandler()

        # Call init method with real instance
        result = await handler.init(
            job_details=job_details,
            sid='real_test_sid',
        )

        # Verify the result
        assert result == (mock_runtime, mock_metadata, mock_config)

        # Verify the mock was called with the real instance
        mock_initialize_agents.assert_called_once_with(
            instance=real_instance,
            llm_config=minimal_llm_config,
            sid='real_test_sid',
            agent_config=_DEFAULT_AGENT_CONFIG,
        )

    @pytest.mark.real_data
    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.evaluate_agent')
    async def test_eval_with_real_instance(
        self, mock_evaluate_agent, real_instance, mock_patch
    ):
        """Test evaluation with real instance data"""
        # Setup mock return values
        expected_result = {'resolved': True, 'passed': 2, 'failed': 1}
        mock_evaluate_agent.return_value = expected_result

        # Create job details with real instance and mock patch
        job_details = JobDetails(
            job_id='real_test_job',
            instance=real_instance,
            run_results={'git_patch': mock_patch, 'status': 'completed'},
        )

        handler = SweAgentHandler()

        # Call eval method
        result = await handler.eval(
            job_details=job_details, sid='real_sid', allow_skip=False
        )

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with real instance
        mock_evaluate_agent.assert_called_once_with(
            git_patch=mock_patch,
            instance=real_instance,
            sid='real_sid',
            allow_skip=False,
        )

    @pytest.mark.real_data
    def test_exception_handlers_with_real_data(self, real_instance):
        """Test exception handlers work with real instance data"""
        handler = SweAgentHandler()
        job_details = JobDetails(job_id='real_job', instance=real_instance)
        test_exception = RuntimeError('Real data test error')

        # Test all exception handlers work
        with patch(
            'openhands.nvidia.swe_agent.swe_agent_handler.initialize_exception'
        ) as mock_init:
            mock_init.return_value = {'error': 'init_error'}
            result = handler.init_exception(job_details, test_exception)
            assert result == {'error': 'init_error'}
            mock_init.assert_called_once_with(job_details, test_exception)

        with patch(
            'openhands.nvidia.swe_agent.swe_agent_handler.run_exception'
        ) as mock_run:
            mock_run.return_value = {'error': 'run_error'}
            result = handler.run_exception(job_details, test_exception)
            assert result == {'error': 'run_error'}
            mock_run.assert_called_once_with(job_details, test_exception)

        with patch(
            'openhands.nvidia.swe_agent.swe_agent_handler.eval_exception'
        ) as mock_eval:
            mock_eval.return_value = {'error': 'eval_error'}
            result = handler.eval_exception(job_details, test_exception)
            assert result == {'error': 'eval_error'}
            mock_eval.assert_called_once_with(job_details, test_exception)


class TestSweAgentHandlerEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.asyncio
    @patch('openhands.nvidia.swe_agent.swe_agent_handler.evaluate_agent')
    async def test_eval_with_none_git_patch_in_results(self, mock_evaluate_agent):
        """Test evaluation when git_patch is explicitly None in run_results"""
        # Setup mock return values
        expected_result = {'resolved': False, 'empty_generation': True}
        mock_evaluate_agent.return_value = expected_result

        # Create test job details with None git_patch
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(
            job_id='test_job',
            instance=instance,
            run_results={'git_patch': None, 'status': 'completed'},
        )

        handler = SweAgentHandler()

        # Call eval method
        result = await handler.eval(job_details=job_details)

        # Verify the result
        assert result == expected_result

        # Verify the mock was called with None (None gets passed through)
        mock_evaluate_agent.assert_called_once_with(
            git_patch=None, instance=instance, sid=None, allow_skip=True
        )

    @pytest.mark.asyncio
    async def test_eval_missing_git_patch_key_raises_keyerror(self):
        """Test that evaluation raises KeyError when git_patch key is missing from run_results"""
        # Create test job details with run_results missing git_patch key
        instance = pd.Series({'instance_id': 'test_instance'})
        job_details = JobDetails(
            job_id='test_job',
            instance=instance,
            run_results={'status': 'completed', 'other_key': 'value'},
        )

        handler = SweAgentHandler()

        # Call eval method - should raise KeyError
        with pytest.raises(KeyError, match='git_patch'):
            await handler.eval(job_details=job_details)

    def test_exception_handlers_with_none_job_details(self):
        """Test exception handlers handle None job_details gracefully"""
        handler = SweAgentHandler()
        test_exception = RuntimeError('Test error')

        # These should not crash even with None job_details
        # (The underlying utils functions should handle this)
        with patch(
            'openhands.nvidia.swe_agent.swe_agent_handler.initialize_exception'
        ) as mock_init:
            mock_init.return_value = {'error': 'init_error'}
            result = handler.init_exception(None, test_exception)
            assert result == {'error': 'init_error'}
            mock_init.assert_called_once_with(None, test_exception)

    def test_inheritance_from_agent_handler(self):
        """Test that SweAgentHandler properly inherits from AgentHandler"""
        from openhands.nvidia.registry import AgentHandler

        handler = SweAgentHandler()
        assert isinstance(handler, AgentHandler)

        # Test that all required abstract methods are implemented
        assert hasattr(handler, 'name')
        assert hasattr(handler, 'init')
        assert hasattr(handler, 'run')
        assert hasattr(handler, 'eval')
        assert hasattr(handler, 'init_exception')
        assert hasattr(handler, 'run_exception')
        assert hasattr(handler, 'eval_exception')
        assert hasattr(handler, 'final_result')

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, minimal_llm_config):
        """Test that multiple handler operations can run concurrently"""
        instance = pd.Series({'instance_id': 'test_instance'})

        # Create multiple handlers to test concurrency
        handlers = [SweAgentHandler() for _ in range(3)]

        # Mock the utils functions
        with (
            patch(
                'openhands.nvidia.swe_agent.swe_agent_handler.initialize_agents'
            ) as mock_init,
            patch(
                'openhands.nvidia.swe_agent.swe_agent_handler.evaluate_agent'
            ) as mock_eval,
        ):
            mock_init.return_value = (Mock(), Mock(), Mock())
            mock_eval.return_value = {'resolved': True}

            # Run multiple init operations concurrently
            job_details_list = [
                JobDetails(
                    job_id=f'test_job_{i}',
                    instance=instance,
                    llm_config=minimal_llm_config,
                )
                for i in range(3)
            ]

            init_tasks = [
                handler.init(
                    job_details=job_details,
                )
                for i, (handler, job_details) in enumerate(
                    zip(handlers, job_details_list)
                )
            ]

            results = await asyncio.gather(*init_tasks)

            # Verify all operations completed
            assert len(results) == 3
            assert mock_init.call_count == 3

            # Test concurrent evaluations
            job_details_list = [
                JobDetails(
                    job_id=f'test_job_{i}',
                    instance=instance,
                    run_results={'git_patch': f'patch_{i}'},
                )
                for i in range(3)
            ]

            eval_tasks = [
                handler.eval(job_details=job_details)
                for handler, job_details in zip(handlers, job_details_list)
            ]

            eval_results = await asyncio.gather(*eval_tasks)

            # Verify all evaluations completed
            assert len(eval_results) == 3
            assert all(result['resolved'] for result in eval_results)
            assert mock_eval.call_count == 3
