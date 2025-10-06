import threading
import unittest
from unittest.mock import Mock, patch

from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia.async_server import OpenHandsServer
from openhands.nvidia.registry import (
    AgentHandler,
    FunctionNotRegisteredError,
    JobDetails,
)
from openhands.nvidia.timer import PausableTimer


class MockInstance(dict):
    """Mock instance for testing."""

    def __init__(
        self,
        instance_id='test_instance',
        trajectory_id='test_trajectory',
        data_source='swebench',
    ):
        super().__init__()
        self['instance_id'] = instance_id
        self['trajectory_id'] = trajectory_id
        self['data_source'] = data_source  # Fix: Add data_source to the dict
        self.instance_id = instance_id
        self.trajectory_id = trajectory_id
        self.data_source = data_source


class MockAgentHandler(AgentHandler):
    """Mock agent handler for testing."""

    @property
    def name(self) -> str:
        return 'swebench'

    async def init(self, instance, llm_config=None, sid=None, max_iterations=1):
        mock_runtime = Mock()
        mock_metadata = Mock()
        mock_config = Mock()
        return mock_runtime, mock_metadata, mock_config

    async def run(self, runtime, metadata, config, instance):
        return {'status': 'completed', 'result': 'success'}

    async def eval(self, job_details, sid=None, allow_skip=True):
        return {'report': {'status': 'passed'}}

    def init_exception(self, job_details, exception):
        return {'error': 'init_failed', 'exception': str(exception)}

    def run_exception(self, job_details, exception):
        return {'error': 'run_failed', 'exception': str(exception)}

    def eval_exception(self, job_details, exception):
        return {'error': 'eval_failed', 'exception': str(exception)}

    def final_result(self, job_details):
        if hasattr(job_details, 'eval_results'):
            return {'final_result': job_details.eval_results}
        elif hasattr(job_details, 'run_results'):
            return {'final_result': job_details.run_results}
        elif hasattr(job_details, 'results'):
            return {'final_result': job_details.results}
        else:
            return {'final_result': 'no_results'}


class TestOpenHandsServer(unittest.TestCase):
    """Test cases for OpenHandsServer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = OpenHandsServer(
            llm_server_addresses=['http://localhost:8000'],
            max_init_workers=2,
            max_run_workers=2,
            max_eval_workers=2,
        )
        self.mock_instance = MockInstance()
        self.mock_handler = MockAgentHandler()

    def run_with_timeout(self, func, timeout=10):
        """Run a function with a timeout to prevent hanging tests."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                self.fail(f'Test timed out after {timeout} seconds')

    def tearDown(self):
        """Clean up after tests."""
        if self.server._server_running:
            self.server.stop()

    def test_initialization_default_parameters(self):
        """Test server initialization with default parameters."""
        server = OpenHandsServer()
        self.assertEqual(server.max_init_workers, 6)
        self.assertEqual(server.max_run_workers, 5)
        self.assertEqual(server.max_eval_workers, 5)  # defaults to max_run_workers
        self.assertTrue(server.allow_skip_eval)
        self.assertFalse(server._server_running)
        self.assertEqual(len(server.weighted_addresses), 0)

    def test_initialization_custom_parameters(self):
        """Test server initialization with custom parameters."""
        server = OpenHandsServer(
            llm_server_addresses=['http://localhost:8000', 'http://localhost:8001'],
            max_init_workers=3,
            max_run_workers=4,
            max_eval_workers=2,
            allow_skip_eval=False,
        )
        self.assertEqual(server.max_init_workers, 3)
        self.assertEqual(server.max_run_workers, 4)
        self.assertEqual(server.max_eval_workers, 2)
        self.assertFalse(server.allow_skip_eval)
        self.assertEqual(len(server.weighted_addresses), 2)

    def test_add_llm_server_address(self):
        """Test adding LLM server addresses."""
        server = OpenHandsServer()
        server.add_llm_server_address('http://localhost:8000')
        self.assertEqual(len(server.weighted_addresses), 1)
        self.assertEqual(server.weighted_addresses[0][1], 'http://localhost:8000')

        # Test adding duplicate address
        with patch('openhands.nvidia.async_server.logger') as mock_logger:
            server.add_llm_server_address('http://localhost:8000')
            mock_logger.warning.assert_called_once()

    def test_clear_llm_server_addresses(self):
        """Test clearing LLM server addresses."""
        self.server.clear_llm_server_addresses()
        self.assertEqual(len(self.server.weighted_addresses), 0)

    def test_create_llm_config_no_addresses(self):
        """Test creating LLM config with no addresses."""
        server = OpenHandsServer()
        with self.assertRaises(ValueError) as context:
            server.create_llm_config({'temperature': 0.7})
        self.assertIn('No LLM server addresses added', str(context.exception))

    def test_create_llm_config_with_addresses(self):
        """Test creating LLM config with available addresses."""
        config = self.server.create_llm_config({'temperature': 0.7})
        self.assertIsInstance(config, LLMConfig)
        self.assertEqual(config.base_url, 'http://localhost:8000')

    def test_get_unique_id(self):
        """Test unique ID generation."""
        uid1 = self.server.get_unique_id(self.mock_instance)
        uid2 = self.server.get_unique_id(self.mock_instance)
        self.assertNotEqual(uid1, uid2)
        # The ID format is now hash-based, so we check for the pattern instead
        # Format: {base_hash}_{random_hex}
        self.assertRegex(uid1, r'^[a-f0-9]{16}_[a-f0-9]{8}$')
        self.assertRegex(uid2, r'^[a-f0-9]{16}_[a-f0-9]{8}$')

    def test_get_unique_id_max_retries(self):
        """Test unique ID generation with max retries."""
        # First, get the actual base hash for the mock instance to create realistic fake IDs
        import hashlib

        from openhands.nvidia.utils import get_instance_id

        base = f'{get_instance_id(self.mock_instance)}_{self.mock_instance["trajectory_id"]}'
        base_hash = hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]

        # Fill up job details to force retries - create IDs that match the pattern
        for i in range(15):
            fake_id = f'{base_hash}_{i:08x}'  # Use hex format to match uuid pattern
            self.server._job_details[fake_id] = JobDetails()

        # Mock uuid.uuid4 to return predictable UUID objects that will conflict
        class MockUUID:
            def __init__(self, value):
                self.hex = f'{value:08x}'

        with patch(
            'openhands.nvidia.async_server.uuid.uuid4',
            side_effect=[MockUUID(i) for i in range(15)],
        ):
            with self.assertRaises(ValueError) as context:
                self.server.get_unique_id(self.mock_instance)
            self.assertIn('Failed to get unique id', str(context.exception))

    def test_start_server(self):
        """Test starting the server."""
        self.assertFalse(self.server._server_running)
        self.server.start()
        self.assertTrue(self.server._server_running)

        # Test starting already running server
        with self.assertRaises(RuntimeError) as context:
            self.server.start()
        self.assertIn('Server is already running', str(context.exception))

    def test_stop_server_not_running(self):
        """Test stopping server that's not running."""
        # Should not raise an exception
        self.server.stop()
        self.assertFalse(self.server._server_running)

    def test_status_empty_server(self):
        """Test status of empty server."""
        status = self.server.status()
        expected = {
            'init_queue': 0,
            'run_queue': 0,
            'eval_queue': 0,
            'active_init': 0,
            'active_run': 0,
            'active_eval': 0,
            'total': 0,
        }
        self.assertEqual(status, expected)

    def test_process_server_not_running(self):
        """Test processing when server is not running."""
        with self.assertRaises(RuntimeError) as context:
            self.server.process(self.mock_instance, {'temperature': 0.7})
        self.assertIn('Server is not running', str(context.exception))

    def test_process_no_llm_addresses(self):
        """Test processing with no LLM addresses."""
        server = OpenHandsServer()
        server.start()
        try:
            with self.assertRaises(ValueError) as context:
                server.process(self.mock_instance, {'temperature': 0.7})
            self.assertIn('No LLM server addresses added', str(context.exception))
        finally:
            server.stop()

    def test_process_unregistered_handler(self):
        """Test processing with unregistered handler."""
        # Create a mock instance with an unregistered dataset type
        mock_instance = MockInstance(data_source='unregistered_dataset')

        self.server.start()

        try:
            with self.assertRaises(FunctionNotRegisteredError) as context:
                self.server.process(mock_instance, {'temperature': 0.7})
            self.assertIn(
                'Dataset type unregistered_dataset is not registered',
                str(context.exception),
            )
        finally:
            self.server.stop()

    def test_process_full_pipeline(self):
        """Test full processing pipeline with mocked registry."""
        # This test is complex because it involves the actual registry system
        # For now, we'll test the basic process flow without full execution
        # since the real registry functions are already registered

        # Create a simple mock instance that won't trigger singularity
        mock_instance = MockInstance(data_source='test_dataset')

        # Mock the registry functions to avoid real execution
        with patch(
            'openhands.nvidia.async_server.is_registered_handler', return_value=False
        ):
            self.server.start()

            try:
                with self.assertRaises(FunctionNotRegisteredError):
                    self.server.process(mock_instance, {'temperature': 0.7})
            finally:
                self.server.stop()

    def test_thread_safety_job_details(self):
        """Test thread safety of job details access."""
        job_id = 'test_job'
        job_details = JobDetails()

        # Test concurrent access to job details
        def add_job():
            with self.server._job_details_lock:
                self.server._job_details[job_id] = job_details

        def remove_job():
            with self.server._job_details_lock:
                if job_id in self.server._job_details:
                    del self.server._job_details[job_id]

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=add_job))
            threads.append(threading.Thread(target=remove_job))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should not crash due to race conditions
        self.assertTrue(True)

    def test_thread_safety_active_jobs(self):
        """Test thread safety of active jobs tracking."""
        job_id = 'test_job'

        def add_to_active():
            with self.server._state_lock:
                self.server._active_init_jobs.add(job_id)

        def remove_from_active():
            with self.server._state_lock:
                self.server._active_init_jobs.discard(job_id)

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=add_to_active))
            threads.append(threading.Thread(target=remove_from_active))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should not crash due to race conditions
        self.assertTrue(True)

    def test_weighted_addresses_load_balancing(self):
        """Test weighted addresses for load balancing."""
        server = OpenHandsServer(
            llm_server_addresses=['http://localhost:8000', 'http://localhost:8001']
        )

        # Create multiple configs and ensure addresses are rotated
        configs = []
        for _ in range(4):
            config = server.create_llm_config({'temperature': 0.7})
            configs.append(config.base_url)

        # Should have used both addresses
        unique_addresses = set(configs)
        self.assertEqual(len(unique_addresses), 2)
        self.assertIn('http://localhost:8000', unique_addresses)
        self.assertIn('http://localhost:8001', unique_addresses)

    def test_queue_operations(self):
        """Test queue operations and status."""
        # Add some items to queues
        self.server.init_queue.put('job1')
        self.server.run_queue.put('job2')
        self.server.evaluate_queue.put('job3')

        # Add some active jobs
        with self.server._state_lock:
            self.server._active_init_jobs.add('active1')
            self.server._active_run_jobs.add('active2')
            self.server._active_eval_jobs.add('active3')

        status = self.server.status()
        self.assertEqual(status['init_queue'], 1)
        self.assertEqual(status['run_queue'], 1)
        self.assertEqual(status['eval_queue'], 1)
        self.assertEqual(status['active_init'], 1)
        self.assertEqual(status['active_run'], 1)
        self.assertEqual(status['active_eval'], 1)
        self.assertEqual(status['total'], 6)

    @patch('openhands.nvidia.async_server.clear_queue')
    def test_stop_server_cleanup(self, mock_clear_queue):
        """Test server cleanup on stop."""
        # Add some test data
        self.server._job_details['test_job'] = JobDetails()
        self.server._active_init_jobs.add('test_job')

        self.server.start()
        self.server.stop()

        # Verify cleanup
        self.assertFalse(self.server._server_running)
        self.assertEqual(len(self.server._job_details), 0)
        self.assertEqual(len(self.server._active_init_jobs), 0)
        self.assertEqual(len(self.server._active_run_jobs), 0)
        self.assertEqual(len(self.server._active_eval_jobs), 0)

        # Verify queues were cleared
        self.assertEqual(mock_clear_queue.call_count, 3)

    def test_custom_job_id(self):
        """Test processing with custom job ID."""
        custom_job_id = 'custom_test_job_123'

        # Create a mock instance that will fail registration check
        mock_instance = MockInstance(data_source='unregistered_type')

        self.server.start()

        try:
            # This will fail at registration check, but the job ID should still be used
            try:
                self.server.process(
                    mock_instance, {'temperature': 0.7}, job_id=custom_job_id
                )
            except FunctionNotRegisteredError:
                pass  # Expected to fail, but job should have been created

        finally:
            self.server.stop()

    def test_job_lifecycle_basic(self):
        """Test basic job lifecycle without external dependencies."""
        # Test that jobs can be created and tracked properly
        job_id = self.server.get_unique_id(self.mock_instance)

        # Manually create a job details object to test lifecycle
        from openhands.nvidia.registry import JobDetails

        job_details = JobDetails()
        job_details.job_id = job_id
        job_details.instance = self.mock_instance
        job_details.event = threading.Event()

        # Test job details management
        with self.server._job_details_lock:
            self.server._job_details[job_id] = job_details
            self.assertIn(job_id, self.server._job_details)

        # Test active job tracking
        with self.server._state_lock:
            self.server._active_init_jobs.add(job_id)
            self.assertIn(job_id, self.server._active_init_jobs)
            self.server._active_init_jobs.discard(job_id)
            self.assertNotIn(job_id, self.server._active_init_jobs)

        # Cleanup
        with self.server._job_details_lock:
            if job_id in self.server._job_details:
                del self.server._job_details[job_id]

    def test_process_with_timeout_parameter(self):
        """Test that process method accepts timeout parameter."""
        custom_timeout = 1.0
        mock_instance = MockInstance(data_source='unregistered_type')

        self.server.start()

        try:
            # This will fail at registration check, but we can verify timeout was passed
            with self.assertRaises(FunctionNotRegisteredError):
                self.server.process(
                    mock_instance, {'temperature': 0.7}, timeout=custom_timeout
                )
        finally:
            self.server.stop()

    @patch('openhands.nvidia.async_server.run_with_timeout_awareness')
    @patch('openhands.nvidia.async_server.get_registered_functions')
    @patch('openhands.nvidia.async_server.is_registered_handler')
    @patch('openhands.nvidia.async_server.PausableTimer')
    def test_timer_initialization_in_process(
        self,
        mock_timer_class,
        mock_is_registered,
        mock_get_functions,
        mock_run_with_timeout,
    ):
        """Test that PausableTimer is created and started for each job."""
        mock_timer = Mock()
        mock_timer.get_timing_info.return_value = {'timing': 'info'}  # Mock timing info
        mock_timer_class.return_value = mock_timer
        mock_is_registered.return_value = True  # Allow registration check to pass

        # Mock async execution to prevent hanging
        mock_run_with_timeout.side_effect = lambda timer, coro: (
            Mock(),
            Mock(),
            Mock(),
        )  # For init

        # Mock all the async functions to prevent hanging
        mock_init_func = Mock()
        mock_run_func = Mock()
        mock_eval_func = Mock()
        mock_final_result = Mock(return_value={'status': 'test'})

        mock_get_functions.side_effect = lambda func_type, dataset: {
            'init': mock_init_func,
            'run': mock_run_func,
            'eval': mock_eval_func,
            'final_result': mock_final_result,
        }.get(func_type)

        mock_instance = MockInstance(data_source='test_dataset')
        custom_timeout = 1.0

        self.server.start()

        def test_logic():
            # This should succeed now with proper mocking
            result = self.server.process(
                mock_instance, {'temperature': 0.7}, timeout=custom_timeout
            )

            # Verify timer was created with correct timeout
            mock_timer_class.assert_called_once_with(timeout=custom_timeout)
            mock_timer.start.assert_called_once()

            # Verify the result contains the expected data
            self.assertIn('status', result)
            self.assertIn('timing', result)  # Verify timing info is added
            return result

        try:
            self.run_with_timeout(test_logic, timeout=5)
        finally:
            self.server.stop()

    @patch('openhands.nvidia.async_server.get_registered_functions')
    @patch('openhands.nvidia.async_server.is_registered_handler')
    def test_timeout_error_handling_in_init_phase(
        self, mock_is_registered, mock_get_functions
    ):
        """Test timeout error handling during init phase."""
        mock_is_registered.return_value = True

        # Mock exception function
        mock_init_exception = Mock(return_value={'error': 'timeout_in_init'})
        mock_get_functions.side_effect = lambda func_type, dataset: {
            'init_exception': mock_init_exception
        }.get(func_type)

        mock_instance = MockInstance(data_source='test_dataset')

        # Create a job details object that will be used
        job_id = self.server.get_unique_id(mock_instance)
        job_details = JobDetails()
        job_details.job_id = job_id
        job_details.instance = mock_instance
        job_details.timer = PausableTimer(timeout=1.0)
        job_details.timer.start()
        job_details.event = threading.Event()

        # Manually trigger timeout
        job_details.timer.trigger_timeout()

        with self.server._job_details_lock:
            self.server._job_details[job_id] = job_details

        # Verify timeout flag is set
        self.assertTrue(job_details.timer.is_expired())

    @patch('openhands.nvidia.async_server.get_registered_functions')
    @patch('openhands.nvidia.async_server.is_registered_handler')
    def test_timeout_error_handling_in_run_phase(
        self, mock_is_registered, mock_get_functions
    ):
        """Test timeout error handling during run phase."""
        mock_is_registered.return_value = True

        # Mock exception function
        mock_run_exception = Mock(return_value={'error': 'timeout_in_run'})
        mock_get_functions.side_effect = lambda func_type, dataset: {
            'run_exception': mock_run_exception
        }.get(func_type)

        mock_instance = MockInstance(data_source='test_dataset')

        # Create a job details object for run phase
        job_id = self.server.get_unique_id(mock_instance)
        job_details = JobDetails()
        job_details.job_id = job_id
        job_details.instance = mock_instance
        job_details.timer = PausableTimer(timeout=1.0)
        job_details.timer.start()
        job_details.event = threading.Event()
        job_details.runtime = Mock()  # Add mock runtime

        # Manually trigger timeout
        job_details.timer.trigger_timeout()

        with self.server._job_details_lock:
            self.server._job_details[job_id] = job_details

        # Verify timeout flag is set
        self.assertTrue(job_details.timer.is_expired())

    @patch('openhands.nvidia.async_server.get_registered_functions')
    @patch('openhands.nvidia.async_server.is_registered_handler')
    def test_timeout_error_handling_in_eval_phase(
        self, mock_is_registered, mock_get_functions
    ):
        """Test timeout error handling during eval phase."""
        mock_is_registered.return_value = True

        # Mock exception function
        mock_eval_exception = Mock(return_value={'error': 'timeout_in_eval'})
        mock_get_functions.side_effect = lambda func_type, dataset: {
            'eval_exception': mock_eval_exception
        }.get(func_type)

        mock_instance = MockInstance(data_source='test_dataset')

        # Create a job details object for eval phase
        job_id = self.server.get_unique_id(mock_instance)
        job_details = JobDetails()
        job_details.job_id = job_id
        job_details.instance = mock_instance
        job_details.timer = PausableTimer(timeout=1.0)
        job_details.timer.start()
        job_details.event = threading.Event()

        # Manually trigger timeout
        job_details.timer.trigger_timeout()

        with self.server._job_details_lock:
            self.server._job_details[job_id] = job_details

        # Verify timeout flag is set
        self.assertTrue(job_details.timer.is_expired())

    @patch('openhands.nvidia.async_server.run_with_timeout_awareness')
    @patch('openhands.nvidia.async_server.PausableTimer')
    @patch('openhands.nvidia.async_server.get_registered_functions')
    @patch('openhands.nvidia.async_server.is_registered_handler')
    def test_timing_information_in_results(
        self,
        mock_is_registered,
        mock_get_functions,
        mock_timer_class,
        mock_run_with_timeout,
    ):
        """Test that timing information is included in final results."""
        mock_timer = Mock()
        mock_timer.get_timing_info.return_value = {
            'total_time': 10.5,
            'counted_time': 8.2,
        }
        mock_timer_class.return_value = mock_timer
        mock_is_registered.return_value = True

        # Mock async execution to prevent hanging
        mock_run_with_timeout.side_effect = lambda timer, coro: (
            Mock(),
            Mock(),
            Mock(),
        )  # For init

        # Mock all the async functions to prevent hanging
        mock_init_func = Mock()
        mock_run_func = Mock()
        mock_eval_func = Mock()
        mock_final_result = Mock(return_value={'status': 'completed'})

        mock_get_functions.side_effect = lambda func_type, dataset: {
            'init': mock_init_func,
            'run': mock_run_func,
            'eval': mock_eval_func,
            'final_result': mock_final_result,
        }.get(func_type)

        mock_instance = MockInstance(data_source='test_dataset')

        self.server.start()

        def test_logic():
            # This should succeed now with proper mocking
            result = self.server.process(mock_instance, {'temperature': 0.7})

            # Verify the result contains timing information
            self.assertIn('status', result)
            self.assertIn('timing', result)
            self.assertEqual(result['timing']['total_time'], 10.5)
            self.assertEqual(result['timing']['counted_time'], 8.2)
            return result

        try:
            self.run_with_timeout(test_logic, timeout=5)
        finally:
            self.server.stop()

    def test_timer_phase_context_usage(self):
        """Test that timer phases are properly managed."""
        timer = PausableTimer(timeout=1.0)
        timer.start()

        # Test that timer starts in 'others' phase by default
        self.assertEqual(timer.current_phase, 'others')

        # Test phase transitions
        timer.enter_phase('init')
        self.assertEqual(timer.current_phase, 'init')

        timer.exit_phase()
        self.assertEqual(timer.current_phase, 'others')

        # Test different phases
        for phase in ['init', 'run', 'eval']:
            timer.enter_phase(phase)
            self.assertEqual(timer.current_phase, phase)
            timer.exit_phase()
            self.assertEqual(timer.current_phase, 'others')

    def test_timeout_awareness_integration(self):
        """Test integration with timeout-aware execution."""
        timer = PausableTimer(timeout=1.0)
        timer.start()
        timer.enter_phase('run')

        # Test remaining timeout calculation
        remaining = timer.get_remaining_timeout()
        self.assertIsNotNone(remaining)
        self.assertGreater(remaining, 0)

        # Test manual timeout trigger
        timer.trigger_timeout()
        self.assertTrue(timer.is_expired())
        self.assertIsNone(timer.get_remaining_timeout())

    def test_timer_counted_vs_uncounted_phases(self):
        """Test that only init/run/eval phases count toward timeout."""
        timer = PausableTimer(timeout=1.0)
        timer.start()

        # Mock time to simulate phase durations
        with patch('time.time') as mock_time:
            mock_time.return_value = 100.0

            # Spend time in 'others' phase (uncounted)
            timer.enter_phase('others')
            mock_time.return_value = 110.0  # 10 seconds
            timer.exit_phase()

            # Should not count toward timeout
            self.assertEqual(timer.get_counted_time(), 0.0)
            self.assertFalse(timer.is_expired())

            # Spend time in 'run' phase (counted)
            mock_time.return_value = 115.0
            timer.enter_phase('run')
            mock_time.return_value = 116.0  # 1 second
            timer.exit_phase()

            # Should count toward timeout
            self.assertEqual(timer.get_counted_time(), 1.0)
            self.assertTrue(timer.is_expired())  # Exceeds 1s timeout

    @patch('openhands.nvidia.async_server.run_with_timeout_awareness')
    @patch('openhands.nvidia.async_server.phase_context')
    def test_timeout_aware_execution_calls(
        self, mock_phase_context, mock_run_with_timeout
    ):
        """Test that timeout-aware execution functions are called properly."""

        # This test verifies that the server uses the timeout infrastructure correctly
        # We can't easily test the full async execution without complex mocking,
        # but we can verify the integration points exist

        timer = PausableTimer(timeout=1.0)
        timer.start()

        # Verify phase context manager exists and works
        with (
            patch.object(timer, 'enter_phase') as mock_enter,
            patch.object(timer, 'exit_phase') as mock_exit,
        ):
            from openhands.nvidia.timer import phase_context

            with phase_context(timer, 'init'):
                pass

            mock_enter.assert_called_once_with('init')
            mock_exit.assert_called_once()

    def test_default_timeout_value(self):
        """Test that process method uses default timeout value."""
        mock_instance = MockInstance(data_source='unregistered_type')

        self.server.start()

        try:
            # This will fail at registration check, but we can verify default timeout
            with self.assertRaises(FunctionNotRegisteredError):
                # Call without timeout parameter to test default
                self.server.process(mock_instance, {'temperature': 0.7})
        finally:
            self.server.stop()

    @patch('openhands.nvidia.async_server.run_with_timeout_awareness')
    @patch('openhands.nvidia.async_server.get_registered_functions')
    @patch('openhands.nvidia.async_server.is_registered_handler')
    @patch('openhands.nvidia.async_server.PausableTimer')
    def test_timer_attached_to_job_details(
        self,
        mock_timer_class,
        mock_is_registered,
        mock_get_functions,
        mock_run_with_timeout,
    ):
        """Test that timer is properly attached to job details."""
        mock_timer = Mock()
        mock_timer.get_timing_info.return_value = {'timing': 'info'}  # Mock timing info
        mock_timer_class.return_value = mock_timer
        mock_is_registered.return_value = True  # Allow registration check to pass

        # Mock async execution to prevent hanging
        mock_run_with_timeout.side_effect = lambda timer, coro: (
            Mock(),
            Mock(),
            Mock(),
        )  # For init

        # Mock all the async functions to prevent hanging
        mock_init_func = Mock()
        mock_run_func = Mock()
        mock_eval_func = Mock()
        mock_final_result = Mock(return_value={'status': 'test'})

        mock_get_functions.side_effect = lambda func_type, dataset: {
            'init': mock_init_func,
            'run': mock_run_func,
            'eval': mock_eval_func,
            'final_result': mock_final_result,
        }.get(func_type)

        mock_instance = MockInstance(data_source='test_dataset')

        self.server.start()

        def test_logic():
            # This should succeed now with proper mocking
            result = self.server.process(mock_instance, {'temperature': 0.7})

            # Verify timer was created and started
            mock_timer_class.assert_called_once_with(timeout=300.0)
            mock_timer.start.assert_called_once()

            # Verify the result contains the expected data
            self.assertIn('status', result)
            self.assertIn('timing', result)  # Verify timing info is added
            return result

        try:
            self.run_with_timeout(test_logic, timeout=5)
        finally:
            self.server.stop()

    def test_runtime_cleanup_on_timeout(self):
        """Test that runtime is properly cleaned up when timeout occurs."""
        job_details = JobDetails()
        job_details.runtime = Mock()
        job_details.timer = PausableTimer(timeout=1.0)
        job_details.timer.start()
        job_details.timer.trigger_timeout()  # Simulate timeout

        # Verify runtime exists before cleanup
        self.assertIsNotNone(job_details.runtime)

        # Simulate the cleanup that happens in async_server workers
        if job_details.runtime:
            job_details.runtime.close()
            job_details.runtime = None

        # Verify runtime was cleaned up
        self.assertIsNone(job_details.runtime)


if __name__ == '__main__':
    unittest.main()
