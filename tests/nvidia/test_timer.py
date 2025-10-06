"""
Unit tests for OpenHands timer module.

Tests cover:
- TimeoutError exception
- PausableTimer class functionality
- Context managers (phase_context and timeout_aware_phase_context)
- Async timeout handling (run_with_timeout_awareness)
"""

import asyncio
from unittest.mock import patch

import pytest

from openhands.nvidia.timer import (
    PausableTimer,
    TimeoutError,
    phase_context,
    run_with_timeout_awareness,
    timeout_aware_phase_context,
)


class TestTimeoutError:
    """Test the custom TimeoutError exception."""

    def test_timeout_error_is_exception(self):
        """Test that TimeoutError is an Exception subclass."""
        assert issubclass(TimeoutError, Exception)

    def test_timeout_error_message(self):
        """Test TimeoutError with message."""
        error = TimeoutError('Test timeout message')
        assert str(error) == 'Test timeout message'

    def test_timeout_error_raising(self):
        """Test raising TimeoutError."""
        with pytest.raises(TimeoutError) as exc_info:
            raise TimeoutError('Operation timed out')
        assert str(exc_info.value) == 'Operation timed out'


class TestPausableTimer:
    """Test the PausableTimer class."""

    def test_init(self):
        """Test PausableTimer initialization."""
        timer = PausableTimer(timeout=30.0)
        assert timer.timeout == 30.0
        assert timer.start_time is None
        assert timer.is_running is False
        assert timer.current_phase is None
        assert timer.phase_start_time is None
        assert timer.is_timeout_triggered is False
        assert timer.phase_times == {
            'init': 0.0,
            'run': 0.0,
            'eval': 0.0,
            'others': 0.0,
        }

    def test_start(self):
        """Test starting the timer."""
        timer = PausableTimer(timeout=30.0)

        with patch('time.time') as mock_time:
            mock_time.return_value = 100.0
            timer.start()

            assert timer.is_running is True
            assert timer.start_time == 100.0
            assert timer.current_phase == 'others'
            assert timer.phase_start_time == 100.0

    def test_start_already_running(self):
        """Test starting timer when already running."""
        timer = PausableTimer(timeout=30.0)
        timer.is_running = True
        timer.start_time = 50.0

        with patch('time.time') as mock_time:
            mock_time.return_value = 100.0
            timer.start()

            # Should not change when already running
            assert timer.start_time == 50.0

    def test_trigger_timeout(self):
        """Test triggering timeout manually."""
        timer = PausableTimer(timeout=30.0)
        assert timer.is_timeout_triggered is False

        timer.trigger_timeout()
        assert timer.is_timeout_triggered is True

        # Should not change if already triggered
        timer.trigger_timeout()
        assert timer.is_timeout_triggered is True

    def test_check_and_trigger_timeout_not_expired(self):
        """Test check_and_trigger_timeout when not expired."""
        timer = PausableTimer(timeout=60.0)
        timer.start()
        timer.enter_phase('run')

        with patch.object(timer, 'get_counted_time') as mock_get_counted:
            mock_get_counted.return_value = 30.0
            timer.check_and_trigger_timeout()

            assert timer.is_timeout_triggered is False

    def test_check_and_trigger_timeout_expired(self):
        """Test check_and_trigger_timeout when expired."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.enter_phase('run')

        with patch.object(timer, 'get_counted_time') as mock_get_counted:
            mock_get_counted.return_value = 35.0
            timer.check_and_trigger_timeout()

            assert timer.is_timeout_triggered is True

    def test_check_and_trigger_timeout_already_triggered(self):
        """Test check_and_trigger_timeout when already triggered."""
        timer = PausableTimer(timeout=30.0)
        timer.is_timeout_triggered = True

        with patch.object(timer, 'get_counted_time') as mock_get_counted:
            timer.check_and_trigger_timeout()
            # Should not call get_counted_time if already triggered
            mock_get_counted.assert_not_called()

    def test_get_remaining_timeout_not_running(self):
        """Test get_remaining_timeout when timer not running."""
        timer = PausableTimer(timeout=30.0)
        assert timer.get_remaining_timeout() is None

    def test_get_remaining_timeout_already_triggered(self):
        """Test get_remaining_timeout when timeout already triggered."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.is_timeout_triggered = True
        assert timer.get_remaining_timeout() is None

    def test_get_remaining_timeout_uncounted_phase(self):
        """Test get_remaining_timeout in uncounted phase."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.current_phase = 'others'
        assert timer.get_remaining_timeout() is None

    def test_get_remaining_timeout_counted_phase(self):
        """Test get_remaining_timeout in counted phase."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.current_phase = 'run'

        with patch.object(timer, 'get_counted_time') as mock_get_counted:
            mock_get_counted.return_value = 10.0
            remaining = timer.get_remaining_timeout()
            assert remaining == 20.0

    def test_get_remaining_timeout_negative_clamp(self):
        """Test get_remaining_timeout clamps negative values to 0."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.current_phase = 'run'

        with patch.object(timer, 'get_counted_time') as mock_get_counted:
            mock_get_counted.return_value = 35.0
            remaining = timer.get_remaining_timeout()
            assert remaining == 0

    def test_enter_phase_not_running(self):
        """Test enter_phase when timer not running."""
        timer = PausableTimer(timeout=30.0)
        timer.enter_phase('run')
        # Should do nothing when not running
        assert timer.current_phase is None

    def test_enter_phase_initial(self):
        """Test entering first phase."""
        timer = PausableTimer(timeout=30.0)
        timer.start()

        with patch('time.time') as mock_time:
            mock_time.return_value = 105.0
            timer.enter_phase('init')

            assert timer.current_phase == 'init'
            assert timer.phase_start_time == 105.0

    def test_enter_phase_transition(self):
        """Test transitioning between phases."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.current_phase = 'others'
        timer.phase_start_time = 100.0

        with patch('time.time') as mock_time:
            mock_time.return_value = 110.0
            timer.enter_phase('run')

            # Should accumulate time for previous phase
            assert timer.phase_times['others'] == 10.0
            assert timer.current_phase == 'run'
            assert timer.phase_start_time == 110.0

    def test_enter_phase_counted_triggers_timeout_check(self):
        """Test entering counted phase triggers timeout check."""
        timer = PausableTimer(timeout=30.0)
        timer.start()

        with patch.object(timer, 'check_and_trigger_timeout') as mock_check:
            timer.enter_phase('run')
            mock_check.assert_called_once()

    def test_enter_phase_uncounted_no_timeout_check(self):
        """Test entering uncounted phase doesn't trigger timeout check."""
        timer = PausableTimer(timeout=30.0)
        timer.start()

        with patch.object(timer, 'check_and_trigger_timeout') as mock_check:
            timer.enter_phase('others')
            mock_check.assert_not_called()

    def test_exit_phase_not_running(self):
        """Test exit_phase when timer not running."""
        timer = PausableTimer(timeout=30.0)
        timer.current_phase = 'run'
        timer.exit_phase()
        # Should do nothing when not running
        assert timer.current_phase == 'run'

    def test_exit_phase(self):
        """Test exiting a phase."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.current_phase = 'run'
        timer.phase_start_time = 100.0

        with patch('time.time') as mock_time:
            mock_time.return_value = 115.0
            timer.exit_phase()

            assert timer.phase_times['run'] == 15.0
            assert timer.current_phase == 'others'
            assert timer.phase_start_time == 115.0

    def test_get_counted_time_basic(self):
        """Test get_counted_time basic functionality."""
        timer = PausableTimer(timeout=30.0)
        timer.phase_times = {
            'init': 5.0,
            'run': 10.0,
            'eval': 3.0,
            'others': 8.0,
        }

        counted_time = timer.get_counted_time()
        assert counted_time == 18.0  # 5 + 10 + 3

    def test_get_counted_time_with_current_counted_phase(self):
        """Test get_counted_time with current counted phase."""
        timer = PausableTimer(timeout=30.0)
        timer.is_running = True
        timer.current_phase = 'run'
        timer.phase_start_time = 100.0
        timer.phase_times = {
            'init': 5.0,
            'run': 10.0,
            'eval': 3.0,
            'others': 8.0,
        }

        with patch('time.time') as mock_time:
            mock_time.return_value = 107.0
            counted_time = timer.get_counted_time()
            # 5 + 10 + 3 + 7 (current run phase)
            assert counted_time == 25.0

    def test_get_counted_time_with_current_uncounted_phase(self):
        """Test get_counted_time with current uncounted phase."""
        timer = PausableTimer(timeout=30.0)
        timer.is_running = True
        timer.current_phase = 'others'
        timer.phase_start_time = 100.0
        timer.phase_times = {
            'init': 5.0,
            'run': 10.0,
            'eval': 3.0,
            'others': 8.0,
        }

        with patch('time.time') as mock_time:
            mock_time.return_value = 107.0
            counted_time = timer.get_counted_time()
            # Should not include current 'others' phase time
            assert counted_time == 18.0

    def test_is_expired_by_counted_time(self):
        """Test is_expired based on counted time."""
        timer = PausableTimer(timeout=30.0)

        with patch.object(timer, 'get_counted_time') as mock_get_counted:
            mock_get_counted.return_value = 35.0
            assert timer.is_expired() is True

            mock_get_counted.return_value = 25.0
            assert timer.is_expired() is False

    def test_is_expired_by_timeout_triggered(self):
        """Test is_expired based on timeout trigger."""
        timer = PausableTimer(timeout=30.0)
        timer.is_timeout_triggered = True

        with patch.object(timer, 'get_counted_time') as mock_get_counted:
            mock_get_counted.return_value = 10.0
            assert timer.is_expired() is True

    def test_get_timing_info_not_running(self):
        """Test get_timing_info when not running."""
        timer = PausableTimer(timeout=30.0)
        timer.phase_times = {
            'init': 5.0,
            'run': 10.0,
            'eval': 3.0,
            'others': 8.0,
        }
        timer.is_timeout_triggered = True

        info = timer.get_timing_info()
        expected = {
            'init_time': 5.0,
            'run_time': 10.0,
            'eval_time': 3.0,
            'others_time': 8.0,
            'counted_elapsed': 18.0,
            'timeout_triggered': True,
        }

        with patch.object(timer, 'get_counted_time') as mock_get_counted:
            mock_get_counted.return_value = 18.0
            info = timer.get_timing_info()
            assert info == expected

    def test_get_timing_info_with_current_phase(self):
        """Test get_timing_info with current active phase."""
        timer = PausableTimer(timeout=30.0)
        timer.is_running = True
        timer.current_phase = 'run'
        timer.phase_start_time = 100.0
        timer.phase_times = {
            'init': 5.0,
            'run': 10.0,
            'eval': 3.0,
            'others': 8.0,
        }

        with (
            patch('time.time') as mock_time,
            patch.object(timer, 'get_counted_time') as mock_get_counted,
        ):
            mock_time.return_value = 107.0
            mock_get_counted.return_value = 25.0

            info = timer.get_timing_info()

            expected = {
                'init_time': 5.0,
                'run_time': 17.0,  # 10 + 7 current
                'eval_time': 3.0,
                'others_time': 8.0,
                'counted_elapsed': 25.0,
                'timeout_triggered': False,
            }
            assert info == expected


class TestPhaseContext:
    """Test the phase_context context manager."""

    def test_phase_context_basic(self):
        """Test basic phase_context usage."""
        timer = PausableTimer(timeout=30.0)
        timer.start()

        with (
            patch.object(timer, 'enter_phase') as mock_enter,
            patch.object(timer, 'exit_phase') as mock_exit,
        ):
            with phase_context(timer, 'run'):
                mock_enter.assert_called_once_with('run')
                mock_exit.assert_not_called()

            mock_exit.assert_called_once()

    def test_phase_context_with_exception(self):
        """Test phase_context with exception."""
        timer = PausableTimer(timeout=30.0)
        timer.start()

        with (
            patch.object(timer, 'enter_phase') as mock_enter,
            patch.object(timer, 'exit_phase') as mock_exit,
        ):
            with pytest.raises(ValueError):
                with phase_context(timer, 'run'):
                    mock_enter.assert_called_once_with('run')
                    raise ValueError('Test exception')

            mock_exit.assert_called_once()


class TestTimeoutAwarePhaseContext:
    """Test the timeout_aware_phase_context async context manager."""

    @pytest.mark.asyncio
    async def test_timeout_aware_phase_context_basic(self):
        """Test basic timeout_aware_phase_context usage."""
        timer = PausableTimer(timeout=30.0)
        timer.start()

        with (
            patch.object(timer, 'enter_phase') as mock_enter,
            patch.object(timer, 'exit_phase') as mock_exit,
        ):
            context_manager = await timeout_aware_phase_context(timer, 'run')

            async with context_manager:
                mock_enter.assert_called_once_with('run')
                mock_exit.assert_not_called()

            mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_aware_phase_context_with_exception(self):
        """Test timeout_aware_phase_context with exception."""
        timer = PausableTimer(timeout=30.0)
        timer.start()

        with (
            patch.object(timer, 'enter_phase') as mock_enter,
            patch.object(timer, 'exit_phase') as mock_exit,
        ):
            with pytest.raises(ValueError):
                context_manager = await timeout_aware_phase_context(timer, 'run')
                async with context_manager:
                    mock_enter.assert_called_once_with('run')
                    raise ValueError('Test exception')

            mock_exit.assert_called_once()


class TestRunWithTimeoutAwareness:
    """Test the run_with_timeout_awareness async function."""

    @pytest.mark.asyncio
    async def test_run_with_timeout_awareness_success(self):
        """Test successful execution with timeout awareness."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.enter_phase('run')

        async def dummy_coro():
            await asyncio.sleep(0.01)
            return 'success'

        with patch.object(timer, 'get_remaining_timeout') as mock_remaining:
            mock_remaining.return_value = 10.0

            result = await run_with_timeout_awareness(timer, dummy_coro())
            assert result == 'success'

    @pytest.mark.asyncio
    async def test_run_with_timeout_awareness_no_remaining_timeout(self):
        """Test when no remaining timeout available."""
        timer = PausableTimer(timeout=30.0)
        timer.start()

        async def dummy_coro():
            return 'success'

        with (
            patch.object(timer, 'get_remaining_timeout') as mock_remaining,
            patch.object(timer, 'trigger_timeout') as mock_trigger,
        ):
            mock_remaining.return_value = None

            with pytest.raises(TimeoutError, match='Operation timed out'):
                await run_with_timeout_awareness(timer, dummy_coro())

            mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_timeout_awareness_zero_remaining(self):
        """Test when remaining timeout is zero."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.enter_phase('run')

        async def dummy_coro():
            return 'success'

        with (
            patch.object(timer, 'get_remaining_timeout') as mock_remaining,
            patch.object(timer, 'trigger_timeout') as mock_trigger,
        ):
            mock_remaining.return_value = 0

            with pytest.raises(TimeoutError, match='Operation timed out'):
                await run_with_timeout_awareness(timer, dummy_coro())

            mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_timeout_awareness_timeout_occurs(self):
        """Test when asyncio timeout occurs."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.enter_phase('run')

        async def slow_coro():
            await asyncio.sleep(1.0)  # Longer than timeout
            return 'success'

        with (
            patch.object(timer, 'get_remaining_timeout') as mock_remaining,
            patch.object(timer, 'trigger_timeout') as mock_trigger,
        ):
            mock_remaining.return_value = 0.01  # Very short timeout

            with pytest.raises(TimeoutError, match='Operation timed out'):
                await run_with_timeout_awareness(timer, slow_coro())

            mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_timeout_awareness_coro_exception(self):
        """Test when coroutine raises an exception."""
        timer = PausableTimer(timeout=30.0)
        timer.start()
        timer.enter_phase('run')

        async def failing_coro():
            raise ValueError('Coroutine failed')

        with patch.object(timer, 'get_remaining_timeout') as mock_remaining:
            mock_remaining.return_value = 10.0

            with pytest.raises(ValueError, match='Coroutine failed'):
                await run_with_timeout_awareness(timer, failing_coro())


class TestIntegrationScenarios:
    """Integration tests for realistic timer usage scenarios."""

    def test_full_workflow_simulation(self):
        """Test a complete workflow with multiple phases."""
        timer = PausableTimer(timeout=10.0)
        timer.start()

        # Simulate init phase
        with patch('time.time') as mock_time:
            mock_time.return_value = 100.0
            timer.enter_phase('init')

            mock_time.return_value = 103.0
            timer.exit_phase()  # 3 seconds in init

            # Simulate run phase
            mock_time.return_value = 105.0
            timer.enter_phase('run')

            mock_time.return_value = 110.0
            timer.exit_phase()  # 5 seconds in run

            # Check that timer correctly tracks time
            assert timer.phase_times['init'] == 3.0
            assert timer.phase_times['run'] == 5.0
            assert timer.get_counted_time() == 8.0
            assert timer.is_expired() is False

    def test_timeout_prevention_in_uncounted_phase(self):
        """Test that uncounted phases don't contribute to timeout."""
        timer = PausableTimer(timeout=5.0)
        timer.start()

        with patch('time.time') as mock_time:
            # Spend 10 seconds in 'others' phase (uncounted)
            mock_time.return_value = 100.0
            timer.enter_phase('others')

            mock_time.return_value = 110.0
            timer.exit_phase()

            # Should not be expired yet
            assert timer.is_expired() is False

            # Now spend 6 seconds in counted phase
            mock_time.return_value = 115.0
            timer.enter_phase('run')

            mock_time.return_value = 121.0
            assert timer.get_counted_time() == 6.0
            assert timer.is_expired() is True

    @pytest.mark.asyncio
    async def test_async_workflow_with_context(self):
        """Test async workflow using context managers."""
        timer = PausableTimer(timeout=5.0)
        timer.start()

        async def simulate_work():
            await asyncio.sleep(0.01)
            return 'completed'

        # Test with timeout_aware_phase_context
        with patch.object(timer, 'get_remaining_timeout') as mock_remaining:
            mock_remaining.return_value = 2.0

            context_manager = await timeout_aware_phase_context(timer, 'run')
            async with context_manager:
                result = await run_with_timeout_awareness(timer, simulate_work())
                assert result == 'completed'

    def test_manual_timeout_trigger(self):
        """Test manual timeout triggering overrides timer logic."""
        timer = PausableTimer(timeout=100.0)  # Long timeout
        timer.start()
        timer.enter_phase('run')

        # Manually trigger timeout
        timer.trigger_timeout()

        # Should be expired regardless of actual time
        assert timer.is_expired() is True
        assert timer.get_remaining_timeout() is None
