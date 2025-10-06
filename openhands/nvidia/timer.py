"""
Timer implementation for OpenHands job processing.

Tracks init/run/eval phases for timeout calculation, excludes others.
"""

import asyncio
import time
from contextlib import contextmanager
from typing import Optional

from openhands.nvidia.logger import nvidia_logger as logger


class TimeoutError(Exception):
    """Custom timeout exception for job processing."""

    pass


class PausableTimer:
    """Timer that only counts init/run/eval phases toward timeout."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        self.start_time: Optional[float] = None
        self.is_running: bool = False
        self.current_phase: Optional[str] = None
        self.phase_start_time: Optional[float] = None

        self.is_timeout_triggered: bool = False

        self.phase_times = {
            'init': 0.0,
            'run': 0.0,
            'eval': 0.0,
            'others': 0.0,
        }

    def start(self):
        """Start the timer."""
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True
            self.current_phase = 'others'
            self.phase_start_time = self.start_time

    def trigger_timeout(self):
        """Trigger timeout by setting the timeout flag."""
        if not self.is_timeout_triggered:
            self.is_timeout_triggered = True

    def check_and_trigger_timeout(self):
        """Check if timeout occurred and trigger if needed."""
        if self.is_timeout_triggered:
            return

        if self.get_counted_time() >= self.timeout:
            self.trigger_timeout()

    def get_remaining_timeout(self) -> Optional[float]:
        """Get remaining timeout for counted phases, or None if not in a counted phase."""
        if not self.is_running or self.is_timeout_triggered:
            return None

        # Only return timeout for counted phases
        if self.current_phase not in ['init', 'run', 'eval']:
            return None

        counted_time = self.get_counted_time()
        remaining = self.timeout - counted_time
        return max(0, remaining)  # Never return negative

    def enter_phase(self, phase: str):
        """Enter a new phase (init, run, or eval)."""
        if not self.is_running:
            return

        current_time = time.time()

        if self.current_phase and self.phase_start_time:
            elapsed = current_time - self.phase_start_time
            self.phase_times[self.current_phase] += elapsed

        self.current_phase = phase
        self.phase_start_time = current_time

        # Check timeout when entering a counted phase (immediate detection)
        if phase in ['init', 'run', 'eval']:
            self.check_and_trigger_timeout()

    def exit_phase(self):
        """Exit current phase and return to 'others'."""
        if not self.is_running:
            return

        current_time = time.time()

        if self.current_phase and self.phase_start_time:
            elapsed = current_time - self.phase_start_time
            self.phase_times[self.current_phase] += elapsed

        self.current_phase = 'others'
        self.phase_start_time = current_time

    def get_counted_time(self) -> float:
        """Get time spent in counted phases (init + run + eval)."""
        counted_time = (
            self.phase_times['init']
            + self.phase_times['run']
            + self.phase_times['eval']
        )

        # Add current phase time if it's a counted phase
        if (
            self.is_running
            and self.current_phase in ['init', 'run', 'eval']
            and self.phase_start_time is not None
        ):
            current_time = time.time()
            current_phase_time = current_time - self.phase_start_time
            counted_time += current_phase_time

        return counted_time

    def is_expired(self) -> bool:
        """Check if the timer has expired (based on counted time only)."""
        return self.get_counted_time() >= self.timeout or self.is_timeout_triggered

    def get_timing_info(self) -> dict[str, float]:
        """Get timing information."""
        current_phase_times = self.phase_times.copy()

        if self.is_running and self.current_phase and self.phase_start_time:
            current_time = time.time()
            current_phase_elapsed = current_time - self.phase_start_time
            current_phase_times[self.current_phase] += current_phase_elapsed

        return {
            'init_time': current_phase_times['init'],
            'run_time': current_phase_times['run'],
            'eval_time': current_phase_times['eval'],
            'others_time': current_phase_times['others'],
            'counted_elapsed': self.get_counted_time(),
            'timeout_triggered': self.is_timeout_triggered,
        }


@contextmanager
def phase_context(timer: PausableTimer, phase: str):
    """Context manager for entering and exiting a phase."""
    timer.enter_phase(phase)
    try:
        yield
    finally:
        timer.exit_phase()


async def timeout_aware_phase_context(timer: PausableTimer, phase: str):
    """Async context manager that can be interrupted by timeout."""

    class TimeoutAwareContext:
        def __init__(self, timer: PausableTimer, phase: str):
            self.timer = timer
            self.phase = phase

        async def __aenter__(self):
            self.timer.enter_phase(self.phase)
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.timer.exit_phase()

    return TimeoutAwareContext(timer, phase)


async def run_with_timeout_awareness(timer: PausableTimer, coro, job_details=None):
    """Run a coroutine with timeout awareness using efficient asyncio.wait_for approach."""
    # Get initial remaining timeout
    remaining_timeout = timer.get_remaining_timeout()

    if remaining_timeout is None:
        # Not in a counted phase, this should not happen
        logger.error(
            'Critical error for timer. Not in a counted phase. This should not happen.'
        )
        timer.trigger_timeout()
        raise TimeoutError('Operation timed out')
    elif remaining_timeout <= 0:
        # Already timed out
        timer.trigger_timeout()
        raise TimeoutError('Operation timed out')
    else:
        # Create a task from the coroutine to enable cancellation
        task = asyncio.create_task(coro)

        # Store task reference in job_details for external cancellation
        if job_details is not None:
            job_details.current_task = task

        try:
            # Use asyncio.wait_for with the task
            return await asyncio.wait_for(task, timeout=remaining_timeout)
        except asyncio.TimeoutError:
            # Timeout occurred, cancel the task and trigger our timeout handling
            task.cancel()
            timer.trigger_timeout()
            raise TimeoutError('Operation timed out')
        except asyncio.CancelledError:
            # Task was cancelled externally
            timer.trigger_timeout()
            raise TimeoutError('Operation cancelled')
        finally:
            # Clear task reference
            if job_details is not None:
                job_details.current_task = None
