import asyncio
import hashlib
import heapq
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Optional, cast

from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.nvidia.registry import (
    FunctionNotRegisteredError,
    JobDetails,
    get_registered_functions,
    is_registered_handler,
)
from openhands.nvidia.reward import Reward
from openhands.nvidia.timer import (
    PausableTimer,
    TimeoutError,
    phase_context,
    run_with_timeout_awareness,
)
from openhands.nvidia.utils import (
    clear_queue,
    get_instance_id,
    get_singularity_job_pids,
    kill_all_singularity_jobs,
)

# add enum for 3 types of jobs


class JobType(Enum):
    INIT = 'init'
    RUN = 'run'
    EVAL = 'eval'


class OpenHandsServer:
    def __init__(
        self,
        llm_server_addresses: list[str] | None = None,
        max_workers: int = 5,
        allow_skip_eval: bool = True,
        reward_server_ip: list[str] | None = None,
    ):
        """Create server.

        max_workers: Maximum number of concurrent jobs across all stages (init/run/eval).

        allow_skip_eval: if True, skip evaluation if run_results (i.e. git_patch) is None or empty.
        Set to False for testing.
        """
        if llm_server_addresses is None:
            llm_server_addresses = []
        self.max_workers = max_workers
        self.allow_skip_eval = allow_skip_eval

        self.job_queue: queue.Queue[str] = queue.Queue()
        self._workers: list[asyncio.AbstractEventLoop | None] = []
        self._active_jobs: set[str] = set()  # Track jobs being processed
        self._discarded_jobs: set[str] = set()  # Track jobs that have been discarded
        self._exclude_pids: set[str] = set()

        self._server_running: bool = False

        # store job detail objects to pass around.
        self._job_details: dict[str, JobDetails] = {}

        self.weighted_addresses = [[0, address] for address in llm_server_addresses]
        heapq.heapify(self.weighted_addresses)

        # THREAD SAFETY: Add locks to protect shared data structures
        self._state_lock = threading.RLock()  # Reentrant lock for active job sets
        self._job_details_lock = threading.RLock()  # Separate lock for job details dict
        self._address_lock = threading.RLock()  # Separate lock for address list

        self.reward: Optional[Reward] = None
        if reward_server_ip is not None:
            logger.info(f'Setting up reward with server IP: {reward_server_ip}')
            self.reward = Reward(server_ip=reward_server_ip)
        else:
            logger.warning(
                'No reward server IP provided. Evaluations would only work for swebench problems.'
            )

        # For sequential VM start-ups to mitigate boot storm
        self._launch_lock = threading.Lock()
        self._last_launch_time = 0
        self._launch_delay_seconds = 15.0  # Wait 15s between starts

    def get_unique_id(self, instance, max_retries=10):
        base = f'{get_instance_id(instance)}_{instance["trajectory_id"]}'
        base_hash = hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]
        for _ in range(max_retries):
            rand = uuid.uuid4().hex[:8]
            uid = f'{base_hash}_{rand}'
            with self._job_details_lock:
                if uid not in self._job_details:
                    return uid
        raise ValueError('Failed to get unique id')

    def add_llm_server_address(self, llm_server_address: str):
        with self._address_lock:
            # Check if address already exists
            for weight, addr in self.weighted_addresses:
                if addr == llm_server_address:
                    logger.warning(
                        f'Warning: LLM server address {llm_server_address} already exists'
                    )
                    return

            heapq.heappush(self.weighted_addresses, [0, llm_server_address])
            logger.info(f'Added LLM server address: {llm_server_address}')

    def clear_llm_server_addresses(self):
        with self._address_lock:
            self.weighted_addresses.clear()
            logger.info('Cleared LLM server addresses')

    def clear_singularity_jobs(self):
        kill_all_singularity_jobs(self._exclude_pids)

    def create_llm_config(self, sampling_params):
        with self._address_lock:
            if len(self.weighted_addresses) == 0:
                raise ValueError('No LLM server addresses added')

            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1  # type: ignore
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        llm_config = LLMConfig(base_url=address, **sampling_params)
        return llm_config

    def _cleanup_job_runtime(self, runtime, job_id: str):
        """Comprehensive cleanup of runtime resources to prevent thread leakage."""

        def close():
            try:
                # 1. Close runtime (handles container processes, plugins, etc.)
                runtime.close()
                # 2. Close event stream and its thread pools
                if hasattr(runtime, 'event_stream') and runtime.event_stream:
                    try:
                        runtime.event_stream.close()
                        logger.debug(f'Event stream closed for job {job_id}')
                    except Exception as e:
                        logger.warning(
                            f'Error closing event stream for job {job_id}: {e}'
                        )
                # 3. Force cleanup any remaining subprocess-related resources
                time.sleep(0.1)  # Brief pause for cleanup to complete

            except Exception as e:
                logger.error(
                    f'Error in comprehensive runtime cleanup for job {job_id}: {e}'
                )
                # Don't re-raise - we want cleanup to continue even if parts fail

        # Run cleanup in background thread, non-blocking
        t = threading.Thread(target=close, daemon=True)
        t.start()

    def start(self):
        if self._server_running:
            raise RuntimeError('Server is already running')
        self._server_running = True

        with self._address_lock:
            self._exclude_pids = get_singularity_job_pids()
            logger.info(f'Excluded Singularity job PIDs: {self._exclude_pids}')

        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Initialize worker list
        self._workers = [None] * self.max_workers

        # Submit workers (each can handle any job type)
        for i in range(self.max_workers):
            self._executor.submit(self._run_worker_in_thread, i)

        self.clear_singularity_jobs()

    def cancel_job(self, job_id: str):
        if not self._server_running:
            raise RuntimeError('Server is not running')

        if job_id not in self._job_details:
            raise ValueError(f'Job {job_id} not found')

        with self._job_details_lock:
            job = self._job_details.get(job_id)
            if job is not None:
                # Mark as timed out and signal completion
                job.timeout_error = True
                if job.event is not None:
                    job.event.set()
                # Clean up runtime if it exists
                if job.runtime:
                    self._cleanup_job_runtime(job.runtime, job_id)
                    job.runtime = None
                # Cancel the current asyncio task if it exists
                if job.current_task is not None and not job.current_task.done():
                    job.current_task.cancel()
                    logger.info(f'Cancelled running task for job {job_id}')
            del self._job_details[job_id]

        with self._state_lock:
            self._active_jobs.discard(job_id)
            # If job_id is in queue, mark it as discarded
            self._discarded_jobs.add(job_id)

        logger.info(f'Job {job_id} canceled')
        return True

    def process(self, instance, sampling_params, job_id=None, timeout: float = 300.0):
        if not self._server_running:
            raise RuntimeError('Server is not running')

        with self._address_lock:
            if len(self.weighted_addresses) == 0:
                raise ValueError('No LLM server addresses added')

        is_reasoning_task = sampling_params.pop('is_reasoning_task', False)
        dataset_type = instance.get('data_source', 'swebench')
        if not is_registered_handler(dataset_type, reasoning=is_reasoning_task):
            raise FunctionNotRegisteredError(
                f'Dataset type {dataset_type} is not registered'
            )

        # Create job details
        if job_id is None:
            job_id = self.get_unique_id(instance)
        job_details = JobDetails()
        job_details.job_id = job_id
        job_details.instance = instance
        job_details.is_reasoning_task = is_reasoning_task
        for agent_config_key in job_details.agent_config:
            if agent_config_key in sampling_params:
                job_details.agent_config[agent_config_key] = sampling_params.pop(
                    agent_config_key
                )
        llm_config = self.create_llm_config(sampling_params)
        job_details.llm_config = llm_config
        job_details.event = threading.Event()

        # Initialize timer - only tracks init/run/eval phases
        # All other time is automatically counted as "others" (not counted toward timeout)
        job_details.timer = PausableTimer(timeout=timeout)
        job_details.timer.start()

        with self._job_details_lock:
            self._job_details[job_id] = job_details
        logger.info(f'Job {job_id} added to job details')

        # Add job to queue
        self.job_queue.put(job_id)
        logger.info(f'Job {job_id} added to job queue')

        # Wait for job to be finished
        job_details.event.wait()

        # Get final result
        _final_result_func = get_registered_functions(
            'final_result', dataset_type, reasoning=is_reasoning_task
        )
        if _final_result_func is None:
            result: dict[str, Any] = {
                'critical_error': 'final_result',
                'error': f'Function not found in registry type final_result for dataset type {dataset_type}',
            }
        else:
            result = _final_result_func(job_details)

        # Add timing information to result
        if job_details.timer:
            timing_info = job_details.timer.get_timing_info()
            result['timing'] = timing_info

        # Close runtime
        if job_details.runtime:
            self._cleanup_job_runtime(job_details.runtime, job_id)
        # Delete job details
        with self._job_details_lock:
            del self._job_details[job_id]
        return result

    async def _worker(self, wid):
        """Worker that processes jobs sequentially through init -> run -> eval."""

        while True:
            logger.info(f'[worker-{wid}] Waiting for job')
            job_id = await asyncio.to_thread(self.job_queue.get)

            # Check for stop sentinel
            if job_id == '__STOP__':
                logger.info(f'[worker-{wid}] Received stop signal, exiting')
                self.job_queue.task_done()
                break

            # Thread-safe job details retrieval
            with self._job_details_lock:
                job_details = self._job_details.get(job_id)
                if job_details is None:
                    logger.warning(f'[worker-{wid}] Job {job_id} not found, skipping')
                    self.job_queue.task_done()
                    continue

            logger.info(f'[worker-{wid}] Got job {job_id}')

            # Thread-safe active jobs tracking
            with self._state_lock:
                if job_id in self._discarded_jobs:
                    logger.warning(f'[worker-{wid}] Job {job_id} discarded, skipping')
                    self.job_queue.task_done()
                    self._discarded_jobs.discard(job_id)
                    continue

                self._active_jobs.add(job_id)

            if job_details.instance is None:
                raise RuntimeError('Instance is not initialized')
            dataset_type = job_details.instance.get('data_source', 'swebench')

            current_stage = 'init'
            try:
                # Process job through all three phases sequentially
                current_stage = 'init'
                await self._process_init(job_id, job_details, dataset_type, wid)
                
                current_stage = 'run'
                await self._process_run(job_id, job_details, dataset_type, wid)
                
                current_stage = 'eval'
                await self._process_eval(job_id, job_details, dataset_type, wid)

            except TimeoutError as e:
                logger.warning(f'[worker-{wid}] Job {job_id} timed out during {current_stage}: {e}')
                job_details.timeout_error = True

                # Get the appropriate exception handler based on current stage
                exception_type = f'{current_stage}_exception'
                exception_func = get_registered_functions(
                    exception_type, dataset_type, reasoning=job_details.is_reasoning_task
                )
                if exception_func is not None:
                    job_details.results = exception_func(job_details, e)
                else:
                    job_details.results = {
                        'error': f'Timeout during {current_stage}: {str(e)}',
                        'timeout': True,
                    }
                if job_details.event is not None:
                    job_details.event.set()
                # Handle runtime cleanup
                if job_details.runtime:
                    self._cleanup_job_runtime(job_details.runtime, job_id)
                    job_details.runtime = None

            except Exception as e:
                logger.error(f'[worker-{wid}] Job {job_id} failed during {current_stage}: {e}')

                # Get the appropriate exception handler based on current stage
                exception_type = f'{current_stage}_exception'
                exception_func = get_registered_functions(
                    exception_type, dataset_type, reasoning=job_details.is_reasoning_task
                )
                if exception_func is not None:
                    job_details.results = exception_func(job_details, e)
                else:
                    job_details.results = {'error': f'Exception during {current_stage}: {str(e)}'}
                if job_details.event is not None:
                    job_details.event.set()
                # Handle runtime cleanup
                if job_details.runtime:
                    self._cleanup_job_runtime(job_details.runtime, job_id)
                    job_details.runtime = None

            finally:
                # Thread-safe cleanup
                with self._state_lock:
                    self._active_jobs.discard(job_id)
                self.job_queue.task_done()

    async def _process_init(self, job_id: str, job_details, dataset_type: str, wid: int):
        """Process the init phase of a job."""
        logger.info(f'[worker-{wid}] Job {job_id} starting init phase')
        func = get_registered_functions(
            'init', dataset_type, reasoning=job_details.is_reasoning_task
        )
        if func is None:
            raise FunctionNotRegisteredError(
                f"Function 'init' not found for dataset type '{dataset_type}'"
            )

        if job_details.timer is None:
            raise RuntimeError('Timer is not initialized')

        # Rate Limit Logic: Prevent Boot Storm
        wait_time = 0.0
        with self._launch_lock:
            now = time.time()
            # The earliest this worker can start is either NOW,
            # or 15s after the last scheduled launch.
            target_start_time = max(now, self._last_launch_time + self._launch_delay_seconds)

            wait_time = target_start_time - now

            # Reserve this slot by updating the global timestamp immediately
            self._last_launch_time = target_start_time

        # Perform the wait asynchronously (outside the lock)
        if wait_time > 0:
            if wait_time > 1.0:
                logger.info(f"Delayed boot-up: waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)

        with phase_context(job_details.timer, 'init'):
            init_coro = func(job_details=job_details, sid=job_id)
            runtime, metadata, config = await run_with_timeout_awareness(
                job_details.timer, init_coro, job_details
            )
            job_details.runtime = runtime
            job_details.metadata = metadata
            job_details.config = config

    async def _process_run(self, job_id: str, job_details, dataset_type: str, wid: int):
        """Process the run phase of a job."""
        logger.info(f'[worker-{wid}] Job {job_id} starting run phase')
        func = get_registered_functions(
            'run', dataset_type, reasoning=job_details.is_reasoning_task
        )
        if func is None:
            raise FunctionNotRegisteredError(
                f"Function 'run' not found for dataset type '{dataset_type}'"
            )

        if job_details.timer is None:
            raise RuntimeError('Timer is not initialized')

        with phase_context(job_details.timer, 'run'):
            run_coro = func(job_details=job_details, sid=job_id)
            run_results = await run_with_timeout_awareness(
                job_details.timer, run_coro, job_details
            )
            job_details.run_results = run_results

        # Close runtime (OSWorld runtime is closed in evaluate function)
        if dataset_type != 'osworld' and job_details.runtime:
            self._cleanup_job_runtime(job_details.runtime, job_id)
            job_details.runtime = None

    async def _process_eval(self, job_id: str, job_details, dataset_type: str, wid: int):
        """Process the eval phase of a job."""
        logger.info(f'[worker-{wid}] Job {job_id} starting eval phase')
        func = get_registered_functions(
            'eval', dataset_type, reasoning=job_details.is_reasoning_task
        )
        if func is None:
            raise FunctionNotRegisteredError(
                f"Function 'eval' not found for dataset type '{dataset_type}'"
            )

        if job_details.timer is None:
            raise RuntimeError('Timer is not initialized')

        with phase_context(job_details.timer, 'eval'):
            eval_coro = func(
                job_details,
                sid=f'eval_{job_id}',
                allow_skip=self.allow_skip_eval,
                reward=self.reward,
            )
            eval_report = await run_with_timeout_awareness(
                job_details.timer, eval_coro, job_details
            )
            # Only keep the 'report' field if present
            if isinstance(eval_report, dict) and 'report' in eval_report:
                job_details.eval_results = eval_report['report']
            else:
                job_details.eval_results = eval_report

        # Close runtime if needed
        if job_details.runtime:
            self._cleanup_job_runtime(job_details.runtime, job_id)
            job_details.runtime = None

        # Signal completion
        if job_details.event is not None:
            job_details.event.set()

    def _run_worker_in_thread(self, worker_id):
        """Run a worker in its own thread with its own event loop. Run until the worker is stopped."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            self._workers[worker_id] = loop
            loop.run_until_complete(self._worker(worker_id))
        finally:
            # Always close the event loop to prevent resource leaks
            try:
                loop.close()
                logger.debug(f'Event loop closed for worker {worker_id}')
            except Exception as e:
                logger.warning(
                    f'Error closing event loop for worker {worker_id}: {e}'
                )

    def stop(self):
        """Stops the server by shutting down all workers and clearing all queues and jobs."""
        if not self._server_running:
            return

        logger.info('Stopping OpenHands server...')

        # Step 1: Set server as not running to prevent new jobs
        self._server_running = False

        # Step 2: Force complete all active jobs with timeout errors
        with self._state_lock:
            active_jobs = list(self._active_jobs)

        logger.info('Signaling active jobs to complete...')
        # Signal all active jobs and mark them with timeout errors
        for job_id in active_jobs:
            try:
                with self._job_details_lock:
                    job = self._job_details.get(job_id)
                    if job is not None:
                        # Mark as timed out and signal completion
                        job.timeout_error = True
                        if job.event is not None:
                            job.event.set()
                        # Clean up runtime if it exists
                        if job.runtime:
                            self._cleanup_job_runtime(job.runtime, job_id)
                            job.runtime = None
            except Exception as e:
                logger.warning(f'Error signaling job {job_id}: {e}')

        time.sleep(1)
        # Step 3: Add stop signals to queue to unblock workers
        logger.info('Sending stop signals to worker queue...')
        for _ in range(self.max_workers):
            try:
                self.job_queue.put_nowait('__STOP__')
            except Exception as e:
                logger.warning(f'Failed to put stop signal in job queue: {e}')
        time.sleep(1)

        # Step 4: Clear all data structures
        logger.info('Clearing internal data structures...')
        try:
            with self._state_lock:
                self._active_jobs.clear()
                self._discarded_jobs.clear()

            with self._job_details_lock:
                self._job_details.clear()

            # Clear worker list (event loops are handled by thread completion)
            self._workers.clear()
        except Exception as e:
            logger.warning(f'Error clearing data structures: {e}')

        # Step 5: Clear the queue
        logger.info('Clearing queue...')
        try:
            clear_queue(self.job_queue)
        except Exception as e:
            logger.warning(f'Error clearing queue: {e}')

        # Step 6: Clean up any remaining singularity jobs
        logger.info('Cleaning up singularity processes...')
        try:
            self.clear_singularity_jobs()
        except Exception as e:
            logger.warning(f'Error cleaning up singularity jobs: {e}')

        logger.info(f'Server stopped. Final status: {self.status()}')

        # Step 7: Shutdown executor and return immediately
        logger.info('Shutting down thread pool executor...')
        if hasattr(self, '_executor') and self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
                logger.info('Thread pool executor shutdown completed')
            except Exception as e:
                logger.warning(f'Error during executor shutdown: {e}')

    def status(self):
        """Returns the number of jobs currently being processed."""
        queue_count = self.job_queue.qsize()

        # Thread-safe status reading
        with self._state_lock:
            active_count = len(self._active_jobs)

        return {
            'queue': queue_count,
            'active': active_count,
            'total': queue_count + active_count,
        }
