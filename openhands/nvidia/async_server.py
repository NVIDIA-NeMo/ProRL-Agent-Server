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
        max_init_workers: int = 6,
        max_run_workers: int = 5,
        max_eval_workers: int | None = None,
        allow_skip_eval: bool = True,
        reward_server_ip: list[str] | None = None,
    ):
        """Create server.

        If *max_eval_workers* is not provided, it defaults to the same value as
        *max_run_workers*, so you only need to specify one number when you want
        these two pools to have the same size.

        allow_skip_eval: if True, skip evaluation if run_results (i.e. git_patch) is None or empty.
        Set to False for testing.
        """
        if llm_server_addresses is None:
            llm_server_addresses = []
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.allow_skip_eval = allow_skip_eval
        # If eval workers not specified, mirror run_workers
        self.max_eval_workers = (
            max_run_workers if max_eval_workers is None else max_eval_workers
        )

        self.init_queue: queue.Queue[str] = queue.Queue()
        self.run_queue: queue.Queue[str] = queue.Queue()
        self.evaluate_queue: queue.Queue[str] = queue.Queue()
        self._init_workers: list[asyncio.AbstractEventLoop | None] = []
        self._run_workers: list[asyncio.AbstractEventLoop | None] = []
        self._eval_workers: list[asyncio.AbstractEventLoop | None] = []
        self._active_init_jobs: set[str] = set()  # Track jobs being initialized
        self._active_run_jobs: set[str] = set()  # Track jobs being run
        self._active_eval_jobs: set[str] = set()  # Track jobs being evaluated
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

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_init_workers
            + self.max_run_workers
            + self.max_eval_workers
        )

        # Initialize worker lists
        self._init_workers = [None] * self.max_init_workers
        self._run_workers = [None] * self.max_run_workers
        self._eval_workers = [None] * self.max_eval_workers

        # Submit init workers
        for i in range(self.max_init_workers):
            self._executor.submit(self._run_worker_in_thread, i, JobType.INIT)

        # Submit run workers
        for i in range(self.max_run_workers):
            self._executor.submit(self._run_worker_in_thread, i, JobType.RUN)

        # Submit evaluation workers
        for i in range(self.max_eval_workers):
            self._executor.submit(self._run_worker_in_thread, i, JobType.EVAL)

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
            self._active_eval_jobs.discard(job_id)
            self._active_run_jobs.discard(job_id)
            self._active_init_jobs.discard(job_id)
            # If job_id is in init/run/eval queue, mark it as discarded
            self._discarded_jobs.add(job_id)

        logger.info(f'Job {job_id} canceled')
        return True

    def process(self, instance, sampling_params, job_id=None, timeout: float = 300.0):
        if not self._server_running:
            raise RuntimeError('Server is not running')

        with self._address_lock:
            if len(self.weighted_addresses) == 0:
                raise ValueError('No LLM server addresses added')

        dataset_type = instance.get('data_source', 'swebench')
        if not is_registered_handler(dataset_type):
            raise FunctionNotRegisteredError(
                f'Dataset type {dataset_type} is not registered'
            )

        # Create job details
        if job_id is None:
            job_id = self.get_unique_id(instance)
        job_details = JobDetails()
        job_details.job_id = job_id
        job_details.instance = instance
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

        # Add job to init queue
        self.init_queue.put(job_id)
        logger.info(f'Job {job_id} added to init queue')

        # Wait for job to be finished
        job_details.event.wait()

        # Get final result
        _final_result_func = get_registered_functions('final_result', dataset_type)
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

    async def _worker(self, wid, job_type: JobType):
        """Unified worker function that handles init, run, and eval jobs based on job_type."""

        # Map job types to their respective queues and tracking sets
        job_config = {
            JobType.INIT: {
                'queue': self.init_queue,
                'active_jobs': self._active_init_jobs,
                'function_type': 'init',
                'exception_type': 'init_exception',
                'worker_name': 'init-worker',
            },
            JobType.RUN: {
                'queue': self.run_queue,
                'active_jobs': self._active_run_jobs,
                'function_type': 'run',
                'exception_type': 'run_exception',
                'worker_name': 'run-worker',
            },
            JobType.EVAL: {
                'queue': self.evaluate_queue,
                'active_jobs': self._active_eval_jobs,
                'function_type': 'eval',
                'exception_type': 'eval_exception',
                'worker_name': 'eval-worker',
            },
        }

        job_config_data = job_config[job_type]
        queue_obj: queue.Queue[str] = cast(queue.Queue[str], job_config_data['queue'])
        active_jobs_set: set[str] = cast(set[str], job_config_data['active_jobs'])
        function_type: str = cast(str, job_config_data['function_type'])
        exception_type: str = cast(str, job_config_data['exception_type'])
        worker_name: str = cast(str, job_config_data['worker_name'])

        while True:
            logger.info(f'[{worker_name}-{wid}] Waiting for job')
            job_id = await asyncio.to_thread(queue_obj.get)

            # Check for stop sentinel
            if job_id == '__STOP__':
                logger.info(f'[{worker_name}-{wid}] Received stop signal, exiting')
                queue_obj.task_done()
                break

            # Thread-safe job details retrieval
            with self._job_details_lock:
                job_details = self._job_details.get(job_id)
                if job_details is None:
                    logger.warning(
                        f'[{worker_name}-{wid}] Job {job_id} not found, skipping'
                    )
                    queue_obj.task_done()
                    continue

            logger.info(f'[{worker_name}-{wid}] Got job {job_id}')

            # Thread-safe active jobs tracking
            with self._state_lock:
                if job_id in self._discarded_jobs:
                    logger.warning(
                        f'[{worker_name}-{wid}] Job {job_id} discarded, skipping'
                    )
                    queue_obj.task_done()
                    self._discarded_jobs.discard(job_id)
                    continue

                active_jobs_set.add(job_id)

            if job_details.instance is None:
                raise RuntimeError('Instance is not initialized')
            dataset_type = job_details.instance.get('data_source', 'swebench')
            func = get_registered_functions(function_type, dataset_type)

            try:
                if func is None:
                    raise FunctionNotRegisteredError(
                        f"Function '{dataset_type}' not found in registry type '{function_type}'"
                    )

                # Enter appropriate phase - all work here counts toward timeout
                if job_details.timer is None:
                    raise RuntimeError('Timer is not initialized')

                with phase_context(job_details.timer, function_type):
                    # Execute the appropriate function based on job type
                    if job_type == JobType.INIT:
                        # Use timeout-aware coroutine execution
                        init_coro = func(
                            job_details=job_details,
                            sid=job_id,
                        )
                        runtime, metadata, config = await run_with_timeout_awareness(
                            job_details.timer, init_coro, job_details
                        )
                        job_details.runtime = runtime
                        job_details.metadata = metadata
                        job_details.config = config
                        # Put in run queue (automatically becomes "others" phase)
                        self.run_queue.put(job_id)

                    elif job_type == JobType.RUN:
                        # Use timeout-aware coroutine execution
                        run_coro = func(
                            job_details=job_details,
                            sid=job_id,
                        )
                        run_results = await run_with_timeout_awareness(
                            job_details.timer, run_coro, job_details
                        )
                        job_details.run_results = run_results
                        # Close runtime (automatically in "others" phase - doesn't count toward timeout)
                        if job_details.runtime:
                            self._cleanup_job_runtime(job_details.runtime, job_id)
                            job_details.runtime = None
                        # Push to evaluation queue (automatically "others" phase)
                        self.evaluate_queue.put(job_id)

                    elif job_type == JobType.EVAL:
                        # Use timeout-aware coroutine execution
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
                        if job_details.event is not None:
                            job_details.event.set()

            except TimeoutError as e:
                logger.warning(
                    f'[{worker_name}-{wid}] Job {job_id} timed out during {function_type}: {e}'
                )
                job_details.timeout_error = True

                # Handle runtime cleanup for run workers
                if job_type == JobType.RUN and job_details.runtime:
                    self._cleanup_job_runtime(job_details.runtime, job_id)
                    job_details.runtime = None

                exception_func = get_registered_functions(exception_type, dataset_type)
                if exception_func is not None:
                    job_details.results = exception_func(job_details, e)
                else:
                    job_details.results = {
                        'error': f'Timeout during {function_type}: {str(e)}',
                        'timeout': True,
                    }
                if job_details.event is not None:
                    job_details.event.set()

            except Exception as e:
                # Handle runtime cleanup for run workers
                if job_type == JobType.RUN and job_details.runtime:
                    self._cleanup_job_runtime(job_details.runtime, job_id)
                    job_details.runtime = None

                exception_func = get_registered_functions(exception_type, dataset_type)
                if exception_func is not None:
                    job_details.results = exception_func(job_details, e)
                else:
                    job_details.results = {
                        'error': f'Exception during {function_type}: {str(e)}'
                    }
                if job_details.event is not None:
                    job_details.event.set()

            finally:
                # Thread-safe cleanup
                with self._state_lock:
                    active_jobs_set.discard(job_id)
                queue_obj.task_done()

    def _run_worker_in_thread(self, worker_id, job_type: JobType):
        """Run a worker in its own thread with its own event loop. Run until the worker is stopped."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            match job_type:
                case JobType.INIT:
                    self._init_workers[worker_id] = loop
                    loop.run_until_complete(self._worker(worker_id, job_type))
                case JobType.RUN:
                    self._run_workers[worker_id] = loop
                    loop.run_until_complete(self._worker(worker_id, job_type))
                case JobType.EVAL:
                    self._eval_workers[worker_id] = loop
                    loop.run_until_complete(self._worker(worker_id, job_type))
        finally:
            # Always close the event loop to prevent resource leaks
            try:
                loop.close()
                logger.debug(
                    f'Event loop closed for {job_type.value} worker {worker_id}'
                )
            except Exception as e:
                logger.warning(
                    f'Error closing event loop for {job_type.value} worker {worker_id}: {e}'
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
            active_jobs = (
                list(self._active_init_jobs)
                + list(self._active_run_jobs)
                + list(self._active_eval_jobs)
            )

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
        # Step 3: Add stop signals to all queues to unblock workers
        logger.info('Sending stop signals to worker queues...')
        for _ in range(self.max_init_workers):
            try:
                self.init_queue.put_nowait('__STOP__')
            except Exception as e:
                logger.warning(f'Failed to put stop signal in init queue: {e}')

        for _ in range(self.max_run_workers):
            try:
                self.run_queue.put_nowait('__STOP__')
            except Exception as e:
                logger.warning(f'Failed to put stop signal in run queue: {e}')

        for _ in range(self.max_eval_workers):
            try:
                self.evaluate_queue.put_nowait('__STOP__')
            except Exception as e:
                logger.warning(f'Failed to put stop signal in eval queue: {e}')
        time.sleep(1)

        # Step 4: Clear all data structures
        logger.info('Clearing internal data structures...')
        try:
            with self._state_lock:
                self._active_init_jobs.clear()
                self._active_run_jobs.clear()
                self._active_eval_jobs.clear()
                self._discarded_jobs.clear()

            with self._job_details_lock:
                self._job_details.clear()

            # Clear worker lists (event loops are handled by thread completion)
            self._init_workers.clear()
            self._run_workers.clear()
            self._eval_workers.clear()
        except Exception as e:
            logger.warning(f'Error clearing data structures: {e}')

        # Step 5: Clear all queues
        logger.info('Clearing all queues...')
        try:
            clear_queue(self.init_queue)
            clear_queue(self.run_queue)
            clear_queue(self.evaluate_queue)
        except Exception as e:
            logger.warning(f'Error clearing queues: {e}')

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
        """Returns the number of jobs currently being processed in both queues and workers."""
        init_queue_count = self.init_queue.qsize()
        run_queue_count = self.run_queue.qsize()
        eval_queue_count = self.evaluate_queue.qsize()

        # Thread-safe status reading
        with self._state_lock:
            active_init_count = len(self._active_init_jobs)
            active_run_count = len(self._active_run_jobs)
            active_eval_count = len(self._active_eval_jobs)

        return {
            'init_queue': init_queue_count,
            'run_queue': run_queue_count,
            'eval_queue': eval_queue_count,
            'active_init': active_init_count,
            'active_run': active_run_count,
            'active_eval': active_eval_count,
            'total': init_queue_count
            + run_queue_count
            + eval_queue_count
            + active_init_count
            + active_run_count
            + active_eval_count,
        }
