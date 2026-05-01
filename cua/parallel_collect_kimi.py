"""
Parallel trajectory collection using Kimi-K2.5.

Same two-stage queue architecture as parallel_collect_trajectories.py:
  - Init workers: boot VMs with rate limiting
  - Collect workers: run flat goal→action loop on initialized VMs

Usage:
    python parallel_collect_kimi.py \
        --model_node pool0-03161 \
        --max_parallel 24 \
        --max_trajectories 10000
"""
import argparse
import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List

from modules_kimi.data_collector import DataCollector
from openhands.core.logger import openhands_logger

# Configure logging
openhands_logger.setLevel(logging.WARNING)
logger = openhands_logger.getChild('kimi_parallel_collector')
logger.setLevel(logging.INFO)


class TrajectoryJobDetails:
    """
    Tracks the state of a specific trajectory generation job (Index 0...N).
    Pre-allocated to ensure we never lose track of a job, even if it fails early.
    """

    def __init__(self, index: int):
        self.index = index
        self.job_id: Optional[str] = None
        self.trajectory_id: Optional[str] = None

        # Concurrency primitives
        self.event = threading.Event()  # Set when job is totally done (success or fail)

        # Result state
        self.completed: bool = False
        self.error: Optional[str] = None

        # Runtime objects (populated during execution)
        self.runtime: Any = None
        self.trajectory_data: Optional[Dict] = None
        self.save_dir: Any = None
        self.osworld_setup: Any = None


class ParallelTrajectoryGenerator:
    def __init__(self, args, data_collector: DataCollector):
        self.data_collector = data_collector
        self.max_parallel = args.max_parallel
        self.max_trajectories = args.max_trajectories
        self.runtime_type = getattr(args, 'runtime', 'singularity')

        # Queues
        self.init_queue: queue.Queue = queue.Queue()
        self.collect_queue: queue.Queue = queue.Queue()

        # Concurrency Control
        self._runtime_semaphore = threading.Semaphore(self.max_parallel)
        self._active_runtime_count = 0
        self._active_runtime_lock = threading.Lock()

        # Job Tracking: Pre-allocate list of all jobs
        self.jobs: List[TrajectoryJobDetails] = [
            TrajectoryJobDetails(i) for i in range(self.max_trajectories)
        ]

        self._executor: Optional[ThreadPoolExecutor] = None
        self._server_running = False

        # For sequential VM start-ups to mitigate boot storm
        self._launch_lock = threading.Lock()
        self._last_launch_time = 0
        self._launch_delay_seconds = 0.0 if self.runtime_type == "nvcf" else 15.0

    def start_workers(self):
        self._server_running = True
        # init + collect workers
        self._executor = ThreadPoolExecutor(max_workers=self.max_parallel * 2)

        logger.info(f"Starting {self.max_parallel} init workers and {self.max_parallel} collect workers...")
        for i in range(self.max_parallel):
            self._executor.submit(self._run_worker_in_thread, self._init_worker, i, "init")
            self._executor.submit(self._run_worker_in_thread, self._collect_worker, i, "collect")

    def stop_workers(self):
        self._server_running = False
        # Send stop signals
        for _ in range(self.max_parallel * 2):
            self.init_queue.put(None)
            self.collect_queue.put(None)

        if self._executor:
            self._executor.shutdown(wait=True)
            logger.info("Worker pool shutdown complete.")

    @staticmethod
    def _run_worker_in_thread(worker_func, worker_id, name):
        """Run an async worker in a dedicated thread with its own loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(worker_func(worker_id))
        except Exception as e:
            logger.error(f"Worker {name}-{worker_id} crashed: {e}")
        finally:
            loop.close()

    async def _init_worker(self, worker_id: int):
        logger.info(f"[init-{worker_id}] Started")
        while True:
            job_idx = await asyncio.to_thread(self.init_queue.get)
            if job_idx is None: break

            job_details = self.jobs[job_idx]

            # Wait for available runtime slot
            await asyncio.to_thread(self._runtime_semaphore.acquire)

            # Rate Limit Logic: Prevent Boot Storm
            wait_time = 0.0
            with self._launch_lock:
                now = time.time()
                target_start_time = max(now, self._last_launch_time + self._launch_delay_seconds)
                wait_time = target_start_time - now
                self._last_launch_time = target_start_time

            if wait_time > 0:
                if wait_time > 1.0:
                    logger.info(f"[init-{worker_id}] Delayed boot-up: waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

            with self._active_runtime_lock:
                self._active_runtime_count += 1

            logger.info(f"[init-{worker_id}] Slot acquired. Active: {self._active_runtime_count}")

            try:
                runtime, traj_data, save_dir, traj_id, setup = \
                    await self.data_collector.init_runtime_for_job(job_idx)

                job_details.job_id = traj_id
                job_details.trajectory_id = traj_id
                job_details.runtime = runtime
                job_details.trajectory_data = traj_data
                job_details.save_dir = save_dir
                job_details.osworld_setup = setup

                # Hand off to collect queue
                self.collect_queue.put(job_idx)

            except Exception as e:
                logger.error(f"[init-{worker_id}] Failed setup for job {job_idx}: {e}")
                job_details.error = str(e)
                job_details.completed = False
                job_details.event.set()

                if job_details.runtime:
                    try:
                        job_details.runtime.close()
                    except Exception:
                        pass

                self._runtime_semaphore.release()
                with self._active_runtime_lock:
                    self._active_runtime_count -= 1
            finally:
                self.init_queue.task_done()

    async def _collect_worker(self, worker_id: int):
        logger.info(f"[collect-{worker_id}] Started")
        while True:
            job_idx = await asyncio.to_thread(self.collect_queue.get)
            if job_idx is None: break

            job_details = self.jobs[job_idx]
            traj_id = job_details.trajectory_id

            try:
                logger.info(f"[collect-{worker_id}] Processing {traj_id}")

                await self.data_collector.collect_trajectory(
                    job_details.runtime,
                    job_details.trajectory_data,
                    job_details.save_dir,
                    job_details.osworld_setup
                )

                job_details.completed = True

            except Exception as e:
                logger.error(f"[collect-{worker_id}] Error in {traj_id}: {e}")
                job_details.error = str(e)
            finally:
                if job_details.runtime:
                    threading.Thread(target=job_details.runtime.close, daemon=True).start()

                self._runtime_semaphore.release()
                with self._active_runtime_lock:
                    self._active_runtime_count -= 1

                job_details.event.set()
                self.collect_queue.task_done()
                logger.info(f"[collect-{worker_id}] Finished {traj_id}. Released slot.")

    async def run(self):
        self.start_workers()

        logger.info(f"Submitting {self.max_trajectories} jobs...")
        for i in range(self.max_trajectories):
            self.init_queue.put(i)

        logger.info("Monitoring jobs...")

        start_time = time.time()
        while True:
            completed_count = 0
            failed_count = 0

            for job in self.jobs:
                if job.event.is_set():
                    if job.error:
                        failed_count += 1
                    else:
                        completed_count += 1

            total_done = completed_count + failed_count

            elapsed = time.time() - start_time
            if total_done > 0:
                rate = elapsed / total_done
                eta = rate * (self.max_trajectories - total_done)
                eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
            else:
                eta_str = "Calculating..."

            msg = (f"Progress: {total_done}/{self.max_trajectories} "
                   f"(Success: {completed_count}, Failed: {failed_count}) "
                   f"Active VMs: {self._active_runtime_count} | ETA: {eta_str}")

            print(f"\r{msg}", end="", flush=True)

            if total_done >= self.max_trajectories:
                print("\n\nAll jobs completed.")
                break

            await asyncio.sleep(2.0)

        self.stop_workers()


def parse_args():
    parser = argparse.ArgumentParser(description="Kimi-K2.5 parallel trajectory collection")

    # Kimi vLLM server
    parser.add_argument("--model_node", type=str, required=True,
                        help="Hostname of the Kimi vLLM server head node")
    parser.add_argument("--kimi_model_name", type=str,
                        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Kimi-K2.5")

    # Runtime selection
    parser.add_argument("--runtime", type=str, choices=["singularity", "nvcf", "nvcf_singularity"], default="singularity",
                        help="Runtime backend: 'singularity' (local KVM), 'nvcf' (NVCF cloud), or 'nvcf_singularity' (local .sif)")

    # Generation mode
    parser.add_argument("--generation_mode", type=str, default="vanilla",
                        choices=["vanilla", "spreadsheetbench", "zenodo"],
                        help="Data generation mode")

    # Environment & Setup
    parser.add_argument("--vm_image_path", type=str,
                        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/OS_images/Ubuntu.qcow2")
    parser.add_argument("--trajectory_save_dir", type=str, default=None,
                        help="Output directory (default: auto based on generation_mode)")

    # Steps
    parser.add_argument("--max_steps_per_trajectory", type=int, default=100)

    # Parallel specific args
    parser.add_argument("--max_parallel", type=int, default=24, help="Max concurrent VMs")
    parser.add_argument("--max_trajectories", type=int, default=10000,
                        help="Total trajectories to generate")
    parser.add_argument("--project_dir", type=str, required=True,
                        help="Project directory (e.g. /path/to/ProRL-Agent-Server/cua)")
    parser.add_argument("--timeout", type=int, default=14400,
                        help="Global timeout in seconds (default: 14400 = 4 hours)")

    args = parser.parse_args()

    PROJECT_DIR = args.project_dir

    if args.generation_mode == "vanilla":
        args.persona_dataset_path = "/lustre/fsw/portfolios/nvr/users/yidong/data/nemotron_data/data/"
        args.example_instructions_path = f"{PROJECT_DIR}/data/agentnet/processed/instructions.txt"
        args.osworld_setup_path = "/lustre/fsw/portfolios/nvr/users/mingjiel/data/osworld/osworld_test_nogdrive.json"
        if args.trajectory_save_dir is None:
            args.trajectory_save_dir = f"{PROJECT_DIR}/trajectories/kimi"
    elif args.generation_mode == "spreadsheetbench":
        args.persona_dataset_path = None
        args.example_instructions_path = f"{PROJECT_DIR}/data/agentnet/processed/instructions.txt"
        args.osworld_setup_path = f"{PROJECT_DIR}/data/custom_configs/spreadsheetbench/osworld_setup_configs.jsonl"
        if args.trajectory_save_dir is None:
            args.trajectory_save_dir = f"{PROJECT_DIR}/trajectories/kimi_spreadsheetbench"
    elif args.generation_mode == "zenodo":
        args.persona_dataset_path = None
        args.example_instructions_path = f"{PROJECT_DIR}/data/agentnet/processed/instructions.txt"
        args.osworld_setup_path = f"{PROJECT_DIR}/data/custom_configs/zenodo/osworld_setup_configs.jsonl"
        if args.trajectory_save_dir is None:
            args.trajectory_save_dir = f"{PROJECT_DIR}/trajectories/kimi_zenodo"

    return args


async def main():
    args = parse_args()

    data_collector = DataCollector(args)
    logger.info("DataCollector initialized (datasets loaded)")

    generator = ParallelTrajectoryGenerator(args, data_collector)

    try:
        await asyncio.wait_for(generator.run(), timeout=args.timeout)
    except asyncio.TimeoutError:
        logger.info(f"Global timeout reached ({args.timeout}s). Shutting down...")
    finally:
        generator.stop_workers()


if __name__ == "__main__":
    asyncio.run(main())
