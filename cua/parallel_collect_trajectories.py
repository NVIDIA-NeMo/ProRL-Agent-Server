import argparse
import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List

from modules.module_data_collector import DataCollector
from openhands.core.logger import openhands_logger

# Configure logging
openhands_logger.setLevel(logging.DEBUG)
logger = openhands_logger.getChild('parallel_collector')
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
        self.nvcf_function_id: Optional[str] = None
        self.nvcf_version_id: Optional[str] = None


class ParallelTrajectoryGenerator:
    def __init__(self, args, data_collector: DataCollector, nvcf_pool=None):
        self.data_collector = data_collector
        self.max_parallel = args.max_parallel
        self.max_trajectories = args.max_trajectories
        self.nvcf_pool = nvcf_pool  # Optional NVCFPool for NVCF runtime
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
        # NVCF uses pre-deployed VMs so no boot storm delay needed
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
            # Blocking get from queue (thread-safe)
            job_idx = await asyncio.to_thread(self.init_queue.get)
            if job_idx is None: break

            job_details = self.jobs[job_idx]

            # Wait for available runtime slot
            # logger.debug(f"[init-{worker_id}] Waiting for slot for job {job_idx}")
            await asyncio.to_thread(self._runtime_semaphore.acquire)

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
                    logger.info(f"[init-{worker_id}] Delayed boot-up: waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

            with self._active_runtime_lock:
                self._active_runtime_count += 1

            logger.info(f"[init-{worker_id}] Slot acquired. Active: {self._active_runtime_count}")

            try:
                # For NVCF, acquire a function from the warm pool
                nvcf_fn_id, nvcf_ver_id = None, None
                if self.nvcf_pool:
                    nvcf_fn_id, nvcf_ver_id = await asyncio.to_thread(self.nvcf_pool.acquire)
                    job_details.nvcf_function_id = nvcf_fn_id
                    job_details.nvcf_version_id = nvcf_ver_id
                    logger.info(f"[init-{worker_id}] Acquired NVCF function {nvcf_fn_id} for job {job_idx}")

                # --- call init_runtime_for_job --- #
                # This creates the runtime and runs setup
                runtime, traj_data, save_dir, traj_id, setup = \
                    await self.data_collector.init_runtime_for_job(
                        job_idx,
                        nvcf_function_id=nvcf_fn_id,
                        nvcf_version_id=nvcf_ver_id,
                    )

                # Store details in the pre-allocated object
                job_details.job_id = traj_id  # Using traj_id as primary ID
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
                job_details.completed = False  # Failed
                job_details.event.set()  # Signal main thread we are done (failed)

                # Close runtime if it was stored (stops keepalive thread, proxies, etc.)
                if job_details.runtime:
                    try:
                        job_details.runtime.close()
                    except Exception:
                        pass

                # Release NVCF function back to pool on failure (health-check first)
                if self.nvcf_pool and job_details.nvcf_function_id:
                    self.nvcf_pool.release_or_replace(job_details.nvcf_function_id, job_details.nvcf_version_id)

                # Release semaphore immediately on failure
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

                # --- call collect_trajectory ---
                # This runs the Planner/Actor loop
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
                # Cleanup Runtime (close HTTP client, but don't undeploy NVCF function)
                if job_details.runtime:
                    threading.Thread(target=job_details.runtime.close, daemon=True).start()

                # Release NVCF function back to pool for reuse
                # If the job errored, health-check first to avoid returning a broken function
                if self.nvcf_pool and job_details.nvcf_function_id:
                    if job_details.error:
                        self.nvcf_pool.release_or_replace(job_details.nvcf_function_id, job_details.nvcf_version_id)
                    else:
                        self.nvcf_pool.release(job_details.nvcf_function_id, job_details.nvcf_version_id)

                # Release semaphore (allows new Init worker to proceed)
                self._runtime_semaphore.release()
                with self._active_runtime_lock:
                    self._active_runtime_count -= 1

                # Signal completion
                job_details.event.set()

                self.collect_queue.task_done()
                logger.info(f"[collect-{worker_id}] Finished {traj_id}. Released slot.")

    async def run(self):
        self.start_workers()

        logger.info(f"Submitting {self.max_trajectories} jobs...")
        # Submit jobs
        for i in range(self.max_trajectories):
            self.init_queue.put(i)

        logger.info("Monitoring jobs...")

        start_time = time.time()
        while True:
            completed_count = 0
            failed_count = 0

            # Check all jobs
            for job in self.jobs:
                if job.event.is_set():
                    if job.error:
                        failed_count += 1
                    else:
                        completed_count += 1

            total_done = completed_count + failed_count

            # Progress reporting
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

            # Overwrite line in terminal (simple progress bar effect)
            print(f"\r{msg}", end="", flush=True)

            # Exit condition
            if total_done >= self.max_trajectories:
                print("\n\nAll jobs completed.")
                break

            await asyncio.sleep(2.0)

        self.stop_workers()


def parse_args():
    parser = argparse.ArgumentParser()
    # Nodes
    parser.add_argument("--planner_node", type=str, required=True)
    parser.add_argument("--actor_node", type=str, required=True)

    # Runtime selection
    parser.add_argument("--runtime", type=str, choices=["singularity", "nvcf"], default="singularity",
                        help="Runtime backend: 'singularity' (local KVM) or 'nvcf' (NVIDIA Cloud Functions)")

    # NVCF-specific args
    parser.add_argument("--nvcf_api_key", type=str, default=None,
                        help="NGC API key (or set NGC_API_KEY env var)")
    parser.add_argument("--nvcf_org", type=str, default=None,
                        help="NGC org name (or set NGC_ORG env var)")

    # Environment & Setup
    parser.add_argument("--vm_image_path", type=str,
                        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/OS_images/Ubuntu.qcow2")
    parser.add_argument("--persona_dataset_path", type=str,
                        default="/lustre/fsw/portfolios/nvr/users/yidong/data/nemotron_data/data/")
    parser.add_argument("--example_instructions_path", type=str,
                        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/cua/data/agentnet/processed/instructions.txt")
    parser.add_argument("--osworld_setup_path", type=str,
                        default="/lustre/fsw/portfolios/nvr/users/mingjiel/data/osworld/osworld_test_nogdrive.json")

    # Steps
    parser.add_argument("--max_steps_per_trajectory", type=int, default=150)
    parser.add_argument("--max_steps_per_goal", type=int, default=10)

    # Models
    parser.add_argument("--planner_model_name", type=str,
                        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking")
    parser.add_argument("--actor_model_name", type=str, default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--min_pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=5120 * 28 * 28)
    parser.add_argument("--max_retry_for_goal_generation", type=int, default=1)
    parser.add_argument("--max_retry_for_action_generation", type=int, default=3)

    # Parallel specific args
    parser.add_argument("--max_parallel", type=int, default=24, help="Max concurrent VMs")
    parser.add_argument(
        "--max_trajectories", type=int, default=10000, help="Total trajectories to generate")
    parser.add_argument(
        "--num_vms_per_instance", type=int, default=1,
        help="Number of VM instances per NVCF function (subdivides one machine into multiple VMs)")
    

    return parser.parse_args()


async def main():
    args = parse_args()

    # 1. Initialize DataCollector ONCE (loads datasets)
    data_collector = DataCollector(args)
    logger.info("DataCollector initialized (datasets loaded)")

    # 2. Set up NVCF pool if using NVCF runtime
    nvcf_pool = None
    if args.runtime == "nvcf":
        import os
        from modules.nvcf_pool import NVCFPool

        api_key = args.nvcf_api_key or os.environ.get("NGC_API_KEY")
        org = args.nvcf_org or os.environ.get("NGC_ORG")
        if not api_key:
            raise ValueError("NGC_API_KEY required for NVCF runtime. Set via --nvcf_api_key or NGC_API_KEY env var.")
        if not org:
            raise ValueError("NGC_ORG required for NVCF runtime. Set via --nvcf_org or NGC_ORG env var.")

        nvcf_pool = NVCFPool(
            pool_size=args.max_parallel,
            num_vms_per_instance=args.num_vms_per_instance,
            nvcf_api_key=api_key,
            nvcf_org=org,
        )
        logger.info(f"Deploying {args.max_parallel} NVCF functions (this may take several minutes)...")
        nvcf_pool.deploy_all(args.max_parallel)
        logger.info("NVCF pool ready.")

    try:
        # 3. Start Parallel Generator
        generator = ParallelTrajectoryGenerator(args, data_collector, nvcf_pool=nvcf_pool)
        await generator.run()
    finally:
        # 4. Cleanup NVCF pool
        if nvcf_pool:
            logger.info("Undeploying NVCF functions...")
            nvcf_pool.undeploy_all()


if __name__ == "__main__":
    asyncio.run(main())
