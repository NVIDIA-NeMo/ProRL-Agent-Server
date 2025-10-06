import asyncio
import json
import logging
import threading
import time
from typing import Any

import aiohttp

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenHandsBatchProcessor:
    """Simplified batch processor for OpenHands requests without verl dependencies."""

    def __init__(
        self,
        openhands_base_urls: list[str],
        openhands_num_workers: int = 4,
        server_addresses: list[str] = None,
        strict: bool = True,
        save_file: str = None,
        log_freq: int = 10,
    ):
        """Initialize the OpenHands batch processor.

        Args:
            openhands_base_urls: List of OpenHands server URLs
            openhands_num_workers: Number of workers per OpenHands server
        """
        self.openhands_urls = openhands_base_urls
        self.openhands_num_workers = openhands_num_workers
        self.server_addresses = server_addresses

        # Default sampling parameters
        self.sampling_params = {
            'model': 'hosted_vllm/Qwen/Qwen3-14B',
            'api_key': 'mykey',
            'modify_params': False,
            'log_completions': False,
            'native_tool_calling': True,
            'temperature': 0.6,
            'top_p': 0.95,
            'max_iterations': 35,
            'max_output_tokens': 3072,
        }

        # Create and start event loop in a separate thread
        self.chat_scheduler_loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        # Wait for the event loop to start
        time.sleep(0.1)

        # Send LLM server addresses to OpenHands servers
        self._send_llm_addresses_to_openhands(strict=strict)

        self.save_file = save_file
        if save_file:
            self.file_lock = threading.Lock()

        self.log_freq = log_freq

    def _run_event_loop(self):
        """Run the event loop in a separate thread"""
        asyncio.set_event_loop(self.chat_scheduler_loop)
        self.chat_scheduler_loop.run_forever()

    def set_sampling_params(self, **kwargs):
        """Update sampling parameters."""
        self.sampling_params.update(kwargs)

    async def generate_sequences(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Generate sequences from OpenHands for a batch of messages.

        Args:
            messages: List of message dictionaries, each containing:
                - instance_id: Unique identifier for the instance
                - trajectory_id: Trajectory identifier (for multiple trajectories per instance)
                - Other message data

        Returns:
            Dictionary with structure {instance_id: {trajectory_id: result_dict}}
        """
        # Start timing
        total_start_time = time.time()

        # Start all OpenHands servers
        server_start_time = time.time()
        await self.start_servers()
        server_end_time = time.time()
        logger.info(
            f'Starting OpenHands servers took: {server_end_time - server_start_time:.2f} seconds'
        )

        try:
            # Request from OpenHands
            request_start_time = time.time()
            output_messages = await self.request_from_openhands(messages)
            request_end_time = time.time()

            total_end_time = time.time()

            logger.info(
                f'OpenHands request took: {request_end_time - request_start_time:.2f} seconds'
            )
            logger.info(f'Total time: {total_end_time - total_start_time:.2f} seconds')

            return output_messages

        finally:
            # Stop all OpenHands servers
            stop_start_time = time.time()
            await self.stop_servers()
            stop_end_time = time.time()
            logger.info(
                f'Stopping OpenHands servers took: {stop_end_time - stop_start_time:.2f} seconds'
            )

    async def start_servers(self):
        """Start all OpenHands servers"""
        if not self.openhands_urls:
            logger.warning('No OpenHands base URLs configured, skipping server start')
            return

        logger.info(f'Starting {len(self.openhands_urls)} OpenHands servers')

        # Start all servers concurrently
        tasks = []
        for openhands_url in self.openhands_urls:
            task = self._start_single_server(openhands_url)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f'Failed to start server {self.openhands_urls[i]}: {result}'
                )
            else:
                success_count += 1

        logger.info(
            f'Successfully started {success_count}/{len(self.openhands_urls)} OpenHands servers'
        )

    async def stop_servers(self):
        """Stop all OpenHands servers"""
        if not self.openhands_urls:
            logger.warning('No OpenHands base URLs configured, skipping server stop')
            return

        logger.info(f'Stopping {len(self.openhands_urls)} OpenHands servers')

        # Stop all servers concurrently
        tasks = []
        for openhands_url in self.openhands_urls:
            task = self._stop_single_server(openhands_url)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f'Failed to stop server {self.openhands_urls[i]}: {result}'
                )
            else:
                success_count += 1

        logger.info(
            f'Successfully stopped {success_count}/{len(self.openhands_urls)} OpenHands servers'
        )

    async def _start_single_server(self, openhands_base_url: str):
        """Start single OpenHands server"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            session = aiohttp.ClientSession(timeout=timeout)

            try:
                url = f'{openhands_base_url}/start'

                async with session.post(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            f'OpenHands server {openhands_base_url} started successfully: {result}'
                        )
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            f'Failed to start OpenHands server {openhands_base_url}, HTTP {response.status}: {error_text}'
                        )
                        raise Exception(f'HTTP {response.status}: {error_text}')

            finally:
                await session.close()

        except asyncio.TimeoutError:
            error_msg = f'Timeout starting OpenHands server {openhands_base_url}'
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f'Error starting OpenHands server {openhands_base_url}: {e}')
            raise

    async def _stop_single_server(self, openhands_base_url: str):
        """Stop single OpenHands server"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            session = aiohttp.ClientSession(timeout=timeout)

            try:
                url = f'{openhands_base_url}/stop'

                async with session.post(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            f'OpenHands server {openhands_base_url} stopped successfully: {result}'
                        )
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            f'Failed to stop OpenHands server {openhands_base_url}, HTTP {response.status}: {error_text}'
                        )
                        raise Exception(f'HTTP {response.status}: {error_text}')

            finally:
                await session.close()

        except asyncio.TimeoutError:
            error_msg = f'Timeout stopping OpenHands server {openhands_base_url}'
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f'Error stopping OpenHands server {openhands_base_url}: {e}')
            raise

    async def request_from_openhands(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Request from OpenHands using server availability tracking for true load balancing."""
        # Start total timing
        total_start_time = time.time()

        # Store all response messages
        all_responses = {}

        if not self.openhands_urls:
            logger.error('No OpenHands base URLs configured')
            return {}

        # Create a job queue
        job_queue = asyncio.Queue()

        # Add all jobs to the queue with retry count
        for i, message in enumerate(messages):
            await job_queue.put((i, message, 0))  # Add retry count as 0

        # Results queue to collect completed work
        results_queue = asyncio.Queue()

        # Server availability queue - servers put themselves here when they finish a job
        available_servers_queue = asyncio.Queue()

        # Initially, all servers are available
        max_active_tasks = 0
        for server_url in self.openhands_urls:
            for worker_idx in range(self.openhands_num_workers):
                await available_servers_queue.put(f'{server_url}|worker_{worker_idx}')
                max_active_tasks += 1

        # Job dispatcher - assigns jobs to available servers
        async def job_dispatcher():
            while True:
                try:
                    # Wait for both a job and an available server
                    job_task = asyncio.create_task(job_queue.get())
                    server_task = asyncio.create_task(available_servers_queue.get())

                    # Wait for both to be available
                    done, pending = await asyncio.wait(
                        [job_task, server_task], return_when=asyncio.ALL_COMPLETED
                    )

                    # If we get here, we have both a job and an available server
                    message_index, message, retry_count = job_task.result()
                    server_worker_id = server_task.result()
                    server_url = server_worker_id.split('|')[0]

                    # Create task to process this job on this specific server
                    asyncio.create_task(
                        process_job_on_server(
                            message_index,
                            message,
                            retry_count,
                            server_url,
                            server_worker_id,
                        )
                    )

                    logger.debug(
                        f'Dispatched message {message_index} to {server_worker_id}'
                    )

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f'Job dispatcher error: {e}')

        # Process a job on a specific server
        async def process_job_on_server(
            message_index, message, retry_count, server_url, server_worker_id
        ):
            try:
                # Process the job
                result, should_retry = await self._send_single_message_to_openhands(
                    message=message,
                    message_index=message_index,
                    openhands_base_url=server_url,
                )

                if should_retry and retry_count < 2:  # Retry up to 3 times
                    await job_queue.put((message_index, message, retry_count + 1))
                    logger.info(
                        f'Retrying message {message_index}, attempt {retry_count + 1}'
                    )
                else:
                    # Put result in results queue
                    await results_queue.put((message_index, message, result))
                    # Mark job as done
                    job_queue.task_done()

            except Exception as e:
                logger.error(
                    f'Error processing message {message_index} on {server_worker_id}: {e}'
                )
                # Put error result
                await results_queue.put((message_index, message, e))
                job_queue.task_done()

            finally:
                # Server is available again - put it back in the available queue
                await available_servers_queue.put(server_worker_id)

        # Start the job dispatcher
        dispatcher_task = asyncio.create_task(job_dispatcher())

        logger.info(
            f'Started job dispatcher with max {max_active_tasks} concurrent tasks to process {len(messages)} messages'
        )

        # Process results as they come in
        processing_start_time = time.time()
        completed_count = 0

        # Collect results
        while completed_count < len(messages):
            try:
                # Wait for a result with timeout to provide progress updates
                message_index, message, result = await asyncio.wait_for(
                    results_queue.get(), timeout=5.0
                )

                # Process the result
                instance_id = message.get('instance_id', f'instance_{message_index}')
                trajectory_id = message.get('trajectory_id', 0)

                if instance_id not in all_responses:
                    all_responses[instance_id] = {}

                if isinstance(result, Exception):
                    logger.error(f'Error processing message {message_index}: {result}')
                    result_data = {
                        'error': str(result),
                        'success': False,
                    }
                    all_responses[instance_id][trajectory_id] = result_data
                elif result is not None:
                    all_responses[instance_id][trajectory_id] = result
                    result_data = result
                else:
                    logger.warning(f'Message {message_index} returned empty response')
                    result_data = {
                        'error': 'Empty response',
                        'success': False,
                    }
                    all_responses[instance_id][trajectory_id] = result_data

                # Save result to file if file saving is enabled
                self._save_result_to_file(
                    instance_id,
                    trajectory_id,
                    all_responses[instance_id][trajectory_id],
                )

                completed_count += 1

                # Log progress periodically
                if completed_count % self.log_freq == 0 or completed_count == len(
                    messages
                ):
                    current_time = time.time()
                    elapsed_time = current_time - processing_start_time
                    progress_percent = (completed_count / len(messages)) * 100
                    pending_jobs = job_queue.qsize()
                    available_servers = available_servers_queue.qsize()
                    active_tasks = max_active_tasks - available_servers

                    logger.info(
                        f'Processing progress: {completed_count}/{len(messages)} ({progress_percent:.1f}%), '
                        f'pending jobs: {pending_jobs}, available servers: {available_servers}, '
                        f'active tasks: {active_tasks}/{max_active_tasks}, elapsed: {elapsed_time:.2f}s'
                    )

            except asyncio.TimeoutError:
                # No result received in timeout period, just log progress
                pending_jobs = job_queue.qsize()
                available_servers = available_servers_queue.qsize()
                current_time = time.time()
                elapsed_time = current_time - processing_start_time
                progress_percent = (completed_count / len(messages)) * 100
                active_tasks = max_active_tasks - available_servers

                logger.info(
                    f'Processing progress: {completed_count}/{len(messages)} ({progress_percent:.1f}%), '
                    f'pending jobs: {pending_jobs}, available servers: {available_servers}, '
                    f'active tasks: {active_tasks}/{max_active_tasks}, elapsed: {elapsed_time:.2f}s'
                )

        # Cancel dispatcher task
        dispatcher_task.cancel()
        await asyncio.gather(dispatcher_task, return_exceptions=True)

        processing_end_time = time.time()
        total_end_time = time.time()

        logger.info(
            f'Request processing took: {processing_end_time - processing_start_time:.2f} seconds'
        )
        logger.info(
            f'Processing completed, received responses for {len(all_responses)} instances'
        )
        logger.info(
            f'Total request time: {total_end_time - total_start_time:.2f} seconds'
        )

        return all_responses

    async def _send_single_message_to_openhands(
        self, message: dict, message_index: int, openhands_base_url: str
    ):
        """Send single message to OpenHands server"""
        try:
            request_data = {
                'instance': message,
                'sampling_params': self.sampling_params,
            }

            timeout = aiohttp.ClientTimeout(
                total=3000
            )  # A large timeout. OpenHands server will timeout if it takes too long.
            session = aiohttp.ClientSession(timeout=timeout)

            try:
                async with session.post(
                    url=f'{openhands_base_url}/process',
                    headers={'Content-Type': 'application/json'},
                    json=request_data,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        out_message = data.get('messages', [])
                        if out_message and len(out_message) > 0:
                            return data, False  # No retry needed
                        else:
                            return None, True  # Retry needed
                    else:
                        error_text = await resp.text()
                        logger.error(
                            f'Failed to process request for message {message_index}: {error_text}'
                        )
                        return None, True  # Retry needed

            finally:
                await session.close()

        except asyncio.TimeoutError as e:
            logger.error(f'Timeout sending message {message_index} to OpenHands: {e}')
            return None, True  # Retry needed
        except Exception as e:
            logger.error(f'Error sending message {message_index} to OpenHands: {e}')
            return None, True  # Retry needed

    def assign_llm_addresses_to_openhands(self, strict=True):
        def url_to_ip(url):
            # remove http:// and https://
            url = url.replace('http://', '').replace('https://', '')
            # remove port
            url = url.split(':')[0]
            return url

        openhands_urls2ips = {url: url_to_ip(url) for url in self.openhands_urls}
        server_addresses2ips = {url: url_to_ip(url) for url in self.server_addresses}
        # Distribute LLM server addresses evenly to OpenHands servers
        num_openhands = len(self.openhands_urls)
        num_llm_servers = len(self.server_addresses)
        # Calculate how many LLM servers each OpenHands server should be assigned
        addresses_per_openhands = num_llm_servers // num_openhands
        assert num_llm_servers % num_openhands == 0, (
            f'num_llm_servers {num_llm_servers} must be divisible by num_openhands {num_openhands}'
        )

        address_assignments = {}

        for i, openhands_url in enumerate(self.openhands_urls):
            # Assign addresses
            assigned_addresses = []
            for j, server_address in enumerate(self.server_addresses):
                if (
                    server_addresses2ips[server_address]
                    == openhands_urls2ips[openhands_url]
                ):
                    assigned_addresses.append(server_address)
            if strict:
                address_assignments[openhands_url] = assigned_addresses
                assert len(assigned_addresses) == addresses_per_openhands, (
                    f'len(assigned_addresses) {len(assigned_addresses)} must be equal to addresses_per_openhands {addresses_per_openhands}'
                )
            else:
                address_assignments[openhands_url] = self.server_addresses

            logger.info(
                f'Assigned LLM server addresses to {openhands_url}: {assigned_addresses}'
            )

        return address_assignments

    def _send_llm_addresses_to_openhands(self, strict=True):
        """Distribute and send LLM server addresses to multiple OpenHands servers"""
        if not self.openhands_urls:
            logger.warning(
                'No OpenHands base URLs configured, skipping LLM address distribution'
            )
            return
        address_assignments = self.assign_llm_addresses_to_openhands(strict=strict)

        # Clear addresses in async event loop
        future = asyncio.run_coroutine_threadsafe(
            self._clear_addresses_async(address_assignments), self.chat_scheduler_loop
        )
        try:
            future.result(timeout=60)  # 60 second timeout
            logger.info('Successfully cleared LLM addresses from all OpenHands servers')
        except Exception as e:
            logger.warning(f'Failed to clear LLM addresses from OpenHands servers: {e}')

        # Send addresses in async event loop
        future = asyncio.run_coroutine_threadsafe(
            self._send_addresses_async(address_assignments, strict=strict),
            self.chat_scheduler_loop,
        )
        try:
            future.result(timeout=60)  # 60 second timeout
            logger.info('Successfully sent LLM addresses to all OpenHands servers')
        except Exception as e:
            logger.warning(f'Failed to send LLM addresses to OpenHands servers: {e}')

    async def _clear_addresses_async(self, address_assignments: dict[str, list[str]]):
        """Asynchronously clear LLM addresses from multiple OpenHands servers"""
        tasks = []

        for openhands_url in self.openhands_urls:
            task = self._clear_llm_servers_from_openhands(openhands_url)
            tasks.append(task)

        # Send all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f'Failed to send address {i}: {result}')
            else:
                success_count += 1

        logger.info(
            f'Successfully sent {success_count}/{len(results)} LLM addresses to OpenHands servers'
        )

    async def _send_addresses_async(
        self, address_assignments: dict[str, list[str]], strict=True
    ):
        """Asynchronously send LLM addresses to multiple OpenHands servers"""
        tasks = []

        for openhands_url, addresses in address_assignments.items():
            for address in addresses:
                task = self._add_llm_server_to_openhands(
                    openhands_url, f'http://{address}/v1' if strict else address
                )
                tasks.append(task)

        # Send all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f'Failed to send address {i}: {result}')
            else:
                success_count += 1

        logger.info(
            f'Successfully sent {success_count}/{len(results)} LLM addresses to OpenHands servers'
        )

    async def _clear_llm_servers_from_openhands(self, openhands_base_url: str):
        """Clear all LLM servers from OpenHands server"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            session = aiohttp.ClientSession(timeout=timeout)

            try:
                url = f'{openhands_base_url}/clear_llm_server'

                async with session.post(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            f'LLM servers cleared successfully from {openhands_base_url}: {result}'
                        )
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            f'Failed to clear LLM servers from {openhands_base_url}, HTTP {response.status}: {error_text}'
                        )
                        raise Exception(f'HTTP {response.status}: {error_text}')

            finally:
                await session.close()

        except asyncio.TimeoutError:
            error_msg = f'Timeout clearing LLM servers from {openhands_base_url}'
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f'Error clearing LLM servers from {openhands_base_url}: {e}')
            raise

    async def _add_llm_server_to_openhands(self, openhands_base_url: str, address: str):
        """Send single LLM server address to OpenHands server"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            session = aiohttp.ClientSession(timeout=timeout)

            try:
                url = f'{openhands_base_url}/add_llm_server'
                payload = {'address': address}

                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            f'LLM server {address} added successfully to {openhands_base_url}: {result}'
                        )
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            f'Failed to add LLM server {address} to {openhands_base_url}, HTTP {response.status}: {error_text}'
                        )
                        raise Exception(f'HTTP {response.status}: {error_text}')

            finally:
                await session.close()

        except asyncio.TimeoutError:
            error_msg = f'Timeout adding LLM server {address} to {openhands_base_url}'
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(
                f'Error adding LLM server {address} to {openhands_base_url}: {e}'
            )
            raise

    def _save_result_to_file(self, instance_id: str, trajectory_id: str, result: dict):
        """Save a single result to file with thread-safe locking."""
        if not self.save_file:
            return

        # Prepare the result data for saving
        result_data = {
            'instance_id': instance_id,
            'trajectory_id': trajectory_id,
            **result,
        }

        # Use file lock for thread-safe writing
        with self.file_lock:
            try:
                with open(self.save_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_data) + '\n')
            except Exception as e:
                logger.error(f'Error saving result to file {self.save_file}: {e}')


# Example usage
async def main(total_jobs: int = 256, openhands_num_workers: int = 70):
    """Example usage of the OpenHands batch processor"""

    # Configure OpenHands servers
    openhands_urls = [
        'http://127.0.0.1:8006',
    ]

    server_addresses = [
        '127.0.0.1:8000',
        '127.0.0.1:8001',
        '127.0.0.1:8002',
        '127.0.0.1:8003',
    ]

    # Create processor
    processor = OpenHandsBatchProcessor(
        openhands_base_urls=openhands_urls,
        openhands_num_workers=openhands_num_workers,
        server_addresses=server_addresses,
    )

    # Example messages
    import numpy as np
    import pandas as pd

    dataset = pd.read_parquet(
        '/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet'
    )
    instance = dataset.iloc[0]['instance']
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    messages = []
    for i in range(total_jobs):
        cur = instance.copy(deep=True)
        cur['trajectory_id'] = i
        messages.append(cur.to_dict())

    start_time = time.time()
    # Process messages
    results = await processor.generate_sequences(messages)

    end_time = time.time()
    print('----------------------------------------------------------------')
    print(f'Time taken: {end_time - start_time} seconds')
    # Print results
    print(f'Received results for {len(results)} instances:')
    for instance_id, trajectories in results.items():
        print(f'Instance {instance_id}: {len(trajectories)} trajectories')
        for trajectory_id, result in trajectories.items():
            messages = result.get('messages', [])
            git_patch = result.get('git_patch', '')
            print(
                f'  Trajectory: {trajectory_id} Messages: {len(messages)} git-patch: {len(git_patch)}'
            )
    print('----------------------------------------------------------------')


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run example
    asyncio.run(main(total_jobs=256, openhands_num_workers=64))
