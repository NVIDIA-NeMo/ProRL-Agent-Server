"""
How to run the server:

1. Start the server. This is run in openhands isolated environment seperate (and before) from running verl training loop.
    python start_server.py --max-init-workers 32 --max-run-workers 32 --timeout 1000 &

2. A one time initialization is needed to add the LLM server address to the server. Make sure address is collocated with openhands server.
They should have same ip but different port. You can add multiple llm addresses to the server. You address should also contain /v1 at the end.
    add_llm_server(address: str)

3. When calling to generate it is recommended:
    (1) wake up the server: start_server()
    (2) send concurrent requests to the server: process_request(instance, sampling_params)
    (3) after all requests are sent, sleep the server: stop_server()

4. Iterate step 3 during training loop.
"""

import asyncio
import copy

import aiohttp


async def start_server():
    async with aiohttp.ClientSession() as session:
        # Replace with your actual server URL
        url = 'http://localhost:8006/start'

        async with session.post(url) as response:
            if response.status == 200:
                result = await response.json()
                print('Server started successfully:', result)
            else:
                error = await response.json()
                print('Failed to start server:', error)


async def stop_server():
    async with aiohttp.ClientSession() as session:
        url = 'http://localhost:8006/stop'

        async with session.post(url) as response:
            if response.status == 200:
                result = await response.json()
                print('Server stopped successfully:', result)
            else:
                error = await response.json()
                print('Failed to stop server:', error)


async def get_status():
    async with aiohttp.ClientSession() as session:
        url = 'http://localhost:8006/status'

        async with session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                print('Server status:', result)
            else:
                error = await response.json()
                print('Failed to get server status:', error)


async def add_llm_server(address: str):
    async with aiohttp.ClientSession() as session:
        url = 'http://localhost:8006/add_llm_server'
        payload = {'address': address}

        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print('LLM server added successfully:', result)
            else:
                error = await response.json()
                print('Failed to add LLM server:', error)


async def process_request(instance: dict, sampling_params: dict | None = None):
    timeout = aiohttp.ClientTimeout(total=3000)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        url = 'http://localhost:8006/process'
        payload = {'instance': instance, 'sampling_params': sampling_params or {}}

        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                # Don't print full messages
                for message in result['messages']:
                    message['content'] = len(message['content'])
                    if 'tool_calls' in message and message['tool_calls'] is not None:
                        message['tool_calls'] = len(message['tool_calls'])
                    if 'input_ids' in message and message['input_ids'] is not None:
                        message['input_ids'] = len(message['input_ids'])
                    message['token_ids'] = len(message['token_ids'])
                result['tools'] = len(result['tools'])
                print('Process completed successfully:', result)
                return result
            else:
                error = await response.json()
                print('Failed to process request:', error)
                return None


# Example usage
async def start_server_api():
    await add_llm_server('http://127.0.0.1:8000')


async def process(instance, num_tasks: int = 4):
    # Start the server
    await start_server()
    # Get server status
    await get_status()
    tasks = []
    for i in range(num_tasks):
        sampling_params = {
            'model': 'hosted_vllm/Qwen/Qwen3-8B',
            'api_key': 'mykey',
            'modify_params': False,
            'log_completions': False,
            'native_tool_calling': True,
            'temperature': 0.6,
            'top_p': 0.9,
            'max_iterations': 35,
            'token_level_generation': True,
            'custom_tokenizer': 'Qwen/Qwen3-8B',
            'max_output_tokens': 4096,
            'max_model_len': 32768,
        }
        instance_copy = copy.deepcopy(instance)
        instance_copy['trajectory_id'] = i
        tasks.append(process_request(instance_copy, sampling_params))

    # Process all requests in parallel
    results = await asyncio.gather(*tasks)

    # Stop the server
    await stop_server()

    return results


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    dataset = pd.read_parquet(
        '/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet'
    )
    instance = dataset.iloc[0]['instance']

    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    instance = instance.to_dict()

    asyncio.run(start_server_api())
    import time

    # Test with 1 task
    start_time = time.time()
    results_single = asyncio.run(process(instance, num_tasks=1))
    end_time = time.time()
    print(f'Time1 taken: {end_time - start_time} seconds')
    for res in results_single:
        assert res['critical_error'] is None, f'Critical error in single task: {res}'

    # Test with 32 tasks
    start_time = time.time()
    results_multi = asyncio.run(process(instance, num_tasks=32))
    end_time = time.time()
    print(f'\nTime2 taken: {end_time - start_time} seconds')
    for res in results_multi:
        assert res['critical_error'] is None, f'Critical error in multiple tasks: {res}'

    print('All tests passed')
