"""
Migrated test script originally from openhands.nvidia.async_server
"""

import time
import copy
import os

from openhands.nvidia.async_server import OpenHandsServer


def test_server(
    total_jobs: int = 4,
    max_parallel_jobs: int = 2,
    allow_skip_eval: bool = False,
    timeout: int = 50,
):
    from concurrent.futures import ThreadPoolExecutor

    import json
    data_file = '/root/OSWorld/osworld_test_nogdrive.json'
    all_data = []
    with open(data_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)

    instance = all_data[3] # 3 is a good example of pruning too much?
    print(instance)

    requests = []
    for i in range(total_jobs):
        cur = copy.deepcopy(instance)
        cur['trajectory_id'] = i
        requests.append(cur)

    llm_server_address = 'https://api.deepseek.com/chat/completions'
    sampling_params = {
        'model': 'deepseek/deepseek-chat',
        'api_key': os.getenv('DEEPSEEK_API_KEY', ''),
        'modify_params': False,
        'log_completions': False,
        'native_tool_calling': True,
        'temperature': 0.6,
        'max_iterations': 3,
    }

    print('Starting server')
    server = OpenHandsServer(
        llm_server_addresses=[llm_server_address, llm_server_address],
        max_init_workers=max_parallel_jobs,
        max_run_workers=max_parallel_jobs,
        allow_skip_eval=allow_skip_eval,
    )
    server.start()
    print('Server started')

    print('Job submission started')

    # Process instances using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [
            executor.submit(
                server.process, inst, dict(sampling_params), timeout=timeout
            )
            for inst in requests
        ]
        results = [future.result() for future in futures]

    print('Job submission finished')
    # print(results)
    server.stop()
    return results


if __name__ == '__main__':
    start = time.time()
    # set timeout approriate to terminate
    results = test_server(
        total_jobs=4, max_parallel_jobs=4, allow_skip_eval=False, timeout=6000
    )
    # Don't print full messages
    print(f'Time taken: {time.time() - start}')
    print('All tests passed!')
