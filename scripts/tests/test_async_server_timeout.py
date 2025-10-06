"""
Migrated test script originally from openhands.nvidia.async_server
"""

import time

from openhands.nvidia.async_server import OpenHandsServer


def test_server(
    total_jobs: int = 4,
    max_parallel_jobs: int = 2,
    allow_skip_eval: bool = False,
    timeout: int = 50,
):
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd

    dataset = pd.read_parquet(
        '/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet'
    )
    instance = dataset.iloc[0]['instance']
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    requests = []
    for i in range(total_jobs):
        cur = instance.copy(deep=True)
        cur['trajectory_id'] = i
        requests.append(cur.to_dict())

    llm_server_address = 'http://127.0.0.1:8000/v1'
    sampling_params = {
        'model': 'hosted_vllm/Qwen/Qwen3-8B',
        'api_key': 'mykey',
        'modify_params': False,
        'log_completions': True,
        'native_tool_calling': True,
        'temperature': 0.6,
        'max_iterations': 35,
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
        total_jobs=5, max_parallel_jobs=5, allow_skip_eval=False, timeout=60
    )
    # Don't print full messages
    for result in results:
        assert type(result['messages']) is list, (
            f'Result is not a list but of type {type(result["messages"])}.'
        )
        if len(result['messages']) == 0:
            print('Result messages is empty.')
            continue
        assert result['messages'][-1]['role'] == 'assistant', (
            f'Last message is not assistant but of role {result["messages"][-1]["role"]}.'
        )
        result['messages'] = len(result['messages'])
        if 'tools' in result:
            result['tools'] = len(result['tools'])
        if 'traceback' in result:
            result.pop('traceback')
    print(results)
    print(f'Time taken: {time.time() - start}')
    print('All tests passed!')
