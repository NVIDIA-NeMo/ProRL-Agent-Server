"""
Migrated test script originally from openhands.nvidia.async_server
"""

import json
import time
from datetime import datetime

from openhands.nvidia.async_server import OpenHandsServer


def test_server(
    total_jobs: int = 4, max_parallel_jobs: int = 2, allow_skip_eval: bool = False
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
        'model': 'hosted_vllm/Qwen/Qwen3-14B',
        'api_key': 'mykey',
        'modify_params': False,
        'log_completions': False,
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
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = [
            executor.submit(server.process, inst, dict(sampling_params), timeout=3000)
            for inst in requests
        ]
        results = [future.result() for future in futures]

    print('Job submission finished')
    # print(results)
    server.stop()
    return results


if __name__ == '__main__':
    start = time.time()
    results = test_server(total_jobs=16, max_parallel_jobs=16, allow_skip_eval=False)
    end = time.time()
    # Prepare output data
    output_data = {
        'test_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_jobs': 16,
            'max_parallel_jobs': 16,
            'allow_skip_eval': False,
            'time_taken': end - start,
            'start_time': start,
            'end_time': end,
        },
        'results': results,
    }

    # Save to JSON file
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'test_latency_results_{timestamp_str}.json'

    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f'Results saved to {filename}')
    # Don't print full messages
    for result in results:
        assert type(result['messages']) is list, (
            f'Result is not a list but of type {type(result["messages"])}.'
        )
        assert result['messages'][-1]['role'] == 'assistant', (
            f'Last message is not assistant but of role {result["messages"][-1]["role"]}.'
        )
        result['messages'] = len(result['messages'])
        result['tools'] = len(result['tools'])
    print(results)
    print(f'Time taken: {time.time() - start}')
    print('All tests passed!')
