"""
Migrated test script originally from openhands.nvidia.async_server
"""

import hashlib
import json
import time

import numpy as np
import pandas as pd

from openhands.nvidia.async_server import OpenHandsServer
from openhands.nvidia.swe_agent.r2egym_parser import ParsedCommit


def pre_process_r2egym_instance(r2egym_instance):
    r2egym_instance = pd.Series(r2egym_instance)
    r2egym_instance = r2egym_instance.apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    r2egym_instance['data_source'] = 'swebench'
    r2egym_instance['instance_id'] = (
        r2egym_instance['docker_image'].replace('/', '_').replace(':', '_')
    )
    docker_image = r2egym_instance['docker_image']
    if ':' in docker_image:
        repo_part, version_part = docker_image.split(':', 1)
    else:
        repo_part, version_part = docker_image, 'latest'
    r2egym_instance['repo'] = repo_part
    r2egym_instance['version'] = version_part

    if ('base_commit' not in r2egym_instance) or pd.isna(
        r2egym_instance['base_commit']
    ):
        if 'commit_hash' in r2egym_instance and not pd.isna(
            r2egym_instance['commit_hash']
        ):
            r2egym_instance['base_commit'] = r2egym_instance['commit_hash']
        else:
            r2egym_instance['base_commit'] = version_part
    r2egym_instance['data_kind'] = 'r2egym'
    parsed_commit = ParsedCommit(**json.loads(r2egym_instance['parsed_commit_content']))
    old_commit = parsed_commit.old_commit_hash
    r2egym_instance['old_commit'] = old_commit
    return r2egym_instance


def test_server(
    total_jobs: int = 4, max_parallel_jobs: int = 2, allow_skip_eval: bool = False
):
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    r2egym_dataset = pd.read_parquet(
        '/lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/data/r2egym/data/train-00003-of-00008.parquet'
    )
    r2egym_instance = r2egym_dataset.iloc[-2]
    r2egym_instance = pre_process_r2egym_instance(r2egym_instance)
    requests = []
    for i in range(total_jobs):
        cur = r2egym_instance.copy(deep=True)
        cur['trajectory_id'] = i
        requests.append(cur.to_dict())

    llm_server_address = 'http://127.0.0.1:8000/v1'
    sampling_params = {
        'model': 'hosted_vllm/Qwen2.5-7B-Instruct',
        'api_key': 'mykey',
        'modify_params': False,
        'log_completions': False,
        'native_tool_calling': True,
        'temperature': 0.6,
        'top_p': 0.9,
        'max_iterations': 30,
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
        futures = []
        for inst in requests:
            short_id = hashlib.sha1(
                f'{inst["instance_id"]}_{inst["trajectory_id"]}'.encode()
            ).hexdigest()[:12]
            futures.append(
                executor.submit(server.process, inst, dict(sampling_params), short_id)
            )
        results = [future.result() for future in futures]
    print('Job submission finished')
    server.stop()
    return results


if __name__ == '__main__':
    start = time.time()
    results = test_server(total_jobs=5, max_parallel_jobs=5, allow_skip_eval=False)
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
