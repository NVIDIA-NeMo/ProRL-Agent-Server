"""
Migrated test script originally from openhands.nvidia.async_server
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from openhands.nvidia.async_server import OpenHandsServer


def test_swebench_server(
    total_jobs: int = 4, max_parallel_jobs: int = 2, allow_skip_eval: bool = False
):
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

    llm_server_address = 'http://127.0.0.1:8000'
    sampling_params = {
        'model': 'hosted_vllm/Qwen/Qwen3-8B',
        'api_key': 'mykey',
        'modify_params': False,
        'log_completions': False,
        'native_tool_calling': True,
        'temperature': 0.6,
        'max_iterations': 2,
        'token_level_generation': True,  # Token level generation is required for token counting.
        'custom_tokenizer': 'Qwen/Qwen3-8B',  # Tokenizer is required
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
            executor.submit(server.process, inst, dict(sampling_params))
            for inst in requests
        ]
        results = [future.result() for future in futures]

    print('Job submission finished')
    # print(results)
    server.stop()
    return results


def test_math_server(
    total_jobs: int = 4,
    max_parallel_jobs: int = 2,
    allow_skip_eval: bool = False,
    reward_server_ip: str | None = None,
):
    dataset = pd.read_parquet(
        '/lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler/aime.parquet'
    )
    instance = dataset.iloc[0]
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    requests = []
    for i in range(total_jobs):
        cur = instance.copy(deep=True)
        cur['trajectory_id'] = i
        requests.append(cur.to_dict())

    llm_server_address = 'http://127.0.0.1:8000'
    reward_server_ip = [reward_server_ip] if reward_server_ip else None
    sampling_params = {
        'model': 'hosted_vllm/Qwen/Qwen3-8B',
        'api_key': 'mykey',
        'modify_params': False,
        'log_completions': False,
        'native_tool_calling': True,
        'temperature': 0.6,
        'max_iterations': 35,  # test full iterations for math
        'token_level_generation': True,  # Token level generation is required for token counting.
        'custom_tokenizer': 'Qwen/Qwen3-8B',  # Tokenizer is required
    }

    print('Starting server')
    server = OpenHandsServer(
        llm_server_addresses=[llm_server_address, llm_server_address],
        max_init_workers=max_parallel_jobs,
        max_run_workers=max_parallel_jobs,
        allow_skip_eval=allow_skip_eval,
        reward_server_ip=reward_server_ip,
    )
    server.start()
    print('Server started')

    print('Job submission started')

    # Process instances using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = [
            executor.submit(server.process, inst, dict(sampling_params))
            for inst in requests
        ]
        results = [future.result() for future in futures]

    print('Job submission finished')
    # print(results)
    server.stop()
    return results


def test_code_server(
    total_jobs: int = 4,
    max_parallel_jobs: int = 2,
    allow_skip_eval: bool = False,
    reward_server_ip: str | None = None,
):
    dataset = pd.read_parquet(
        '/lustre/fsw/portfolios/nvr/users/mingjiel/data/eurus2-rl-data/codecontests.parquet'
    )
    instance = dataset.iloc[0]
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    requests = []
    for i in range(total_jobs):
        cur = instance.copy(deep=True)
        cur['trajectory_id'] = i
        requests.append(cur.to_dict())

    llm_server_address = 'http://127.0.0.1:8000'
    reward_server_ip = [reward_server_ip] if reward_server_ip else None
    sampling_params = {
        'model': 'hosted_vllm/Qwen/Qwen3-8B',
        'api_key': 'mykey',
        'modify_params': False,
        'log_completions': False,
        'native_tool_calling': True,
        'temperature': 0.6,
        'max_iterations': 35,  # test full iterations for math
        'token_level_generation': True,  # Token level generation is required for token counting.
        'custom_tokenizer': 'Qwen/Qwen3-8B',  # Tokenizer is required
    }

    print('Starting server')
    server = OpenHandsServer(
        llm_server_addresses=[llm_server_address, llm_server_address],
        max_init_workers=max_parallel_jobs,
        max_run_workers=max_parallel_jobs,
        allow_skip_eval=allow_skip_eval,
        reward_server_ip=reward_server_ip,
    )
    server.start()
    print('Server started')

    print('Job submission started')

    # Process instances using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = [
            executor.submit(server.process, inst, dict(sampling_params))
            for inst in requests
        ]
        results = [future.result() for future in futures]

    print('Job submission finished')
    # print(results)
    server.stop()
    return results


def check_results(results, start_time):
    # Don't print full messages
    for result in results:
        assert type(result['messages']) is list, (
            f'Result is not a list but of type {type(result["messages"])}.'
        )
        assert result['messages'][-1]['role'] == 'assistant', (
            f'Last message is not assistant but of role {result["messages"][-1]["role"]}.'
        )
        for message in result['messages']:
            message['content'] = len(message['content'])
            message['token_ids'] = len(message['token_ids'])
            if 'input_ids' in message and message['input_ids'] is not None:
                message['input_ids'] = len(message['input_ids'])
            if 'tool_calls' in message:
                message['tool_calls'] = len(message['tool_calls'])
        result['tools'] = len(result['tools'])
    print(results)
    print(f'Time taken: {time.time() - start_time}')


def parse_args():
    parser = argparse.ArgumentParser(description='OpenHands async server tests')
    parser.add_argument(
        '--total-jobs',
        type=int,
        default=5,
        help='Total number of jobs to process (default: 5)',
    )
    parser.add_argument(
        '--reward-server-ip',
        type=str,
        default='localhost',
        help='IP address of the reward server (default: localhost)',
    )
    parser.add_argument(
        '--skip-math-test', action='store_true', help='Skip the math server test'
    )
    parser.add_argument(
        '--skip-code-test', action='store_true', help='Skip the code server test'
    )
    parser.add_argument(
        '--skip-swebench-test',
        action='store_true',
        help='Skip the swebench server test',
    )
    parser.add_argument(
        '--allow-skip-eval',
        action='store_true',
        help='Skip evaluation if the agent fails to complete the task',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f'Running tests with {args.total_jobs} total jobs')
    print(f'Using reward server IP: {args.reward_server_ip}')

    if not args.skip_math_test:
        print('Running math server test...')
        start_time = time.time()
        results = test_math_server(
            total_jobs=args.total_jobs,
            max_parallel_jobs=args.total_jobs,
            allow_skip_eval=False,
            reward_server_ip=args.reward_server_ip,
        )
        check_results(results, start_time)
        print('math tests passed!')

    if not args.skip_code_test:
        print('Running code server test...')
        start_time = time.time()
        results = test_code_server(
            total_jobs=args.total_jobs,
            max_parallel_jobs=args.total_jobs,
            allow_skip_eval=False,
            reward_server_ip=args.reward_server_ip,
        )
        check_results(results, start_time)
        print('code tests passed!')

    if not args.skip_swebench_test:
        print('Running swebench server test...')
        start_time = time.time()
        results = test_swebench_server(
            total_jobs=args.total_jobs,
            max_parallel_jobs=args.total_jobs,
            allow_skip_eval=args.allow_skip_eval,
        )
        check_results(results, start_time)
        print('swebench tests passed!')

    print('All tests passed!')
