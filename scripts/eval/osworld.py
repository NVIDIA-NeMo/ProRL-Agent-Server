"""
Migrated test script originally from openhands.nvidia.async_server
"""

import argparse
import time
import copy
import os
import json 

from openhands.nvidia.async_server_osworld import OpenHandsServer

def get_existing_results(output_dir: str) -> set[str]:
    """Get set of instance IDs that already have results saved."""
    existing_results = set()
    if not os.path.exists(output_dir):
        return existing_results
    for file in os.listdir(output_dir):
        if file.endswith('.json') and not file.startswith('error_'):
            try:
                with open(os.path.join(output_dir, file), 'r') as f:
                    data = json.load(f)
                    existing_results.add(data['instance_id'])
            except (json.JSONDecodeError, KeyError):
                # Skip files that can't be parsed or don't have instance_id
                continue
    return existing_results


def test_server(args):
    """Run OSWorld evaluation with the given arguments."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import json
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
 
    existing_results = get_existing_results(args.output_dir)
    print(f'Found {len(existing_results)} existing results')
    
    all_data = []
    skipped_count = 0
    with open(args.data_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['id'] in existing_results:
                skipped_count += 1
                continue
            data['trajectory_id'] = 0
            data['instance_id'] = data['id']
            all_data.append(data)

    print(f'Skipped {skipped_count} instances with existing results')
    print(f'Total instances to process: {len(all_data)}')
    
    if args.total_jobs is not None:
        requests = all_data[:args.total_jobs]
        print(f'Running first {len(requests)} jobs')
    else:
        requests = all_data

    sampling_params = {
        'model': args.model,
        'modify_params': False,
        'log_completions': False,
        'native_tool_calling': True,
        'temperature': args.temperature,
        'max_iterations': args.max_iterations,
        'enable_vision': args.enable_vision,
        'enable_a11y_tree': args.enable_a11y_tree,
        'max_image_history': args.max_image_history,
    }

    print('Starting server')
    server = OpenHandsServer(
        llm_server_addresses=[args.llm_server_address],
        max_workers=args.max_parallel_jobs,
        allow_skip_eval=True,
    )
    server.start()
    print('Server started')

    print('Job submission started')
    
    results = []
    
    # Process instances using ThreadPoolExecutor with as_completed for async saving
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        # Submit all jobs and map futures to their instance data
        future_to_instance = {
            executor.submit(
                server.process, inst, dict(sampling_params), timeout=args.timeout
            ): inst
            for inst in requests
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_instance), 1):
            instance = future_to_instance[future]
            try:
                result = future.result()
                results.append(result)
                
                # Save result immediately to file
                instance_id = result.get('instance_id', instance.get('id', f'unknown_{i}'))
                output_file = os.path.join(args.output_dir, f'{instance_id}.json')
                
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f'[{i}/{len(requests)}] Completed and saved: {instance_id}')
                    
            except Exception as e:
                print(f'[{i}/{len(requests)}] Job failed with error: {e}')
                # Save error result as well
                error_result = {
                    'instance_id': instance.get('id', f'unknown_{i}'),
                    'error': str(e),
                    'instance': instance
                }
                results.append(error_result)
                error_file = os.path.join(args.output_dir, f'error_{i}.json')
                with open(error_file, 'w') as f:
                    json.dump(error_result, f, indent=2)

    print('Job submission finished')
    # print(results)
    server.stop()
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run OSWorld evaluation with OpenHands',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Job configuration
    parser.add_argument(
        '--total-jobs',
        type=int,
        default=None,
        help='Number of jobs to run. If None, runs all jobs in the data file.'
    )
    parser.add_argument(
        '--max-parallel-jobs',
        type=int,
        default=2,
        help='Maximum number of parallel jobs to run'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=6000,
        help='Timeout in seconds for each job'
    )
    
    # Data and output
    parser.add_argument(
        '--data-file',
        type=str,
        default='/lustre/fsw/portfolios/nvr/users/mingjiel/data/osworld/osworld_test_nogdrive.json',
        help='Path to the data file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./osworld_results',
        help='Directory to save results'
    )
    
    # LLM configuration
    parser.add_argument(
        '--llm-server-address',
        type=str,
        default='http://localhost:8000/v1',
        help='LLM server address'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='hosted_vllm/Qwen/Qwen3-VL-235B-A22B-Instruct',
        help='Model name'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=15,
        help='Maximum number of agent iterations'
    )
    
    # Vision and accessibility
    parser.add_argument(
        '--enable-vision',
        action='store_true',
        default=True,
        help='Enable vision capabilities'
    )
    parser.add_argument(
        '--no-vision',
        action='store_false',
        dest='enable_vision',
        help='Disable vision capabilities'
    )
    parser.add_argument(
        '--enable-a11y-tree',
        action='store_true',
        default=False,
        help='Enable accessibility tree'
    )
    parser.add_argument(
        '--max-image-history',
        type=int,
        default=3,
        help='Maximum number of images to keep in history'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print('Configuration:')
    for arg, value in vars(args).items():
        print(f'  {arg}: {value}')
    print()
    
    start = time.time()
    
    results = test_server(args)
    
    print(f'\nTime taken: {time.time() - start:.2f}s')
    print(f'Total results: {len(results)}')
    print('All tests passed!')
