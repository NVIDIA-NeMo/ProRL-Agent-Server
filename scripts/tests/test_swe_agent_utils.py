"""
Migrated test script originally from openhands.nvidia.swe_agent.utils
"""

import asyncio

import numpy as np
import pandas as pd

from openhands.core.logger import openhands_logger as logger
from openhands.nvidia.swe_agent.utils import evaluate_agent

if __name__ == '__main__':
    # Load one multimodal and one non-multimodal example
    dataset_mm = pd.read_parquet(
        '/lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/data/swe-bench-multimodal/data/train.parquet'
    )
    dataset_non_mm = pd.read_parquet(
        '/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet'
    )

    row_mm = dataset_mm.iloc[0]
    row_non_mm = dataset_non_mm.iloc[0]

    def to_instance(series: pd.Series) -> pd.Series:
        if 'instance' in series:
            inst_dict = series['instance']
        else:
            inst_dict = series.to_dict()
        inst = pd.Series(inst_dict)
        # Convert any numpy arrays to lists for serialization
        return inst.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    instance_mm = to_instance(row_mm)
    instance_non_mm = to_instance(row_non_mm)

    # Store (instance, dataset_name) pairs for testing
    test_cases = [
        (instance_mm, 'princeton-nlp/SWE-bench_Multimodal'),
        (instance_non_mm, 'swebench'),  # non-multimodal
    ]

    # try:
    #     # Try to get the current event loop
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     # No event loop exists in this thread, create a new one
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)

    # result = loop.run_until_complete(run(instance))
    # print("BEGIN RESULTS.")
    # print(result)
    # print("END RESULTS.")

    mock_patch = """
    diff --git a/openhands_patch_test.txt b/openhands_patch_test.txt
    new file mode 100644
    index 0000000..e69de29
    --- /dev/null
    +++ b/openhands_patch_test.txt
    @@
    +This is an OpenHands test patch.
    """

    ###### async evaluate ##########
    async def run_parallel_async():
        tasks = []
        for inst, ds_name in test_cases:
            inst_clone = inst.copy()
            gold_patch = inst_clone['patch']
            tasks.append(evaluate_agent(gold_patch, inst_clone))
        return await asyncio.gather(*tasks, return_exceptions=True)

    try:
        all_reports = asyncio.run(run_parallel_async())
        for i, rep in enumerate(all_reports):
            print(f'\nBEGIN EVAL REPORT [{i}]')
            print(rep)
            print(f'END EVAL REPORT [{i}]')
    except Exception as eval_err:
        logger.error(f'Failed to parallel-evaluate patches: {eval_err}')
        raise

    ###### sequential evaluate (non-async) ##########
    async def run_sequential_async():
        results = []
        for i, (inst, ds_name) in enumerate(test_cases):
            res = await evaluate_agent(mock_patch, inst)
            print(f'\nBEGIN EVAL REPORT SEQ [{i}]')
            print(res)
            print(f'END EVAL REPORT SEQ [{i}]')
            results.append(res)
        return results

    try:
        asyncio.run(run_sequential_async())
    except Exception as seq_err:
        logger.error(f'Sequential evaluation failed: {seq_err}')
        raise
