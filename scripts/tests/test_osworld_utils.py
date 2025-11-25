import asyncio

import numpy as np
import pandas as pd

from openhands.core.config.llm_config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.nvidia.os_world.osworld_utils import (
    initialize_agents,
    run_agent,
    evaluate_agent,
)
from openhands.nvidia.registry import JobDetails
from openhands.nvidia.timer import PausableTimer


async def run(instance):
    max_iterations = 35
    sampling_params = {
        'model': 'deepseek/deepseek-chat',
        'api_key': 'sk-697e5dc7145849a3b2ad1595718b35f2',
        'modify_params': False,
        'log_completions': True,
        'native_tool_calling': True,
        'temperature': 0.6,
    }
    llm_config = LLMConfig(base_url='https://api.deepseek.com/chat/completions', **sampling_params)


    job_details = JobDetails(
        job_id='test_job',
        instance=instance,
        llm_config=llm_config,
    )
    job_details.agent_config['max_iterations'] = max_iterations
    job_details.timer = PausableTimer(timeout=500)
    job_details.timer.start()

    # run agent
    runtime, metadata, config = await initialize_agents(
        instance, llm_config=llm_config, agent_config=job_details.agent_config
    )
    job_details.runtime = runtime
    job_details.metadata = metadata
    job_details.config = config

    import pdb; pdb.set_trace()
    run_results = await run_agent(job_details)   

    import pdb; pdb.set_trace()
    eval_results = await evaluate_agent(run_results, instance, runtime)
    return eval_results


if __name__ == '__main__':
    instance = {
        "id": "06fe7178-4491-4589-810f-2e2bc9502122",
        "snapshot": "chrome",
        "instruction": "Can you make my computer bring back the last tab I shut down?",
        #"instruction": "Go on to ArXiv and search for 'ProRL' paper from Nvidia. Open the pdf.",
        "source": "https://www.wikihow.com/Switch-Tabs-in-Chrome",
        "config": [
        {
            "type": "launch",
            "parameters": {
            "command": [
                "google-chrome",
                "--remote-debugging-port=1337"
            ]
            }
        },
        {
            "type": "launch",
            "parameters": {
            "command": [
                "socat",
                "tcp-listen:9222,fork",
                "tcp:localhost:1337"
            ]
            }
        },
        {
            "type": "chrome_open_tabs",
            "parameters": {
            "urls_to_open": [
                "https://www.lonelyplanet.com",
                "https://www.airbnb.com",
                "https://www.tripadvisor.com"
            ]
            }
        },
        {
            "type": "chrome_close_tabs",
            "parameters": {
            "urls_to_close": [
                "https://www.tripadvisor.com"
            ]
            }
        }
        ],
        "trajectory": "trajectories/",
        "related_apps": [
        "chrome"
        ],
        "evaluator": {
        "func": "is_expected_tabs",
        "result": {
            "type": "open_tabs_info"
        },
        "expected": {
            "type": "rule",
            "rules": {
            "type": "url",
            "urls": [
                "https://www.lonelyplanet.com",
                "https://www.airbnb.com",
                "https://www.tripadvisor.com"
            ]
            }
        }
        },
        "proxy": True,
        "fixed_ip": False,
        "possibility_of_env_change": "low"
    }

    import json
    data_file = '/root/OSWorld/osworld_test_nogdrive.json'
    all_data = []
    with open(data_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)
    instance = all_data[3] # 3 is a good example of pruning too much?
    print(instance)

    # Initialize the agents
    results = asyncio.run(run(instance))

    logger.info(f'Run Results: {results}')
    logger.info('Agents initialized successfully!')
