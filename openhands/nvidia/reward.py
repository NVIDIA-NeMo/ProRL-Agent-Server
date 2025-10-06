import heapq
import json
import threading

import aiohttp

from openhands.nvidia.logger import nvidia_logger as logger


class Reward:
    def __init__(self, server_ip: list[str]):
        self.server_ip = server_ip
        self.weighted_addresses = [[0, ip] for ip in server_ip]
        heapq.heapify(self.weighted_addresses)

        self._address_lock = threading.Lock()

    async def get_reward(self, instance, solution_str, session=None):
        with self._address_lock:
            if len(self.weighted_addresses) == 0:
                raise ValueError('No LLM server addresses added')

            ip = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1  # type: ignore
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        data_source = instance['data_source']
        if isinstance(instance['reward_model'], str):
            ground_truth = json.loads(instance['reward_model'])['ground_truth']
        elif isinstance(instance['reward_model'], dict):
            ground_truth = instance['reward_model']['ground_truth']
        else:
            raise ValueError(
                f'Invalid reward_model type: {type(instance["reward_model"])}'
            )
        extra_info = instance.get('extra_info', None)

        # please pass in session, spinning each session is very bad
        if session is None:
            local_session = True
        else:
            local_session = False
        session = session or aiohttp.ClientSession()
        if data_source == 'reasoning_gym':
            try:
                async with session.post(
                    f'http://{ip}:8288/score',
                    json={
                        'answer': solution_str,
                        'entry': extra_info['reward_model']['entry'],
                        'task': extra_info['reward_model']['reasoning_task'],
                    },
                ) as res:
                    res = await res.json()
                    res = res['score']
            except Exception as e:
                logger.error(f'Error: {e}, ip: {ip}')
                logger.info(
                    f'answer: {solution_str[:10]}, task: {extra_info["reward_model"]["reasoning_task"]}, entry: {extra_info["reward_model"]["entry"]}'[
                        :100
                    ]
                )
                res = 0
        else:
            try:
                async with session.post(
                    f'http://{ip}:8388/compute_score',
                    json={
                        'solution_str': solution_str,
                        'ground_truth': ground_truth,
                        'data_source': data_source,
                    },
                ) as res:
                    res = await res.json()
                    res = res['score']
            except Exception as e:
                logger.error(f'Error: {e}, ip: {ip}')
                logger.info(
                    f'answer: {solution_str[:10]}, data_source: {data_source}, ground_truth: {ground_truth}'[
                        :100
                    ]
                )
                res = 0

        if isinstance(res, (int, float, bool)):
            score = float(res)
        else:
            score = float(res[0])

        if local_session:
            await session.close()

        return {'resolved': score > 0.99, 'reward': score}
