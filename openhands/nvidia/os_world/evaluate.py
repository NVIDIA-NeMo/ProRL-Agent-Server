from typing import Any, Union, Dict, List
import inspect

from openhands.nvidia.os_world.controllers.python import PythonController
from openhands.nvidia.os_world import metrics, getters
from openhands.core.logger import openhands_logger as logger

async def function_wrapper(func, *args, **kwargs):
    # wrap function to handle coroutine functions
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)

class Evaluator:
    def __init__(self, task_config: Dict[str, Any], controller):
        self.setup_controller = controller
        self.vm_ip = controller.vm_ip
        self.server_port = controller.server_port
        self.chromium_port = controller.chromium_port
        self.vlc_port = controller.vlc_port
        self.http_server = controller.http_server
        self.http_server_setup_root = controller.http_server_setup_root
        self.cache_dir = controller.cache_dir
        self.client_password = controller.client_password
        self.screen_width = controller.screen_width
        self.screen_height = controller.screen_height

        # Assume Linux platform for OS World VMs
        # TODO: get from runtime/controller, mismatch initial letter is lowercase 
        self.vm_platform = 'Linux'

        self.controller = PythonController(self.vm_ip, self.server_port)
        self._set_evaluator_info(task_config)
 
    def _set_evaluator_info(self, task_config: Dict[str, Any]):
        """Set evaluator information from task config"""
        # evaluator dict
        # func -> metric function string, or list of metric function strings
        # conj -> conjunction of multiple metrics if func is a list with length > 1, "and"/"or"
        # result -> result getter config, or list of result getter configs
        # expected (optional) -> expected getter config, or list of expected getter configs
        # options (optional) -> metric options, or list of metric options
        # if func is a str list, then result, expected (if exists), options (if exists) should also be lists of the same length
        # even if one of the metrics does not need expected or options field, it should be included in the list with None
        self.evaluator = task_config["evaluator"]
        self.metric = [getattr(metrics, func) for func in self.evaluator["func"]] \
            if isinstance(self.evaluator["func"], list) \
            else getattr(metrics, self.evaluator["func"])
        self.metric_conj: str = self.evaluator.get("conj", "and")  # take conjunction of multiple metrics
        if "result" in self.evaluator and len(self.evaluator["result"]) > 0:
            self.result_getter = [getattr(getters, "get_{:}".format(res["type"])) for res in
                                          self.evaluator["result"]] \
                if isinstance(self.evaluator["result"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["result"]["type"]))
        else:
            self.result_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None

        if "expected" in self.evaluator and len(self.evaluator["expected"]) > 0:
            self.expected_getter = [getattr(getters, "get_{:}".format(exp["type"])) if exp else None for exp in
                                            self.evaluator["expected"]] \
                if isinstance(self.evaluator["expected"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["expected"]["type"]))
        else:
            self.expected_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None
        self.metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = [opt if opt else {} for opt in
                                                                            self.evaluator["options"]] \
            if isinstance(self.evaluator.get("options", {}), list) \
            else self.evaluator["options"] \
            if "options" in self.evaluator \
            else [{}] * len(self.metric) \
            if isinstance(self.metric, list) \
            else {}

        assert (not isinstance(self.evaluator["func"], list)
                or (len(self.metric) == len(self.result_getter) == len(self.expected_getter) == len(
                    self.metric_options)))

    async def evaluate(self, action_history = []):
        """
        Evaluate whether the task is successfully completed.
        """

        def last_action_is_fail(last_action):
            try:
                function_type = last_action['tool_calls'][0]['function']['name']
                return function_type == 'fail'
            except:
                return False

        # Special handling for infeasible tasks
        # TODO: Currently working on litellm json dumped format. Might need to be modified for other formats.
        if self.evaluator['func'] == "infeasible":
            if len(action_history) > 0:
                last_action = action_history[-1]
                if last_action_is_fail(last_action):
                    return 1
            return 0
        else:
            if len(action_history) > 0:
                last_action = action_history[-1]
                if last_action_is_fail(last_action):
                    return 0

        postconfig = self.evaluator.get("postconfig", [])
        await self.setup_controller.setup(postconfig)

        if type(self.metric) == list:
            # Multiple metrics to evaluate whether the task is successfully completed
            results = []
            assert len(self.metric) == len(self.result_getter), "The number of metrics and result getters must be the same"
            if "expected" in self.evaluator:
                assert len(self.metric) == len(self.expected_getter), "The number of metrics and expected getters must be the same"
            for idx, metric in enumerate(self.metric):
                try:
                    config = self.evaluator["result"][idx]
                    result_state = await function_wrapper(self.result_getter[idx], self, config)
                except FileNotFoundError:
                    logger.error("File not found!")
                    if self.metric_conj == 'and':
                        return 0

                if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                    expected_state = await function_wrapper(self.expected_getter[idx], self, self.evaluator["expected"][idx])
                    metric: int = await function_wrapper(metric, result_state, expected_state, **self.metric_options[idx])
                else:
                    metric: int = await function_wrapper(metric, result_state, **self.metric_options[idx])

                if self.metric_conj == 'and' and float(metric) == 0.0:
                    return 0
                elif self.metric_conj == 'or' and float(metric) == 1.0:
                    return 1
                else:
                    results.append(metric)

            return sum(results) / len(results) if self.metric_conj == 'and' else max(results)
        else:
            # Single metric to evaluate whether the task is successfully completed
            try:
                result_state = await function_wrapper(self.result_getter, self, self.evaluator["result"])
            except FileNotFoundError:
                logger.error("File not found!")
                return 0

            if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                expected_state = await function_wrapper(self.expected_getter, self, self.evaluator["expected"])
                metric: float = await function_wrapper(self.metric, result_state, expected_state, **self.metric_options)
            else:
                metric: float = await function_wrapper(self.metric, result_state, **self.metric_options)

        return metric