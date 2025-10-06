from openhands.nvidia.math_coder.math_code_handler import CodeHandler, MathHandler
from openhands.nvidia.registry import add_name_mapping, register_agent_handler
from openhands.nvidia.swe_agent.swe_agent_handler import SweAgentHandler

register_agent_handler(SweAgentHandler())
register_agent_handler(MathHandler())
register_agent_handler(CodeHandler())

for code_dataset in ['codecontests', 'apps', 'codeforces', 'taco']:
    add_name_mapping(code_dataset, 'deepcoder')
