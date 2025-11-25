from openhands.nvidia.gui_agent.gui_handler import GuiAgentHandler
from openhands.nvidia.math_coder.math_code_handler import CodeHandler, MathHandler
from openhands.nvidia.math_coder.prorl_handler import ProRLHandler
from openhands.nvidia.registry import add_name_mapping, register_agent_handler
from openhands.nvidia.stem_agent.stem_handler import STEMHandler
from openhands.nvidia.swe_agent.swe_agent_handler import SweAgentHandler
from openhands.nvidia.os_world.osworld_handler import OSWorldHandler

register_agent_handler(SweAgentHandler())
register_agent_handler(MathHandler())
register_agent_handler(CodeHandler())
register_agent_handler(ProRLHandler(), reasoning=True)

for code_dataset in ['codecontests', 'apps', 'codeforces', 'taco']:
    add_name_mapping(code_dataset, 'deepcoder')
for prorl_dataset in [
    'deepscaler',
    'deepcoder',
    'codecontests',
    'apps',
    'codeforces',
    'taco',
    'rStar',
    'tool_call',
    'if_eval',
    'llm_judge',
]:
    add_name_mapping(prorl_dataset, 'prorl', reasoning=True)
register_agent_handler(STEMHandler())
register_agent_handler(GuiAgentHandler())
register_agent_handler(OSWorldHandler())

for code_dataset in ['codecontests', 'apps', 'codeforces', 'taco']:
    add_name_mapping(code_dataset, 'deepcoder')

# Optional aliases for GUI tasks
for gui_name in ['gui', 'visual_browsing', 'browsergym']:
    add_name_mapping(gui_name, 'gui')
