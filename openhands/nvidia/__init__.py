# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from openhands.nvidia.gui_agent.gui_handler import GuiAgentHandler
from openhands.nvidia.math_coder.math_code_handler import CodeHandler, MathHandler
from openhands.nvidia.math_coder.prorl_handler import ProRLHandler
from openhands.nvidia.near_agent.near_agent_handler import NeARHandler
from openhands.nvidia.registry import add_name_mapping, register_agent_handler
from openhands.nvidia.stem_agent.stem_handler import STEMHandler
from openhands.nvidia.swe_agent.swe_agent_handler import SweAgentHandler

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
register_agent_handler(NeARHandler())

for code_dataset in ['codecontests', 'apps', 'codeforces', 'taco']:
    add_name_mapping(code_dataset, 'deepcoder')

# Optional aliases for GUI tasks
for gui_name in ['gui', 'visual_browsing', 'browsergym']:
    add_name_mapping(gui_name, 'gui')

# Optional aliases for NeAR research tasks
for near_name in ['near', 'research', 'deep_research']:
    add_name_mapping(near_name, 'near')
