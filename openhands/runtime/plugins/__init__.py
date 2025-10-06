# Requirements
from openhands.runtime.plugins.agent_skills import (
    AgentSkillsPlugin,
    AgentSkillsRequirement,
)
from openhands.runtime.plugins.jupyter import JupyterPlugin, JupyterRequirement, DirectJupyterRequirement, DirectJupyterPlugin
from openhands.runtime.plugins.requirement import Plugin, PluginRequirement
from openhands.runtime.plugins.vscode import VSCodePlugin, VSCodeRequirement

__all__ = [
    'Plugin',
    'PluginRequirement',
    'AgentSkillsRequirement',
    'AgentSkillsPlugin',
    'JupyterRequirement',
    'JupyterPlugin',
    'VSCodeRequirement',
    'VSCodePlugin',
    'DirectJupyterRequirement',
    'DirectJupyterPlugin',
]

ALL_PLUGINS = {
    'jupyter': JupyterPlugin,
    'agent_skills': AgentSkillsPlugin,
    'vscode': VSCodePlugin,
    'direct_jupyter': DirectJupyterPlugin,
}
