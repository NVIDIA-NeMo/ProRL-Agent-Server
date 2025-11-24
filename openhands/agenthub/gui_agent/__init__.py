from openhands.agenthub.gui_agent.gui_agent import (
    GuiAgent,
)

from openhands.agenthub.gui_agent.osworld_agent import OSWorldAgent
from openhands.controller.agent import Agent
Agent.register('GuiAgent', GuiAgent)
Agent.register('OSWorldAgent', OSWorldAgent)
