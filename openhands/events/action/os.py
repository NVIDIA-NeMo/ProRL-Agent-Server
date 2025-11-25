from dataclasses import dataclass
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class OSInteractiveAction(Action):
    os_actions: str
    thought: str = ''
    action: str = ActionType.OS_INTERACTIVE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'I am interacting with the operating system:\n```\n{self.os_actions}\n```'

    def __str__(self) -> str:
        ret = '**OSInteractiveAction**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'OS_ACTIONS: {self.os_actions}'
        return ret


@dataclass
class OSWorldInteractiveAction(Action):
    """Action for interacting with OSWorld VM.
    
    Attributes:
        method: The method to call (e.g., 'execute_action', 'get_screenshot', 'run_python_script')
        params: Parameters for the method (dict)
        thought: Optional thought/reasoning for this action
    """
    method: str  # Method name from PythonController
    params: dict = None  # Parameters for the method
    thought: str = ''
    action: str = ActionType.OSWORLD_INTERACTIVE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    pause_time: float = 0.0

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    @property
    def message(self) -> str:
        if self.params:
            return f'I am interacting with the OSWorld virtual machine:\nMethod: {self.method}\nParams: {self.params}'
        return f'I am interacting with the OSWorld virtual machine:\nMethod: {self.method}'

    def __str__(self) -> str:
        ret = '**OSWorldInteractiveAction**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'METHOD: {self.method}\n'
        if self.params:
            ret += f'PARAMS: {self.params}'
        return ret

