from dataclasses import dataclass, field
from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class OSWorldOutputObservation(Observation):
    """This data class represents the output of a browser."""

    observation: str = ObservationType.OSWORLD
    command: str = field(default='')
    content: str = field(default='')
    screenshot: str | None = None
    accessibility_tree: str | None = None
    tool_call_id: str | None = None
    name: str = ''

    @property
    def message(self) -> str:
        return f'OSWorld action {self.command}'

    @property
    def image_urls(self) -> list[str]:
        if self.screenshot:
            return [f'data:image/png;base64,{self.screenshot}']
        else:
            return []

    def __str__(self) -> str:
        ret = (
            '**OSWorldOutputObservation**\n'
            f'Action: {self.command}\n'
            f'Observation: {self.accessibility_tree}'
        )
        return ret
    