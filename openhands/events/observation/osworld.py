from dataclasses import dataclass, field
from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class OSWorldOutputObservation(Observation):
    """This data class represents the output of a browser."""

    observation: str = ObservationType.OSWORLD
    command: str = field(default='')
    content: str = field(default='')
    screenshot: str = field(repr=False, default='')  # don't show in repr, in base64 format
    accessibility_tree: str = field(repr=False, default='')
    tool_call_id: str | None = None
    name: str = ''

    @property
    def message(self) -> str:
        return f'OSWorld action {self.command}'

    @property
    def image_urls(self) -> list[str]:
        return [f'data:image/png;base64,{self.screenshot}']

    def __str__(self) -> str:
        ret = (
            '**OSWorldOutputObservation**\n'
            f'Action: {self.command}\n'
            f'Observation: {self.accessibility_tree}'
        )
        return ret
    