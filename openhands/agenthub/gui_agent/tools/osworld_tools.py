from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# Not supported: KeyDownTool, KeyUpTool, MouseDownTool, MouseUpTool

# Click at (x, y)
ClickTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='click',
        description='Click at the given screen coordinates (x, y).',
        parameters={
            'type': 'object',
            'properties': {
                'x': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized X coordinate in range [0,1]'},
                'y': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized Y coordinate in range [0,1]'},
                'clicks': {'type': 'integer', 'minimum': 1, 'default': 1, 'description': 'Number of clicks to perform (default 1)'},
                'interval': {'type': 'number', 'default': 0.0, 'description': 'Seconds between clicks when clicks > 1 (default 0.0)'},
                'button': {'type': 'string', 'default': 'left', 'description': "Mouse button to use: 'left' | 'middle' | 'right' (default 'left')"},
                'duration': {'type': 'number', 'default': 0.0, 'description': 'Seconds to take moving to the target before clicking (default 0.0)'},
            },
            'required': ['x', 'y'],
            'additionalProperties': False,
        },
    ),
)

# Middle click at (x, y)
MiddleClickTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='middleClick',
        description='Middle-click at the given screen coordinates (x, y).',
        parameters={
            'type': 'object',
            'properties': {
                'x': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized X coordinate in range [0,1]'},
                'y': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized Y coordinate in range [0,1]'},
                'interval': {'type': 'number', 'default': 0.0, 'description': 'Seconds between clicks (default 0.0)'},
                'duration': {'type': 'number', 'default': 0.0, 'description': 'Seconds to take moving to the target before clicking (default 0.0)'},
            },
            'required': ['x', 'y'],
            'additionalProperties': False,
        },
    ),
)

# Double click at (x, y)
DoubleClickTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='doubleClick',
        description='Double-click at the given screen coordinates (x, y).',
        parameters={
            'type': 'object',
            'properties': {
                'x': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized X coordinate in range [0,1]'},
                'y': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized Y coordinate in range [0,1]'},
                'button': {'type': 'string', 'default': 'left', 'description': "Mouse button to use: 'left' | 'middle' | 'right' (default 'left')"},
                'interval': {'type': 'number', 'default': 0.0, 'description': 'Seconds between the two clicks (default 0.0)'},
                'duration': {'type': 'number', 'default': 0.0, 'description': 'Seconds to take moving to the target before clicking (default 0.0)'},
            },
            'required': ['x', 'y'],
            'additionalProperties': False,
        },
    ),
)

# Triple click at (x, y)
TripleClickTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='tripleClick',
        description='Triple-click at the given screen coordinates (x, y).',
        parameters={
            'type': 'object',
            'properties': {
                'x': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized X coordinate in range [0,1]'},
                'y': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized Y coordinate in range [0,1]'},
                'button': {'type': 'string', 'default': 'left', 'description': "Mouse button to use: 'left' | 'middle' | 'right' (default 'left')"},
                'interval': {'type': 'number', 'default': 0.0, 'description': 'Seconds between clicks (default 0.0)'},
                'duration': {'type': 'number', 'default': 0.0, 'description': 'Seconds to take moving to the target before clicking (default 0.0)'},
            },
            'required': ['x', 'y'],
            'additionalProperties': False,
        },
    ),
)

# Right click at (x, y)
RightClickTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='rightClick',
        description='Right-click at the given screen coordinates (x, y).',
        parameters={
            'type': 'object',
            'properties': {
                'x': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized X coordinate in range [0,1]'},
                'y': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized Y coordinate in range [0,1]'},
                'interval': {'type': 'number', 'default': 0.0, 'description': 'Seconds between clicks (default 0.0)'},
                'duration': {'type': 'number', 'default': 0.0, 'description': 'Seconds to take moving to the target before clicking (default 0.0)'},
            },
            'required': ['x', 'y'],
            'additionalProperties': False,
        },
    ),
)

# Move mouse to (x, y)
MoveToTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='moveTo',
        description='Move the mouse pointer to the given screen coordinates (x, y).',
        parameters={
            'type': 'object',
            'properties': {
                'x': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized X coordinate in range [0,1]'},
                'y': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized Y coordinate in range [0,1]'},
                'duration': {'type': 'number', 'default': 0.0, 'description': 'Seconds to take moving to the target (default 0.0)'},
            },
            'required': ['x', 'y'],
            'additionalProperties': False,
        },
    ),
)

# Drag mouse to (x, y)
DragToTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='dragTo',
        description='Drag the mouse pointer to the given screen coordinates (x, y).',
        parameters={
            'type': 'object',
            'properties': {
                'x': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized X coordinate in range [0,1]'},
                'y': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'Normalized Y coordinate in range [0,1]'},
                'duration': {'type': 'number', 'default': 0.0, 'description': 'Seconds to take dragging to the target (default 0.0)'},
                'button': {'type': 'string', 'default': 'left', 'description': "Mouse button to hold while dragging: 'left' | 'middle' | 'right' (default 'left')"},
                'mouseDownUp': {'type': 'boolean', 'default': True, 'description': ' When true, the mouseUp/Down actions are not performed. Which allows dragging over multiple (small) actions (default True)'},
            },
            'required': ['x', 'y'],
            'additionalProperties': False,
        },
    ),
)

# Scroll by amount (positive = up, negative = down)
ScrollTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='scroll',
        description='Scroll vertically by the given amount. Positive scrolls up; negative scrolls down.',
        parameters={
            'type': 'object',
            'properties': {
                'amount': {'type': 'integer', 'default': 1, 'description': 'Scroll amount (positive up, negative down). Default 1.'},
                'x': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'The x position on the screen where the click happens. None by default. If provided, it will scroll at the given x position. Optional normalized X coordinate in range [0,1]'},
                'y': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'The y position on the screen where the click happens. None by default. If provided, it will scroll at the given y position. Optional normalized Y coordinate in range [0,1]'},
            },
            'required': ['amount'],
            'additionalProperties': False,
        },
    ),
)

HorizontalScrollTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='hscroll',
        description='Scroll horizontally by the given amount. Positive scrolls right; negative scrolls left.',
        parameters={
            'type': 'object',
            'properties': {
                'amount': {'type': 'integer', 'default': 1, 'description': 'Scroll amount (positive left, negative right). Default 1.'},
                'x': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'The x position on the screen where the click happens. None by default. If provided, it will scroll at the given x position. Optional normalized X coordinate in range [0,1]'},
                'y': {'type': 'number', 'format': 'float', 'minimum': 0.0, 'maximum': 1.0, 'description': 'The y position on the screen where the click happens. None by default. If provided, it will scroll at the given y position. Optional normalized Y coordinate in range [0,1]'},
            },
            'required': ['amount'],
            'additionalProperties': False,
        },
    ),
)

# Type text
WriteTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='write',
        description='Type the given text.',
        parameters={
            'type': 'object',
            'properties': {
                'text': {'type': 'string', 'description': 'The text to type'},
                'interval': {'type': 'number', 'default': 0.0, 'description': 'Seconds between each character (default 0.0)'},
            },
            'required': ['text'],
            'additionalProperties': False,
        },
    ),
)

# Press a key (optionally multiple times)
PressTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='press',
        description='Press a single key one or more times.',
        parameters={
            'type': 'object',
            'properties': {
                'key': {'type': 'string', 'description': 'The key to press (e.g., enter, esc, tab)'},
            },
            'required': ['key'],
            'additionalProperties': False,
        },
    ),
)

# Press a combination of keys as a hotkey
HotkeyTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='hotkey',
        description='Press a combination of keys together as a hotkey.',
        parameters={
            'type': 'object',
            'properties': {
                'keys': {
                    'type': 'array',
                    'description': 'Ordered list of keys to press together',
                    'items': {'type': 'string'},
                    'minItems': 2,
                },
            },
            'required': ['keys'],
            'additionalProperties': False,
        },
    ),
)
