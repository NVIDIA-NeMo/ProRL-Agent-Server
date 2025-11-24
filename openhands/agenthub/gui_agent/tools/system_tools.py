from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

FailTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='fail',
        description="Use this tool if you can not complete the task. Do not easily use this tool, try your best to do the task.",
        parameters={
            'type': 'object',
            'properties': {},
        },
    ),
)

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish',
        description="Use this tool when you have successfully completed the user's requested task",
        parameters={
            'type': 'object',
            'properties': {},
        },
    ),
)

WaitTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='wait',
        description="Use this tool when you think the screen UI is not updated. Waiting for 1 second is usually enough for the UI to update.",
        parameters={
            'type': 'object',
            'properties': {
                'seconds': {'type': 'integer', 'default': 1, 'description': 'The number of seconds to wait. Default 1.'},
            },
        },
    ),
)