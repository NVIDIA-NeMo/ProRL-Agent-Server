OSWORLD_OBSERVATION_FEEDBACK_PROMPT = """Action executed. Please generate the next move according to the UI screenshot and instruction. And you can refer to the previous actions and observations for reflection.

Instruction: {instruction}
"""

ERROR_OBSERVATION_FEEDBACK_PROMPT = """Action failed. Please refer to previous message for the UI screenshot. Please continue working on the task according to the instruction.
Error message: {error_message}.

Instruction: {instruction}
"""