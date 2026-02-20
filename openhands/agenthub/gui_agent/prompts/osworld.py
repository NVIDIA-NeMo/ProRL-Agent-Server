OSWORLD_OBSERVATION_FEEDBACK_PROMPT = """Action executed. Please generate the next move according to the UI screenshot and instruction.

Instruction: {instruction}
"""

ERROR_OBSERVATION_FEEDBACK_PROMPT = """Action failed. Please continue working on the task according to the instruction.
Error message: {error_message}

Instruction: {instruction}
"""