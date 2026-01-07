# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""NeAR workspace initialization and filesystem setup."""

from evaluation.utils.shared import EvalMetadata
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.runtime.base import Runtime


def setup_near_workspace(
    runtime: Runtime,
    instance: dict,
    metadata: EvalMetadata,
) -> None:
    """Initialize NeAR workspace structure in container.

    Creates:
    - /workspace/notes.md - Session notes
    - /workspace/assets/ - Resources directory
    - /workspace/mounted/ - User filesystem mounts
    - /workspace/outputs/ - Output deliverables
    - /skills/ - Empty skills directory (for future use)

    Args:
        runtime: Runtime instance
        instance: Task instance
        metadata: Evaluation metadata

    Raises:
        RuntimeError: If workspace initialization fails
    """
    logger.info('-' * 30)
    logger.info('BEGIN NeAR Workspace Setup')
    logger.info('-' * 30)

    # Extract task description for notes.md
    task_description = instance.get('problem_statement') or instance.get(
        'query', 'No task specified'
    )

    # Create workspace directory structure and initialize notes.md
    workspace_cmd = f"""
mkdir -p /workspace/assets /workspace/mounted /workspace/outputs /skills && \\
cat > /workspace/notes.md << 'NOTES_EOF'
# Research Notes

## Task
{task_description}

## Findings
- Start documenting your research findings here
- Add timestamps and sources for your discoveries
- Organize information systematically

## Next Steps
- [ ] List action items as you progress

NOTES_EOF
echo 'NeAR workspace initialized' && \\
echo 'Workspace structure:' && \\
ls -la /workspace/ && \\
echo 'Skills directory:' && \\
ls -la /skills/
"""

    action = CmdRunAction(command=workspace_cmd)
    action.set_hard_timeout(10)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f'Failed to initialize NeAR workspace: {str(obs)}')

    # Verify critical directories exist
    verify_cmd = """
test -d /workspace && test -d /workspace/outputs && test -d /skills && \\
test -f /workspace/notes.md && \\
echo 'All directories verified successfully'
"""

    action = CmdRunAction(command=verify_cmd)
    action.set_hard_timeout(5)
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(
            'Workspace verification failed: Critical directories not found'
        )

    logger.info('NeAR workspace setup complete')
    logger.info('-' * 30)
