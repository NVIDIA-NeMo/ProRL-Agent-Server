from __future__ import annotations

import pytest
from pydantic import ValidationError

from polar.runtime.models import PrepareAction


def test_exec_prepare_action_accepts_bounded_retry_policy() -> None:
    action = PrepareAction(
        type="exec",
        command="git config --global core.pager ''",
        max_attempts=3,
        retry_backoff_seconds=0.1,
    )

    assert action.max_attempts == 3
    assert action.retry_backoff_seconds == 0.1


def test_upload_prepare_action_rejects_exec_retry_policy() -> None:
    with pytest.raises(ValidationError, match="does not support exec retry settings"):
        PrepareAction(
            type="upload_dir",
            source="source",
            target="target",
            max_attempts=2,
        )
