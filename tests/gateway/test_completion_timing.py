from __future__ import annotations

from types import SimpleNamespace

from polar.gateway.engine import POLAR_INFERENCE_TIMINGS_KEY
from polar.gateway.server import _completion_metadata


def test_completion_metadata_moves_private_inference_timing_out_of_response() -> None:
    response = {
        "choices": [{"message": {"content": "ok"}}],
        POLAR_INFERENCE_TIMINGS_KEY: [{"e2e_ms": 12.5, "queue_ms": 2.0}],
    }
    session = SimpleNamespace(
        session_id="session-1",
        task_id="task-1",
        metadata={"source": "test"},
    )

    metadata = _completion_metadata(session, response)

    assert POLAR_INFERENCE_TIMINGS_KEY not in response
    assert metadata == {
        "source": "test",
        "session_id": "session-1",
        "task_id": "task-1",
        "inference_timings": [{"e2e_ms": 12.5, "queue_ms": 2.0}],
    }


def test_completion_metadata_drops_malformed_private_timing() -> None:
    response = {POLAR_INFERENCE_TIMINGS_KEY: {"e2e_ms": 12.5}}

    metadata = _completion_metadata(None, response)

    assert metadata == {}
    assert response == {}


def test_completion_metadata_revalidates_private_timing_fields() -> None:
    response = {
        POLAR_INFERENCE_TIMINGS_KEY: [
            {
                "e2e_ms": 12.5,
                "queue_ms": float("nan"),
                "secret": "must-not-be-persisted",
            }
        ]
    }

    metadata = _completion_metadata(None, response)

    assert metadata == {"inference_timings": [{"e2e_ms": 12.5}]}
    assert response == {}
