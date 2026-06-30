from __future__ import annotations

from polar.agent.models import AgentSpec
from polar.rollout.manager import _request_for_sample
from polar.rollout.models import TaskRequest


def _request() -> TaskRequest:
    return TaskRequest(
        task_id="eval-task",
        instruction="Fix it",
        num_samples=2,
        agent=AgentSpec(
            harness="mini_swe_agent",
            model_name="Qwen/Qwen3.5-4B",
            settings={"model_kwargs": {"temperature": 0.2}},
        ),
        metadata={"eval_sampling_seed_base": 1234},
    )


def test_eval_samples_receive_distinct_stable_seeds_without_mutating_request() -> None:
    request = _request()

    first = _request_for_sample(request, 0)
    second = _request_for_sample(request, 1)
    repeated = _request_for_sample(_request(), 1)

    assert first.agent.settings["sampling_seed"] == 1234
    assert second.agent.settings["sampling_seed"] == 1235
    assert repeated.agent.settings["sampling_seed"] == 1235
    assert "sampling_seed" not in request.agent.settings
    assert first.agent.settings["model_kwargs"] == {"temperature": 0.2}
    assert first.metadata["eval_sampling_seed"] == 1234
    assert second.metadata["eval_sampling_seed"] == 1235


def test_training_request_without_eval_seed_metadata_is_unchanged() -> None:
    request = _request().model_copy(update={"metadata": {}})

    assert _request_for_sample(request, 0) is request
