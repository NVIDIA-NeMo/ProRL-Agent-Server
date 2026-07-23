from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slime_bridge import rollout as rollout_module
from slime_bridge.config import resolve_polar_slime_config


def _args(**overrides) -> SimpleNamespace:
    values = {
        "polar_rollout_url": "http://rollout:8080",
        "polar_task_template": {
            "agent": {
                "harness": "mini_swe_agent",
                "model_name": "Qwen/Qwen3.5-4B",
                "settings": {"step_limit": 20},
            }
        },
        "polar_task_id_template": "eval-{rollout_id}-{sample.group_index}",
        "polar_max_async_level": 1,
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 2,
        "update_weights_interval": 1,
        "polar_min_complete_accept_fraction": 0.5,
        "polar_early_stop_grace_sessions": 0,
        "hf_checkpoint": "tokenizer",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(prompt: str = "Fix the bug") -> SimpleNamespace:
    return SimpleNamespace(
        prompt=prompt,
        metadata={},
        group_index=3,
        index=0,
    )


class _EvalOutput:
    def __init__(self, *, data, metrics) -> None:
        self.data = data
        self.metrics = metrics


@pytest.mark.asyncio
async def test_external_eval_dataset_metrics_keep_terminal_bench_namespace(
    monkeypatch,
) -> None:
    dataset = SimpleNamespace(name="terminal_bench_2_1")
    args = _args(eval_datasets=[dataset])

    async def run_dataset(**_kwargs):
        return (
            "terminal_bench_2_1",
            {"rewards": [1.0], "valid_count": 1},
            {
                "polar/reward_mean": 1.0,
                "polar/valid_count": 1.0,
                "timing/session_ms/e2e_mean": 123.0,
            },
        )

    monkeypatch.setattr(rollout_module, "_run_eval_dataset", run_dataset)
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_eval_output_type",
        lambda: _EvalOutput,
    )

    result = await rollout_module._run_eval_rollout(args, 0, object())

    assert set(result.data) == {"terminal_bench_2_1"}
    assert result.metrics == {
        "eval/terminal_bench_2_1/reward_mean": 1.0,
        "eval/terminal_bench_2_1/valid_count": 1.0,
        "timing/eval/terminal_bench_2_1/session_ms/e2e_mean": 123.0,
    }


@pytest.mark.asyncio
async def test_fixed_eval_does_not_touch_resumed_training_cursor(monkeypatch) -> None:
    dataset = SimpleNamespace(name="tmax_holdout")
    args = _args(eval_datasets=[dataset])

    class CursorSentinel:
        cursor = 40

        def __getattr__(self, name):
            raise AssertionError(f"fixed eval touched training data source: {name}")

        def __iter__(self):
            raise AssertionError("fixed eval iterated the training data source")

        def __len__(self):
            raise AssertionError("fixed eval measured the training data source")

    async def run_dataset(*, rollout_id, dataset_cfg, **_kwargs):
        assert rollout_id == 39
        assert dataset_cfg is dataset
        return (
            "tmax_holdout",
            {"rewards": [0.0], "valid_count": 1},
            {"polar/reward_mean": 0.0},
        )

    monkeypatch.setattr(rollout_module, "_run_eval_dataset", run_dataset)
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_eval_output_type",
        lambda: _EvalOutput,
    )
    source = CursorSentinel()

    await rollout_module._run_eval_rollout(args, 39, source)

    assert source.cursor == 40


@pytest.mark.asyncio
async def test_multiple_external_eval_datasets_keep_disjoint_metric_namespaces(
    monkeypatch,
) -> None:
    args = _args(
        eval_datasets=[
            SimpleNamespace(name="terminal_bench_2_1"),
            SimpleNamespace(name="secondary_holdout"),
        ]
    )

    async def run_dataset(*, dataset_cfg, **_kwargs):
        return (
            dataset_cfg.name,
            {"rewards": [1.0], "valid_count": 1},
            {
                "polar/reward_mean": 1.0,
                "timing/session_ms/e2e_mean": 10.0,
            },
        )

    monkeypatch.setattr(rollout_module, "_run_eval_dataset", run_dataset)
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_eval_output_type",
        lambda: _EvalOutput,
    )

    result = await rollout_module._run_eval_rollout(args, 3, object())

    assert set(result.data) == {"terminal_bench_2_1", "secondary_holdout"}
    assert result.metrics == {
        "eval/terminal_bench_2_1/reward_mean": 1.0,
        "timing/eval/terminal_bench_2_1/session_ms/e2e_mean": 10.0,
        "eval/secondary_holdout/reward_mean": 1.0,
        "timing/eval/secondary_holdout/session_ms/e2e_mean": 10.0,
    }


@pytest.mark.asyncio
async def test_duplicate_external_eval_dataset_names_fail_instead_of_overwriting(
    monkeypatch,
) -> None:
    args = _args(
        eval_datasets=[
            SimpleNamespace(name="terminal_bench_2_1"),
            SimpleNamespace(name="terminal_bench_2_1"),
        ]
    )

    async def run_dataset(*, dataset_cfg, **_kwargs):
        return dataset_cfg.name, {"rewards": [1.0]}, {"polar/reward_mean": 1.0}

    monkeypatch.setattr(rollout_module, "_run_eval_dataset", run_dataset)
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_eval_output_type",
        lambda: _EvalOutput,
    )

    with pytest.raises(ValueError, match="Duplicate eval dataset name"):
        await rollout_module._run_eval_rollout(args, 0, object())


@pytest.mark.asyncio
async def test_legacy_eval_source_metrics_are_prefixed_before_slime_logging(
    monkeypatch,
) -> None:
    args = _args(eval_datasets=[])
    config = resolve_polar_slime_config(args)

    async def submit_groups(**_kwargs):
        return (
            {"rewards": [0.5], "valid_count": 1},
            {
                "polar/reward_mean": 0.5,
                "timing/session_ms/e2e_mean": 25.0,
            },
        )

    monkeypatch.setattr(rollout_module, "_pull_sample_groups", lambda *_args: [[_sample()]])
    monkeypatch.setattr(rollout_module, "_submit_eval_groups", submit_groups)
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_eval_output_type",
        lambda: _EvalOutput,
    )

    result = await rollout_module._run_eval_rollout(args, 0, object())

    assert set(result.data) == {config.eval_dataset_name}
    assert result.metrics == {
        f"eval/{config.eval_dataset_name}/reward_mean": 0.5,
        f"timing/eval/{config.eval_dataset_name}/session_ms/e2e_mean": 25.0,
    }


def test_eval_payload_carries_runtime_sampling_overrides_and_stable_seed_base() -> None:
    args = _args()
    config = resolve_polar_slime_config(args)
    dataset = SimpleNamespace(
        temperature=0.2,
        top_p=0.9,
        top_k=32,
        max_response_len=4096,
        stop=["</answer>"],
        stop_token_ids=[7, 8],
        min_new_tokens=4,
        repetition_penalty=1.1,
        skip_special_tokens=False,
        no_stop_trim=True,
    )

    baseline = rollout_module._build_task_payload(
        args=args,
        config=config,
        group=[_sample(), _sample()],
        rollout_id=0,
        task_position=5,
        eval_dataset_cfg=dataset,
        eval_dataset_name="holdout",
    )
    final = rollout_module._build_task_payload(
        args=args,
        config=config,
        group=[_sample(), _sample()],
        rollout_id=40,
        task_position=5,
        eval_dataset_cfg=dataset,
        eval_dataset_name="holdout",
    )

    assert baseline["agent"]["settings"]["model_kwargs"] == {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 4096,
        "stop": ["</answer>"],
        "extra_body": {
            "top_k": 32,
            "stop_token_ids": [7, 8],
            "min_tokens": 4,
            "repetition_penalty": 1.1,
            "skip_special_tokens": False,
            "no_stop_trim": True,
        },
    }
    assert baseline["early_stop_min_usable_sessions"] == 2
    assert baseline["dispatch_priority"] == rollout_module._EVAL_DISPATCH_PRIORITY
    seed_key = rollout_module._EVAL_SAMPLING_SEED_METADATA_KEY
    assert baseline["metadata"][seed_key] == final["metadata"][seed_key]

    other_prompt = rollout_module._build_task_payload(
        args=args,
        config=config,
        group=[_sample(), _sample()],
        rollout_id=0,
        task_position=6,
        eval_dataset_cfg=dataset,
        eval_dataset_name="holdout",
    )
    assert other_prompt["metadata"][seed_key] != baseline["metadata"][seed_key]


def test_training_payload_is_not_given_eval_sampling_overrides() -> None:
    args = _args(eval_temperature=0.2, eval_top_p=0.9)
    config = resolve_polar_slime_config(args)

    payload = rollout_module._build_task_payload(
        args=args,
        config=config,
        group=[_sample(), _sample()],
        rollout_id=7,
        task_position=0,
    )

    assert payload["agent"]["settings"] == {"step_limit": 20}
    assert payload["early_stop_min_usable_sessions"] == 1
    assert payload.get("dispatch_priority", 0) == 0
    assert rollout_module._EVAL_SAMPLING_SEED_METADATA_KEY not in payload.get("metadata", {})


def test_eval_max_prompt_len_filters_before_task_submission(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "short", "metadata": {"id": 1}}),
                json.dumps({"prompt": "too long", "metadata": {"id": 2}}),
            ]
        )
    )

    class Sample:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(rollout_module, "_load_sample_type", lambda: Sample)
    monkeypatch.setattr(
        rollout_module,
        "_eval_prompt_token_length",
        lambda _args, prompt: 3 if prompt == "short" else 9,
    )
    args = _args(eval_max_prompt_len=5, input_key="prompt", metadata_key="metadata")
    dataset = SimpleNamespace(
        path=str(path),
        input_key="prompt",
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        n_samples_per_eval_prompt=2,
        custom_generate_function_path=None,
    )

    groups = rollout_module._load_eval_sample_groups(args, dataset)

    assert len(groups) == 1
    assert [sample.prompt for sample in groups[0]] == ["short", "short"]


@pytest.mark.asyncio
async def test_all_failed_eval_zero_fills_rewards_and_returns_error_counts(
    monkeypatch,
) -> None:
    args = _args(min_eval_samples=2)
    config = resolve_polar_slime_config(args)
    failed = SimpleNamespace(task_id="failed", status="failed", results=[])
    placeholder = SimpleNamespace(
        reward={"score": 0.0},
        status="aborted",
        metadata={
            "polar": {
                "session_id": "failed-session",
                "session_status": "ERROR",
                "placeholder": True,
                "trajectory_metadata": {
                    "evaluation": {
                        "verifier_reward_accepted": True,
                        "verifier_exit_code": 0,
                    }
                },
            }
        },
    )

    async def submit(*_args, **_kwargs):
        return failed

    monkeypatch.setattr(rollout_module, "_submit_and_wait_for_task", submit)
    monkeypatch.setattr(
        rollout_module,
        "_convert_eval_task_result_to_samples",
        lambda *_args, **_kwargs: [placeholder],
    )
    monkeypatch.setattr(rollout_module, "_build_metrics", lambda *_args, **_kwargs: {})

    data, metrics = await rollout_module._submit_eval_groups(
        args=args,
        config=config,
        dataset_name="holdout",
        rollout_id=0,
        sample_groups=[[_sample(), _sample()]],
        dataset_cfg=SimpleNamespace(min_eval_samples=2),
    )

    assert data["rewards"] == [0.0, 0.0]
    assert data["all_rewards"] == [0.0, 0.0]
    assert data["valid_count"] == 0
    assert data["accounted_count"] == 2
    assert data["completed_count"] == 0
    assert data["model_failure_count"] == 0
    assert data["error_count"] == 2
    assert data["min_eval_samples"] == 2
    assert metrics["polar/valid_count"] == 0.0
    assert metrics["polar/accounted_count"] == 2.0
    assert metrics["polar/completed_count"] == 0.0
    assert metrics["polar/model_failure_count"] == 0.0
    assert metrics["polar/error_count"] == 2.0
    assert metrics["polar/min_valid_count"] == 2.0
    assert metrics["polar/reward_mean"] == 0.0
    assert metrics["polar/reward_std"] == 0.0
    assert "polar/reward_mean_valid" not in metrics


@pytest.mark.asyncio
async def test_eval_submission_error_is_logged_and_zero_filled(
    monkeypatch,
    caplog,
) -> None:
    args = _args(min_eval_samples=1)
    config = resolve_polar_slime_config(args)

    async def submit(*_args, **_kwargs):
        raise RuntimeError("rollout server unavailable")

    monkeypatch.setattr(rollout_module, "_submit_and_wait_for_task", submit)
    monkeypatch.setattr(rollout_module, "_build_metrics", lambda *_args, **_kwargs: {})

    data, metrics = await rollout_module._submit_eval_groups(
        args=args,
        config=config,
        dataset_name="holdout",
        rollout_id=0,
        sample_groups=[[_sample()]],
        dataset_cfg=SimpleNamespace(min_eval_samples=1),
    )

    assert data["rewards"] == [0.0]
    assert data["valid_count"] == 0
    assert data["accounted_count"] == 1
    assert data["error_count"] == 1
    assert metrics["polar/reward_mean"] == 0.0
    assert "assigning reward 0" in caplog.text
    assert "zero-filled 1 failed/missing sample" in caplog.text


@pytest.mark.asyncio
async def test_eval_payload_build_error_is_isolated_and_zero_filled(
    monkeypatch,
    caplog,
) -> None:
    args = _args(min_eval_samples=1)
    config = resolve_polar_slime_config(args)
    original_build_payload = rollout_module._build_task_payload

    def build_payload(**kwargs):
        if kwargs["task_position"] == 0:
            raise ValueError("sample metadata is malformed")
        return original_build_payload(**kwargs)

    async def submit(_client, _base_url, payload):
        return SimpleNamespace(
            task_id=payload["task_id"],
            status="completed",
            results=[object()],
        )

    completed = SimpleNamespace(
        reward={"score": 1.0},
        status="completed",
        metadata={
            "polar": {
                "session_id": "healthy-session",
                "session_status": "COMPLETED",
                "placeholder": False,
            }
        },
        response_length=1,
    )

    monkeypatch.setattr(rollout_module, "_build_task_payload", build_payload)
    monkeypatch.setattr(rollout_module, "_submit_and_wait_for_task", submit)
    monkeypatch.setattr(
        rollout_module,
        "_convert_eval_task_result_to_samples",
        lambda _config, task_result, *_args, **_kwargs: (
            [completed] if task_result.status == "completed" else []
        ),
    )
    monkeypatch.setattr(rollout_module, "_build_metrics", lambda *_args, **_kwargs: {})

    data, metrics = await rollout_module._submit_eval_groups(
        args=args,
        config=config,
        dataset_name="holdout",
        rollout_id=3,
        sample_groups=[[_sample("broken")], [_sample("healthy")]],
        dataset_cfg=SimpleNamespace(min_eval_samples=1),
    )

    assert data["rewards"] == [1.0, 0.0]
    assert data["valid_count"] == 1
    assert data["accounted_count"] == 2
    assert data["error_count"] == 1
    assert metrics["polar/reward_mean"] == 0.5
    assert metrics["polar/reward_mean_valid"] == 1.0
    assert metrics["polar/reward_std_valid"] == 0.0
    assert "payload-error-eval-holdout-3-0" in caplog.text
    assert "sample metadata is malformed" in caplog.text


@pytest.mark.parametrize(
    "bad_reward",
    ["not-a-number", float("nan"), float("inf"), True, None],
)
@pytest.mark.asyncio
async def test_malformed_eval_reward_is_logged_and_zero_filled(
    monkeypatch,
    caplog,
    bad_reward,
) -> None:
    args = _args(min_eval_samples=1)
    config = resolve_polar_slime_config(args)
    task_result = SimpleNamespace(task_id="eval", status="completed", results=[object()])
    malformed = SimpleNamespace(
        reward={"score": bad_reward},
        status="completed",
        metadata={
            "polar": {
                "session_id": "bad-reward",
                "session_status": "COMPLETED",
                "placeholder": False,
            }
        },
        response_length=1,
    )

    async def submit(*_args, **_kwargs):
        return task_result

    monkeypatch.setattr(rollout_module, "_submit_and_wait_for_task", submit)
    monkeypatch.setattr(
        rollout_module,
        "_convert_eval_task_result_to_samples",
        lambda *_args, **_kwargs: [malformed],
    )

    data, metrics = await rollout_module._submit_eval_groups(
        args=args,
        config=config,
        dataset_name="holdout",
        rollout_id=0,
        sample_groups=[[_sample()]],
        dataset_cfg=SimpleNamespace(min_eval_samples=1),
    )

    assert data["rewards"] == [0.0]
    assert data["valid_count"] == 0
    assert data["error_count"] == 1
    assert metrics["polar/reward_mean"] == 0.0
    assert metrics["polar/reward_std"] == 0.0
    assert metrics["polar/reward_accounted_sessions"] == 1.0
    assert "polar/reward_mean_all_samples" not in metrics
    assert "session bad-reward has malformed reward" in caplog.text


@pytest.mark.asyncio
async def test_eval_optional_metric_error_does_not_discard_valid_reward(
    monkeypatch,
    caplog,
) -> None:
    args = _args(min_eval_samples=1)
    config = resolve_polar_slime_config(args)
    task_result = SimpleNamespace(task_id="eval", status="completed", results=[object()])
    completed = SimpleNamespace(
        reward={"score": 1.0},
        status="completed",
        metadata={
            "polar": {
                "session_id": "valid-reward",
                "session_status": "COMPLETED",
                "placeholder": False,
            }
        },
        response_length=1,
    )

    async def submit(*_args, **_kwargs):
        return task_result

    monkeypatch.setattr(rollout_module, "_submit_and_wait_for_task", submit)
    monkeypatch.setattr(
        rollout_module,
        "_convert_eval_task_result_to_samples",
        lambda *_args, **_kwargs: [completed],
    )
    monkeypatch.setattr(
        rollout_module,
        "_build_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad timing field")),
    )

    data, metrics = await rollout_module._submit_eval_groups(
        args=args,
        config=config,
        dataset_name="holdout",
        rollout_id=0,
        sample_groups=[[_sample()]],
        dataset_cfg=SimpleNamespace(min_eval_samples=1),
    )

    assert data["rewards"] == [1.0]
    assert data["valid_count"] == 1
    assert data["error_count"] == 0
    assert metrics["polar/reward_mean"] == 1.0
    assert metrics["polar/reward_mean_valid"] == 1.0
    assert "optional metric aggregation failed" in caplog.text
    assert "bad timing field" in caplog.text


@pytest.mark.asyncio
async def test_eval_error_with_positive_verifier_reward_is_zero_filled_and_logged(
    monkeypatch,
    caplog,
) -> None:
    args = _args(min_eval_samples=2)
    config = resolve_polar_slime_config(args)
    task_result = SimpleNamespace(task_id="eval", status="completed", results=[object(), object()])

    def eval_sample(
        session_id: str,
        reward: float,
        *,
        status: str,
        trajectory_evaluation: dict | None = None,
    ) -> SimpleNamespace:
        polar = {
            "session_id": session_id,
            "session_status": status,
            "placeholder": False,
        }
        if trajectory_evaluation is not None:
            polar["trajectory_metadata"] = {"evaluation": trajectory_evaluation}
        return SimpleNamespace(
            reward={"score": reward},
            metadata={"polar": polar},
            response_length=1,
        )

    completed_trace_1 = eval_sample("completed", 1.0, status="COMPLETED")
    completed_trace_2 = eval_sample("completed", 1.0, status="COMPLETED")
    trusted_model_failure = eval_sample(
        "model-failure",
        0.75,
        status="FAILED",
        trajectory_evaluation={
            "verifier_reward_accepted": True,
            "verifier_exit_code": 0,
        },
    )

    async def submit(*_args, **_kwargs):
        return task_result

    monkeypatch.setattr(rollout_module, "_submit_and_wait_for_task", submit)
    monkeypatch.setattr(
        rollout_module,
        "_convert_eval_task_result_to_samples",
        lambda *_args, **_kwargs: [
            completed_trace_1,
            completed_trace_2,
            trusted_model_failure,
        ],
    )
    monkeypatch.setattr(
        rollout_module,
        "_build_metrics",
        lambda *_args, **_kwargs: {
            "polar/reward_mean": 0.875,
            "polar/reward_std": 0.125,
        },
    )

    data, metrics = await rollout_module._submit_eval_groups(
        args=args,
        config=config,
        dataset_name="holdout",
        rollout_id=0,
        sample_groups=[[_sample(), _sample()]],
        dataset_cfg=SimpleNamespace(min_eval_samples=2),
    )

    assert data["rewards"] == [1.0, 0.0]
    assert data["all_rewards"] == [1.0, 0.0]
    assert data["valid_count"] == 1
    assert data["accounted_count"] == 2
    assert data["completed_count"] == 1
    assert data["model_failure_count"] == 1
    assert data["error_count"] == 1
    assert metrics["polar/reward_mean"] == 0.5
    assert metrics["polar/reward_std"] == 0.5
    assert metrics["polar/reward_mean_valid"] == 1.0
    assert metrics["polar/reward_std_valid"] == 0.0
    assert metrics["polar/valid_count"] == 1.0
    assert metrics["polar/accounted_count"] == 2.0
    assert metrics["polar/completed_count"] == 1.0
    assert metrics["polar/model_failure_count"] == 1.0
    assert metrics["polar/error_count"] == 1.0
    assert "session model-failure status=FAILED" in caplog.text
    assert "assigning reward 0" in caplog.text
