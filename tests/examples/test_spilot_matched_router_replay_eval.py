from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest
import yaml

from polar.agent.presets.spilot_router_runner import (
    _assign_slots,
    build_initial_router_messages,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "examples"
    / "spilot_router_slime_grpo"
    / "matched_router_replay_eval.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "spilot_matched_router_replay_eval", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


module = _load_script()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _policy_config(tmp_path: Path, *, shuffle_slots: bool = False) -> Path:
    config = {
        "polar_task_template": {
            "agent": {
                "harness": "spilot_router",
                "settings": {
                    "model_pool": {
                        "M0": {
                            "model": "pool/qwen3.6-27b",
                            "card": {"name": "Qwen", "strength": "efficient"},
                            "cost_weight": 1.0,
                            "model_kwargs": {"temperature": 1.0},
                        },
                        "M1": {
                            "model": "pool/gpt-5.5",
                            "card": {"name": "GPT", "strength": "capable"},
                            "cost_weight": 15.0,
                            "model_kwargs": {"max_completion_tokens": 16_384},
                        },
                    },
                    "shuffle_slots": shuffle_slots,
                    "shuffle_seed": 0,
                    "router_max_tokens": 192,
                    "router_model_kwargs": {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    },
                },
            },
            "evaluator": {
                "strategy": "spilot_harbor",
                "config": {
                    "require_valid_action": True,
                    "cost_penalty_lambda": 0.2,
                    "cost_normalizer": 30.0,
                },
            },
        }
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "eval.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt": [{"role": "user", "content": "Fix the deterministic bug."}],
                "metadata": {"task_name": "task-0"},
            }
        ],
    )
    return path


def _actor_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_manifest = checkpoint_dir / ".export_complete.json"
    _write_json(checkpoint_manifest, {"schema_version": 1, "iteration": 59})
    _write_json(checkpoint_dir / "config.json", {"model_type": "qwen3_5"})

    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    ready = actor_dir / "router-ready.json"
    _write_json(
        ready,
        {
            "schema_version": 1,
            "base_url": "http://router.test/v1",
            "model_id": "spilot/frozen-router-iter-59",
            "slurm_job_id": "14046109",
        },
    )
    _write_json(
        actor_dir / "config.json",
        {
            "model_id": "spilot/frozen-router-iter-59",
            "model_path": str(checkpoint_dir.resolve()),
            "slurm_job_id": "14046109",
        },
    )
    api_key = actor_dir / "api-key"
    api_key.write_text("test-secret-that-must-not-be-persisted\n", encoding="utf-8")
    return ready, checkpoint_manifest, api_key


def _router_response(
    url: str,
    _headers: dict[str, str],
    payload: dict[str, Any],
    _timeout: float,
) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"x-request-id": "router-request-1"},
        json={
            "id": "completion-1",
            "model": payload["model"],
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"action":"ROUTE","model_slot":"M0"}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 677,
                "completion_tokens": 12,
                "total_tokens": 689,
            },
        },
        request=httpx.Request("POST", url),
    )


def _prepare_collection(tmp_path: Path) -> dict[str, Path]:
    data = _dataset(tmp_path)
    policy = _policy_config(tmp_path)
    ready, checkpoint_manifest, api_key = _actor_files(tmp_path)
    output = tmp_path / "matched-replay"
    args = SimpleNamespace(
        data=data,
        policy_config=policy,
        router_ready_json=ready,
        router_api_key_file=api_key,
        checkpoint_manifest=checkpoint_manifest,
        output_dir=output,
        start_index=0,
        max_tasks=1,
        replicates=1,
        seed=20260715,
        request_timeout=17.0,
    )
    assert module.run_collect(args, request_fn=_router_response) == 0
    return {
        "data": data,
        "policy": policy,
        "output": output,
        "ready": ready,
        "checkpoint_manifest": checkpoint_manifest,
        "api_key": api_key,
    }


def _write_complete_forced_ledger(
    tmp_path: Path,
    *,
    collection_output: Path,
) -> Path:
    collection_manifest = json.loads(
        (collection_output / "collection_manifest.json").read_text(encoding="utf-8")
    )
    collection_plan = collection_manifest["plan"]
    dataset_row = collection_plan["dataset_rows"][0]
    plan = {
        "schema_version": module.FORCED_LEDGER_SCHEMA_VERSION,
        "run_id": "strict-forced-pair",
        "eval_only": True,
        "actor_invoked": False,
        "include_qwen35_baseline": False,
        "forward_seed_to_pool": True,
        "data_sha256": collection_plan["data_sha256"],
        "start_index": collection_plan["start_index"],
        "max_tasks": collection_plan["max_tasks"],
        "replicates": collection_plan["replicates"],
        "seed": collection_plan["seed"],
        "dataset_indices": [dataset_row["dataset_index"]],
        "dataset_rows": [dataset_row],
        "candidates": [
            {
                "pool_model": "pool/qwen3.6-27b",
                "endpoint_model": "nvidia/qwen/qwen3.6-27b",
                "label": "Qwen3.6-27B",
            },
            {
                "pool_model": "pool/gpt-5.5",
                "endpoint_model": "openai/openai/gpt-5.5",
                "label": "GPT-5.5",
            },
        ],
        "expected_result_count": 2,
        "semantic_config": {"sha256": "1" * 64},
        "semantic_identity_sha256": "2" * 64,
        "implementation_manifest": {"sha256": "3" * 64},
        "task_asset_manifest": {"sha256": "4" * 64},
    }
    plan_sha = module.canonical_sha256(plan)
    allocation_attempts = [
        {
            "allocation_attempt_id": "a" * 64,
            "slurm_job_id": "14050000",
            "teardown_verification": {"status": "verified", "reason": None},
        }
    ]
    collection = {
        "status": "complete",
        "result_set_complete": True,
        "expected_result_count": 2,
        "collected_result_count": 2,
        "missing_result_count": 0,
        "valid_result_count": 2,
        "invalid_result_count": 0,
    }
    manifest = {
        **plan,
        "plan_sha256": plan_sha,
        "allocation_attempts": allocation_attempts,
        "collection": collection,
        "content_integrity": {"status": "verified", "failures": []},
        "publication": {
            "status": "teardown_verified",
            "allocation_attempt_id": "a" * 64,
        },
    }
    summary = {
        **manifest,
        "publication": {"status": "published", "allocation_attempt_id": "a" * 64},
        "final_metrics_status": "published",
        "final_metrics": {"candidate_metrics": {}},
    }
    seed = module.pair_seed(
        collection_plan["seed"], dataset_row["dataset_index"], 0
    )
    rewards = {
        "pool/qwen3.6-27b": 0.5,
        "pool/gpt-5.5": 1.0,
    }
    endpoint_models = {
        candidate["pool_model"]: candidate["endpoint_model"]
        for candidate in plan["candidates"]
    }
    results: list[dict[str, Any]] = []
    for index, model in enumerate(module.PRIMARY_MODELS):
        reward = rewards[model]
        work_sha256 = module.canonical_sha256(
            {
                "plan_sha256": plan_sha,
                "dataset_index": dataset_row["dataset_index"],
                "dataset_row_sha256": dataset_row["dataset_row_sha256"],
                "replicate": 0,
                "pair_seed": seed,
                "candidate_model": model,
                "candidate_endpoint_model": endpoint_models[model],
            }
        )
        results.append(
            {
                "schema_version": module.FORCED_LEDGER_SCHEMA_VERSION,
                "dataset_index": dataset_row["dataset_index"],
                "dataset_row_sha256": dataset_row["dataset_row_sha256"],
                "replicate": 0,
                "pair_seed": seed,
                "candidate_model": model,
                "candidate_endpoint_model": endpoint_models[model],
                "forced_eval_plan_sha256": plan_sha,
                "forced_eval_work_sha256": work_sha256,
                "task_id": f"forced-task-{index}",
                "session_id": f"forced-session-{index}",
                "allocation_attempt_id": "a" * 64,
                "valid": True,
                "task_status": "completed",
                "session_status": "COMPLETED",
                "actor_invoked": False,
                "eval_only": True,
                "pool_status": "completed",
                "pool_return_code": 0,
                "pool_timed_out": False,
                "reward": reward,
                "harbor_outcome_reward": reward,
                "reported_reward": reward,
            }
        )

    output = tmp_path / "forced-ledger"
    output.mkdir()
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "summary.json", summary)
    _write_jsonl(output / "results.jsonl", results)
    return output


def _rewrite_tampered_decisions(
    output: Path,
    mutate: Any,
) -> list[dict[str, Any]]:
    decisions_path = output / "decisions.jsonl"
    rows = module.read_jsonl(decisions_path)
    mutate(rows[0])
    rows[0]["decision_sha256"] = module.canonical_sha256(
        {
            key: value
            for key, value in rows[0].items()
            if key != "decision_sha256"
        }
    )
    _write_jsonl(decisions_path, rows)
    manifest_path = output / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decisions_sha256"] = module.sha256_file(decisions_path)
    _write_json(manifest_path, manifest)
    return rows


def _refresh_request_provenance(
    row: dict[str, Any],
    *,
    instruction: str,
) -> None:
    mapping = row["slot_mapping"]
    candidates = [
        module.Candidate(
            slot=slot,
            model=metadata["model"],
            card=metadata["card"],
            cost_weight=metadata["cost_weight"],
        )
        for slot, metadata in sorted(
            mapping.items(), key=lambda item: int(item[0][1:])
        )
    ]
    messages = build_initial_router_messages(instruction, candidates, label_mode="anonymous")
    row["slot_mapping_fingerprint"] = module.canonical_sha256(mapping)
    row["messages_sha256"] = module.canonical_sha256(messages)
    payload = module._openai_chat_payload(
        model=row["request_model"],
        messages=messages,
        model_kwargs=row["model_kwargs"],
    )
    row["request_payload_sha256"] = module.canonical_sha256(payload)


def test_production_and_counterfactual_assignments_swap_semantic_models(
    tmp_path: Path,
) -> None:
    policy_path = _policy_config(tmp_path, shuffle_slots=True)
    policy = module.load_policy_snapshot(policy_path)
    cell = module.DatasetCell(
        dataset_index=0,
        prompt="Fix it",
        metadata={},
        instruction="Fix it",
        dataset_row_sha256="d" * 64,
    )

    production_payload, production = module.build_router_request(
        cell=cell,
        policy=policy,
        request_model="spilot/frozen-router",
        sampling_seed=71,
        permutation="production",
    )
    swapped_payload, swapped = module.build_router_request(
        cell=cell,
        policy=policy,
        request_model="spilot/frozen-router",
        sampling_seed=71,
        permutation="counterfactual_swap",
    )

    raw_pool = yaml.safe_load(policy_path.read_text(encoding="utf-8"))[
        "polar_task_template"
    ]["agent"]["settings"]["model_pool"]
    production_candidates = _assign_slots(
        raw_pool,
        shuffle=True,
        seed=0,
        stable_seed=71,
        session_id="ignored-when-stable-seed-is-set",
        task_id="ignored-when-stable-seed-is-set",
        label_mode="anonymous",
    )
    production_mapping = {
        candidate.slot: candidate.public_metadata()
        for candidate in production_candidates
    }
    assert production["slot_mapping"] == production_mapping
    assert swapped["slot_mapping"]["M0"] == production_mapping["M1"]
    assert swapped["slot_mapping"]["M1"] == production_mapping["M0"]
    assert production["slot_mapping_fingerprint"] != swapped["slot_mapping_fingerprint"]
    assert production_payload["seed"] == swapped_payload["seed"] == 71
    assert production_payload["model"] == swapped_payload["model"]
    assert production_payload["messages"] != swapped_payload["messages"]


def test_router_request_uses_exact_production_prompt_helper(tmp_path: Path) -> None:
    policy = module.load_policy_snapshot(_policy_config(tmp_path))
    cell = module.DatasetCell(
        dataset_index=0,
        prompt="Repair unicode: λ",
        metadata={},
        instruction="Repair unicode: λ",
        dataset_row_sha256="d" * 64,
    )
    assigned = module._candidate_assignment(
        policy, sampling_seed=9, permutation="production"
    )
    payload, provenance = module.build_router_request(
        cell=cell,
        policy=policy,
        request_model="spilot/frozen-router",
        sampling_seed=9,
        permutation="production",
    )

    exact = build_initial_router_messages(cell.instruction, assigned, label_mode="anonymous")
    assert payload["messages"] == exact
    assert provenance["messages_sha256"] == module.canonical_sha256(exact)
    assert exact == [
        {
            "role": "system",
            "content": (
                "You are SPilot, a routing policy for software-engineering agents. "
                "Do not solve the task yourself. Select one candidate to run a full "
                "coding-agent attempt. Your response must be exactly one JSON object, "
                "with no markdown, commentary, or extra keys."
            ),
        },
        {
            "role": "user",
            "content": (
                "TASK:\nRepair unicode: λ\n\nAVAILABLE MODEL SLOTS:\n"
                '[{"model_card": {"name": "Qwen", "strength": "efficient"}, '
                '"model_slot": "M0"}, {"model_card": {"name": "GPT", '
                '"strength": "capable"}, "model_slot": "M1"}]\n\n'
                "Choose the first action using exactly this schema:\n"
                '{"action":"ROUTE","model_slot":"M0"}\n'
                "Replace M0 with one available slot. Output only the JSON object."
            ),
        },
    ]


def test_collect_records_both_permutations_from_mocked_http(tmp_path: Path) -> None:
    policy = module.load_policy_snapshot(_policy_config(tmp_path))
    cell = module.DatasetCell(
        dataset_index=3,
        prompt="Fix it",
        metadata={},
        instruction="Fix it",
        dataset_row_sha256="d" * 64,
    )
    requests: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def request(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        requests.append((url, headers, payload, timeout))
        return _router_response(url, headers, payload, timeout)

    decisions = module.collect_router_decisions(
        cells=[cell],
        replicates=1,
        base_seed=123,
        policy=policy,
        actor_identity={
            "base_url": "http://router.test/v1",
            "request_model": "spilot/frozen-router",
        },
        api_key="ephemeral-secret",
        collection_plan_sha256="c" * 64,
        request_timeout=17.0,
        request_fn=request,
    )

    assert [row["permutation"] for row in decisions] == list(module.PERMUTATIONS)
    assert [row["selected_slot"] for row in decisions] == ["M0", "M0"]
    assert [row["selected_model"] for row in decisions] == [
        "pool/qwen3.6-27b",
        "pool/gpt-5.5",
    ]
    assert len({row["pair_seed"] for row in decisions}) == 1
    assert all(row["valid"] is True for row in decisions)
    assert all(row["response_headers"] == {"x-request-id": "router-request-1"} for row in decisions)
    assert all(row["decision_sha256"] == module.canonical_sha256({
        key: value for key, value in row.items() if key != "decision_sha256"
    }) for row in decisions)
    assert [request[0] for request in requests] == [
        "http://router.test/v1/chat/completions",
        "http://router.test/v1/chat/completions",
    ]
    assert all(request[1]["Authorization"] == "Bearer ephemeral-secret" for request in requests)
    assert all(request[3] == 17.0 for request in requests)
    assert "ephemeral-secret" not in json.dumps(decisions)


@pytest.mark.parametrize(
    "tampering",
    ["paired_seed", "selected_model", "request_prompt_mapping"],
)
def test_collection_validation_rejects_self_rehashed_provenance_tampering(
    tmp_path: Path,
    tampering: str,
) -> None:
    context = _prepare_collection(tmp_path)

    def mutate(row: dict[str, Any]) -> None:
        if tampering == "paired_seed":
            row["pair_seed"] += 1
            row["sampling_seed"] += 1
            row["model_kwargs"]["seed"] += 1
            _refresh_request_provenance(
                row, instruction="Fix the deterministic bug."
            )
        elif tampering == "selected_model":
            selected_slot = row["selected_slot"]
            other_slot = next(
                slot for slot in row["slot_mapping"] if slot != selected_slot
            )
            row["selected_model"] = row["slot_mapping"][other_slot]["model"]
        elif tampering == "request_prompt_mapping":
            row["slot_mapping"]["M0"]["card"]["name"] = "tampered-card"
            row["selected_model"] = row["slot_mapping"][row["selected_slot"]][
                "model"
            ]
            _refresh_request_provenance(
                row, instruction="Fix the deterministic bug."
            )
        else:  # pragma: no cover - protects the parametrized fixture itself
            raise AssertionError(f"unknown tampering case: {tampering}")

    rows = _rewrite_tampered_decisions(context["output"], mutate)
    unhashed = {
        key: value
        for key, value in rows[0].items()
        if key != "decision_sha256"
    }
    assert rows[0]["decision_sha256"] == module.canonical_sha256(unhashed)
    manifest = json.loads(
        (context["output"] / "collection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["decisions_sha256"] == module.sha256_file(
        context["output"] / "decisions.jsonl"
    )

    with pytest.raises(module.LedgerError):
        module.validate_collection_output(context["output"])


def test_finalize_publishes_strict_three_arm_matched_outcomes(
    tmp_path: Path,
) -> None:
    context = _prepare_collection(tmp_path)
    forced = _write_complete_forced_ledger(
        tmp_path, collection_output=context["output"]
    )

    result = module.run_finalize(
        SimpleNamespace(
            output_dir=context["output"],
            forced_output_dir=forced,
            data=context["data"],
            policy_config=context["policy"],
        )
    )

    assert result == 0
    summary = json.loads(
        (context["output"] / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["final_metrics_status"] == "published_strict_complete_replay"
    assert summary["publication"] == {
        "status": "published_strict_complete_replay",
        "strict_denominator_complete": True,
    }
    metrics = summary["final_metrics"]
    assert metrics["denominator"] == {
        "expected_cell_count": 1,
        "complete_cell_count": 1,
        "missing_cell_count": 0,
        "strict_complete_case": True,
    }
    assert metrics["arms"]["forced_qwen3.6_27b"]["accuracy_outcome"]["mean"] == 0.5
    assert metrics["arms"]["forced_gpt_5.5"]["accuracy_outcome"]["mean"] == 1.0
    production = metrics["arms"]["router_production_one_call_replay"]
    assert production["selected_model_counts"] == {"pool/qwen3.6-27b": 1}
    assert production["cost"]["mean"] == 1.0
    assert production["reward"]["mean"] == pytest.approx(0.5 * (1.0 - 0.2 / 30.0))
    permutation = metrics["slot_permutation_diagnostic"]
    assert permutation["counterfactual_selected_model_counts"] == {"pool/gpt-5.5": 1}
    assert permutation["semantic_selection_agreement_fraction"] == 0.0
    assert permutation["presented_slot_agreement_fraction"] == 1.0

    replay_rows = module.read_jsonl(context["output"] / "replay_rows.jsonl")
    assert len(replay_rows) == 1
    assert replay_rows[0]["router_arms"]["production"]["selected_model"] == "pool/qwen3.6-27b"
    assert replay_rows[0]["router_arms"]["counterfactual_swap"]["selected_model"] == "pool/gpt-5.5"
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in context["output"].iterdir()
        if path.is_file()
    )
    assert "test-secret-that-must-not-be-persisted" not in persisted


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("incomplete", "forced ledger collection is not complete"),
        ("unpublished", "forced ledger parent teardown is not verified"),
        ("wrong_seed", "forced outcome has wrong pair seed"),
    ],
)
def test_finalize_fails_closed_for_non_authoritative_forced_ledgers(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    context = _prepare_collection(tmp_path)
    forced = _write_complete_forced_ledger(
        tmp_path, collection_output=context["output"]
    )
    manifest_path = forced / "manifest.json"
    summary_path = forced / "summary.json"
    results_path = forced / "results.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if failure == "incomplete":
        manifest["collection"].update(
            {
                "status": "partial",
                "result_set_complete": False,
                "collected_result_count": 1,
                "missing_result_count": 1,
            }
        )
        summary["collection"] = dict(manifest["collection"])
        _write_json(manifest_path, manifest)
        _write_json(summary_path, summary)
    elif failure == "unpublished":
        manifest["publication"] = {
            "status": "pending_teardown",
            "allocation_attempt_id": "a" * 64,
        }
        _write_json(manifest_path, manifest)
    elif failure == "wrong_seed":
        rows = module.read_jsonl(results_path)
        rows[0]["pair_seed"] += 1
        _write_jsonl(results_path, rows)
    else:  # pragma: no cover - protects the parametrized fixture itself
        raise AssertionError(f"unknown failure case: {failure}")

    with pytest.raises(module.LedgerError, match=message):
        module.run_finalize(
            SimpleNamespace(
                output_dir=context["output"],
                forced_output_dir=forced,
                data=context["data"],
                policy_config=context["policy"],
            )
        )

    collection_summary = json.loads(
        (context["output"] / "summary.json").read_text(encoding="utf-8")
    )
    assert collection_summary["final_metrics"] is None
    assert collection_summary["final_metrics_status"] == "awaiting_forced_outcomes"
    assert not (context["output"] / "replay_rows.jsonl").exists()


def test_finalize_rejects_valid_shape_but_tampered_forced_work_identity(
    tmp_path: Path,
) -> None:
    context = _prepare_collection(tmp_path)
    forced = _write_complete_forced_ledger(
        tmp_path, collection_output=context["output"]
    )
    results_path = forced / "results.jsonl"
    rows = module.read_jsonl(results_path)
    assert rows[0]["forced_eval_work_sha256"] != "f" * 64
    rows[0]["forced_eval_work_sha256"] = "f" * 64
    _write_jsonl(results_path, rows)

    with pytest.raises(module.LedgerError, match="work identity"):
        module.run_finalize(
            SimpleNamespace(
                output_dir=context["output"],
                forced_output_dir=forced,
                data=context["data"],
                policy_config=context["policy"],
            )
        )

    collection_summary = json.loads(
        (context["output"] / "summary.json").read_text(encoding="utf-8")
    )
    assert collection_summary["final_metrics"] is None
    assert not (context["output"] / "replay_rows.jsonl").exists()
