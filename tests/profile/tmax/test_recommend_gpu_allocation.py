from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "tmax_slime_grpo" / "profile"))
import recommend_gpu_allocation as recommendation  # noqa: E402


def _comparability(
    *,
    harness: str = "test-harness",
    model: str = "test/model-9b",
    code_revision: str = "a" * 40,
) -> dict:
    result = {
        "model": model,
        "checkpoint": "release/checkpoint-000",
        "data_sha256": "d" * 64,
        "global_batch_size": 256,
        "code_revision": code_revision,
        "slime_revision": "b" * 40,
        "megatron_revision": "c" * 40,
        "harness": harness,
        "rollout_batch_size": 8,
        "samples_per_prompt": 32,
        "context_parallel_size": 1,
        "max_tokens_per_gpu": 32768,
        "allow_single_sample_over_token_cap": False,
        "optimizer_cpu_offload": False,
        "min_complete_accept_fraction": 1.0,
        "early_stop_grace_sessions": 64,
    }
    result["fingerprint"] = recommendation.comparability_fingerprint(result)
    return result


def _job(
    label: str,
    actor: int,
    rollout: int,
    allocated: int,
    sessions_per_second: float,
    *,
    mode: str = "fully_async",
    total_steps: int = 6,
    warmup_steps: int = 1,
    staleness: float = 0.25,
    inference_e2e_ms: float = 100.0,
    comparability: dict | None = None,
) -> dict:
    contract = dict(comparability or _comparability())
    step_time = 10.0
    steps = []
    for step_id in range(total_steps):
        steps.append(
            {
                "step": step_id,
                "steady": step_id >= warmup_steps,
                "sources": ["train", "rollout"],
                "step_time_s": step_time,
                "train_wait_time_s": 1.0,
                "accepted_session_count": sessions_per_second * step_time,
                "accepted_session_count_metric": recommendation.TRAINABLE_SESSION_SOURCE,
                "staleness_mean": staleness,
                "inference_e2e_ms_mean": inference_e2e_ms,
                "rollout_success_rate": 1.0,
                "terminal_timeout_session_count": 0.0,
                "terminal_error_session_count": 0.0,
                "queue_backlog_groups": 0.0,
                "trainable_session_fraction": 1.0,
            }
        )
    return {
        "job": label,
        "job_id": label,
        "job_status": "SUCCEEDED",
        "run_path": f"/runs/{label}",
        "comparability": contract,
        "config": {
            "label": label,
            "mode": mode,
            "actor_gpus": actor,
            "rollout_gpus": rollout,
            "allocated_gpus": allocated,
            "async_level": 4 if mode == "fully_async" else 1,
            "global_batch_size": contract["global_batch_size"],
            "rollout_batch_size": contract["rollout_batch_size"],
            "samples_per_prompt": contract["samples_per_prompt"],
            "context_parallel_size": contract["context_parallel_size"],
            "max_tokens_per_gpu": contract["max_tokens_per_gpu"],
            "allow_single_sample_over_token_cap": contract[
                "allow_single_sample_over_token_cap"
            ],
            "optimizer_cpu_offload": contract["optimizer_cpu_offload"],
            "min_complete_accept_fraction": contract[
                "min_complete_accept_fraction"
            ],
            "early_stop_grace_sessions": contract["early_stop_grace_sessions"],
        },
        "warmup_step_ids": list(range(warmup_steps)),
        "step_count_total": total_steps,
        "step_count_steady": total_steps - warmup_steps,
        "steps": steps,
        "steady_state": {
            "throughput": {
                "accepted_trainable_tokens_per_s": sessions_per_second * 100,
                "accepted_trainable_tokens_per_gpu_hour": (
                    sessions_per_second * 3600 / allocated * 100
                ),
                "accepted_group_fraction": 1.0,
            }
        },
        "gpu": {
            "steady_coverage_fraction": 1.0,
            "steady": {"utilization_gpu_pct": {"mean": 50.0}},
            "roles": {
                "actor": {"utilization_gpu_pct": {"mean": 60.0}},
                "rollout": {"utilization_gpu_pct": {"mean": 40.0}},
            },
        },
        "warnings": [],
    }


def _summary(tmp_path: Path, name: str, jobs: list[dict]) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    return path


def test_cross_suite_contract_valid_recommendation_selects_efficiency(
    tmp_path: Path,
) -> None:
    first_contract = _comparability(harness="spilot")
    second_contract = _comparability(harness="tmax", code_revision="e" * 40)
    common_winner_a = _job(
        "a-16t16r", 16, 16, 32, 10.0, comparability=first_contract
    )
    common_winner_b = _job(
        "b-16t16r", 16, 16, 32, 20.0, comparability=second_contract
    )
    slower_40_a = _job("a-8t32r", 8, 32, 40, 10.2, comparability=first_contract)
    slower_40_b = _job("b-8t32r", 8, 32, 40, 20.4, comparability=second_contract)
    failed = _job(
        "failed", 32, 32, 32, 100.0, mode="collocate", comparability=first_contract
    )
    failed["job_status"] = "FAILED"

    analysis = recommendation.build_analysis(
        [
            ("spilot", _summary(tmp_path, "first", [common_winner_a, slower_40_a, failed])),
            ("tmax", _summary(tmp_path, "second", [common_winner_b, slower_40_b])),
        ],
        expected_steps=6,
        warmup_steps=1,
    )

    assert analysis["suites"][0]["recommended_signature"] == "fully_async:16t:16r:32g"
    assert analysis["suites"][1]["recommended_signature"] == "fully_async:16t:16r:32g"
    assert analysis["consensus"]["signature"] == "fully_async:16t:16r:32g"
    assert analysis["consensus"]["status"] == "measured_consistent"
    failed_row = next(
        row for row in analysis["suites"][0]["rows"] if row["label"] == "failed"
    )
    assert failed_row["valid"] is False
    assert any("not SUCCEEDED" in reason for reason in failed_row["exclusion_reasons"])


def test_exact_step_records_sources_timing_and_provenance_are_required(
    tmp_path: Path,
) -> None:
    extra = _job("extra", 8, 24, 32, 10.0, total_steps=7)
    missing_rollout = _job("missing-rollout", 8, 24, 32, 10.0)
    missing_rollout["steps"][1]["sources"] = ["train"]
    missing_time = _job("missing-time", 8, 24, 32, 10.0)
    missing_time["steps"][2]["step_time_s"] = None
    fallback_metric = _job("fallback-metric", 8, 24, 32, 10.0)
    fallback_metric["steps"][3]["accepted_session_count_metric"] = (
        "polar/candidate/trainable_sessions"
    )

    analysis = recommendation.build_analysis(
        [
            (
                "suite",
                _summary(
                    tmp_path,
                    "strict",
                    [extra, missing_rollout, missing_time, fallback_metric],
                ),
            )
        ],
        expected_steps=6,
        warmup_steps=1,
    )
    by_label = {row["label"]: row for row in analysis["suites"][0]["rows"]}
    assert "completed 7/6 expected steps exactly" in by_label["extra"][
        "exclusion_reasons"
    ]
    assert any("missing rollout perf record" in reason for reason in by_label["missing-rollout"]["exclusion_reasons"])
    assert any("no positive step_time_s" in reason for reason in by_label["missing-time"]["exclusion_reasons"])
    assert any("is not proven from" in reason for reason in by_label["fallback-metric"]["exclusion_reasons"])


def test_terminal_success_and_comparability_fingerprint_are_required(
    tmp_path: Path,
) -> None:
    missing_status = _job("missing-status", 8, 24, 32, 10.0)
    missing_status.pop("job_status")
    bad_fingerprint = _job("bad-fingerprint", 8, 24, 32, 10.0)
    bad_fingerprint["comparability"]["fingerprint"] = "0" * 64

    analysis = recommendation.build_analysis(
        [("suite", _summary(tmp_path, "terminal", [missing_status, bad_fingerprint]))],
        expected_steps=6,
        warmup_steps=1,
    )
    rows = analysis["suites"][0]["rows"]
    assert all(row["valid"] is False for row in rows)
    assert any("status is not SUCCEEDED" in reason for reason in rows[0]["exclusion_reasons"])
    assert "comparability fingerprint does not match its fields" in rows[1][
        "exclusion_reasons"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_parallel_size", 2),
        ("max_tokens_per_gpu", 67584),
        ("allow_single_sample_over_token_cap", True),
        ("optimizer_cpu_offload", True),
    ],
)
def test_memory_contract_mismatch_blocks_suite_ranking(
    tmp_path: Path, field: str, value: object
) -> None:
    baseline = _job("baseline", 16, 16, 32, 10.0)
    changed_contract = _comparability()
    changed_contract[field] = value
    changed_contract["fingerprint"] = recommendation.comparability_fingerprint(
        changed_contract
    )
    changed = _job(
        "changed",
        8,
        24,
        32,
        11.0,
        comparability=changed_contract,
    )

    analysis = recommendation.build_analysis(
        [("suite", _summary(tmp_path, field, [baseline, changed]))],
        expected_steps=6,
        warmup_steps=1,
    )

    assert analysis["suites"][0]["status"] == "inconclusive"


def test_suite_fingerprint_mismatch_blocks_ranking(tmp_path: Path) -> None:
    first = _job("first", 16, 16, 32, 10.0)
    second = _job(
        "second",
        8,
        24,
        32,
        11.0,
        comparability=_comparability(model="test/different-model"),
    )
    analysis = recommendation.build_analysis(
        [("suite", _summary(tmp_path, "mismatch", [first, second]))],
        expected_steps=6,
        warmup_steps=1,
    )
    suite = analysis["suites"][0]
    assert suite["status"] == "inconclusive"
    assert suite["recommended_signature"] is None
    assert suite["directional_candidate_signature"] is None


def test_confounders_block_definitive_winner(tmp_path: Path) -> None:
    normal = _job("normal", 16, 16, 32, 10.0, inference_e2e_ms=100.0)
    slow_provider = _job("slow-provider", 8, 24, 32, 10.2, inference_e2e_ms=120.0)
    analysis = recommendation.build_analysis(
        [("suite", _summary(tmp_path, "provider", [normal, slow_provider]))],
        expected_steps=6,
        warmup_steps=1,
    )
    suite = analysis["suites"][0]
    assert suite["status"] == "directional"
    assert suite["recommended_signature"] is None
    assert suite["directional_candidate_signature"] is not None
    assert any("latency differs" in blocker for blocker in suite["decision_blockers"])


def test_missing_confounder_telemetry_is_inconclusive(tmp_path: Path) -> None:
    complete = _job("complete", 16, 16, 32, 10.0)
    missing = _job("missing", 8, 24, 32, 9.8)
    missing["steps"][2]["rollout_success_rate"] = None
    analysis = recommendation.build_analysis(
        [("suite", _summary(tmp_path, "missing", [complete, missing]))],
        expected_steps=6,
        warmup_steps=1,
    )
    suite = analysis["suites"][0]
    assert suite["status"] == "inconclusive"
    assert suite["recommended_signature"] is None
    assert any("missing rollout_success_rate" in item for item in suite["decision_blockers"])


def test_efficiencies_within_five_percent_use_lower_staleness() -> None:
    fast = recommendation.extract_row(
        "suite", _job("fast", 16, 16, 32, 10.0, staleness=0.5), 6, 1
    )
    calm = recommendation.extract_row(
        "suite", _job("calm", 8, 24, 32, 9.6, staleness=0.1), 6, 1
    )
    suite = recommendation.analyze_suite("suite", [fast, calm])
    assert suite["status"] == "measured"
    assert calm["trainable_sessions_per_gpu_hour"] < fast[
        "trainable_sessions_per_gpu_hour"
    ]
    assert calm["efficiency_relative_to_best"] >= 0.95
    assert suite["recommended_label"] == "calm"


def test_conflicting_suite_winners_make_consensus_inconclusive(tmp_path: Path) -> None:
    first_contract = _comparability(harness="spilot")
    second_contract = _comparability(harness="tmax", code_revision="e" * 40)
    analysis = recommendation.build_analysis(
        [
            (
                "spilot",
                _summary(
                    tmp_path,
                    "spilot",
                    [
                        _job("16-16", 16, 16, 32, 10, comparability=first_contract),
                        _job("8-24", 8, 24, 32, 8, comparability=first_contract),
                    ],
                ),
            ),
            (
                "tmax",
                _summary(
                    tmp_path,
                    "tmax",
                    [
                        _job("16-16", 16, 16, 32, 8, comparability=second_contract),
                        _job("8-24", 8, 24, 32, 10, comparability=second_contract),
                    ],
                ),
            ),
        ],
        expected_steps=6,
        warmup_steps=1,
    )
    assert analysis["consensus"]["status"] == "inconclusive"
    assert analysis["consensus"]["signature"] is None
    assert "winners differ" in analysis["consensus"]["reason"]


def test_legacy_summary_is_excluded_without_crashing(tmp_path: Path) -> None:
    legacy = {
        "job": "legacy",
        "config": {"label": "legacy", "mode": "fully_async"},
        "step_count_total": 3,
        "step_count_steady": 2,
        "steps": [{"step": index, "sources": ["train", "rollout"]} for index in range(3)],
    }
    analysis = recommendation.build_analysis(
        [("legacy", _summary(tmp_path, "legacy", [legacy]))],
        expected_steps=3,
        warmup_steps=1,
    )
    row = analysis["suites"][0]["rows"][0]
    assert row["valid"] is False
    assert analysis["suites"][0]["status"] == "no_valid_arms"
    assert "missing comparability contract" in row["exclusion_reasons"]


def test_markdown_and_html_have_required_technical_sections(tmp_path: Path) -> None:
    summary = _summary(tmp_path, "summary", [_job("baseline", 8, 24, 32, 10)])
    analysis = recommendation.build_analysis(
        [("tmax", summary)], expected_steps=6, warmup_steps=1
    )
    assert analysis["suites"][0]["status"] == "inconclusive"
    assert analysis["suites"][0]["recommended_signature"] is None
    rendered_markdown = recommendation.render_markdown(analysis)
    rendered_html = recommendation.render_html(analysis)
    assert "# Training GPU allocation profile" in rendered_markdown
    assert "test/model-9b" in rendered_markdown
    assert "## Technical summary" in rendered_markdown
    assert "## Selection methodology" in rendered_markdown
    assert "## Limitations and robustness checks" in rendered_markdown
    assert "<title>Training GPU allocation profile</title>" in rendered_html
    assert "Technical summary" in rendered_html
    assert "Selection methodology" in rendered_html
    assert "Limitations and robustness checks" in rendered_html

    output_html = tmp_path / "report.html"
    assert (
        recommendation.main(
            [
                "--suite",
                f"tmax={summary}",
                "--expected-steps",
                "6",
                "--warmup-steps",
                "1",
                "--markdown",
                str(tmp_path / "report.md"),
                "--html",
                str(output_html),
                "--json",
                str(tmp_path / "report.json"),
                "--arm-csv",
                str(tmp_path / "arms.csv"),
                "--gpu-role-csv",
                str(tmp_path / "gpu.csv"),
            ]
        )
        == 0
    )
    assert output_html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_invalid_warmup_contract_is_rejected(tmp_path: Path) -> None:
    summary = _summary(tmp_path, "summary", [])
    with pytest.raises(ValueError, match="less than expected_steps"):
        recommendation.build_analysis(
            [("suite", summary)], expected_steps=3, warmup_steps=3
        )
