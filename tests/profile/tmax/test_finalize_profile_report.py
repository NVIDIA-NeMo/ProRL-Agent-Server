from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


PROFILE_DIR = (
    Path(__file__).resolve().parents[3] / "examples" / "tmax_slime_grpo" / "profile"
)
sys.path.insert(0, str(PROFILE_DIR))

import finalize_profile_report as finalizer  # noqa: E402
import run_profile_report as strict_report  # noqa: E402
from test_run_profile_report import _prepare_inputs, _run_args  # noqa: E402


def _complete_strict_bundle(tmp_path: Path) -> Path:
    data_root, log_root, spilot_manifest, tmax_manifest = _prepare_inputs(
        tmp_path / "inputs"
    )
    strict_dir = tmp_path / "strict"
    result = strict_report.main(
        _run_args(
            data_root,
            log_root,
            spilot_manifest,
            tmax_manifest,
            strict_dir,
        )
    )
    assert result == 0
    assert (strict_dir / "REPORT_COMPLETE").is_file()
    _write_concurrent_campaign(strict_dir)
    return strict_dir


def _write_concurrent_campaign(strict_dir: Path) -> Path:
    analysis = json.loads((strict_dir / "gpu-allocation-analysis.json").read_text())
    suites = {}
    for suite in analysis["suites"]:
        suites[suite["suite"]] = {
            "jobs": {
                row["label"]: int(row["job_id"])
                for row in suite["rows"]
            }
        }
    campaign = {
        "campaign_id": "test-concurrent-campaign",
        "authoritative": True,
        "experiment_contract": {
            "profile_jobs_have_dependencies": False,
            "scheduler_policy": "All eight arms were submitted independently.",
            "requested_topology": {
                "profile_jobs": 8,
                "total_gpus": 272,
                "total_nodes": 34,
            },
        },
        "suites": suites,
        "snapshot": {
            "authoritative_concurrent_profile_jobs": 8,
            "authoritative_concurrent_profile_gpus": 272,
            "authoritative_concurrent_profile_nodes": 34,
            "total_concurrent_profile_jobs": 13,
            "total_concurrent_profile_gpus": 448,
            "overlapping_preliminary_jobs": [91, 92, 93, 94, 95],
            "overlapping_preliminary_gpus": 176,
        },
    }
    campaign_path = strict_dir / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    return campaign_path


def _fake_delivery(artifact_path: Path, html_path: Path) -> dict[str, object]:
    artifact = json.loads(artifact_path.read_text())
    assert artifact["surface"] == "report"
    html_path.write_text(
        "<!doctype html><html><head><title>GPU profile</title></head>"
        f"<body><h1>{artifact['manifest']['title']}</h1></body></html>\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "stages": {
            "validation": "passed",
            "package": "passed",
            "verification": "passed",
        },
        "counts": {"charts": len(artifact["manifest"]["charts"])},
    }


def test_complete_eight_arm_bundle_builds_directional_canonical_report(
    tmp_path: Path,
) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    output_dir = tmp_path / "report"
    # Existing interim artifacts are out of scope and must be preserved.
    output_dir.mkdir()
    interim = output_dir / "gpu_allocation_profiling_interim_report.html"
    interim.write_text("interim sentinel\n")

    result = finalizer.finalize_bundle(
        strict_dir,
        output_dir,
        workspace_root=tmp_path,
        delivery_runner=_fake_delivery,
    )

    assert result["status"] == "complete"
    assert result["strict_valid_arms"] == 8
    assert result["evidence_strength"] == "directional"
    assert interim.read_text() == "interim sentinel\n"
    for name in (
        finalizer.FINAL_EVIDENCE,
        finalizer.FINAL_ARTIFACT,
        finalizer.FINAL_HTML,
        finalizer.FINAL_RECEIPT,
        finalizer.FINAL_COMPLETE,
    ):
        assert (output_dir / name).is_file()
        assert (output_dir / name).stat().st_size > 0

    evidence = json.loads((output_dir / finalizer.FINAL_EVIDENCE).read_text())
    artifact = json.loads((output_dir / finalizer.FINAL_ARTIFACT).read_text())
    complete = json.loads((output_dir / finalizer.FINAL_COMPLETE).read_text())
    assert evidence["completion"]["valid_arms"] == 8
    assert evidence["completion"]["steady_steps_per_arm"] == 2
    assert evidence["evidence_strength"] == "directional"
    assert evidence["presentation_provenance"]["finalizer_path"].endswith(
        "examples/tmax_slime_grpo/profile/finalize_profile_report.py"
    )
    assert evidence["presentation_provenance"]["finalizer_sha256"] == finalizer._sha256(
        finalizer.FINALIZER_PATH
    )
    assert evidence["presentation_qa"]["verification_mode"] == "browser"
    assert "implicit structural-only fallback is rejected" in evidence[
        "presentation_qa"
    ]["verification_scope"]
    assert len(evidence["arm_metrics"]) == 8
    assert len(evidence["chart_map"]) == 6
    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "ready"
    assert len(artifact["snapshot"]["datasets"]["arm_metrics"]) == 8
    assert len(artifact["manifest"]["charts"]) == 6
    assert all(chart["type"] == "bar" for chart in artifact["manifest"]["charts"])
    assert all(
        chart["settings"]["orientation"] == "vertical"
        for chart in artifact["manifest"]["charts"]
    )
    assert {
        chart["encodings"]["y"]["field"] for chart in artifact["manifest"]["charts"]
    } == {
        "sessions_per_second",
        "sessions_per_gpu_hour",
        "weighted_wait_ratio",
        "staleness_mean",
    }
    blocks = artifact["manifest"]["blocks"]
    for index, block in enumerate(blocks):
        if block["type"] == "chart":
            assert index > 0
            assert blocks[index - 1]["type"] == "markdown"
            assert blocks[index - 1]["sourceId"] == "final_evidence"
    assert all(card.get("sourceId") for card in artifact["manifest"]["cards"])
    assert all(chart.get("sourceId") for chart in artifact["manifest"]["charts"])
    assert all(table.get("sourceId") for table in artifact["manifest"]["tables"])
    gpu_role_rows = artifact["snapshot"]["datasets"]["gpu_role_metrics"]
    assert len(gpu_role_rows) == 8
    assert {
        "coverage",
        "overall_utilization",
        "actor_utilization",
        "rollout_utilization",
        "shared_utilization",
        "publishability",
    } <= set(gpu_role_rows[0])
    gpu_role_table = next(
        table
        for table in artifact["manifest"]["tables"]
        if table["id"] == "gpu_role_table"
    )
    assert len(gpu_role_table["columns"]) == 7
    assert next(
        block for block in blocks if block["id"] == "gpu_role_finding"
    )["sourceId"] == "final_evidence"
    summary = next(block for block in blocks if block["id"] == "technical_summary")
    assert "all eight planned arms" in summary["body"]
    assert "Ray and Slurm terminal-state" in summary["body"]
    assert "directional, not a production optimum" in summary["body"]
    assert complete["status"] == "complete"
    assert complete["evidence_strength"] == "directional"
    assert complete["delivery_verification"] == "passed"
    assert complete["presentation_provenance"]["finalizer_sha256"] == evidence[
        "presentation_provenance"
    ]["finalizer_sha256"]
    assert complete["presentation_provenance"]["portable_builder_name"] == (
        "injected_delivery_runner"
    )
    assert complete["qa_limitations"] == []
    assert evidence["completion"]["ray_terminal_gate_complete"] is True
    assert evidence["completion"]["slurm_terminal_gate_complete"] is True
    assert evidence["source_snapshot"]["snapshot_id"] == complete[
        "source_snapshot"
    ]["snapshot_id"]
    snapshot_dir = tmp_path / complete["source_snapshot"]["directory"]
    assert snapshot_dir.name.startswith(finalizer.SOURCE_SNAPSHOT_PREFIX)
    assert evidence["strict_output_dir"] == complete["source_snapshot"]["directory"]
    snapshot_source_ids = {
        "strict_analysis",
        "strict_inputs",
        "slurm_terminal",
        "campaign_metadata",
    }
    assert all(
        source["path"].startswith(complete["source_snapshot"]["directory"] + "/")
        for source in artifact["manifest"]["sources"]
        if source["id"] in snapshot_source_ids
    )
    assert finalizer._sha256(
        snapshot_dir / "slurm-terminal-evidence.json"
    ) == evidence["source_hashes"]["slurm-terminal-evidence.json"]
    for name, digest in complete["outputs"].items():
        assert finalizer._sha256(output_dir / name) == digest


def test_artifact_generation_is_deterministic_for_one_strict_bundle(
    tmp_path: Path,
) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    validated = finalizer.validate_strict_bundle(strict_dir)
    evidence_one = finalizer.build_evidence(validated, tmp_path)
    evidence_two = finalizer.build_evidence(validated, tmp_path)
    assert evidence_one == evidence_two

    output_dir = tmp_path / "report"
    artifact_one = finalizer.build_artifact(
        evidence_one,
        workspace_root=tmp_path,
        output_dir=output_dir,
    )
    artifact_two = finalizer.build_artifact(
        evidence_two,
        workspace_root=tmp_path,
        output_dir=output_dir,
    )
    assert artifact_one == artifact_two
    assert finalizer._json_text(artifact_one) == finalizer._json_text(artifact_two)


def test_concurrent_campaign_metadata_is_validated_and_disclosed(
    tmp_path: Path,
) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    campaign_path = _write_concurrent_campaign(strict_dir)
    validated = finalizer.validate_strict_bundle(strict_dir)
    evidence = finalizer.build_evidence(validated, tmp_path)
    artifact = finalizer.build_artifact(
        evidence,
        workspace_root=tmp_path,
        output_dir=tmp_path / "report",
    )

    execution = evidence["execution_context"]
    assert execution["mode"] == "concurrent"
    assert execution["profile_jobs_have_dependencies"] is False
    assert execution["peak_concurrent_profile_jobs"] == 13
    assert execution["peak_concurrent_profile_gpus"] == 448
    assert len(execution["overlapping_preliminary_job_ids"]) == 5
    assert evidence["source_hashes"]["campaign.json"] == finalizer._sha256(
        campaign_path
    )
    summary = next(
        block
        for block in artifact["manifest"]["blocks"]
        if block["id"] == "technical_summary"
    )["body"]
    limitations = next(
        block
        for block in artifact["manifest"]["blocks"]
        if block["id"] == "limitations"
    )["body"]
    next_steps = next(
        block
        for block in artifact["manifest"]["blocks"]
        if block["id"] == "next_steps"
    )["body"]
    assert "submitted independently with no job dependencies" in summary
    assert "13 profile jobs using 448 GPUs" in summary
    assert "Concurrent-load confounding" in limitations
    assert "13 profile jobs using 448 GPUs" in limitations
    assert "Serialized-order" not in limitations
    assert "without preliminary-job overlap" in next_steps
    assert "six post-warmup steady optimizer steps per repeat" in next_steps
    assert any(
        source["id"] == "campaign_metadata"
        for source in artifact["manifest"]["sources"]
    )


def test_legacy_r4_overlap_fields_remain_compatible(tmp_path: Path) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    campaign_path = strict_dir / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    snapshot = campaign["snapshot"]
    snapshot["overlapping_preliminary_r4_jobs"] = snapshot.pop(
        "overlapping_preliminary_jobs"
    )
    snapshot["overlapping_preliminary_r4_gpus"] = snapshot.pop(
        "overlapping_preliminary_gpus"
    )
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")

    validated = finalizer.validate_strict_bundle(strict_dir)
    execution = finalizer.build_evidence(validated, tmp_path)["execution_context"]
    assert execution["recorded_concurrent_profile_jobs"] == 13
    assert execution["recorded_concurrent_profile_gpus"] == 448
    assert len(execution["overlapping_preliminary_job_ids"]) == 5


def test_noncompleted_slurm_allocation_fails_finalizer_validation(
    tmp_path: Path,
) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    slurm_path = strict_dir / "slurm-terminal-evidence.json"
    slurm = json.loads(slurm_path.read_text())
    first_job_id = next(iter(slurm["jobs"]))
    slurm["jobs"][first_job_id]["state"] = "FAILED"
    slurm_path.write_text(json.dumps(slurm, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        finalizer.FinalizationError,
        match=f"Slurm terminal job {first_job_id} is not COMPLETED",
    ):
        finalizer.validate_strict_bundle(strict_dir)


def test_campaign_job_id_mismatch_fails_closed(tmp_path: Path) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    campaign_path = _write_concurrent_campaign(strict_dir)
    campaign = json.loads(campaign_path.read_text())
    first_suite = next(iter(campaign["suites"].values()))
    first_arm = next(iter(first_suite["jobs"]))
    first_suite["jobs"][first_arm] = 99999999
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        finalizer.FinalizationError,
        match="campaign job ids differ",
    ):
        finalizer.validate_strict_bundle(strict_dir)


def test_missing_campaign_metadata_fails_closed(tmp_path: Path) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    (strict_dir / "campaign.json").unlink()

    with pytest.raises(
        finalizer.FinalizationError,
        match="authoritative campaign.json is missing",
    ):
        finalizer.validate_strict_bundle(strict_dir)


def test_swapped_campaign_arm_bindings_fail_closed(tmp_path: Path) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    campaign_path = strict_dir / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    jobs = next(iter(campaign["suites"].values()))["jobs"]
    first, second = list(jobs)[:2]
    jobs[first], jobs[second] = jobs[second], jobs[first]
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        finalizer.FinalizationError,
        match="arm-to-job bindings differ",
    ):
        finalizer.validate_strict_bundle(strict_dir)


def test_inconsistent_concurrency_arithmetic_fails_closed(tmp_path: Path) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    campaign_path = strict_dir / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["snapshot"]["total_concurrent_profile_jobs"] = 1
    campaign["snapshot"]["total_concurrent_profile_gpus"] = 1
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        finalizer.FinalizationError,
        match="snapshot job count does not reconcile",
    ):
        finalizer.validate_strict_bundle(strict_dir)


def test_incomplete_bundle_fails_closed_without_touching_final_outputs(
    tmp_path: Path,
) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    (strict_dir / "REPORT_COMPLETE").unlink()
    (strict_dir / "REPORT_INCOMPLETE.json").write_text(
        json.dumps({"status": "incomplete", "reasons": ["7/8 valid arms"]})
    )
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    existing = {
        name: f"old {name}\n"
        for name in (
            finalizer.FINAL_EVIDENCE,
            finalizer.FINAL_ARTIFACT,
            finalizer.FINAL_HTML,
            finalizer.FINAL_RECEIPT,
            finalizer.FINAL_COMPLETE,
        )
    }
    for name, body in existing.items():
        (output_dir / name).write_text(body)

    with pytest.raises(finalizer.FinalizationError, match="REPORT_COMPLETE is missing"):
        finalizer.finalize_bundle(
            strict_dir,
            output_dir,
            workspace_root=tmp_path,
            delivery_runner=_fake_delivery,
        )

    assert {name: (output_dir / name).read_text() for name in existing} == existing


def test_delivery_runner_without_verified_stages_cannot_publish(
    tmp_path: Path,
) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    output_dir = tmp_path / "report"

    def incomplete_receipt(artifact_path: Path, html_path: Path) -> dict[str, object]:
        assert artifact_path.is_file()
        html_path.write_text("<!doctype html><title>unverified</title>\n")
        return {"ok": True}

    with pytest.raises(
        finalizer.FinalizationError,
        match="delivery receipt has no stages",
    ):
        finalizer.finalize_bundle(
            strict_dir,
            output_dir,
            workspace_root=tmp_path,
            delivery_runner=incomplete_receipt,
        )

    assert not (output_dir / finalizer.FINAL_COMPLETE).exists()
    assert not (output_dir / finalizer.FINAL_HTML).exists()


def test_structural_only_mode_is_explicit_and_recorded(tmp_path: Path) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    output_dir = tmp_path / "report"

    def structural_delivery(
        artifact_path: Path, html_path: Path
    ) -> dict[str, object]:
        artifact = json.loads(artifact_path.read_text())
        html_path.write_text("<!doctype html><title>structural</title>\n")
        assert artifact["surface"] == "report"
        return {
            "ok": True,
            "stages": {
                "validation": "passed",
                "package": "passed",
                "verification": "structural_only",
            },
        }

    finalizer.finalize_bundle(
        strict_dir,
        output_dir,
        workspace_root=tmp_path,
        delivery_runner=structural_delivery,
        structural_only=True,
    )
    evidence = json.loads((output_dir / finalizer.FINAL_EVIDENCE).read_text())
    complete = json.loads((output_dir / finalizer.FINAL_COMPLETE).read_text())
    assert evidence["presentation_qa"]["verification_mode"] == "structural_only"
    assert "browser_isolation" in evidence["presentation_qa"]
    assert complete["delivery_verification"] == "structural_only"
    assert complete["qa_limitations"]


def test_implicit_structural_only_delivery_is_rejected(tmp_path: Path) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    output_dir = tmp_path / "report"

    def implicit_structural(
        artifact_path: Path, html_path: Path
    ) -> dict[str, object]:
        assert artifact_path.is_file()
        html_path.write_text("<!doctype html><title>structural</title>\n")
        return {
            "ok": True,
            "stages": {
                "validation": "passed",
                "package": "passed",
                "verification": "structural_only",
            },
        }

    with pytest.raises(
        finalizer.FinalizationError,
        match="verification must be passed",
    ):
        finalizer.finalize_bundle(
            strict_dir,
            output_dir,
            workspace_root=tmp_path,
            delivery_runner=implicit_structural,
        )
    assert not (output_dir / finalizer.FINAL_COMPLETE).exists()


def test_strict_source_mutation_during_delivery_cannot_change_frozen_snapshot(
    tmp_path: Path,
) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    output_dir = tmp_path / "report"
    original_analysis = (strict_dir / "gpu-allocation-analysis.json").read_bytes()

    def mutate_source(artifact_path: Path, html_path: Path) -> dict[str, object]:
        assert artifact_path.is_file()
        html_path.write_text("<!doctype html><title>mutated source</title>\n")
        analysis_path = strict_dir / "gpu-allocation-analysis.json"
        analysis = json.loads(analysis_path.read_text())
        analysis["post_validation_mutation"] = True
        analysis_path.write_text(json.dumps(analysis, sort_keys=True) + "\n")
        return {
            "ok": True,
            "stages": {
                "validation": "passed",
                "package": "passed",
                "verification": "passed",
            },
        }

    result = finalizer.finalize_bundle(
        strict_dir,
        output_dir,
        workspace_root=tmp_path,
        delivery_runner=mutate_source,
    )
    assert result["status"] == "complete"
    complete = json.loads((output_dir / finalizer.FINAL_COMPLETE).read_text())
    evidence = json.loads((output_dir / finalizer.FINAL_EVIDENCE).read_text())
    snapshot_dir = tmp_path / complete["source_snapshot"]["directory"]
    snapshot_analysis = snapshot_dir / "gpu-allocation-analysis.json"
    assert snapshot_analysis.read_bytes() == original_analysis
    assert (strict_dir / "gpu-allocation-analysis.json").read_bytes() != original_analysis
    assert evidence["strict_output_dir"] == complete["source_snapshot"]["directory"]
    assert evidence["source_hashes"]["gpu-allocation-analysis.json"] == (
        finalizer._sha256(snapshot_analysis)
    )
    assert complete["source_snapshot"]["manifest_sha256"] == finalizer._sha256(
        tmp_path / complete["source_snapshot"]["manifest"]
    )


def test_mid_publish_failure_invalidates_old_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    output_dir = tmp_path / "report"
    finalizer.finalize_bundle(
        strict_dir,
        output_dir,
        workspace_root=tmp_path,
        delivery_runner=_fake_delivery,
    )
    assert (output_dir / finalizer.FINAL_COMPLETE).is_file()

    original_replace = os.replace

    def fail_on_artifact(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name == finalizer.FINAL_ARTIFACT:
            raise OSError("simulated payload publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(finalizer.os, "replace", fail_on_artifact)
    with pytest.raises(OSError, match="simulated payload publication failure"):
        finalizer.finalize_bundle(
            strict_dir,
            output_dir,
            workspace_root=tmp_path,
            delivery_runner=_fake_delivery,
        )

    assert not (output_dir / finalizer.FINAL_COMPLETE).exists()


def test_two_steady_steps_cannot_be_labeled_measured(tmp_path: Path) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    analysis_path = strict_dir / "gpu-allocation-analysis.json"
    analysis = json.loads(analysis_path.read_text())
    for suite in analysis["suites"]:
        suite["status"] = "measured"
        suite["recommended_signature"] = suite["directional_candidate_signature"]
        suite["recommended_label"] = suite["directional_candidate_label"]
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        finalizer.FinalizationError,
        match="claims measured status with fewer than the required steady steps",
    ):
        finalizer.validate_strict_bundle(strict_dir)


def test_non_utc_strict_inputs_are_rejected(tmp_path: Path) -> None:
    strict_dir = _complete_strict_bundle(tmp_path)
    inputs_path = strict_dir / "report-inputs.json"
    inputs = json.loads(inputs_path.read_text())
    inputs["log_timezone"] = "America/Los_Angeles"
    inputs_path.write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n")

    with pytest.raises(finalizer.FinalizationError, match="did not use UTC"):
        finalizer.validate_strict_bundle(strict_dir)
