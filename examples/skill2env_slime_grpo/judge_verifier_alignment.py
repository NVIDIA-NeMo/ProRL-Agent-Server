#!/usr/bin/env python3
"""Judge Skill2Env prompt/verifier alignment with an Inference Hub model."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_SKILL_DIR = Path("/home/haozh/.codex/skills/inference-hub")
TEXT_SUFFIXES = {
    "", ".cfg", ".conf", ".csv", ".html", ".ini", ".js", ".json", ".jsonl",
    ".md", ".ndjson", ".py", ".sh", ".sql", ".toml", ".ts", ".txt", ".xml",
    ".yaml", ".yml",
}
VERDICTS = {"reasonable", "partially_reasonable", "unreasonable", "uncertain"}


def private_config(skill_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in (skill_dir / "credentials.env").read_text().splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            key, value = raw.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def request_response(
    config: dict[str, str],
    prompt: str,
    *,
    max_output_tokens: int,
    timeout: float,
) -> dict:
    body = json.dumps({
        "model": config["OPENAI_MODEL"],
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": "high"},
    }).encode()
    request = urllib.request.Request(
        config["OPENAI_BASE_URL"].rstrip("/") + "/responses",
        data=body,
        headers={
            "Authorization": "Bearer " + config["OPENAI_API_KEY"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status // 100 != 2:
            raise RuntimeError(f"Inference Hub returned HTTP {response.status}")
        return json.load(response)


def response_text(payload: dict) -> str:
    texts: list[str] = []
    for item in payload.get("output") or []:
        for content in item.get("content") or []:
            text = content.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts)


def parse_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("judge response did not contain a JSON object")


def smoke(config: dict[str, str], timeout: float) -> None:
    payload = request_response(
        config,
        "Reply with exactly OK",
        max_output_tokens=16,
        timeout=timeout,
    )
    if response_text(payload).strip() != "OK":
        raise RuntimeError("Inference Hub smoke response was not exactly OK")


def load_manifest(path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row["metadata"]["task_name"].rsplit("/", 1)[-1]
        result[task_id] = row
    return result


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\x00" not in sample


def render_files(
    files: list[Path],
    *,
    root: Path,
    total_limit: int,
    per_file_limit: int,
) -> tuple[str, list[str]]:
    blocks: list[str] = []
    omitted: list[str] = []
    used = 0
    for path in sorted(set(files)):
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = str(path)
        if not path.is_file() or not is_text_file(path):
            omitted.append(relative + " [non-text]")
            continue
        try:
            content = path.read_text(errors="replace")
        except OSError as exc:
            omitted.append(relative + f" [read error: {exc}]")
            continue
        if len(content) > per_file_limit:
            content = content[:per_file_limit] + "\n[TRUNCATED]\n"
            omitted.append(relative + " [truncated]")
        block = f"\n===== {relative} =====\n{content}\n"
        if used + len(block) > total_limit:
            omitted.append(relative + " [total limit]")
            continue
        blocks.append(block)
        used += len(block)
    return "".join(blocks), omitted


def output_root(eval_root: Path, task_id: str) -> Path | None:
    for run_name in ("run", "replacements"):
        candidate = eval_root / run_name / "tasks" / task_id
        if candidate.is_dir():
            return candidate
    return None


def make_prompt(
    task_id: str,
    row: dict,
    result: dict,
    eval_root: Path,
) -> tuple[str, dict]:
    task_dir = Path(row["metadata"]["task_dir"])
    environment = task_dir / "environment"
    tests = task_dir / "tests"
    out = output_root(eval_root, task_id)

    user_files = [task_dir / "instruction.md", task_dir / "task.toml"]
    if environment.is_dir():
        user_files.extend(path for path in environment.rglob("*") if path.is_file())
    user_text, user_omitted = render_files(
        user_files,
        root=task_dir,
        total_limit=100_000,
        per_file_limit=30_000,
    )

    verifier_files: list[Path] = []
    if tests.is_dir():
        verifier_files.extend(path for path in tests.rglob("*") if path.is_file())
    verifier_text, verifier_omitted = render_files(
        verifier_files,
        root=task_dir,
        total_limit=90_000,
        per_file_limit=50_000,
    )

    artifact_text = ""
    artifact_omitted: list[str] = []
    diagnostics = ""
    if out is not None:
        workspace = out / "overlay" / "upper" / "workspace"
        if workspace.is_dir():
            artifacts = [
                path for path in workspace.rglob("*")
                if path.is_file() and "/.codex/" not in str(path)
            ]
            artifact_text, artifact_omitted = render_files(
                artifacts,
                root=workspace,
                total_limit=60_000,
                per_file_limit=25_000,
            )
        verifier_log = out / "logs" / "verifier"
        if verifier_log.is_dir():
            log_files = [
                path for path in verifier_log.iterdir()
                if path.is_file() and path.name != "reward.json"
            ]
            diagnostics, _ = render_files(
                log_files,
                root=verifier_log,
                total_limit=20_000,
                per_file_limit=10_000,
            )

    system_instruction = """You are auditing whether an evaluation task and its
hidden verifier are fair and aligned. Judge the task design, not whether the
candidate answer is elegant.

Evidence policy:
1. "USER-VISIBLE MATERIAL" is disclosed to the agent. Requirements explicitly
   stated there, including schemas/checkers referenced in those files, are fair.
2. "HIDDEN VERIFIER" was not visible to the agent. A hidden check is fair only
   when it tests an explicit requirement or a necessary, unambiguous semantic
   consequence of user-visible material.
3. The official solution is intentionally absent and must not be treated as a
   disclosure source.
4. Exact hidden field names, enum spellings, counts, planted IDs, oracle labels,
   or lexical patterns are unfair unless user-visible material specifies them.
5. Flag brittle false negatives: naive substring/regex checks that reject a
   semantically correct negation, ordering checks not requested, one acceptable
   answer hard-coded as the only answer, or metrics that reward hidden oracle
   overlap.
6. Do not call a task unreasonable merely because it is difficult or strict.
7. Focus especially on checks that caused this candidate to receive less than
   full reward, while also note any major verifier flaw visible in the code.

Verdicts:
- reasonable: failed checks are materially disclosed and verifier is robust.
- partially_reasonable: core task is valid, but at least one scored check is
  hidden, ambiguous, or brittle enough to cause an unfair deduction.
- unreasonable: core/full success substantially depends on undisclosed or
  invalid requirements, or verifier is dominated by false-negative-prone checks.
- uncertain: supplied material is insufficient to decide.

Return exactly one JSON object with this schema:
{
  "verdict": "reasonable|partially_reasonable|unreasonable|uncertain",
  "confidence": 0.0,
  "prompt_contract_score": 0,
  "verifier_robustness_score": 0,
  "failed_checks_assessment": "fair|mixed|unfair|uncertain",
  "issues": [
    {
      "severity": "minor|major|critical",
      "category": "hidden_contract|brittle_match|ambiguous_requirement|oracle_dependency|unsupported_check|other",
      "verifier_requirement": "...",
      "disclosed_where": "... or not disclosed",
      "explanation": "..."
    }
  ],
  "fair_failed_checks": ["..."],
  "unfair_failed_checks": ["..."],
  "reasoning": "concise evidence-based conclusion",
  "recommended_action": "keep|clarify_prompt|fix_verifier|clarify_and_fix|remove_task"
}
Scores are integers 0..4, where 4 is fully aligned/robust."""

    prompt = f"""{system_instruction}

TASK ID: {task_id}

OBSERVED VERIFIER RESULT:
{json.dumps(result, ensure_ascii=False, indent=2)}

USER-VISIBLE MATERIAL:
{user_text}

USER-VISIBLE FILES OMITTED OR TRUNCATED:
{json.dumps(user_omitted, ensure_ascii=False)}

HIDDEN VERIFIER:
{verifier_text}

HIDDEN VERIFIER FILES OMITTED OR TRUNCATED:
{json.dumps(verifier_omitted, ensure_ascii=False)}

CANDIDATE-GENERATED ARTIFACTS:
{artifact_text}

GENERATED ARTIFACTS OMITTED OR TRUNCATED:
{json.dumps(artifact_omitted, ensure_ascii=False)}

VERIFIER DIAGNOSTICS:
{diagnostics}
"""
    metadata = {
        "user_omitted": user_omitted,
        "verifier_omitted": verifier_omitted,
        "artifact_omitted": artifact_omitted,
        "prompt_chars": len(prompt),
    }
    return prompt, metadata


def validate_judgment(value: dict) -> None:
    if value.get("verdict") not in VERDICTS:
        raise ValueError(f"invalid verdict: {value.get('verdict')!r}")
    for key in ("prompt_contract_score", "verifier_robustness_score"):
        score = value.get(key)
        if not isinstance(score, int) or not 0 <= score <= 4:
            raise ValueError(f"invalid {key}: {score!r}")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError(f"invalid confidence: {confidence!r}")


def judge_one(
    task_id: str,
    row: dict,
    result: dict,
    args: argparse.Namespace,
    config: dict[str, str],
) -> dict:
    started = time.time()
    prompt, context_metadata = make_prompt(task_id, row, result, args.eval_root)
    error = None
    raw_text = ""
    judgment = None
    attempts = 0
    for attempts in range(1, args.max_retries + 2):
        try:
            payload = request_response(
                config,
                prompt,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
            )
            raw_text = response_text(payload)
            judgment = parse_json_object(raw_text)
            validate_judgment(judgment)
            break
        except Exception as exc:
            error = str(exc)
            if attempts <= args.max_retries:
                time.sleep(min(2 ** attempts, 10))
    record = {
        "task": task_id,
        "status": "completed" if judgment is not None else "error",
        "judgment": judgment,
        "attempts": attempts,
        "elapsed_seconds": round(time.time() - started, 1),
        "error": error if judgment is None else None,
        "context": context_metadata,
    }
    task_out = args.output_dir / "tasks" / task_id
    task_out.mkdir(parents=True, exist_ok=True)
    (task_out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if judgment is None:
        (task_out / "raw_response.txt").write_text(raw_text)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path, default=DEFAULT_SKILL_DIR)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=2400)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = private_config(args.skill_dir)
    smoke(config, args.timeout)
    manifest = load_manifest(args.manifest)
    evaluation = json.loads(args.eval_summary.read_text())
    non_full = {
        result["task"]: result
        for result in evaluation["results"]
        if result["reward"] < 1.0
    }
    if args.task_id:
        missing = [task for task in args.task_id if task not in non_full]
        if missing:
            raise SystemExit(f"requested tasks are not non-full results: {missing}")
        selected = args.task_id
    else:
        selected = sorted(non_full)

    records: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                judge_one, task_id, manifest[task_id], non_full[task_id],
                args, config,
            ): task_id
            for task_id in selected
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            compact = {
                "task": record["task"],
                "status": record["status"],
                "verdict": (
                    record["judgment"].get("verdict")
                    if record["judgment"] else None
                ),
                "elapsed_seconds": record["elapsed_seconds"],
                "error": record["error"],
            }
            print(json.dumps(compact), flush=True)

    records.sort(key=lambda item: item["task"])
    completed = [item for item in records if item["judgment"] is not None]
    counts = {
        verdict: sum(item["judgment"]["verdict"] == verdict for item in completed)
        for verdict in sorted(VERDICTS)
    }
    summary = {
        "requested": len(selected),
        "completed": len(completed),
        "model": config["OPENAI_MODEL"],
        "verdict_counts": counts,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    return 0 if len(completed) == len(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
