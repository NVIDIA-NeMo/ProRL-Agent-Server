#!/usr/bin/env python3
"""Select a Skill2Env subset whose task distribution resembles TB 2.1.

The selector is intentionally dependency-free.  It combines prompt/name TF-IDF
similarity with observable task-shape features, excludes non-technical task
families, and caps the number of rows assigned to any one Terminal-Bench task.
The caps prevent a large cluster such as data processing or security auditing
from dominating merely because Skill2Env contains many variants of it.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[a-z][a-z0-9+#._-]{1,}|[0-9]+", re.IGNORECASE)
UUID_SUFFIX_RE = re.compile(r"_[0-9a-f]{32}$")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "create", "do", "for", "from", "have", "help", "i", "if", "in", "into",
    "is", "it", "its", "me", "must", "of", "on", "or", "our", "please",
    "provided", "should", "task", "that", "the", "their", "then", "there",
    "these", "this", "to", "under", "use", "using", "want", "we", "with",
    "within", "write", "you", "your",
}

FEATURE_PATTERNS = {
    "build_compile": re.compile(
        r"\b(build|compile|compiler|linker|makefile|cmake|extension|binary)\b", re.I
    ),
    "debug_repair": re.compile(
        r"\b(debug|diagnos|fix|repair|recover|restore|regression|broken|failure)\w*\b",
        re.I,
    ),
    "systems_infra": re.compile(
        r"\b(server|daemon|process|network|nginx|qemu|kernel|memory|"
        r"container|runtime|distributed|parallel|ssh|linux|filesystem|proxy)\w*\b",
        re.I,
    ),
    "code": re.compile(
        r"\b(implement|code|source|script|function|class|library|package|"
        r"python|javascript|typescript|rust|c\+\+|java|bash|shell)\w*\b",
        re.I,
    ),
    "data_db": re.compile(
        r"\b(data|dataset|database|sql|sqlite|postgres|csv|etl|query|schema|"
        r"pipeline|log|token|index|graph)\w*\b",
        re.I,
    ),
    "ml_science": re.compile(
        r"\b(model|pytorch|torch|tensorflow|inference|training|sampl|statistics|"
        r"scientific|protein|dna|matrix|optimization|simulation|probability)\w*\b",
        re.I,
    ),
    "security": re.compile(
        r"\b(security|vulnerab|exploit|password|hash|crypt|sanitize|xss|secret|"
        r"forensic|malicious|attack|certificate)\w*\b",
        re.I,
    ),
    "install_config": re.compile(
        r"\b(install|configure|setup|provision|deploy|start|run|dependency|"
        r"environment)\w*\b",
        re.I,
    ),
    "media": re.compile(
        r"\b(video|image|audio|render|graphics|html|pdf|document)\w*\b", re.I
    ),
}

REPORT_RE = re.compile(
    r"\b(memo|executive summary|operating review|board|stakeholder|marketing|"
    r"sales|seo|narrative|ethics|psychology|persuasion|strategy offsite|"
    r"recommendation report)\b",
    re.I,
)
NON_TB_FAMILY_RE = re.compile(
    r"(^s4h-|writing|narrative|salary|resume|seo|sales|marketing|stakeholder|"
    r"storyboard|voiceover|social-media|investor|pitch-deck|ux-writing|"
    r"sexual-health|skin-health|sleep-analyzer|weightloss|tcm-|workshop|"
    r"design-sprint|skill-(creator|improver|author|personalizer|miner|template)|"
    r"tool-design-sprint|pm-workflow|contract-negotiation|semantic-gap|"
    r"summarize-|interview|jobs-to-be-done|testimonial|terms-analyzer|"
    r"wireframe|ux-heuristics|sales-engineer|mom-test|shape-up|"
    r"site-architecture|skill-tester|scientific-brainstorming|"
    r"scientific-visualization|usability-test|utility-pm|utm-builder|"
    r"ask-questions|solutions-architect|well-architected)",
    re.I,
)
TECHNICAL_FAMILY_TOKENS = {
    "aflpp", "algorand", "api", "appium", "atheris", "backend", "build",
    "c", "cairo", "cargo", "cloud", "code", "codeql", "compiler", "constant",
    "cosmos", "coverage", "database", "debug", "deps", "detox", "devops",
    "dwarf", "entry", "firebase", "frontend", "fullstack", "fuzzing",
    "genotoxic", "git", "github", "harness", "java", "javascript", "junit",
    "kotlin", "legacy", "libafl", "mssql", "mutation", "mysql", "network",
    "next", "ossfuzz", "playwright", "postgres", "property", "puppeteer",
    "pytest", "python", "qdrant", "replay", "rspec", "rust", "ruzzy", "sarif",
    "scanpy", "scikit", "scvelo", "scvi", "sdk", "seatbelt", "secrets",
    "security", "selenium", "semgrep", "service", "shap", "simpy", "simulation",
    "slurm", "snowflake", "software", "solana", "spark", "sql", "sre", "state",
    "statistical", "statsmodels", "substrate", "supabase", "swift", "sympy",
    "systematic", "terraform", "test", "testing", "testng", "testunit",
    "timesfm", "tinybird", "ton", "torch", "torchdrug", "trailmark",
    "typescript", "umap", "unittest", "vaex", "varlock", "video", "vitest",
    "vue", "vulnerability", "webdriverio", "whisper", "wordpress", "wp",
    "wycheproof", "xunit", "yara", "zarr", "zeroize",
}
ACTION_RE = re.compile(
    r"\b(implement|debug|fix|repair|build|compile|install|configure|recover|"
    r"restore|run|server|service|script|code|database|pipeline)\w*\b",
    re.I,
)
EXT_RE = re.compile(
    r"\b[\w.-]+\.(py|c|cpp|cc|h|rs|go|java|js|ts|sh|sql|json|csv|txt|html|"
    r"xml|png|jpg|mp4|pt|bin|so|yaml|yml|toml|md|pdf|xlsx|pptx)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill2env", required=True, type=Path)
    parser.add_argument("--tb21", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--selection-report",
        type=Path,
        help="Per-row JSONL report; defaults beside --output.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Aggregate JSON report; defaults beside --output.",
    )
    parser.add_argument("--target-size", type=int, default=120)
    parser.add_argument(
        "--core-size",
        type=int,
        default=120,
        help="Number of rows selected under the stricter nearest-task cap.",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=1,
        help="Maximum selected variants of one Skill2Env task family.",
    )
    parser.add_argument(
        "--max-per-tb-task",
        type=int,
        default=5,
        help="Maximum rows assigned to one nearest Terminal-Bench task.",
    )
    parser.add_argument(
        "--fallback-max-per-tb-task",
        type=int,
        default=15,
        help=(
            "Relaxed nearest-task cap used only when the primary cap cannot "
            "reach --target-size."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("row has no prompt messages")
    return "\n".join(
        str(message.get("content", ""))
        for message in prompt
        if isinstance(message, dict)
    )


def task_name(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("task_name", "")).split("/", 1)[-1]


def family_name(row: dict[str, Any]) -> str:
    name = UUID_SUFFIX_RE.sub("", task_name(row))
    name = re.sub(r"^task_(?:s4h-)?", "", name)
    return name


def is_technical_family(family: str) -> bool:
    return bool(set(family.lower().split("-")) & TECHNICAL_FAMILY_TOKENS)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in TOKEN_RE.finditer(text))
        if token not in STOPWORDS
    ]


def tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    tokenized = [tokenize(text) for text in texts]
    document_frequency: collections.Counter[str] = collections.Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    document_count = len(tokenized)
    idf = {
        token: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }
    vectors = []
    for tokens in tokenized:
        counts = collections.Counter(tokens)
        vector = {
            token: (1.0 + math.log(count)) * idf[token]
            for token, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append({token: value / norm for token, value in vector.items()})
    return vectors


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def feature_set(text: str) -> set[str]:
    return {
        feature for feature, pattern in FEATURE_PATTERNS.items() if pattern.search(text)
    }


def extension_set(text: str) -> set[str]:
    return {match.group(1).lower() for match in EXT_RE.finditer(text)}


def set_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.target_size <= 0:
        raise ValueError("--target-size must be positive")
    if args.core_size <= 0:
        raise ValueError("--core-size must be positive")
    if args.max_per_family <= 0:
        raise ValueError("--max-per-family must be positive")
    if args.max_per_tb_task <= 0:
        raise ValueError("--max-per-tb-task must be positive")
    if args.fallback_max_per_tb_task < args.max_per_tb_task:
        raise ValueError(
            "--fallback-max-per-tb-task must be at least --max-per-tb-task"
        )

    skill_rows = load_jsonl(args.skill2env)
    tb_rows = load_jsonl(args.tb21)
    if args.target_size > len(skill_rows):
        raise ValueError("--target-size exceeds the Skill2Env row count")

    skill_texts = [
        f"{family_name(row).replace('-', ' ')}\n{prompt_text(row)}"
        for row in skill_rows
    ]
    tb_texts = [
        f"{task_name(row).replace('-', ' ')}\n{prompt_text(row)}" for row in tb_rows
    ]
    vectors = tfidf_vectors(skill_texts + tb_texts)
    skill_vectors = vectors[: len(skill_rows)]
    tb_vectors = vectors[len(skill_rows) :]

    skill_features = [feature_set(text) for text in skill_texts]
    tb_features = [feature_set(text) for text in tb_texts]
    skill_extensions = [extension_set(text) for text in skill_texts]
    tb_extensions = [extension_set(text) for text in tb_texts]

    best_candidate: dict[int, dict[str, Any]] = {}
    for skill_index, (row, text, vector) in enumerate(
        zip(skill_rows, skill_texts, skill_vectors)
    ):
        family = family_name(row)
        features = skill_features[skill_index]
        strong_technical_features = {
            "build_compile",
            "debug_repair",
            "systems_infra",
            "code",
            "security",
            "install_config",
        }
        eligible = bool(features & strong_technical_features) or {
            "data_db",
            "ml_science",
        }.issubset(features)
        if (
            task_name(row).startswith("task_s4h-")
            or NON_TB_FAMILY_RE.search(family)
            or not is_technical_family(family)
        ):
            eligible = False
        if not eligible:
            continue

        report_penalty = 0.0
        if REPORT_RE.search(text) and not ACTION_RE.search(text):
            report_penalty = 0.18
        elif REPORT_RE.search(text):
            report_penalty = 0.07

        for tb_index, tb_vector in enumerate(tb_vectors):
            text_score = cosine(vector, tb_vector)
            shape_score = set_similarity(
                skill_features[skill_index], tb_features[tb_index]
            )
            artifact_score = set_similarity(
                skill_extensions[skill_index], tb_extensions[tb_index]
            )
            score = (
                0.84 * text_score
                + 0.13 * shape_score
                + 0.03 * artifact_score
                - report_penalty
            )
            candidate = {
                "skill_index": skill_index,
                "skill_task": task_name(row),
                "family": family,
                "matched_tb_index": tb_index,
                "matched_tb_task": task_name(tb_rows[tb_index]),
                "score": score,
                "text_similarity": text_score,
                "shape_similarity": shape_score,
                "artifact_similarity": artifact_score,
                "report_penalty": report_penalty,
            }
            previous = best_candidate.get(skill_index)
            if previous is None or score > previous["score"]:
                best_candidate[skill_index] = candidate

    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    family_counts: collections.Counter[str] = collections.Counter()
    tb_counts: collections.Counter[str] = collections.Counter()

    # Rank globally, while limiting both repeated Skill2Env families and any one
    # TB nearest-neighbour bucket. This preserves quality without letting a
    # large source cluster dominate the selected distribution.
    ranked_candidates = sorted(
        best_candidate.values(),
        key=lambda item: (
            item["score"],
            item["text_similarity"],
            item["skill_task"],
        ),
        reverse=True,
    )

    def add_candidates(max_per_tb_task: int, tier: str, limit: int) -> None:
        for candidate in ranked_candidates:
            if len(selected) >= limit:
                return
            skill_index = candidate["skill_index"]
            family = candidate["family"]
            matched_tb_task = candidate["matched_tb_task"]
            if skill_index in selected_indices:
                continue
            if family_counts[family] >= args.max_per_family:
                continue
            if tb_counts[matched_tb_task] >= max_per_tb_task:
                continue
            selected_candidate = dict(candidate)
            selected_candidate["selection_tier"] = tier
            selected.append(selected_candidate)
            selected_indices.add(skill_index)
            family_counts[family] += 1
            tb_counts[matched_tb_task] += 1

    core_limit = min(args.core_size, args.target_size)
    add_candidates(args.max_per_tb_task, "core", core_limit)
    if len(selected) < args.target_size:
        add_candidates(
            args.fallback_max_per_tb_task, "expanded", args.target_size
        )

    for rank, candidate in enumerate(selected, 1):
        candidate["selection_rank"] = rank

    if len(selected) != args.target_size:
        raise RuntimeError(
            f"selected only {len(selected)} rows; increase --max-per-family "
            "or --fallback-max-per-tb-task"
        )

    # Preserve selection rank in the data and report. The core prefix of a
    # larger run is therefore identical to a smaller run with the same inputs.
    output_rows = [skill_rows[item["skill_index"]] for item in selected]
    report_path = args.selection_report or args.output.with_suffix(
        ".selection.jsonl"
    )
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    write_jsonl(args.output, output_rows)
    write_jsonl(report_path, selected)

    matched_counts = collections.Counter(
        item["matched_tb_task"] for item in selected
    )
    tier_counts = collections.Counter(item["selection_tier"] for item in selected)
    scores = sorted(item["score"] for item in selected)
    summary = {
        "schema_version": 1,
        "method": {
            "name": "tb21_tfidf_task_shape_with_distribution_caps",
            "score_weights": {
                "tfidf_cosine": 0.84,
                "task_shape_jaccard": 0.13,
                "artifact_extension_jaccard": 0.03,
            },
            "target_size": args.target_size,
            "core_size": core_limit,
            "max_per_family": args.max_per_family,
            "max_per_tb_task": args.max_per_tb_task,
            "fallback_max_per_tb_task": args.fallback_max_per_tb_task,
        },
        "source": {
            "skill2env_path": str(args.skill2env.resolve()),
            "skill2env_rows": len(skill_rows),
            "skill2env_sha256": sha256(args.skill2env),
            "tb21_path": str(args.tb21.resolve()),
            "tb21_rows": len(tb_rows),
            "tb21_sha256": sha256(args.tb21),
        },
        "output": {
            "path": str(args.output.resolve()),
            "rows": len(output_rows),
            "sha256": sha256(args.output),
            "selection_report_path": str(report_path.resolve()),
            "selection_report_sha256": sha256(report_path),
        },
        "selection": {
            "unique_families": len(family_counts),
            "covered_tb_tasks": len(matched_counts),
            "tier_counts": dict(sorted(tier_counts.items())),
            "score_min": scores[0],
            "score_median": scores[len(scores) // 2],
            "score_max": scores[-1],
            "matched_tb_task_counts": dict(sorted(matched_counts.items())),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
