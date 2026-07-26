from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH_SAFETY = ROOT / "examples" / "path_safety.sh"
SWEGYM = ROOT / "examples" / "swegym_slime_grpo"
TMAX_DATA = ROOT / "examples" / "tmax-15k"
TMAX_TRAIN = ROOT / "examples" / "tmax_slime_grpo"


def _validate(path: str, root: str, prefix: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; shift; polar_validate_removal_path "$@"',
            "path-safety",
            str(PATH_SAFETY),
            "TEST_PATH",
            path,
            root,
            prefix,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _require_absolute(path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; shift; polar_require_absolute_path "$@"',
            "path-safety",
            str(PATH_SAFETY),
            "TEST_PATH",
            path,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_path_safety_accepts_only_normalized_descendants(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    target = root / "polar-job"

    accepted = _validate(str(target), str(root), "polar-")
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == str(target)

    for unsafe in (
        "",
        "relative/path",
        "/",
        str(root),
        str(root / "polar-job" / ".." / ".."),
        str(root / "not-polar"),
    ):
        rejected = _validate(unsafe, str(root), "polar-")
        assert rejected.returncode != 0, unsafe


def test_absolute_output_path_rejects_relative_and_root(tmp_path: Path) -> None:
    assert _require_absolute(str(tmp_path / "logs")).returncode == 0
    assert _require_absolute("relative/logs").returncode != 0
    assert _require_absolute("/").returncode != 0


def test_path_safety_rejects_allowed_root_symlinked_to_root(tmp_path: Path) -> None:
    root_link = tmp_path / "root-link"
    root_link.symlink_to(Path("/"), target_is_directory=True)

    rejected = _validate(str(root_link / "etc"), str(root_link))

    assert rejected.returncode != 0
    assert "resolves to /" in rejected.stderr


def test_path_safety_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "polar-escape").symlink_to(outside, target_is_directory=True)

    rejected = _validate(str(root / "polar-escape"), str(root), "polar-")
    assert rejected.returncode != 0
    assert "outside" in rejected.stderr


def test_safe_remove_deletes_only_validated_target(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    target = root / "polar-job"
    sibling = root / "keep"
    target.mkdir(parents=True)
    sibling.write_text("sentinel\n")
    (target / "generated").write_text("data\n")

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; shift; polar_safe_remove_tree "$@"',
            "path-safety",
            str(PATH_SAFETY),
            "TEST_PATH",
            str(target),
            str(root),
            "polar-",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not target.exists()
    assert sibling.read_text() == "sentinel\n"


@pytest.mark.parametrize(
    "script",
    [
        PATH_SAFETY,
        SWEGYM / "build_sif_shard.sh",
        SWEGYM / "prepare_agent_cli.sh",
        SWEGYM / "prepare_assets.sh",
        SWEGYM / "reconvert_checkpoint.sh",
        SWEGYM / "submit_build_sifs_slurm.sh",
        SWEGYM / "submit_slurm.sh",
        TMAX_DATA / "build_sif_shard.sh",
        TMAX_DATA / "submit_build_sifs_slurm.sh",
        TMAX_TRAIN / "prepare_mini_swe_agent.sh",
    ],
)
def test_guarded_launcher_scripts_are_valid_bash(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)


def test_user_overridable_deletions_and_log_dirs_are_guarded() -> None:
    scripts = "\n".join(
        path.read_text()
        for path in (
            SWEGYM / "build_sif_shard.sh",
            SWEGYM / "prepare_agent_cli.sh",
            SWEGYM / "prepare_assets.sh",
            SWEGYM / "reconvert_checkpoint.sh",
            TMAX_DATA / "build_sif_shard.sh",
            TMAX_TRAIN / "prepare_mini_swe_agent.sh",
        )
    )
    assert 'rm -rf "${POLAR_JOB_CACHE_ROOT}"' not in scripts
    assert 'rm -rf "${AGENT_CLI_DIR}"' not in scripts
    assert "polar_safe_remove_tree" in scripts

    assert "polar_require_absolute_path POLAR_SLURM_LOG_DIR" in (SWEGYM / "submit_slurm.sh").read_text()
    assert "polar_require_absolute_path SIF_BUILD_LOG_DIR" in (SWEGYM / "submit_build_sifs_slurm.sh").read_text()
    assert "polar_require_absolute_path TMAX_SIF_BUILD_LOG_DIR" in (TMAX_DATA / "submit_build_sifs_slurm.sh").read_text()
