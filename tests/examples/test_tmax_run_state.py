from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_STATE = ROOT / "examples" / "tmax_slime_grpo" / "run_state.sh"


def _init_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    (path / "tracked.txt").write_text("initial\n")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=TMax Test",
            "-c",
            "user.email=tmax@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def test_run_state_pins_and_persists_source_revisions(tmp_path: Path) -> None:
    prorl = tmp_path / "prorl"
    slime = tmp_path / "slime"
    megatron = tmp_path / "megatron"
    prorl_commit = _init_repo(prorl)
    slime_commit = _init_repo(slime)
    megatron_commit = _init_repo(megatron)
    state = tmp_path / "state.env"
    env = os.environ.copy()
    env.update(
        PRORL_ROOT=str(prorl),
        SLIME_ROOT=str(slime),
        MEGATRON_ROOT=str(megatron),
        STATE=str(state),
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUN_STATE}; "
            'tmax_pin_source_revisions "$PRORL_ROOT" "$SLIME_ROOT" "$MEGATRON_ROOT"; '
            'tmax_write_run_state "$STATE"; '
            'printf "%s|%s|%s" "$TMAX_PRORL_GIT_COMMIT" "$TMAX_SLIME_GIT_COMMIT" "$TMAX_MEGATRON_GIT_COMMIT"',
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout == f"{prorl_commit}|{slime_commit}|{megatron_commit}"
    content = state.read_text()
    assert f"export MEGATRON_DIR={megatron}\n" in content
    assert f"export TMAX_PRORL_GIT_COMMIT={prorl_commit}\n" in content
    assert f"export TMAX_SLIME_GIT_COMMIT={slime_commit}\n" in content
    assert f"export TMAX_MEGATRON_GIT_COMMIT={megatron_commit}\n" in content


def test_source_revision_pin_rejects_checkout_drift(tmp_path: Path) -> None:
    prorl = tmp_path / "prorl"
    slime = tmp_path / "slime"
    megatron = tmp_path / "megatron"
    _init_repo(prorl)
    _init_repo(slime)
    _init_repo(megatron)
    env = os.environ.copy()
    env.update(
        PRORL_ROOT=str(prorl),
        SLIME_ROOT=str(slime),
        MEGATRON_ROOT=str(megatron),
        TMAX_PRORL_GIT_COMMIT="0" * 40,
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUN_STATE}; "
            'tmax_pin_source_revisions "$PRORL_ROOT" "$SLIME_ROOT" "$MEGATRON_ROOT"',
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "ProRL source revision changed" in result.stderr


def test_source_revision_pin_rejects_dirty_worktree(tmp_path: Path) -> None:
    prorl = tmp_path / "prorl"
    slime = tmp_path / "slime"
    megatron = tmp_path / "megatron"
    _init_repo(prorl)
    _init_repo(slime)
    _init_repo(megatron)
    (slime / "untracked.txt").write_text("not pinned\n")
    env = os.environ.copy()
    env.update(
        PRORL_ROOT=str(prorl),
        SLIME_ROOT=str(slime),
        MEGATRON_ROOT=str(megatron),
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUN_STATE}; "
            'tmax_pin_source_revisions "$PRORL_ROOT" "$SLIME_ROOT" "$MEGATRON_ROOT"',
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Slime source worktree is dirty" in result.stderr


def test_source_revision_verify_requires_complete_lock(tmp_path: Path) -> None:
    prorl = tmp_path / "prorl"
    slime = tmp_path / "slime"
    megatron = tmp_path / "megatron"
    _init_repo(prorl)
    _init_repo(slime)
    _init_repo(megatron)
    env = os.environ.copy()
    env.update(
        PRORL_ROOT=str(prorl),
        SLIME_ROOT=str(slime),
        MEGATRON_ROOT=str(megatron),
        TMAX_PRORL_GIT_COMMIT="0" * 40,
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUN_STATE}; "
            'tmax_verify_source_revisions "$PRORL_ROOT" "$SLIME_ROOT" "$MEGATRON_ROOT"',
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "source revision lock is incomplete" in result.stderr
    assert "TMAX_SLIME_GIT_COMMIT" in result.stderr


def test_explicit_run_id_cannot_override_existing_run_state(tmp_path: Path) -> None:
    state = tmp_path / "state.env"
    state.write_text("export RUN_ID=locked-run\nexport SAVE_DIR=/tmp/locked-run\n")
    env = os.environ.copy()
    env["STATE"] = str(state)

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUN_STATE}; "
            'tmax_load_selected_run_state "$STATE" requested-run',
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "requested RUN_ID requested-run does not match locked run state locked-run" in result.stderr
