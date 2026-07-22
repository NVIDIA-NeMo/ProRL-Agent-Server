from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "examples" / "tmax_slime_grpo" / "verify_python_provenance.py"


def _run_checker(project_root: Path, pythonpath: Path, harness: str = ""):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--project-root",
            str(project_root),
            "--harness",
            harness,
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_fake_polar(source_root: Path, *, include_router_method: bool) -> None:
    package = source_root / "polar"
    presets = package / "agent" / "presets"
    presets.mkdir(parents=True)
    for init in (
        package / "__init__.py",
        package / "agent" / "__init__.py",
        presets / "__init__.py",
    ):
        init.write_text("")
    (package / "cli.py").write_text("VALUE = 1\n")
    method = (
        "\n    def _router_completion(self):\n        return None\n"
        if include_router_method
        else ""
    )
    (presets / "spilot_router_runner.py").write_text(
        f"class SpilotOrchestrator:{method or ' pass'}\n"
    )


def test_checker_accepts_frozen_spilot_source_and_attests_hash(tmp_path: Path) -> None:
    project_root = tmp_path / "frozen"
    source_root = project_root / "src"
    _write_fake_polar(source_root, include_router_method=True)

    result = _run_checker(project_root, source_root, "spilot_router")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.removeprefix("[tmax provenance] "))
    assert payload["source_root"] == str(source_root.resolve())
    assert payload["spilot_runner"].startswith(str(source_root.resolve()))
    assert len(payload["spilot_runner_sha256"]) == 64


def test_checker_rejects_mutable_editable_source_outside_frozen_root(
    tmp_path: Path,
) -> None:
    frozen_root = tmp_path / "frozen"
    (frozen_root / "src").mkdir(parents=True)
    mutable_source = tmp_path / "mutable" / "src"
    _write_fake_polar(mutable_source, include_router_method=True)

    result = _run_checker(frozen_root, mutable_source, "spilot_router")

    assert result.returncode != 0
    assert "resolved outside the frozen project source" in result.stderr


def test_checker_rejects_spilot_runner_without_router_completion(tmp_path: Path) -> None:
    project_root = tmp_path / "frozen"
    source_root = project_root / "src"
    _write_fake_polar(source_root, include_router_method=False)

    result = _run_checker(project_root, source_root, "spilot_router")

    assert result.returncode != 0
    assert "missing callable SpilotOrchestrator._router_completion" in result.stderr


def test_tmax_and_spilot_wrappers_pin_project_source_before_shared_launcher() -> None:
    tmax_launcher = (ROOT / "examples" / "tmax_slime_grpo" / "run.sh").read_text()
    spilot_wrapper = (ROOT / "examples" / "spilot_router_slime_grpo" / "run.sh").read_text()

    export = 'export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"'
    assert export in tmax_launcher
    assert '"${SCRIPT_DIR}/verify_python_provenance.py"' in tmax_launcher
    assert 'exec bash "${SCRIPT_DIR}/../tmax_slime_grpo/run.sh" "$@"' in spilot_wrapper
