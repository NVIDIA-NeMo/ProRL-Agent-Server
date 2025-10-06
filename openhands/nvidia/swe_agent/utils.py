# type: ignore
import time
import pandas as pd
import numpy as np
import os
import asyncio
import json
import copy
import traceback
import tempfile
from pathlib import Path
from evaluation.benchmarks.swe_bench.run_infer import (  # type: ignore
    initialize_runtime,
    get_instruction,
    complete_runtime,
    is_fatal_evaluation_error,
    codeact_user_response,
    EvalException
)
from evaluation.utils.shared import (  # type: ignore
    EvalMetadata,
    get_default_sandbox_config_for_eval,
    update_llm_config_for_completions_logging,
)
from evaluation.benchmarks.swe_bench.resource.mapping import get_instance_resource_factor

from openhands.core.config.llm_config import LLMConfig
from openhands.runtime.base import Runtime
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
)
from openhands.core.setup import create_agent, create_controller
from openhands.core.main import create_runtime
from openhands.nvidia.controller import run_controller_with_controller
from openhands.controller.state.state import State
from openhands.nvidia.logger import nvidia_logger as logger

from openhands.nvidia.registry import JobDetails, _DEFAULT_AGENT_CONFIG
from openhands.nvidia.utils import process_messages_from_agent_state, is_last_action_finish, get_messages_from_partial_result

DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')

RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
from typing import Callable

def infer_instance_type(instance: dict) -> str:
    _RULES: list[tuple[str, Callable[[dict], bool]]] = [
        ("r2egym", lambda inst: "docker_image" in inst and not pd.isna(inst['docker_image'])),
        ("swebench_multimodal", lambda inst: "image_assets" in inst and not pd.isna(inst['image_assets'])),
        ("swebench", lambda inst: True),
    ]
    if 'data_kind' in instance:
        return instance['data_kind']
    return next(kind for kind, predicate in _RULES if predicate(instance))

from evaluation.benchmarks.swe_bench.eval_infer import (  # type: ignore
    process_instance as _eval_process_instance,
    process_git_patch as _process_git_patch,
    ConditionalImports as _EvalConditionalImports,
    ConditionalImports as ConditionalImports,
)
from evaluation.utils.shared import EvalMetadata  # type: ignore
from openhands.core.config import LLMConfig

from swegym.harness.grading import get_eval_report
from swegym.harness.run_evaluation import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
)
from swegym.harness.test_spec import make_test_spec


def get_instance_docker_image(instance, data_kind = "swebench") -> str:
    if data_kind == "r2egym":
        logger.debug("data_kind is r2egym")
        return instance["docker_image"]
    elif data_kind == "swebench_multimodal":
        logger.debug("data_kind is swebench_multimodal")
        instance_id = instance["instance_id"]
        try:
            repo, issue = instance_id.split('__', 1)
        except ValueError:
            repo, issue = instance_id, ''

        _issue = issue[:-2] if issue.endswith("_0") else issue
        image_name = f"sweb.eval.x86_64.{repo}_1776_{_issue}:latest"
        docker_prefix = "docker.io/swebench"
    elif data_kind == "swebench":
        logger.debug("data_kind is swebench")
        if isinstance(instance, str):
            instance_id = instance
        else:
            instance_id = instance["instance_id"]
        image_name = f"sweb.eval.x86_64.{instance_id}".replace("__", "_s_")
        docker_prefix = DOCKER_IMAGE_PREFIX.rstrip("/")
    else:
        raise ValueError(f"Invalid data kind: {data_kind}")
    result = f"{docker_prefix}/{image_name}".lower()
    return result

def get_config(
    instance: dict,
    metadata: EvalMetadata,
    agent_config: dict=_DEFAULT_AGENT_CONFIG,
) -> OpenHandsConfig:
    data_kind = infer_instance_type(instance)
    base_container_image = get_instance_docker_image(instance, data_kind)
    logger.debug(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
    )
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.runtime_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'
    if data_kind != "r2egym":
        sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
            dataset_name=metadata.dataset or "swebench",
            instance_id=instance['instance_id'],
        )
    else:
        sandbox_config.remote_runtime_resource_factor = 1

    # run as fakeroot
    # Currently set to False as some container require GLIBC_2.38
    sandbox_config.run_as_fakeroot = False

    # Disable browser, stops openhands from spawning 100+ threads
    sandbox_config.browsergym_eval_env = 'skip'

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime='singularity',
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    # Mount OpenHands source directory if provided via OVERWRITE_OPENHANDS_DIR
    mount_dir_env = os.environ.get("OVERWRITE_OPENHANDS_DIR", "").strip()
    if mount_dir_env:
        try:
            mount_path = Path(mount_dir_env).expanduser().resolve()
            if mount_path.exists() and mount_path.is_dir():
                config.sandbox.volumes = f"{mount_path}:/openhands/code:ro"
            else:
                logger.warning(
                    f"OVERWRITE_OPENHANDS_DIR is not a valid directory: {mount_dir_env}. Skipping mount."
                )
        except Exception as e:
            logger.error(
                f"Failed to set mount from OVERWRITE_OPENHANDS_DIR='{mount_dir_env}': {e}. Skipping mount."
            )
    # https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/core/config/agent_config.py

    # Think Tool
    # https://github.com/yidong72/OpenHands_internal/blob/5fe7578f45a847f39e6a22947b504f2167e98117/openhands/agenthub/codeact_agent/tools/think.py#L14
    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=False,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        enable_think=False, # not too sure what this does.
        enable_history_truncation=False, # turn off history truncation
        ensure_thinking_end_properly=agent_config['ensure_thinking_end_properly'], # set to true only if using text based server for training.
        action_timeout=30.0, # 30 seconds per action
        strict_loop_detector=agent_config['strict_loop_detector'], # set to true only if training
    )
    config.set_agent_config(agent_config)
    return config

async def initialize_agents(
        instance: dict,
        llm_config: LLMConfig | None = None,
        sid:str | None = None,
        eval_output_dir:str = "/root",
        git_commit:str = "9f93e8a1532d6e1da4ea702f3dbd31d0f6b2fb3a",
        dataset:str = "swebench",
        data_split:str = "train",
        agent_config: dict = dict(_DEFAULT_AGENT_CONFIG),
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
    # Fall back to a sensible default if the caller does not provide an
    # explicit ``llm_config`` (mirrors the behaviour of the old
    # ``initialize_agents`` implementation that lived in
    # ``scripts/test_local_agent.py``).
    dataset = infer_instance_type(instance)

    if llm_config is None:
        raise ValueError('LLM config is None, cannot initialize.')

    metadata = EvalMetadata(
        agent_class="CodeActAgent",
        llm_config=llm_config,
        agent_config=None,
        max_iterations=agent_config['max_iterations'],
        eval_output_dir=eval_output_dir,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=git_commit,
        dataset=dataset,
        data_split=data_split,
        details={'mode': 'swe'},
        condenser_config=NoOpCondenserConfig(),
    )


    config = get_config(instance, metadata, agent_config)

    metadata.details['runtime_failure_count'] = 0  # type: ignore[index]
    metadata.details['remote_runtime_resource_factor'] = (  # type: ignore[index]
        config.sandbox.remote_runtime_resource_factor
    )

    runtime = create_runtime(config, sid=sid)

    await runtime.connect()
    logger.debug(f"Runtime connected {runtime.sid}")

    try:
        initialize_runtime(runtime, instance, metadata)

    except Exception as e:
        logger.error(f"Error initializing runtime: {e}")
        raise e

    # Return the same triple expected by all current call-sites. The caller
    # already has easy access to *instance* so we no longer return it here
    # (this avoids the previous mismatch where some sites expected three
    # return values).
    return runtime, metadata, config

async def run_agent(
        job_details: JobDetails,
        sid: str | None = None,
    ) -> dict[str, object]:
    runtime = job_details.runtime
    metadata = job_details.metadata
    config = job_details.config
    instance = job_details.instance

    message_action = get_instruction(instance, metadata)
    try:
        agent = create_agent(config)
        job_details.agent = agent
        controller, initial_state = create_controller(
            agent=agent,
            runtime=runtime,
            config=config,
            replay_events=None,
        )
        job_details.controller = controller
        state: State | None = await run_controller_with_controller(
                config=config,
                initial_user_action=message_action,
                sid=sid,
                runtime=runtime,
                agent=agent,
                fake_user_response_fn=codeact_user_response,
                controller=controller,
                initial_state=initial_state,
            )
        job_details.state = state

        # Try to get git patch first
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.debug(
            f'Got git diff for instance {instance['instance_id']}:\n--------\n{git_patch}\n--------'
        )

        # if fatal error, throw EvalError to log.
        if state is None:
            raise EvalException('Final state is None')
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

    except Exception as e:
        logger.error(f"Error running agent: {e}")

    # get messages from agent history
    try:
        run_results = process_messages_from_agent_state(agent, state, job_details) # type: ignore
    except Exception as e:
        logger.error(f"Error while running, failed to retrieve agent messages: {e}")
        raise Exception(f"Failed to retrieve agent messages: {str(e)}")

    return {
        "git_patch": git_patch,
        'success': not bool(state.last_error if state else True),
        'error': state.last_error if state and state.last_error else None,
        'finish': is_last_action_finish(state),
        **run_results,
    }

async def run(instance):
    #agent = initialize_agents(instance)
    #await initialize_runtime_for_agent(agent, instance)
    #return await run_agent(agent, instance)
    runtime, metadata, config = await initialize_agents(instance)
    results  = await run_agent(runtime, metadata, config, instance)
    return results


def _apply_patch_and_evaluate_r2egym(runtime, git_patch: str, instance: dict):

    from openhands.events.action import CmdRunAction, FileReadAction
    from openhands.events.observation import CmdOutputObservation
    import tempfile, json, re

    def _normalise(nodeid: str) -> str:
        nodeid = nodeid.split(" - ")[0]
        parts  = nodeid.split("::")

        if len(parts) >= 3:
            return ".".join(parts[-2:])
        if len(parts) == 2:
            return parts[-1]
        return nodeid

    def _basic_pytest_parser(log: str) -> dict[str, str]:
        status_map: dict[str, str] = {}
        ansi_re = re.compile(r"\u001b\[[0-9;]*m")

        for line in log.splitlines():
            line = ansi_re.sub("", line).strip()
            if not line.startswith(("PASSED", "FAILED", "ERROR", "SKIPPED")):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            status, nodeid_raw = parts
            test_name = _normalise(nodeid_raw)
            if test_name:
                status_map[test_name] = status

        return status_map

    def parse_log_fn(_repo: str):
        return _basic_pytest_parser

    def decolor_dict_keys(d: dict[str, str]) -> dict[str, str]:
        ansi_re = re.compile(r"\u001b\[[0-9;]*m")
        return {ansi_re.sub("", k): v for k, v in d.items()}

    git_patch = _process_git_patch(git_patch)

    try:
        status_action = CmdRunAction(command="cd /testbed && git status --porcelain")
        status_action.set_hard_timeout(10)
        status_obs = runtime.run_action(status_action)
        modified_in_repo: set[str] = set()
        if isinstance(status_obs, CmdOutputObservation):
            for _line in status_obs.content.splitlines():
                _line = _line.rstrip("\n")
                if not _line:
                    continue
                if _line.startswith("??"):
                    path_part = _line[3:].strip()
                    if path_part:
                        modified_in_repo.add(path_part)
                    continue

                if len(_line) < 3:
                    continue
                x_status, y_status = _line[0], _line[1]
                if y_status != " ":
                    path_part = _line[3:].strip()
                    if path_part:
                        modified_in_repo.add(path_part)
        if modified_in_repo:
            filtered_lines: list[str] = []
            include_current = True
            for _l in git_patch.splitlines(keepends=True):
                if _l.startswith("diff --git"):
                    try:
                        parts = _l.split()
                        current_file = parts[3][2:] if len(parts) >= 4 else ""
                        include_current = current_file not in modified_in_repo
                    except Exception:
                        include_current = True
                    if include_current:
                        filtered_lines.append(_l)
                else:
                    if include_current:
                        filtered_lines.append(_l)
            git_patch = "".join(filtered_lines)
    except Exception as e:
        logger.error(f"Failed to filter git patch against repo changes: {e}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        patch_path = os.path.join(tmp_dir, "patch.diff")
        with open(patch_path, "w") as f:
            f.write(git_patch)
        runtime.copy_to(patch_path, "/tmp/")

    apply_cmd = (
        "cd /testbed && "
        f"(git apply -v /tmp/patch.diff && echo '{APPLY_PATCH_PASS}' || "
        f"(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        f"(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo '{APPLY_PATCH_PASS}' || "
        f"echo '{APPLY_PATCH_FAIL}')))"
    )
    action = CmdRunAction(command=apply_cmd)
    action.set_hard_timeout(60)
    obs = runtime.run_action(action)
    assert isinstance(obs, CmdOutputObservation)
    patch_result = obs.content
    # Early return if patch failed to apply
    if APPLY_PATCH_FAIL in patch_result or APPLY_PATCH_PASS not in patch_result:
        return {
            "report": {
                "empty_generation": len(git_patch.strip()) == 0,
                "resolved": False,
                "failed_apply_patch": True,
                "error_eval": False,
                "test_timeout": False,
            },
            "apply_patch_output": patch_result,
            "test_output": "",
        }
    action = CmdRunAction(command="ln -s /r2e_tests /testbed/r2e_tests && bash /testbed/run_tests.sh")
    action.set_hard_timeout(60)
    obs = runtime.run_action(action)
    assert isinstance(obs, CmdOutputObservation)
    test_output = obs.content
    test_output_clean = re.sub(r"\x1b\[[0-9;]*m|\r", "", test_output)
    repo_name = instance.get("repo_name") or instance.get("repo", "").split("/")[-1]
    if repo_name.endswith("_final"):
        repo_name = repo_name[: -len("_final")]

    parsed = parse_log_fn(repo_name)(test_output_clean)
    parsed = decolor_dict_keys(parsed)

    expected_json = instance.get("expected_output_json")
    try:
        print("try to load expected_json")
        expected = json.loads(expected_json)
    except Exception:
        print("failed to load expected_json")
        expected = {}
    expected = decolor_dict_keys(expected)
    strip_suffix = lambda d: {k.split(" - ")[0]: v for k, v in d.items()}
    parsed = strip_suffix(parsed)
    expected = strip_suffix(expected)

    parse_norm     = { _normalise(k): v for k, v in parsed.items() }
    expected_norm  = { _normalise(k): v for k, v in expected.items() }

    # compare
    match = parse_norm == expected_norm

    resolved = len(parse_norm) == len(expected_norm) and all(
        k in expected_norm and parse_norm[k] == expected_norm[k] for k in parse_norm
    )
    return {
        "report": {
            "empty_generation": False,
            "resolved": resolved,
            "failed_apply_patch": False,
            "error_eval": False,
            "test_timeout": False,
        },
        "apply_patch_output": patch_result,
        "test_output": test_output,
    }

def _apply_patch_and_evaluate(
    runtime,
    git_patch: str,
    instance: dict,
):

    kind = infer_instance_type(instance)

    if kind == "r2egym":
        return _apply_patch_and_evaluate_r2egym(runtime, git_patch, instance)

    is_multimodal = kind == "multimodal"

    if is_multimodal:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.run_evaluation import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
        )
        from swebench.harness.test_spec.test_spec import make_test_spec
    else:
        try:
            from swegym.harness.grading import get_eval_report
            from swegym.harness.run_evaluation import (
                APPLY_PATCH_FAIL,
                APPLY_PATCH_PASS,
            )
            from swegym.harness.test_spec import make_test_spec
        except ModuleNotFoundError:
            from swebench.harness.grading import get_eval_report
            from swebench.harness.run_evaluation import (
                APPLY_PATCH_FAIL,
                APPLY_PATCH_PASS,
            )
            from swebench.harness.test_spec.test_spec import make_test_spec

    from openhands.events.action import CmdRunAction
    from openhands.events.observation import CmdOutputObservation

    instance["instance_id"] = instance["instance_id"].lower()
    if "version" not in instance and "base_commit" in instance:
        instance["version"] = instance["base_commit"]

    model_patch = _process_git_patch(git_patch)
    instance_id: str = instance["instance_id"]

    test_spec = make_test_spec(instance)
    import tempfile, os, time

    with tempfile.TemporaryDirectory() as tmp_dir:
        patch_path = os.path.join(tmp_dir, "patch.diff")
        with open(patch_path, "w") as f:
            f.write(model_patch)
        runtime.copy_to(patch_path, "/tmp")

        eval_script_path = os.path.join(tmp_dir, "eval.sh")
        with open(eval_script_path, "w") as f:
            f.write(test_spec.eval_script)
        runtime.copy_to(eval_script_path, "/tmp")

    # Make script executable
    action = CmdRunAction(command="chmod +x /tmp/eval.sh")
    action.set_hard_timeout(5)
    runtime.run_action(action)

    apply_cmd = (
        "cd /testbed && "
        f"(git apply -v /tmp/patch.diff && echo '{APPLY_PATCH_PASS}' || "
        f"(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        f"(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo '{APPLY_PATCH_PASS}' || "
        f"echo '{APPLY_PATCH_FAIL}')))"
    )
    action = CmdRunAction(command=apply_cmd)
    action.set_hard_timeout(30)
    obs = runtime.run_action(action)
    assert isinstance(obs, CmdOutputObservation)
    patch_result = obs.content  # type: ignore[attr-defined]

    if APPLY_PATCH_FAIL in patch_result:
        raise RuntimeError(f"{instance_id}: {APPLY_PATCH_FAIL}\n{patch_result}")
    if APPLY_PATCH_PASS not in patch_result:
        raise RuntimeError(
            f"{instance_id}: Unexpected output when applying patch:\n{patch_result}"
        )

    logger.debug(f"[{instance_id}] {APPLY_PATCH_PASS}:\n{patch_result}")


    pass_string = f"[{instance_id}] {APPLY_PATCH_PASS}:\n{patch_result}"
    log_file = "/tmp/eval_output.log"

    # Run evaluation script and append its output to the existing log
    action = CmdRunAction(command=f"/tmp/eval.sh >> {log_file} 2>&1 & echo $!")
    action.set_hard_timeout(30)
    obs = runtime.run_action(action)
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        raise RuntimeError("Failed to launch evaluation script")

    pid = obs.content.split()[-1].strip()
    logger.debug(f"[{instance_id}] Evaluation started (PID={pid})")

    start_time = time.time()
    timeout = 20 * 60  # 20 minutes
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Evaluation timed out after {timeout} seconds")
        check_action = CmdRunAction(command=f"ps -p {pid} > /dev/null; echo $?")
        check_action.set_hard_timeout(5)
        check_obs = runtime.run_action(check_action)
        if (
            isinstance(check_obs, CmdOutputObservation)
            and check_obs.content.split()[-1].strip() == "1"
        ):
            logger.debug(f"[{instance_id}] Evaluation finished after {elapsed:.0f}s")
            break
        logger.debug(f"[{instance_id}] [{elapsed:.0f}s] Evaluation in progress …")
        time.sleep(30)

    cat_action = CmdRunAction(command=f"cat {log_file}")
    cat_action.set_hard_timeout(5)
    cat_obs = runtime.run_action(cat_action)
    if not (isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0):
        raise RuntimeError("Failed to read evaluation output")

    test_output: str = cat_obs.content  # type: ignore[attr-defined]
    test_output = pass_string + "\n" + test_output

    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = os.path.join(tmp_dir, "logs", instance_id.lower())
        os.makedirs(logs_dir, exist_ok=True)
        test_out_path = os.path.join(logs_dir, "test_output.txt")
        with open(test_out_path, "w") as f:
            f.write(test_output)

        try:
            grading_report = get_eval_report(
                test_spec=test_spec,
                prediction={"model_patch": model_patch, "instance_id": instance_id},
                log_path=test_out_path,
                include_tests_status=True,
            )
        except Exception as e:
            if "got an unexpected keyword argument" in str(e):
                grading_report = get_eval_report(
                    test_spec=test_spec,
                    prediction={"model_patch": model_patch, "instance_id": instance_id},
                    test_log_path=test_out_path,
                    include_tests_status=True,
                )
            else:
                raise  # re-raise if it's not the error we expect
    report = grading_report[instance_id]
    logger.debug(f"[{instance_id}] Grading report: {report}")

    test_result = {
        "report": {
            "empty_generation": False,
            "resolved": report.get("resolved", False),
            "failed_apply_patch": APPLY_PATCH_FAIL in patch_result,
            "error_eval": False,
            "test_timeout": False,
        },
        "apply_patch_output": patch_result,
        "test_output": test_output,
    }

    return test_result

async def evaluate_agent(git_patch: str | None, instance: dict, sid: str | None = None, allow_skip=True):
    # skip evaluation if git_patch is None or empty
    if allow_skip:
        if git_patch is None or len(git_patch) == 0:
            return {'resolved': False}
    else:
        if git_patch is None:
            raise ValueError('Patch is None, cannot evaluate')
    from openhands.utils.async_utils import call_sync_from_async
    runtime = None

    # Create a dummy LLM config to avoid the error of no LLM config
    llm_config = LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        modify_params=False,
        log_completions=False,
    )

    try:
        runtime, _, _ = await initialize_agents(instance, llm_config=llm_config, sid=sid)
        if instance.get("data_kind") == "r2egym":
            test_result = await call_sync_from_async(
                _apply_patch_and_evaluate_r2egym, runtime, git_patch, instance
            )
        else:
            test_result = await call_sync_from_async(
                _apply_patch_and_evaluate, runtime, git_patch, instance
            )
    except Exception as e:
        logger.error(f"Error evaluating agent: {e}")
        raise e
    finally:
        if runtime is not None:
            try:
                runtime.event_stream.close()
            except Exception:
                pass
            await call_sync_from_async(runtime.close)

    return test_result
