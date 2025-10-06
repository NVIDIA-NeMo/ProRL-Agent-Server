
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
import json
from openhands.nvidia.swe_agent.r2egym_parser import ParsedCommit

def merge_multiple_dataframes(dataframes):
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
    return merged_df

def pre_process_r2egym_instance(r2egym_instance):
    r2egym_instance = pd.Series(r2egym_instance)
    r2egym_instance = r2egym_instance.apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    r2egym_instance['data_source'] = 'swebench'
    r2egym_instance['instance_id'] = (
        r2egym_instance['docker_image'].replace('/', '_').replace(':', '_')
    )
    docker_image = r2egym_instance['docker_image']
    if ':' in docker_image:
        repo_part, version_part = docker_image.split(':', 1)
    else:
        repo_part, version_part = docker_image, 'latest'
    r2egym_instance['repo'] = repo_part
    r2egym_instance['version'] = version_part
    if ('base_commit' not in r2egym_instance) or pd.isna(
        r2egym_instance['base_commit']
    ):
        if 'commit_hash' in r2egym_instance and not pd.isna(
            r2egym_instance['commit_hash']
        ):
            r2egym_instance['base_commit'] = r2egym_instance['commit_hash']
        else:
            r2egym_instance['base_commit'] = version_part
    r2egym_instance['data_kind'] = 'r2egym'
    parsed_commit = ParsedCommit(**json.loads(r2egym_instance['parsed_commit_content']))
    old_commit = parsed_commit.old_commit_hash
    r2egym_instance['old_commit'] = old_commit
    return r2egym_instance

def pre_process_swegym_instance(swebench_instance):
    swebench_instance['instance']["data_kind"] = "swebench"
    swebench_instance['instance']['data_source'] = 'swebench'

    # Convert prompt: string -> numpy.ndarray of a single {role, content} dict
    original_prompt = swebench_instance.get("prompt")
    if isinstance(original_prompt, str):
        swebench_instance["prompt"] = np.array([
            {"role": "user", "content": original_prompt}
        ], dtype=object)
    elif original_prompt is not None and not isinstance(original_prompt, np.ndarray):
        swebench_instance["prompt"] = np.array(original_prompt, dtype=object)

    return swebench_instance

def pre_process_swebench_mm_instance(swebench_mm_instance):
    swebench_mm_instance['data_kind'] = 'swebench_multimodal'
    swebench_mm_instance['data_source'] = 'swebench'
    return swebench_mm_instance

def _normalise(v):
    if isinstance(v, np.ndarray):
        list_v = v.tolist()
        if isinstance(list_v, dict):
            list_v = [list_v]
        return np.array(list_v, dtype=object)
    else:
        raise ValueError(f"Invalid type: {type(v)}")

def _collect_keys(v):
    if isinstance(v, dict):
        return set(v.keys())
    if isinstance(v, pd.Series):
        return set(v.to_dict().keys())
    return set()

def _align_instance(v, all_instance_keys):
    # print(v)
    if isinstance(v, pd.Series):
        v = v.to_dict()
    if not isinstance(v, dict):
        return {k: None for k in all_instance_keys}
    return {k: v.get(k, None) for k in all_instance_keys}

def main() -> None:

    parser = argparse.ArgumentParser(description="Prepare data: merge and pre-process parquet files from a directory (SWE-Bench, R2E-Gym, etc.).")
    parser.add_argument("--data-dir", type=Path, default="/lustre/fsw/portfolios/llmservice/users/shaokunz/project/data/recipe/swe_r2e", help="Directory containing parquet files to process.")

    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"[ERROR] --data-dir {data_dir} is not a valid directory")
    parquet_paths_all = sorted(p.resolve() for p in data_dir.rglob("*.parquet"))
    parquet_paths = [p for p in parquet_paths_all if p.name != "prepared_data.parquet"]

    final_list = []
    for p in parquet_paths:
        df = pd.read_parquet(p)
        # r2egym
        if "docker_image" in df.columns:
            df = df.apply(pre_process_r2egym_instance, axis=1)
            def _wrap_r2egym_row(s: pd.Series) -> pd.Series:
                instance_dict = s.to_dict()
                prompt_text = instance_dict.get("problem_statement", "")
                return pd.Series({
                    "instance": instance_dict,
                    "ability": "coding",
                    "source": "swebench",
                    "prompt": np.array([{ "role": "user", "content": prompt_text }], dtype=object),
                })
            df = df.apply(_wrap_r2egym_row, axis=1)
            final_list.append(df)
        # swebench_mm
        elif "image_assets" in df.columns:
            df = df.apply(pre_process_swebench_mm_instance, axis=1)
            final_list.append(df)
        # swebench
        else:
            df = df.apply(pre_process_swegym_instance, axis=1)
            final_list.append(df)
    final_df = merge_multiple_dataframes(final_list)
    
    if 'prompt' in final_df.columns:
        final_df['prompt'] = final_df['prompt'].apply(_normalise)

    # print(final_df['prompt'][0])

    # Align keys in `instance` across different datasets
    if 'instance' in final_df.columns:
        all_keys_sets = final_df['instance'].apply(_collect_keys).tolist()
        all_instance_keys = set().union(*all_keys_sets) if len(all_keys_sets) > 0 else set()
        final_df['instance'] = final_df['instance'].apply(lambda v: _align_instance(v, all_instance_keys))

    final_df.to_parquet(data_dir / "prepared_data.parquet")
    
if __name__ == "__main__":
    main()
