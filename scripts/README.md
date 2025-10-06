# Scripts Directory

This directory contains small utilities for running OpenHands and managing datasets/images. Each section starts with what the script is for, then shows usage and parameters.

## start_server.py

Launches the OpenHands async server (FastAPI) to process evaluation requests. Manages lifecycle, worker pools, LLM endpoints, and timeouts.

Usage:
```bash
export LOG_LEVEL=ERROR
export DEBUG=False
python start_server.py [OPTIONS]
```

Parameters:
- `--max-init-workers`  Maximum initialization workers (default: 64)
- `--max-run-workers`   Maximum run workers (default: 64)
- `--timeout`           Global job timeout in seconds (default: 300)
- `--host`              Bind host (default: 0.0.0.0)
- `--port`              Bind port (default: 8006)
- `--allow-skip-eval`   Skip eval if git_patch is None/empty (default: True)
- `--reward-server-ip`  Reward server IP for math/code/reasoning gym (default: [])

Notes:
- Endpoints include `/start`, `/stop`, `/status`, `/add_llm_server`, `/clear_llm_server`, `/process`.
- Add LLM addresses before sending `/process` requests.

## pull_swe_images.py

Builds and manages Singularity images required by SWE‑Bench/SWE‑Bench multimodal. Converts Docker images referenced in parquet to `.sif` files.

Usage:
```bash
# Optional: set where images are stored
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/path/to/singularity_images

python pull_swe_images.py [OPTIONS]
```

Required:
- `--parquet-file`  Path to a SWE‑Bench parquet file

Optional:
- `--prefix`     Override Docker image namespace prefix
- `--dest-dir`   Directory to store `.sif` images (default: `<workspace>/singularity_images`)
- `--temp-base`  Base directory for temporary build folders (default: `<dest>/temp_dif`)
- `--start-index`  1-based index of first image to process (default: 1)
- `--end-index`    1-based index of last image (inclusive)
- `--log-name`     Write combined logs under `images_process/log/`

Tip:
- If the shared image repo already has images, copy them locally for faster access.

## run_swe.py

Bulk evaluation on SWE‑Bench datasets. Starts the async server, sends requests with concurrency, balances across multiple LLM endpoints, and writes results to `.jsonl`.

Usage:
```bash
python run_swe.py [OPTIONS]
```

Common options:
- `--dataset-path`   Path to SWE‑Bench parquet (train/val)
- `--output`         Output results file (`.jsonl`) (default: `eval_results.jsonl`)
- `--llm-addresses`  One or more LLM base URLs (e.g., `http://10.0.0.2:8000/v1 ...`)
- `--host`           Host for the OpenHands server (default: `localhost`)
- `--port`           Port for the server (default: `8006`)
- `--concurrency`    Max concurrent eval requests (default: `32`)
- `--num-instances`  Limit number of instances (for quick testing)
- `--sampling-params` JSON string merged into default sampling params

Notes:
- Requests are distributed round‑robin across provided `--llm-addresses`.
- Adjust `--concurrency` based on LLM capacity to avoid overload.

Example:
```bash
python run_swe.py \
  --dataset-path /path/to/train.parquet \
  --output swe_results.jsonl \
  --llm-addresses http://10.0.0.2:8000/v1 http://10.0.0.3:8000/v1 \
  --concurrency 64 \
  --sampling-params '{"temperature": 0.3, "top_p": 0.95}'
```

## prepare_data.py

Merges and standardizes parquet datasets (SWE‑Bench, SWE‑Bench multimodal, R2E‑Gym) into a single `merged.parquet`. Automatically detects dataset type and normalizes fields.

Usage:
```bash
python prepare_data.py --data-dir /path/to/parquet_directory
```

Parameters:
- `--data-dir`  Directory containing one or more parquet files (recursive)

Output:
- Writes `merged.parquet` under the same `--data-dir`.
