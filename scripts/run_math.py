import argparse
import asyncio
import json
import time
from pathlib import Path
import os
import signal
import subprocess
import sys
import aiohttp, asyncio, time
import aiohttp
import numpy as np
import pandas as pd

DEFAULT_SAMPLING_PARAMS = {
    "model": "hosted_vllm/Qwen/Qwen3-8B",
    "api_key": "mykey",
    "modify_params": False,
    "log_completions": False,
    "native_tool_calling": True,
    "temperature": 0.6,
    "top_p": 0.9,
    "max_iterations": 35,
}

def _url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"

async def _post(session: aiohttp.ClientSession, url: str, payload: dict | None = None):
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()

async def start_server(session: aiohttp.ClientSession, host: str, port: int):
    return await _post(session, _url(host, port, "/start"))

async def stop_server(session: aiohttp.ClientSession, host: str, port: int):
    return await _post(session, _url(host, port, "/stop"))

async def add_llm_server(session: aiohttp.ClientSession, host: str, port: int, addr: str):
    return await _post(session, _url(host, port, "/add_llm_server"), {"address": addr})

async def process_instance(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    inst: dict,
    params: dict,
):
    payload = {"instance": inst, "sampling_params": params}
    return await _post(session, _url(host, port, "/process"), payload)

async def evaluate(args):
    df = pd.read_parquet(args.dataset_path)
    if args.num_instances:
        df = df.head(args.num_instances)

    instances: list[dict] = []
    for idx, row in df.iterrows():
        s = pd.Series(row).apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        s["trajectory_id"] = idx
        instances.append(s.to_dict())

    params = DEFAULT_SAMPLING_PARAMS.copy()
    if args.sampling_params:
        params.update(json.loads(args.sampling_params))

    if args.disable_thinking:
        params["enable_thinking"] = False

    # Set timeout to None to avoid timeout errors
    # timeout will be controlled by the server
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for addr in args.llm_addresses:
            await add_llm_server(session, args.host, args.port, addr)
        await start_server(session, args.host, args.port)

        semaphore = asyncio.Semaphore(args.concurrency)
        results: list[dict] = []

        async def worker(i: int, inst: dict):
            async with semaphore:
                try:
                    res = await process_instance(session, args.host, args.port, inst, params)
                    return res
                except Exception as e:
                    return {
                        "instance_id": inst.get("instance_id", i),
                        "trajectory_id": inst.get("trajectory_id", i),
                        "resolved": False,
                        "critical_error": str(e),
                    }

        start_time = time.time()
        tasks = [worker(i, inst) for i, inst in enumerate(instances)]
        for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
            results.append(await coro)
            if idx % 50 == 0:
                print(f"[progress] {idx}/{len(instances)} instances finished")

        await stop_server(session, args.host, args.port)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for r in results:
            fp.write(json.dumps(r) + "\n")
    print(f"Results saved to {out_path}")
    print("Resolved", sum(r.get("resolved") for r in results))

def parse_args():
    p = argparse.ArgumentParser("Simple bulk evaluation with OpenHands async server")
    # for code dataset use: /lustre/fsw/portfolios/nvr/users/mingjiel/data/eurus2-rl-data/train_code.parquet
    p.add_argument("--dataset-path", default="/lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler/train.parquet")
    p.add_argument("--output", default="eval_results.jsonl")
    p.add_argument("--llm-addresses", nargs="+", default=["http://127.0.0.1:8000/v1"])
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8006)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--num-instances", type=int)
    # need to launch reward server. Then pass in the ip address of the reward server.
    p.add_argument("--reward-server-ip", type=str, nargs="+", default=[])
    # Turn thinking off if using code dataset.
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument(
        "--sampling-params",
        default="",
        help="JSON string to merge into default sampling params",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start_server_path = Path(__file__).with_name("start_server.py")
    cmd = [sys.executable, str(start_server_path), "--port", str(args.port), "--reward-server-ip", *args.reward_server_ip, "--timeout", "500"]
    server_proc = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=None,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    async def _wait_until_ready(timeout: int = 60):


        url = _url(args.host, args.port, "/status")
        start_t = time.time()
        while time.time() - start_t < timeout:
            if server_proc.poll() is not None:
                raise RuntimeError("start_server.py exited unexpectedly")
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(url) as resp:
                        if resp.status in {200, 503}:
                            return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Async server not ready after {timeout}s on {url}")

    asyncio.run(_wait_until_ready())

    try:
        asyncio.run(evaluate(args))
    finally:
        try:
            if hasattr(os, "killpg") and server_proc.poll() is None:
                os.killpg(server_proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            server_proc.terminate()
        except Exception:
            pass
        try:
            server_proc.wait(timeout=10)
        except Exception:
            server_proc.kill()
