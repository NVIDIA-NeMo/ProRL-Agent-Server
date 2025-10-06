# OpenHands Profiling Scripts

Quick scripts to profile OpenHands performance: launch LLM servers, measure latency, tune server, and stress test batches.

## launch_llm_servers.sh

Launch multiple vLLM servers for load-balanced profiling.

Usage:
```bash
./launch_llm_servers.sh
```

Notes:
- Sets up Python env and installs deps
- Launches 4 vLLM servers on ports 8000–8003
- 2 GPUs per server (tensor parallel), Qwen3‑14B with tool calling/reasoning
- Waits until all servers are ready
- GPU mapping: 8000→0,1 · 8001→2,3 · 8002→4,5 · 8003→6,7

## profile_latency.py

Measure per‑request latency to choose a proper timeout.

Usage:
```bash
python profile_latency.py
```

Defaults and output:
- 16 parallel jobs on SWE‑Bench for realism
- Prints per‑request timings and saves JSON results
- Use these timings to set `--timeout` in the server

Example output:
```
Time taken: 45.2 seconds
All tests passed!
```

## Start OpenHands server

Start the main server with tuned parameters from latency results.

Usage:
```bash
python ../start_server.py [OPTIONS]
```

Key options:
- `--max-init-workers`  init workers (CPU‑bound)
- `--max-run-workers`   run workers (LLM/GPU‑bound)
- `--timeout`           request timeout (seconds)

Example:
```bash
python ../start_server.py --max-init-workers 32 --max-run-workers 32 --timeout 500
```

## profile_batch.py

Stress test batch processing and measure throughput.

Usage:
```bash
python profile_batch.py
```

What it reports:
- Total batch time, requests/sec, success/failure rates
- Basic server utilization stats

Notes:
- Sends batch requests concurrently (default: 256)
- Distributes load across configured OpenHands/LLM servers
- Includes simple retry/error handling

## Monitoring

Watch during runs:
- GPU/CPU utilization, memory
- Network latency, success rates
- `active_run` via server status

Check status:
```bash
curl -X GET http://127.0.0.1:8006/status
```

## Troubleshooting

- GPU OOM: reduce tensor parallel size or number of servers
- Timeouts: increase `--timeout` based on latency profiling
- Startup failures: verify GPU availability and model access
- Port conflicts: free/choose ports 8000–8003
