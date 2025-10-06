# Configuration Guide for start_server.py

## Overview

This guide provides best practices and recommended configurations for running the OpenHands server using the `start_server.py` script. The goal is to maintain high GPU utilization while keeping CPU utilization at reasonable levels.

## Best Practices

### Core Principle
- **Maintain GPU high utilization** while **not consuming too much CPU utilization**
- **Recommended setting**: Each vLLM server should handle approximately **8 concurrent active running agent jobs**

## Configuration Examples

### Example 1: Qwen-14B Model Configuration

For a Qwen-14B model using tensor parallelism of 2:

#### System Setup
- **Each node hosts**: 4 LLM servers
- **Concurrent jobs per LLM server**: 8 agent jobs
- **Total concurrent jobs**: 4 × 8 = 32 concurrent jobs for OpenHands servers

#### Recommended Command
```bash
export LOG_LEVEL=ERROR
export DEBUG=False
python start_server.py --max-init-workers 32 --max-run-workers 32 --timeout 500
```

#### Parameter Breakdown
- `--max-init-workers 32`: Maximum number of workers for initialization tasks
- `--max-run-workers 32`: Maximum number of workers for running tasks
- `--timeout 500`: Timeout setting in seconds

## Timeout Configuration

### SWE-bench Tasks
- **Initial recommendation**: Start with a timeout of **500 seconds**
- **Monitoring**: Watch for timeout jobs in your system
- **Adjustment**: If observing **large amounts of timeout jobs**, slightly increase the timeout value until timeout jobs are under control

### Timeout Tuning Strategy
1. Start with 500s timeout
2. Monitor job completion rates
3. If high timeout rate (>5%), incrementally increase timeout
4. Continue adjusting until timeout jobs are manageable

## Integration with VERL

### Concurrency Calculation
For VERL integration, treat the OpenHands server concurrency as:
```
Total Concurrency = max_init_workers + max_run_workers
```

**Example**: With the configuration above (32 + 32 = 64 total concurrency)
`config.rollout.openhands_num_workers=64`

### Benefits
- Ensures more jobs are scheduled and awaiting in the run queue
- Improves overall system throughput
- Better resource utilization

## Performance Expectations

### Batch Completion Times
Given the recommended settings above:
- **1 batch of 32×8 requests** should be completed in approximately **~1200 seconds**

*See `./profile` folder for detailed performance analysis and benchmarking scripts.*

