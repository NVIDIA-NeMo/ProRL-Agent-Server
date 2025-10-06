# OpenHands NVIDIA Module

The OpenHands NVIDIA module is a high‑performance, asynchronous system for running and evaluating software‑engineering agents at scale. It uses a small registry of pluggable handlers and a three‑stage async pipeline.

## What it provides

- **Registry**: Register agent types (init/run/eval) behind a simple interface for different applications
- **Agent Handlers**: Small classes that implement how a given agent works on specific task
- **Async Server**: A fast, three‑stage pipeline (init → run → eval) with worker pools

## How it works

The `ProRLAgent Server` provides high-performance asynchronous processing of agent jobs through a sophisticated three-stage pipeline architecture. Each job flows through distinct stages with dedicated worker pools, ensuring optimal resource utilization and fault isolation.

### Server Architecture

The server implements a producer-consumer pattern with three specialized stages:

1. **Initialization Stage**: Creates runtime environments, metadata, and configurations
2. **Run Stage**: Executes the agent logic within the prepared runtime
3. **Evaluation Stage**: Analyzes and scores the agent's output

Each stage maintains its own:
- **Queue**: Thread-safe job queue for that stage
- **Worker Pool**: Configurable number of async workers
- **Active Job Tracking**: Set of currently processing job IDs
- **Error Handling**: Stage-specific exception handlers

### Server Configuration

```python
server = OpenHandsServer(
    llm_server_addresses=['http://server1:8000/v1', 'http://server2:8000/v1'],
    max_init_workers=6,     # Workers for initialization stage
    max_run_workers=5,      # Workers for execution stage
    max_eval_workers=5,     # Workers for evaluation stage (defaults to max_run_workers)
    allow_skip_eval=True    # Skip evaluation if no result is generated (such as empty git-patch for swebench)
)
```

### Detailed Pipeline Flow

**Job Submission and Setup:**
```python
def process(self, instance, sampling_params, job_id=None):
    # 1. Create JobDetails container
    job_details = JobDetails()
    job_details.job_id = job_id or self.get_unique_id(instance)
    job_details.instance = instance
    job_details.max_iterations = sampling_params.pop('max_iterations', 2)
    job_details.llm_config = self.create_llm_config(sampling_params)
    job_details.start_time = time.time()
    job_details.event = threading.Event()  # For synchronization

    # 2. Store job details and add to init queue
    self._job_details[job_id] = job_details
    self.init_queue.put(job_id)

    # 3. Wait for completion
    job_details.event.wait()
```

**Stage 1: Initialization Worker (`_init_worker`)**

The initialization stage prepares the execution environment:

```python
async def _init_worker(self, wid):
    job_id = await asyncio.to_thread(self.init_queue.get)
    job_details = self._job_details[job_id]

    # Determine agent type from instance
    dataset_type = getattr(job_details.instance, 'data_source', 'swebench')
    _init_func = get_registered_functions('init', dataset_type)

    # Call registered init function
    runtime, metadata, config = await _init_func(
        job_details.instance,      # Input: Problem instance data
        job_details.llm_config,    # Input: LLM configuration
        sid=job_id,                # Input: Session ID for tracking
        max_iterations=job_details.max_iterations  # Input: Iteration limit
    )

    # Store results in job_details
    job_details.runtime = runtime    # Runtime environment (Docker, etc.)
    job_details.metadata = metadata  # Evaluation metadata
    job_details.config = config      # OpenHands configuration

    # Move to next stage
    self.run_queue.put(job_id)
```

**What the `init` function receives and produces:**
- **Inputs**:
  - `instance` (pd.Series): Task instance data (format depends on agent type)
  - `llm_config` (LLMConfig): LLM server configuration and parameters
  - `sid` (str): Session ID for logging and tracking
  - `max_iterations` (int): Maximum number of agent iterations allowed
- **Outputs**:
  - `runtime` (Runtime): Execution environment for the agent
  - `metadata` (EvalMetadata): Agent-specific evaluation configuration
  - `config` (OpenHandsConfig): Agent configuration including LLM settings

**Stage 2: Run Worker (`_run_worker`)**

The run stage executes the agent within the prepared environment:

```python
async def _run_worker(self, wid):
    job_id = await asyncio.to_thread(self.run_queue.get)
    job_details = self._job_details[job_id]
    job_details.start_run_time = time.time()

    # Get registered run function
    dataset_type = getattr(job_details.instance, 'data_source', 'swebench')
    _run_func = get_registered_functions('run', dataset_type)

    # Execute agent
    run_results = await _run_func(
        job_details.runtime,   # Input: Prepared runtime environment
        job_details.metadata,  # Input: Evaluation metadata
        job_details.config,    # Input: Agent configuration
        job_details.instance,  # Input: Problem instance
    )

    # Store results and clean up runtime
    job_details.run_results = run_results  # Agent's output (git_patch, messages, etc.)

    # IMPORTANT: Runtime is closed immediately after run completes
    if job_details.runtime:
        job_details.runtime.close()
        job_details.runtime = None

    # Move to evaluation
    self.evaluate_queue.put(job_id)
```

**What the `run` function receives and produces:**
- **Inputs**:
  - `runtime` (Runtime): Active execution environment from init stage
  - `metadata` (EvalMetadata): Evaluation metadata from init stage
  - `config` (OpenHandsConfig): Agent configuration from init stage
  - `instance` (pd.Series): Original task instance
- **Outputs**:
  - `dict[str, object]` containing agent-specific results (format varies by agent type)
  - Common fields include execution traces, generated outputs, performance metrics
  - Results are stored in `job_details.run_results` for use in evaluation stage

**Stage 3: Evaluation Worker (`_eval_worker`)**

The evaluation stage assesses the agent's performance:

```python
async def _eval_worker(self, wid):
    job_id = await asyncio.to_thread(self.evaluate_queue.get)
    job_details = self._job_details[job_id]
    job_details.start_eval_time = time.time()

    # Get registered eval function
    dataset_type = getattr(job_details.instance, 'data_source', 'swebench')
    _eval_func = get_registered_functions('eval', dataset_type)

    # Evaluate results
    eval_report = await _eval_func(
        job_details,                    # Input: Complete job details
        sid=f'eval_{job_id}',          # Input: Evaluation session ID
        allow_skip=self.allow_skip_eval # Input: Skip if no patch
    )

    # Store evaluation results
    if isinstance(eval_report, dict) and 'report' in eval_report:
        job_details.eval_results = eval_report['report']
    else:
        job_details.eval_results = eval_report

    # Signal completion
    job_details.event.set()
```

**What the `eval` function receives and produces:**
- **Inputs**:
  - `job_details` (JobDetails): Complete job information including:
    - `job_details.run_results`: Agent's output from run stage
    - `job_details.instance`: Original task instance
    - `job_details.metadata`: Evaluation metadata from init stage
  - `sid` (str): Evaluation session ID
  - `allow_skip` (bool): Whether to skip evaluation based on run results
- **Outputs**:
  - `dict[str, Any]` containing agent-specific evaluation results
  - Results are stored in `job_details.eval_results` and used for final result processing

## Quick start

```python
from openhands.nvidia.async_server import OpenHandsServer

server = OpenHandsServer(
    llm_server_addresses=["http://localhost:8000/v1"],
    max_init_workers=4,
    max_run_workers=4,
)
server.start()

result = server.process(instance, sampling_params)
print(result)

server.stop()
```

Check `status()` at any time:

```python
server.status()  # { 'init_queue': ..., 'run_queue': ..., 'eval_queue': ..., 'active_*': ..., 'total': ... }
```

## Add a new agent type

Implement a handler and register it. Handlers expose `name`, `init`, `run`, `eval`.

```python
from openhands.nvidia.registry import AgentHandler, register_agent_handler

class MyAgentHandler(AgentHandler):
    @property
    def name(self) -> str:
        return "my_agent"

    async def init(self, instance, llm_config=None, sid=None, max_iterations=1):
        # prepare runtime, metadata, config
        return runtime, metadata, config

    async def run(self, runtime, metadata, config, instance):
        # execute agent logic
        return {"git_patch": "...", "messages": [...]}  # shape is agent-specific

    async def eval(self, job_details, sid=None, allow_skip=True):
        # evaluate run_results against the instance
        return {"score": 1.0}

register_agent_handler(MyAgentHandler())
```

Instances are routed by `instance.data_source` to the matching handler `name`.

## Built‑in example: SWE Agent

A concrete handler for SWEBench‑style tasks, used for SWE‑Bench, SWE‑Bench Multimodal, and R2E‑Gym.

- **SWE‑Gym/SWE‑Bench**
  - Inference: container image derived from `instance_id`
  - Eval: `swegym.harness.*`
  - Extra: `pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git`
- **SWE‑Bench Multimodal**
  - Inference: official image path, mounts dataset `image_assets`
  - Eval: `swebench.harness`
  - Extra: `pip install swebench` (or the package above)
- **R2E‑Gym**
  - Inference: explicit `docker_image` per instance (resource factor fixed at 1)
  - Eval: internal helper with pre‑filtering of hunks that modify files already changed inside the image


#### JobDetails State Evolution

The `JobDetails` object accumulates state as it flows through the pipeline:

**After Job Creation:**
```python
job_details.job_id = "unique_job_id"
job_details.instance = pd.Series(instance_data)
job_details.max_iterations = 2
job_details.llm_config = LLMConfig(...)
job_details.start_time = 1234567890.0
job_details.event = threading.Event()
```

**After Init Stage:**
```python
job_details.runtime = DockerRuntime(...)
job_details.metadata = EvalMetadata(...)
job_details.config = OpenHandsConfig(...)
```

**After Run Stage:**
```python
job_details.run_results = {
    # Agent-specific results - format varies by agent type
    # Common: execution outputs, traces, generated content, metrics
}
job_details.start_run_time = 1234567935.0
job_details.runtime = None  # Closed for resource management
```

**After Eval Stage:**
```python
job_details.eval_results = {
    # Agent-specific evaluation results - format varies by agent type
    # Common: scores, test results, performance metrics
}
job_details.start_eval_time = 1234567980.0
```

## LLM load balancing

Weighted round‑robin across `llm_server_addresses`. Sampling params are passed via the generated LLM config.

## Error handling & cleanup

- Stage‑specific exception handlers for init/run/eval
- On failures, errors are recorded and the job is completed
- Runtimes are closed right after the run stage

## Performance tips

- Tune `max_init_workers`, `max_run_workers`, and (optionally) `max_eval_workers`
- Use multiple LLM servers to distribute load
- Set timeouts appropriate to your environment
