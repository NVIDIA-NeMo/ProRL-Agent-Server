# SWE-bench Utils Testing

This directory contains comprehensive tests for the SWE-bench utilities, including both unit tests and integration tests that can work with real data from the `__main__` section examples.

## Test Structure

### Test Categories

1. **Unit Tests**: Fast, isolated tests using mocks
2. **Integration Tests**: Tests with real data when available
3. **Real Data Tests**: Tests specifically requiring the actual parquet dataset
4. **Main Examples Tests**: Tests that validate the examples from `utils.py`'s `__main__` section
5. **Timer Tests**: Tests for timeout-sensitive job processing with phase-aware timing

### Files

- `test_swebench_utils.py` - Original comprehensive unit tests + new real data tests
- `test_swebench_utils_integration.py` - Integration tests using real examples
- `test_swe_agent_handler.py` - Unit tests for the SweAgentHandler class
- `test_async_server.py` - Unit tests for the OpenHandsServer async server class
- `test_timer.py` - Unit tests for the OpenHands timer module (timeout handling)
- `conftest.py` - Pytest fixtures and configuration
- `pytest.ini` - Pytest configuration and markers
- `run_tests.py` - Test runner script with different options

## Running Tests

### Quick Start

```bash
# Run all unit tests (fast, mocked)
python run_tests.py --unit

# Run integration tests (if real data is available)
python run_tests.py --integration

# Run tests with real data specifically
python run_tests.py --real-data

# Run SweAgentHandler tests specifically
python run_tests.py --swe-agent-handler

# Run OpenHandsServer async tests specifically
python run_tests.py --async-server

# Run timer tests specifically
python run_tests.py --timer

# Test the main examples from utils.py
python run_tests.py --main-examples

# Run all tests
python run_tests.py --all
```

### Using pytest directly

```bash
# Unit tests only
pytest -m "not integration and not slow" tests/nvidia/

# Integration tests
pytest -m integration tests/nvidia/

# Real data tests
pytest -m real_data tests/nvidia/

# Skip slow tests
pytest -m "not slow" tests/nvidia/

# Run SweAgentHandler tests specifically
pytest tests/nvidia/test_swe_agent_handler.py -v

# Run OpenHandsServer async tests specifically
pytest tests/nvidia/test_async_server.py -v

# Run timer tests specifically
pytest tests/nvidia/test_timer.py -v

# Run with coverage
pytest --cov=openhands.nvidia --cov-report=term-missing tests/nvidia/
```

## Real Data Setup

### Requirements

The real data tests require access to the SWE-bench dataset:
- Path: `/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet`
- Tests will be automatically skipped if this file is not available

### What Real Data Tests Cover

1. **Data Loading**: Validates the exact data loading pattern from `__main__`
2. **Instance Processing**: Tests numpy array conversion and serialization
3. **Parallel Evaluation**: Tests the parallel async evaluation pattern
4. **Sequential Evaluation**: Tests the sequential evaluation pattern
5. **Docker Image Generation**: Tests with real instance IDs
6. **Configuration**: Tests configuration generation with real data

## SweAgentHandler Tests

The `test_swe_agent_handler.py` file contains comprehensive unit tests for the `SweAgentHandler` class, which provides the integration layer for SWE-bench evaluation:

### Test Coverage

1. **Basic Functionality Tests**:
   - Name property verification
   - Method signature validation
   - Inheritance from AgentHandler

2. **Initialization Tests (`init` method)**:
   - Successful initialization with various parameters
   - Default parameter handling
   - Exception propagation

3. **Run Tests (`run` method)**:
   - Successful execution
   - Exception handling and propagation

4. **Evaluation Tests (`eval` method)**:
   - Evaluation with git patches
   - Evaluation without patches (None run_results)
   - Evaluation with empty git patches
   - Missing git_patch key handling (KeyError)
   - Exception propagation

5. **Exception Handlers**:
   - `init_exception`, `run_exception`, `eval_exception`
   - Proper delegation to utils functions

6. **Integration Tests**: Tests with real data when available

7. **Edge Cases**:
   - None values handling
   - Concurrent operations
   - Missing keys in run_results

### Test Categories

- **Unit Tests**: Fast tests using mocks for all external dependencies
- **Integration Tests**: Tests using real data structures (marked with `@pytest.mark.real_data`)
- **Edge Cases**: Tests for error conditions and boundary cases

## OpenHandsServer Tests

The `test_async_server.py` file contains comprehensive unit tests for the `OpenHandsServer` class, which provides the asynchronous job processing server for handling multiple concurrent OpenHands evaluation tasks.

### Test Coverage

1. **Server Lifecycle Tests**:
   - Server initialization with default and custom parameters
   - Starting and stopping the server
   - Thread pool management and cleanup
   - Server status reporting

2. **Configuration Management Tests**:
   - LLM server address management (add, clear, load balancing)
   - LLM configuration creation
   - Worker pool configuration

3. **Job Management Tests**:
   - Unique job ID generation and collision handling
   - Job details storage and retrieval
   - Job lifecycle tracking (init, run, eval phases)
   - Custom job ID support

4. **Queue Operations Tests**:
   - Job queuing across init, run, and evaluation phases
   - Queue status monitoring
   - Queue cleanup on server shutdown

5. **Thread Safety Tests**:
   - Concurrent access to job details
   - Active job tracking across worker threads
   - Lock contention handling

6. **Error Handling Tests**:
   - Unregistered handler detection
   - Server state validation
   - Exception propagation and cleanup

7. **Load Balancing Tests**:
   - Weighted address distribution
   - Round-robin server selection

8. **Timeout Integration Tests**:
   - Timeout parameter acceptance in process method
   - PausableTimer initialization and lifecycle
   - Timeout error handling across all phases (init, run, eval)
   - Timing information inclusion in results
   - Phase-aware timeout management
   - Runtime cleanup on timeout events

### Test Categories

- **Unit Tests**: Fast tests using mocks for registry functions and external dependencies
- **Thread Safety Tests**: Tests for concurrent operations and race condition prevention
- **Lifecycle Tests**: Tests for proper resource management and cleanup
- **Integration Tests**: Basic integration with registry system (without external services)
- **Timeout Tests**: Tests for timeout functionality and phase-aware timing behavior

## Test Examples from __main__

The `__main__` section of `utils.py` contains several patterns that are now testable:

### 1. Data Loading Pattern
```python
dataset = pd.read_parquet("/path/to/train.parquet")
instance = dataset.iloc[0]['instance']
instance = pd.Series(instance)
instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
```

### 2. Parallel Evaluation Pattern
```python
async def run_parallel_async():
    tasks = []
    for idx in range(1):
        inst_clone = instance.copy()
        inst_clone["instance_id"] = f"{instance['instance_id']}_{idx}"
        gold_patch = inst_clone['patch']
        tasks.append(evaluate_agent(gold_patch, inst_clone))
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. Sequential Evaluation Pattern
```python
async def run_sequential_async():
    results = []
    for i in range(2):
        res = await evaluate_agent(mock_patch, instance)
        results.append(res)
    return results
```

## Test Fixtures

The `conftest.py` file provides several useful fixtures:

- `real_dataset`: Loads the actual parquet dataset (skipped if not available)
- `real_instance`: Processed real instance from the dataset
- `mock_patch`: The exact mock patch used in `__main__`
- `minimal_llm_config`: LLM configuration for testing
- `mock_runtime`: Mock runtime for testing
- `sample_evaluation_result`: Sample evaluation result structure

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.real_data`: Tests requiring real data
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.asyncio`: Async tests

## Mocking Strategy

The tests use a hybrid approach:

1. **Unit Tests**: Mock all external dependencies (runtime, LLM calls, etc.)
2. **Integration Tests**: Use real data structures but mock expensive operations
3. **Real Data Tests**: Use actual data files but mock runtime/network operations

This allows testing the real data flow without requiring expensive Docker containers or LLM API calls.

## Environment Variables

Some tests may require or check for environment variables:

- `OPENAI_API_KEY`: For LLM configuration (can be dummy for tests)
- `EVAL_DOCKER_IMAGE_PREFIX`: Docker image prefix for testing
- `RUN_WITH_BROWSING`: Browsing configuration

## Troubleshooting

### Real Data Not Available
If real data tests are skipped:
1. Check if the parquet file exists at the expected path
2. Run `python run_tests.py --unit` to run tests without real data
3. The existing unit tests provide comprehensive coverage without real data

### Import Errors
Make sure the project root is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/OpenHands_internal"
```

### Async Test Issues
Async tests require proper event loop handling. The test runner and fixtures handle this automatically.

## Contributing

When adding new tests:

1. Use appropriate markers (`@pytest.mark.real_data`, `@pytest.mark.asyncio`, etc.)
2. Add fixtures to `conftest.py` for reusable test data
3. Mock expensive operations (Docker, LLM calls, network, time-dependent operations)
4. Test both the happy path and error conditions
5. For timer tests, use `patch('time.time')` to avoid actual delays
6. Update this README if adding new test categories

## Examples

### Testing a New Function with Real Data

```python
@pytest.mark.real_data
def test_my_function_with_real_data(real_instance):
    """Test my function with real instance data"""
    result = my_function(real_instance)
    assert result is not None
    assert 'instance_id' in result
```

### Testing Async Patterns

```python
@pytest.mark.asyncio
@pytest.mark.real_data
async def test_async_pattern(real_instance, mock_runtime):
    """Test async pattern from __main__"""
    with patch('module.expensive_function') as mock_func:
        mock_func.return_value = expected_result
        result = await my_async_function(real_instance)
        assert result == expected_result
```

### Testing SweAgentHandler

```python
@pytest.mark.asyncio
@patch('openhands.nvidia.swe_agent.swe_agent_handler.initialize_agents')
async def test_handler_init(mock_initialize_agents, minimal_llm_config):
    """Test SweAgentHandler initialization"""
    mock_initialize_agents.return_value = (Mock(), Mock(), Mock())

    handler = SweAgentHandler()
    instance = pd.Series({'instance_id': 'test'})

    result = await handler.init(instance=instance, llm_config=minimal_llm_config)
    assert len(result) == 3  # runtime, metadata, config
```

### Testing with JobDetails

```python
@pytest.mark.asyncio
@patch('openhands.nvidia.swe_agent.swe_agent_handler.evaluate_agent')
async def test_handler_eval(mock_evaluate_agent):
    """Test SweAgentHandler evaluation"""
    mock_evaluate_agent.return_value = {'resolved': True}

    handler = SweAgentHandler()
    job_details = JobDetails(
        job_id='test',
        instance=pd.Series({'instance_id': 'test'}),
        run_results={'git_patch': 'test_patch'}
    )

    result = await handler.eval(job_details=job_details)
    assert result['resolved'] is True
```

### Testing OpenHandsServer

```python
def test_server_initialization():
    """Test server initialization with custom parameters"""
    server = OpenHandsServer(
        llm_server_addresses=["http://localhost:8000"],
        max_init_workers=3,
        max_run_workers=4,
        max_eval_workers=2
    )

    assert server.max_init_workers == 3
    assert server.max_run_workers == 4
    assert server.max_eval_workers == 2
    assert len(server.weighted_addresses) == 1

def test_job_lifecycle():
    """Test basic job lifecycle management"""
    server = OpenHandsServer(llm_server_addresses=["http://localhost:8000"])
    mock_instance = MockInstance()

    # Test unique ID generation
    job_id = server.get_unique_id(mock_instance)
    assert "test_instance" in job_id
    assert "test_trajectory" in job_id

    # Test job details storage
    job_details = JobDetails(job_id=job_id, instance=mock_instance)
    with server._job_details_lock:
        server._job_details[job_id] = job_details
        assert job_id in server._job_details

def test_thread_safety():
    """Test thread-safe operations"""
    server = OpenHandsServer()
    job_id = "test_job"

    # Test concurrent job tracking
    with server._state_lock:
        server._active_init_jobs.add(job_id)
        assert job_id in server._active_init_jobs
        server._active_init_jobs.discard(job_id)
        assert job_id not in server._active_init_jobs
```
