import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from openhands.nvidia.async_server_process import OpenHandsServer as OpenHandsServer_Process
from openhands.nvidia.async_server import OpenHandsServer as OpenHandsServer_Thread
from openhands.nvidia.registry import FunctionNotRegisteredError
from openhands.nvidia.utils import (
    JobTimeoutError,
    LLMServerRequest,
    NoLLMServerError,
    ProcessRequest,
    CancelRequest,
    ServerNotRunningError,
    process_with_timeout,
)
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.runtime.impl.singularity.singularity_runtime import kill_process_tree

import multiprocessing
from multiprocessing import Process, Queue
import threading
import uuid
import traceback
from typing import Optional, Dict, Any, List

app = FastAPI(title='OpenHands Async Server API')

# Global timeout configuration (in seconds)
DEFAULT_TIMEOUT = 300.0  # 5 minutes
global_timeout = DEFAULT_TIMEOUT

# New globals for multiprocessing-based architecture
server_process: Optional[multiprocessing.Process] = None
request_queue: Optional[multiprocessing.queues.Queue] = None
job_result_queue: Optional[multiprocessing.queues.Queue] = None
control_response_queue: Optional[multiprocessing.queues.Queue] = None
response_thread: Optional[threading.Thread] = None
result_futures: Dict[str, "asyncio.Future"] = {}
futures_lock = threading.Lock()
accepting_requests = False
accepting_requests_lock = threading.Lock()
server_config: Dict[str, Any] = {}
llm_server_addresses_buffer: List[str] = []
config_lock = threading.Lock()

thread_based_server = False


def _reject_all_pending_futures(error_message: str):
    """Set exception on all pending request futures and clear the map."""
    global result_futures
    with futures_lock:
        items = list(result_futures.items())
        result_futures.clear()
    for _, fut in items:
        if not fut.done():
            try:
                fut.set_exception(RuntimeError(error_message))
            except Exception:
                logger.debug(f'Could not set exception on future: {error_message}')


def _is_server_running() -> bool:
    return server_process is not None and server_process.is_alive()


def _response_listener():
    """Background thread that receives results from the child process and fulfills futures by job_id."""
    global job_result_queue, result_futures
    while True:
        try:
            if job_result_queue is None:
                return
            msg = job_result_queue.get()
            if msg is None:
                # Sentinel to stop the listener
                return
            if not isinstance(msg, dict):
                continue
            if msg.get('type') != 'result':
                # Ignore any non-result messages on this queue
                continue
            job_id = msg.get('job_id')
            ok = msg.get('ok', False)
            with futures_lock:
                fut = result_futures.pop(job_id, None)
            if fut is None:
                continue

            # Check if future is already done (completed, cancelled, or has exception)
            if fut.done():
                logger.debug(f'Future for job {job_id} is already done (cancelled: {fut.cancelled()}), skipping result')
                continue

            if ok:
                try:
                    fut.set_result(msg.get('result'))
                except Exception as e:
                    # Only set exception if future is not already done
                    if not fut.done():
                        try:
                            fut.set_exception(e)
                        except Exception:
                            logger.debug(f'Could not set exception on future for job {job_id}: {e}')
                    else:
                        logger.debug(f'Future for job {job_id} was completed while setting exception: {e}')
            else:
                try:
                    fut.set_exception(RuntimeError(msg.get('error', 'Unknown error')))
                except Exception:
                    if not fut.done():
                        logger.debug(f'Could not set exception on future for job {job_id}')
                    else:
                        logger.debug(f'Future for job {job_id} was completed while setting exception')
        except Exception as e:
            logger.warning(f'Response listener encountered error: {e}')


def server_worker(
    req_q,
    result_q,
    ctrl_resp_q,
    config: Dict[str, Any],
):
    """Child process entry point: owns OpenHandsServer and executes jobs concurrently."""
    try:
        # Ensure this child runs in its own process group/session
        try:
            os.setsid()
        except Exception:
            pass

        if thread_based_server:
            openhands_server_class = OpenHandsServer_Thread
            logger.info('Using thread-based server')
        else:
            openhands_server_class = OpenHandsServer_Process
            logger.info('Using process-based server')
        # Initialize server with provided config
        server = openhands_server_class(
            llm_server_addresses=config.get('llm_server_addresses', []),
            max_init_workers=config.get('max_init_workers'),
            max_run_workers=config.get('max_run_workers'),
            max_eval_workers=config.get('max_eval_workers'),
            allow_skip_eval=config.get('allow_skip_eval', True),
            reward_server_ip=config.get('reward_server_ip'),
        )
        global_timeout_local = config.get('timeout', DEFAULT_TIMEOUT)

        # Start server
        server.start()

        # Internal executors for job concurrency and any threaded work inside process_with_timeout
        job_executor_workers = server.max_init_workers + server.max_run_workers + server.max_eval_workers + 30
        job_executor = ThreadPoolExecutor(max_workers=job_executor_workers)
        inner_thread_pool = ThreadPoolExecutor(max_workers=job_executor_workers)

        running = True
        submitted_futures: Dict[str, "concurrent.futures.Future"] = {}

        def _submit_job(job_id: str, instance: dict, sampling_params: dict, timeout: Optional[float]):
            t = timeout if timeout is not None else global_timeout_local

            def job_fn():
                try:
                    # Run the async processing in its own event loop
                    result = asyncio.run(
                        process_with_timeout(
                            server,
                            instance,
                            sampling_params,
                            t,
                            inner_thread_pool,
                            job_id=job_id,
                        )
                    )
                    result_q.put({'type': 'result', 'job_id': job_id, 'ok': True, 'result': result})
                except Exception as ex:
                    result_q.put(
                        {
                            'type': 'result',
                            'job_id': job_id,
                            'ok': False,
                            'error': f'{type(ex).__name__}: {str(ex)}',
                        }
                    )

            fut = job_executor.submit(job_fn)
            submitted_futures[job_id] = fut

        while running:
            try:
                msg = req_q.get()
                if msg is None:
                    break
                if not isinstance(msg, dict):
                    continue
                msg_type = msg.get('type')

                if msg_type == 'process':
                    _submit_job(
                        job_id=msg['job_id'],
                        instance=msg['instance'],
                        sampling_params=msg.get('sampling_params', {}),
                        timeout=msg.get('timeout'),
                    )

                elif msg_type == 'cancel':
                    job_id = msg.get('job_id')
                    fut = submitted_futures.pop(job_id, None)
                    try:
                        server.cancel_job(job_id)
                    except Exception:
                        pass
                    if fut is not None:
                        fut.cancel()
                    # Acknowledge cancel
                    ctrl_resp_q.put({'type': 'cancel_ack', 'request_id': msg.get('request_id'), 'ok': True})

                elif msg_type == 'add_llm_server':
                    try:
                        server.add_llm_server_address(msg['address'])
                        ctrl_resp_q.put({'type': 'add_llm_server_ack', 'request_id': msg.get('request_id'), 'ok': True})
                    except Exception as ex:
                        ctrl_resp_q.put(
                            {
                                'type': 'add_llm_server_ack',
                                'request_id': msg.get('request_id'),
                                'ok': False,
                                'error': str(ex),
                            }
                        )

                elif msg_type == 'clear_llm_server':
                    try:
                        server.clear_llm_server_addresses()
                        ctrl_resp_q.put({'type': 'clear_llm_server_ack', 'request_id': msg.get('request_id'), 'ok': True})
                    except Exception as ex:
                        ctrl_resp_q.put(
                            {
                                'type': 'clear_llm_server_ack',
                                'request_id': msg.get('request_id'),
                                'ok': False,
                                'error': str(ex),
                            }
                        )

                elif msg_type == 'status':
                    try:
                        status = server.status()
                    except Exception as ex:
                        status = {'error': str(ex)}
                    ctrl_resp_q.put({'type': 'status_ack', 'request_id': msg.get('request_id'), 'ok': True, 'status': status})

                elif msg_type == 'stop':
                    running = False
                    # Attempt to cancel all jobs
                    for jid, fut in list(submitted_futures.items()):
                        try:
                            server.cancel_job(jid)
                        except Exception:
                            pass
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                        submitted_futures.pop(jid, None)
                    try:
                        server.clear_singularity_jobs()
                    except Exception:
                        pass
                    try:
                        server.stop()
                    except Exception:
                        pass
                    ctrl_resp_q.put({'type': 'stop_ack', 'request_id': msg.get('request_id'), 'ok': True})
                    break

                else:
                    # Unknown message type; ignore
                    continue

            except Exception as loop_ex:
                # Log and continue
                try:
                    logger.warning(f'Worker loop error: {loop_ex}\n{traceback.format_exc()}')
                except Exception:
                    pass

    except Exception as e:
        try:
            logger.error(f'Child process fatal error: {e}\n{traceback.format_exc()}')
        except Exception:
            pass
    finally:
        # Best-effort cleanup happens here; executors will be GC'd on process exit
        pass


def init_server(
    max_init_workers: int = 6,
    max_run_workers: int = 5,
    max_eval_workers: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
    allow_skip_eval: bool = True,
    reward_server_ip: Optional[list] = None,
):
    logger.info(
        f'Initializing server with max_init_workers={max_init_workers}, max_run_workers={max_run_workers}, max_eval_workers={max_eval_workers}, timeout={timeout}'
    )
    if allow_skip_eval:
        logger.info(
            'Allowing skipping evaluation if git_patch is None or empty. Please set allow_skip_eval=False for testing.'
        )
    else:
        logger.info(
            'Not allowing skipping evaluation if git_patch is None or empty. Please set allow_skip_eval=True for production.'
        )
    global global_timeout, server_config
    # Store configuration; actual server is created in the child process on /start
    with config_lock:
        server_config = {
            'max_init_workers': max_init_workers,
            'max_run_workers': max_run_workers,
            'max_eval_workers': max_eval_workers,
            'timeout': timeout,
            'allow_skip_eval': allow_skip_eval,
            'reward_server_ip': reward_server_ip,
        }
    global_timeout = timeout


@app.exception_handler(ServerNotRunningError)
async def server_not_running_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={'detail': 'Server is not running. Please start the server first.'},
    )


@app.exception_handler(NoLLMServerError)
async def no_llm_server_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={
            'detail': 'No LLM server addresses configured. Please add at least one LLM server address.'
        },
    )


@app.exception_handler(JobTimeoutError)
async def job_timeout_handler(request, exc):
    return JSONResponse(status_code=504, content={'detail': str(exc)})


@app.exception_handler(FunctionNotRegisteredError)
async def function_not_registered_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            'detail': f'Invalid dataset type or function not registered: {str(exc)}'
        },
    )


def _start_child_process():
    """Start the child process and background response listener."""
    global server_process, request_queue, job_result_queue, control_response_queue, response_thread, accepting_requests
    if _is_server_running():
        raise RuntimeError('Server process already running')

    # Build an effective config snapshot with buffered LLM addresses
    with config_lock:
        effective_config = dict(server_config)
        effective_config['llm_server_addresses'] = list(llm_server_addresses_buffer)

    request_queue = multiprocessing.Queue()
    job_result_queue = multiprocessing.Queue()
    control_response_queue = multiprocessing.Queue()

    server_process = Process(
        target=server_worker,
        args=(request_queue, job_result_queue, control_response_queue, effective_config),
        daemon=False,
    )
    server_process.start()

    # Start background thread to collect job results
    response_thread = threading.Thread(target=_response_listener, name='response-listener', daemon=False)
    response_thread.start()

    with accepting_requests_lock:
        accepting_requests = True


@app.post('/start')
async def start_server():
    global accepting_requests
    # Previously relied on global server; now we only check the child process state
    if _is_server_running():
        logger.warning('Server is already running. But user requested to start.')
        raise HTTPException(status_code=400, detail='Server is already running')

    try:
        _start_child_process()
        return {'status': 'Server started successfully'}
    except Exception as e:
        logger.error(f'Failed to start server process: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to start server: {str(e)}')


@app.post('/stop')
async def stop_server():
    global server_process, request_queue, job_result_queue, control_response_queue, response_thread, accepting_requests
    if not _is_server_running():
        logger.warning('Server is not running. But user requested to stop.')
        raise ServerNotRunningError()
    try:
        # Stop accepting new requests immediately and fail in-flight ones
        with accepting_requests_lock:
            accepting_requests = False
        _reject_all_pending_futures('Server is stopping; job cancelled')

        # Send stop control message and wait for ack briefly
        req_id = str(uuid.uuid4())
        try:
            if request_queue is not None:
                request_queue.put({'type': 'stop', 'request_id': req_id})
            # Await stop ack
            if control_response_queue is not None:
                ack = control_response_queue.get(timeout=10)
                # Ignore ack content; proceed to cleanup
        except Exception:
            logger.warning('Stop ack not received; proceeding to terminate')

        # Kill the entire process tree of the server process using project helper
        if server_process is not None and server_process.is_alive():
            try:
                kill_process_tree(server_process.pid)
            except Exception as e:
                logger.warning(f'kill_process_tree failed for pid {server_process.pid}: {e}')
            server_process.join(timeout=10)

        # Clean up queues and response listener
        try:
            if job_result_queue is not None:
                job_result_queue.put(None)
        except Exception:
            pass
        if response_thread is not None:
            try:
                response_thread.join(timeout=10)
            except Exception:
                pass

        server_process = None
        request_queue = None
        job_result_queue = None
        control_response_queue = None
        response_thread = None

        return {'status': 'Server stopped successfully'}
    except Exception as e:
        logger.warning(f'Failed to stop server process cleanly: {str(e)}')
        # Best-effort hard cleanup
        try:
            if server_process is not None and server_process.is_alive():
                kill_process_tree(server_process.pid)
        except Exception:
            pass
        return {'status': 'Force killed all server process resources.'}


@app.get('/status')
async def get_status():
    if not _is_server_running():
        logger.warning('Server is not running. But user requested to get status.')
        raise ServerNotRunningError()
    # Minimal status from coordinator; child has richer status but we avoid contention
    with futures_lock:
        pending = len(result_futures)
    # Ask child for its status
    child_status = None
    try:
        req_id = str(uuid.uuid4())
        if request_queue is not None:
            request_queue.put({'type': 'status', 'request_id': req_id})
        if control_response_queue is not None:
            ack = control_response_queue.get(timeout=10)
            if isinstance(ack, dict) and ack.get('type') == 'status_ack':
                child_status = ack.get('status')
    except Exception:
        # If control path fails, still return local info
        logger.warning('Failed to get child status; proceeding to return local info')
        pass
    if child_status:
        return {'status': 'running', 'pending_jobs': pending, **child_status}
    else:
        return {'status': 'running', 'pending_jobs': pending}


@app.post('/cancel')
async def cancel(request: CancelRequest):
    if not _is_server_running():
        logger.warning('Server is not running. Cannot cancel job.')
        raise ServerNotRunningError()

    try:
        req_id = str(uuid.uuid4())
        if request_queue is not None:
            request_queue.put({'type': 'cancel', 'job_id': request.job_id, 'request_id': req_id})
        # Await acknowledgment from control response queue
        if control_response_queue is not None:
            try:
                ack = control_response_queue.get(timeout=10)
                if not ack.get('ok', False):
                    raise HTTPException(status_code=500, detail=f"Failed to cancel job: {ack.get('error', 'unknown')}")
            except Exception:
                pass
        return {'status': f'Cancel requested for job {request.job_id}'}
    except Exception as e:
        logger.error(f'Failed to cancel job {request.job_id}: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Failed to cancel job {request.job_id}: {str(e)}'
        )


@app.post('/add_llm_server')
async def add_llm_server(request: LLMServerRequest):
    try:
        address = request.address
        # Always keep buffer in sync
        with config_lock:
            if address not in llm_server_addresses_buffer:
                llm_server_addresses_buffer.append(address)
        if not _is_server_running():
            return {'status': f'Buffered LLM server address: {address}'}

        req_id = str(uuid.uuid4())
        if request_queue is not None:
            request_queue.put({'type': 'add_llm_server', 'address': address, 'request_id': req_id})

        if control_response_queue is not None:
            try:
                ack = control_response_queue.get(timeout=5)
                if not ack.get('ok', False):
                    raise HTTPException(status_code=500, detail=f"Failed to add LLM server: {ack.get('error', 'unknown')}")
            except Exception:
                pass
        return {'status': f'Added LLM server address: {address}'}
    except Exception as e:
        logger.error(f'Failed to add LLM server: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to add LLM server: {str(e)}')


@app.post('/clear_llm_server')
async def clear_llm_server():
    try:
        # Always clear buffer
        with config_lock:
            llm_server_addresses_buffer.clear()
        if not _is_server_running():
            return {'status': 'Cleared buffered LLM server addresses'}

        req_id = str(uuid.uuid4())
        if request_queue is not None:
            request_queue.put({'type': 'clear_llm_server', 'request_id': req_id})

        if control_response_queue is not None:
            try:
                ack = control_response_queue.get(timeout=5)
                if not ack.get('ok', False):
                    raise HTTPException(status_code=500, detail=f"Failed to clear LLM servers: {ack.get('error', 'unknown')}")
            except Exception:
                pass
        return {'status': 'Cleared all LLM server addresses'}
    except Exception as e:
        logger.error(f'Failed to clear LLM servers: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to clear LLM servers: {str(e)}')


@app.post('/process')
async def process(request: ProcessRequest):
    logger.debug(f'Processing instance: {request.instance}')
    logger.debug(f'Sampling params: {request.sampling_params}')
    global accepting_requests

    with accepting_requests_lock:
        _accepting = accepting_requests
    if not _is_server_running() or not _accepting:
        logger.warning('Server is not running or not accepting requests.')
        raise ServerNotRunningError()

    # Validate instance extraction
    try:
        instance = request.instance
    except Exception as e:
        logger.error(f'Invalid instance data: {str(e)}')
        raise HTTPException(status_code=400, detail=f'Invalid instance data: {str(e)}')

    # Submit job to child process and await result with timeout
    job_id = request.job_id if getattr(request, 'job_id', None) else str(uuid.uuid4())
    try:
        # Create a future that the response listener will fulfill
        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        # Add callback to clean up the future from result_futures when done
        def cleanup_future(fut):
            try:
                with futures_lock:
                    result_futures.pop(job_id, None)
            except Exception as e:
                logger.debug(f'Error in cleanup callback for job {job_id}: {e}')

        # Use a weak reference to avoid circular references
        fut.add_done_callback(cleanup_future)

        with futures_lock:
            result_futures[job_id] = fut

        # Send request to child process
        if request_queue is not None:
            request_queue.put(
                {
                    'type': 'process',
                    'job_id': job_id,
                    'instance': instance,
                    'sampling_params': request.sampling_params,
                    'timeout': global_timeout,
                }
            )

        # Await result with timeout
        try:
            # WARNING: We set the timeout to be 2x the global timeout for final timeout
            result = await asyncio.wait_for(fut, timeout=global_timeout*2+10)
            # result = await fut
        except asyncio.TimeoutError:
            # On timeout, send cancel and raise 504
            try:
                if request_queue is not None:
                    request_queue.put({'type': 'cancel', 'job_id': job_id, 'request_id': str(uuid.uuid4())})
            except Exception:
                pass
            # Clean up the future from result_futures and cancel it
            with futures_lock:
                fut_to_cancel = result_futures.pop(job_id, None)
            if fut_to_cancel and not fut_to_cancel.done():
                try:
                    fut_to_cancel.cancel()
                except Exception:
                    pass
            raise HTTPException(status_code=504, detail='Processing timed out')
        except Exception as e:
            # Clean up the future from result_futures on any exception
            with futures_lock:
                fut_to_cancel = result_futures.pop(job_id, None)
            if fut_to_cancel and not fut_to_cancel.done():
                try:
                    fut_to_cancel.cancel()
                except Exception:
                    pass
            raise
        return result
    except ServerNotRunningError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to process job: {str(e)}')


def start_api_server(host: str = '0.0.0.0', port: int = 8000):
    uvicorn.run(app, host=host, port=port)


def parse_args():
    parser = argparse.ArgumentParser(description='OpenHands Async Server API')
    parser.add_argument(
        '--max-init-workers',
        type=int,
        default=64,
        help='Maximum number of initialization workers (default: 64)',
    )
    parser.add_argument(
        '--max-run-workers',
        type=int,
        default=64,
        help='Maximum number of run workers (default: 64)',
    )
    parser.add_argument(
        '--max-eval-workers',
        type=int,
        default=None,
        help='Maximum number of run workers (default: None)',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f'Global timeout for job processing in seconds (default: {DEFAULT_TIMEOUT})',
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8006,
        help='Port to bind the server to (default: 8006)',
    )
    parser.add_argument(
        '--allow-skip-eval',
        type=bool,
        default=True,
        help='Allow skipping evaluation if git_patch is None or empty. Set to False for testing (default: True).',
    )
    parser.add_argument(
        '--reward-server-ip',
        type=str,
        nargs='*',
        default=[],
        help='List of reward server IP addresses (default: [])',
    )
    parser.add_argument(
        '--use-thread-based-server',
        action='store_true',
        default=False,
        help='Use thread-based server (default: False)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    thread_based_server = args.use_thread_based_server
    if not thread_based_server:
        # For process-based server, we need to set the start method to spawn
        import multiprocessing
        multiprocessing.set_start_method('fork')
    logger.info(f'Using thread-based server: {thread_based_server}')
    init_server(
        max_init_workers=args.max_init_workers,
        max_run_workers=args.max_run_workers,
        max_eval_workers=args.max_eval_workers,
        timeout=args.timeout,
        allow_skip_eval=args.allow_skip_eval,
        reward_server_ip=args.reward_server_ip,
    )
    start_api_server(host=args.host, port=args.port)
