import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from openhands.nvidia.async_server import OpenHandsServer
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

app = FastAPI(title='OpenHands Async Server API')

# Global server instance
server = None
# Thread pool for submitting jobs
thread_pool = None

# Global timeout configuration (in seconds)
DEFAULT_TIMEOUT = 300.0  # 5 minutes
global_timeout = DEFAULT_TIMEOUT


def _initialize_thread_pool(force_reinit: bool = False):
    """Initialize or reinitialize the global thread pool if needed."""
    global thread_pool, server
    
    if server is None:
        logger.error('Cannot initialize thread pool: server is not initialized')
        return
    
    if thread_pool is None or thread_pool._shutdown or force_reinit:
        # Clean up existing thread pool if it exists
        if thread_pool is not None and not thread_pool._shutdown:
            try:
                thread_pool.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.warning(f'Error shutting down old thread pool: {e}')
        
        # Create new thread pool
        thread_pool_count = server.max_init_workers * 3
        thread_pool = ThreadPoolExecutor(max_workers=thread_pool_count)
        logger.info(f'Initialized thread pool with {thread_pool_count} threads')


def init_server(
    max_init_workers: int = 6,
    max_run_workers: int = 5,
    max_eval_workers: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    allow_skip_eval: bool = True,
    reward_server_ip: list | None = None,
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
    global server, global_timeout
    server = OpenHandsServer(
        llm_server_addresses=[],
        max_init_workers=max_init_workers,
        max_run_workers=max_run_workers,
        max_eval_workers=max_eval_workers,
        allow_skip_eval=allow_skip_eval,
        reward_server_ip=reward_server_ip,
    )
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


@app.post('/start')
async def start_server():
    global server, thread_pool
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    if server._server_running:
        logger.warning('Server is already running. But user requested to start.')
        raise HTTPException(status_code=400, detail='Server is already running')

    try:
        server.start()
        _initialize_thread_pool(force_reinit=True)
        return {'status': 'Server started successfully'}
    except Exception as e:
        logger.error(f'Failed to start server: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to start server: {str(e)}')


@app.post('/stop')
async def stop_server():
    global server, thread_pool
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    if not server._server_running:
        logger.warning('Server is not running. But user requested to stop.')
        raise ServerNotRunningError()
    try:
        # Run the stop operation in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, server.stop)
        server.clear_singularity_jobs()
        
        # Clean up the global thread pool to prevent dead threads
        if thread_pool is not None:
            thread_pool.shutdown(wait=False, cancel_futures=True)
            thread_pool = None
            logger.info('Shutting down global thread pool')
        return {'status': 'Server stopped successfully'}
    except Exception as e:
        logger.warning(f'Failed to stop server: {str(e)}. Force kill all singularity jobs.')
        server.clear_singularity_jobs()
        
        # Still try to clean up thread pool even if server stop failed
        if thread_pool is not None:
            try:
                thread_pool.shutdown(wait=False, cancel_futures=True)
                thread_pool = None
            except Exception as thread_e:
                logger.warning(f'Failed to shutdown thread pool: {thread_e}')  
        return {'status': 'Force killed all singularity jobs.'}


@app.get('/status')
async def get_status():
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    if not server._server_running:
        logger.warning('Server is not running. But user requested to get status.')
        raise ServerNotRunningError()

    try:
        return server.status()
    except Exception as e:
        logger.error(f'Server probably not running. Failed to get server status: {str(e)}')
        raise HTTPException(
            status_code=500,
            detail=f'Server probably not running. Failed to get server status: {str(e)}',
        )


@app.post('/cancel')
async def cancel(request: CancelRequest):
    """Cancel a specific job by job_id."""
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    if not server._server_running:
        logger.warning('Server is not running. Cannot cancel job.')
        raise ServerNotRunningError()

    try:
        success = server.cancel_job(request.job_id)
        if success:
            return {'status': f'Job {request.job_id} canceled successfully'}
        else:
            raise HTTPException(
                status_code=404, detail=f'Job {request.job_id} not found'
            )
    except Exception as e:
        logger.error(f'Failed to cancel job {request.job_id}: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Failed to cancel job {request.job_id}: {str(e)}'
        )

@app.post('/add_llm_server')
async def add_llm_server(request: LLMServerRequest):
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    try:
        server.add_llm_server_address(request.address)
        return {'status': f'Added LLM server address: {request.address}'}
    except Exception as e:
        logger.error(f'Failed to add LLM server: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Failed to add LLM server: {str(e)}'
        )


@app.post('/clear_llm_server')
async def clear_llm_server():
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    try:
        server.clear_llm_server_addresses()
        return {'status': 'Cleared all LLM server addresses'}
    except Exception as e:
        logger.error(f'Failed to clear LLM servers: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Failed to clear LLM servers: {str(e)}'
        )


@app.post('/process')
async def process(request: ProcessRequest):
    logger.debug(f'Processing instance: {request.instance}')
    logger.debug(f'Sampling params: {request.sampling_params}')
    global server, thread_pool
    if server is None or thread_pool is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    if not server._server_running:
        logger.warning('Server is not running. But user requested to process.')
        raise ServerNotRunningError()

    if len(server.weighted_addresses) == 0:
        logger.error('No LLM server addresses configured. Please add at least one LLM server address.')
        raise NoLLMServerError()

    # Convert instance dict to pandas Series
    try:
        instance = request.instance
    except Exception as e:
        logger.error(f'Invalid instance data: {str(e)}')
        raise HTTPException(status_code=400, detail=f'Invalid instance data: {str(e)}')

    try:
        result = await process_with_timeout(
            server, instance, request.sampling_params, global_timeout, thread_pool, job_id=request.job_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to process job: {str(e)}')
    return result


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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init_server(
        max_init_workers=args.max_init_workers,
        max_run_workers=args.max_run_workers,
        max_eval_workers=args.max_eval_workers,
        timeout=args.timeout,
        allow_skip_eval=args.allow_skip_eval,
        reward_server_ip=args.reward_server_ip,
    )
    start_api_server(host=args.host, port=args.port)
