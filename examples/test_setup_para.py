#!/usr/bin/env python3
"""
Example script demonstrating OSWorld runtime usage with OSWorldInteractiveAction.

This script shows how to:
1. Configure and create an OSWorld runtime
2. Connect to a VM
3. Use OSWorldInteractiveAction to interact with the VM
4. Perform GUI actions (click, type, press keys)
5. Take screenshots
6. Get VM information (platform, screen size)
7. Clean up resources

Prerequisites:
- Singularity/Apptainer installed
- QEMU and OVMF installed
- VM image at ./OS_images/Ubuntu.qcow2 with OSWorld server
"""

import asyncio
import json
import sys
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from openhands.core.config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.runtime.impl.singularity.osworld_singularity_runtime import (
    OSWorldSingularityRuntime,
)
from openhands.storage import get_file_store
from examples.setup import SetupController


async def main(config_path, results_dir):
    """Main example function."""
    
    print("=" * 80)
    print("OSWorld Runtime Example")
    print("=" * 80)
    print()
    
    # 1. Create configuration
    print("1. Creating configuration...")
    config = OpenHandsConfig()
    config.runtime = 'osworld'
    config.sandbox.base_container_image = 'ubuntu:24.04'
    config.sandbox.run_as_fakeroot = True
    
    # Check if VM image exists
    vm_image_path = './OS_images/Ubuntu.qcow2'
    if not Path(vm_image_path).exists():
        print(f"ERROR: VM image not found at {vm_image_path}")
        print("Please place your Ubuntu VM image with OSWorld server at this location.")
        return 1
    
    print(f"✓ Configuration created")
    print(f"  Runtime: {config.runtime}")
    print(f"  Base image: {config.sandbox.base_container_image}")
    print(f"  VM image: {vm_image_path}")
    print()
    
    # 2. Create event stream
    print("2. Creating event stream...")
    file_store = get_file_store('local', '/tmp/osworld_example')
    event_stream = EventStream(sid='osworld-example', file_store=file_store)
    print("✓ Event stream created")
    print()
    
    # 3. Create OSWorld runtime
    print("3. Creating OSWorld runtime...")
    runtime = OSWorldSingularityRuntime(
        config=config,
        event_stream=event_stream,
        sid='osworld-example',
        os_type='linux',
        vm_image_path=vm_image_path,
        attach_to_existing=False,
    )
    print("✓ Runtime created")
    print(f"  OS Type: {runtime.os_type}")
    print()
    
    try:
        # 4. Connect to runtime (starts VM)
        print("4. Connecting to runtime (this may take 1-2 minutes)...")
        print("   - Building Singularity image (if needed)")
        print("   - Starting QEMU VM")
        print("   - Waiting for VM to boot")
        print("   - Waiting for OSWorld server to start")
        await runtime.connect()
        print("✓ Runtime connected and VM is ready!")
        print(f"  VM Server URL: {runtime.osworld_vm_url}")
        print(f"  VNC URL: {runtime.vnc_url}")
        print(f"  Chromium URL: {runtime._chromium_port}")
        print()
        
        # 5. Check if VM is alive
        print("5. Checking VM health...")
        runtime.check_if_alive()
        print("✓ VM is alive and responding")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.exception("Failed to run example")
        return 1
    
    setup_controller = SetupController(
        vm_ip="127.0.0.1",
        server_port=runtime._vm_server_port,
        chromium_port=runtime._chromium_port,
        cache_dir="/tmp/osworld_example",
        client_password="password",
        runtime=runtime  # Pass your runtime object here
    )

    with open(config_path, 'r') as f:
        setup_config = json.load(f)
    assert 'config' in setup_config, "Setup config not found in setup JSON"
    await setup_controller.setup(setup_config['config'])

    await asyncio.sleep(10)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(setup_config, f)

    screenshot = runtime.get_vm_screenshot()
    with open(os.path.join(results_dir, 'screenshot.png'), 'wb') as f:
        f.write(screenshot)
    
    print("Cleaning up...")
    runtime.close()
    print("✓ Runtime closed")
    print()


def run_main_wrapper(config_path, results_dir, example_name):
    """Wrapper function to run async main in a separate process."""
    print(f"\n[Process {os.getpid()}] Starting processing for {example_name}")
    
    # Check if already processed
    if os.path.exists(os.path.join(results_dir, 'screenshot.png')):
        print(f"[Process {os.getpid()}] Skipping {example_name} because it already exists")
        return 0
    
    try:
        exit_code = asyncio.run(main(config_path, results_dir))
        print(f"[Process {os.getpid()}] Completed {example_name} with exit code {exit_code}")
        return exit_code if exit_code is not None else 0
    except Exception as e:
        print(f"[Process {os.getpid()}] Error processing {example_name}: {e}")
        return 1


if __name__ == '__main__':
    # Configuration
    examples_path = '/home/jayliu/OSWorld/evaluation_examples/examples'
    output_path = '/home/jayliu/ProRL-Agent-Server/two_results'
    
    # ============= CONCURRENCY CONTROL =============
    # Set the maximum number of concurrent processes
    # Change this value to control concurrency: 1, 2, 4, etc.
    MAX_WORKERS = 4
    # ===============================================
    
    # Define 2 examples to run in parallel
    # You can modify these to test different examples
    tasks = [
        {
            'category': 'chrome',
            'example': '44ee5668-ecd5-4366-a6ce-c1c9b8d4e938.json'
        },
        {
            'category': 'chrome',
            'example': 'fc6d8143-9452-4171-9459-7f515143419a.json'  # Change this to a different example
        },
        {
            'category': 'chrome',
            'example': 'e1e75309-3ddb-4d09-92ec-de869c928143.json'  # Change this to a different example
        },
        {
            'category': 'chrome',
            'example': '47543840-672a-467d-80df-8f7c3b9788c9.json'  # Change this to a different example
        },
    ]
    """
        {
            'category': 'chrome',
            'example': '030eeff7-b492-4218-b312-701ec99ee0cc.json'
        },
        {
            'category': 'chrome',
            'example': '93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9.json'  # Change this to a different example
        },
        {
            'category': 'chrome',
            'example': '1704f00f-79e6-43a7-961b-cedd3724d5fd.json'  # Change this to a different example
        },
        {
            'category': 'chrome',
            'example': 'a728a36e-8bf1-4bb6-9a03-ef039a5233f0.json'  # Change this to a different example
        },
        
    ]
    """
    # Prepare arguments for each process
    process_args = []
    for task in tasks:
        category = task['category']
        example = task['example']
        example_path = os.path.join(examples_path, category, example)
        results_dir = os.path.join(output_path, category, example)
        os.makedirs(results_dir, exist_ok=True)
        process_args.append((example_path, results_dir, example))
    
    # Run processes with controlled concurrency using ProcessPoolExecutor
    print("=" * 80)
    print(f"Starting {len(tasks)} tasks with max concurrency: {MAX_WORKERS}")
    print("=" * 80)
    print()
    
    # Use ProcessPoolExecutor for structured concurrency control
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the executor
        future_to_task = {}
        for i, args in enumerate(process_args):
            future = executor.submit(run_main_wrapper, *args)
            future_to_task[future] = (i + 1, args[2])  # (task_number, example_name)
            print(f"Submitted task {i+1}/{len(tasks)}: {args[2]}")
        
        print(f"\nRunning with {MAX_WORKERS} worker(s)...")
        print()
        
        # Process completed tasks as they finish
        completed = 0
        for future in as_completed(future_to_task):
            task_num, example_name = future_to_task[future]
            completed += 1
            
            try:
                exit_code = future.result()
                status = "✓ SUCCESS" if exit_code == 0 else f"✗ FAILED (exit code: {exit_code})"
                print(f"[{completed}/{len(tasks)}] Task {task_num} ({example_name}): {status}")
            except Exception as e:
                print(f"[{completed}/{len(tasks)}] Task {task_num} ({example_name}): ✗ EXCEPTION - {e}")
    
    print()
    print("=" * 80)
    print("All tasks completed!")
    print("=" * 80)
    
    sys.exit(0)

