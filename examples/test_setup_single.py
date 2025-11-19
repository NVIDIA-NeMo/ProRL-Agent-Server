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
from examples.setup import SetupController, Evaluator


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
    
    import pdb; pdb.set_trace()
    with open(config_path, 'r') as f:
        setup_config = json.load(f)
    assert 'config' in setup_config, "Setup config not found in setup JSON"
    await setup_controller.setup(setup_config['config'])
    #evaluator = Evaluator(setup_config)

    #await asyncio.sleep(5)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(setup_config, f)

    def get_final_state():
        action = OSWorldInteractiveAction(method='run_bash_script', params={'script': 'wmctrl -l'})
        observation = runtime.run_action(action)
        with open(os.path.join(results_dir, 'final_state.txt'), 'w') as f:
            f.write(observation.content)

    get_final_state()

    await asyncio.sleep(3)

    screenshot = runtime.get_vm_screenshot()
    with open(os.path.join(results_dir, 'screenshot.png'), 'wb') as f:
        f.write(screenshot)
    
    print("Start Working. Continue to evaluate...")
    import pdb; pdb.set_trace()
    
    #evaluator.evaluate(setup_controller)
    
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
    MAX_WORKERS = 1
    # ===============================================
    
    # Define 2 examples to run in parallel
    # You can modify these to test different examples
    tasks = [
        {
            'category': 'os',
            'example': '5ced85fc-fa1a-4217-95fd-0fb530545ce2.json'
        },
    ]
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
    
    asyncio.run(main(process_args[0][0], process_args[0][1]))
    print()
    print("=" * 80)
    print("All tasks completed!")
    print("=" * 80)
    
    sys.exit(0)

