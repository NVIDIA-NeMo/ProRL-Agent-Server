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

google_drive_problems = [
    '4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc.json',
    '897e3b53-5d4d-444b-85cb-2cdc8a97d903.json',
    '22a4636f-8179-4357-8e87-d1743ece1f81.json',
    'b52b40a5-ad70-4c53-b5b0-5650a8387052.json',
    'a0b9dc9c-fc07-4a88-8c5d-5e3ecad91bcb.json',
    '46407397-a7d5-4c6b-92c6-dbe038b1457b.json',
    '0c825995-5b70-4526-b663-113f4c999dd2.json',
    '78aed49a-a710-4321-a793-b611a7c5b56b.json',
    '897e3b53-5d4d-444b-85cb-2cdc8a97d903.json',
    '46407397-a7d5-4c6b-92c6-dbe038b1457b.json'
]

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

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(setup_config, f)
    
    def get_final_state():
        action = OSWorldInteractiveAction(method='run_bash_script', params={'script': 'wmctrl -l'})
        observation = runtime.run_action(action)
        with open(os.path.join(results_dir, 'final_state.txt'), 'w') as f:
            f.write(observation.content)
            
    get_final_state()

    screenshot = runtime.get_vm_screenshot()
    with open(os.path.join(results_dir, 'screenshot.png'), 'wb') as f:
        f.write(screenshot)

    print("Cleaning up...")
    runtime.close()
    print("✓ Runtime closed")
    print()

if __name__ == '__main__':
    # Run async main
    import os
    examples_path = '/home/jayliu/OSWorld/evaluation_examples/examples'
    output_path = '/home/jayliu/ProRL-Agent-Server/results'
    categories = os.listdir(examples_path)
    categories = ['gimp']
    for category in categories:
        print(f"Running {category}...")
        examples = os.listdir(os.path.join(examples_path, category))
        for example in examples:
            if example in google_drive_problems:
                print(f"Skipping {example} because it is a Google Drive problem")
                continue
            print(f"Running {example}...")
            example_path = os.path.join(examples_path, category, example)
            results_dir = os.path.join(output_path, category, example)
            os.makedirs(results_dir, exist_ok=True)
            if os.path.exists(os.path.join(results_dir, 'screenshot.png')):
                print(f"Skipping {example} because it already exists")
                continue
            exit_code = asyncio.run(main(example_path, results_dir))
    sys.exit(0)

