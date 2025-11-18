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


async def main():
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
        
    import pdb; pdb.set_trace()
    setup_controller = SetupController(
        vm_ip="127.0.0.1",
        server_port=runtime._vm_server_port,
        chromium_port=runtime._chromium_port,
        cache_dir="/tmp/osworld_example",
        client_password="password",
        runtime=runtime  # Pass your runtime object here
    )

    files = [
          {
            "url": "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/gimp/2a729ded-3296-423d-aec4-7dd55ed5fbb3/dog_with_background.png",
            "path": "/home/user/Desktop/dog_with_background.png"
          },
          {
            "url": "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/0810415c-bde4-4443-9047-d5f70165a697/Novels_Intro_Packet.docx",
            "path": "/home/user/Desktop/Novels_Intro_Packet.docx"
          }
        ]

    setup_controller._download_setup(files)
    setup_controller._change_wallpaper_setup('/home/user/Desktop/dog_with_background.png')
    #setup_controller._launch_setup(command=['gimp', '/home/user/Desktop/dog_with_background.png'])
    #setup_controller._open_setup('/home/user/Desktop/Novels_Intro_Packet.docx')
    command = [
          "python",
          "-c",
          "import pyautogui; import time; time.sleep(0.5); pyautogui.hotkey('ctrl', 'alt', 't'); time.sleep(0.5); pyautogui.write('stty size'); time.sleep(0.5); pyautogui.press('enter')"
        ]
    setup_controller._execute_setup(command=command)

    setup_controller._launch_setup(command=['google-chrome', '--remote-debugging-port=1337'])
    setup_controller._launch_setup(command=['socat', 'tcp-listen:9222,fork', 'tcp:localhost:1337'])
    await setup_controller._chrome_open_tabs_setup(urls_to_open=["https://drugs.com"])

    import pdb; pdb.set_trace()
    print("Cleaning up...")
    runtime.close()
    print("✓ Runtime closed")
    print()


if __name__ == '__main__':
    # Run async main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

