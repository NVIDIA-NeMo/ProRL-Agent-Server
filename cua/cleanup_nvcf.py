#!/usr/bin/env python3
"""List and clean up NVCF functions.

Usage:
    # List all functions in the org:
    python cleanup_nvcf.py --list

    # Undeploy and delete only YOUR pool functions (nvcf-pool-*):
    python cleanup_nvcf.py --cleanup

    # Undeploy and delete ALL functions (careful — includes other users'):
    python cleanup_nvcf.py --cleanup --all

    # Undeploy and delete a specific function:
    python cleanup_nvcf.py --delete FUNCTION_ID VERSION_ID
"""
import argparse
import os
import sys

sys.path.insert(0, "/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server")

from openhands.nvidia.os_world.nvcf import OSWorldDeployer

POOL_NAME_PREFIX = "nvcf-pool-"


def main():
    parser = argparse.ArgumentParser(description="NVCF function cleanup utility")
    parser.add_argument("--list", action="store_true", help="List all functions in the org")
    parser.add_argument("--cleanup", action="store_true",
                        help="Undeploy and delete pool functions (nvcf-pool-* only, unless --all)")
    parser.add_argument("--all", action="store_true",
                        help="With --cleanup: delete ALL functions, not just pool ones")
    parser.add_argument("--delete", nargs=2, metavar=("FUNC_ID", "VER_ID"),
                        help="Delete a specific function")
    args = parser.parse_args()

    api_key = os.environ.get("NGC_API_KEY")
    org = os.environ.get("NGC_ORG")
    if not api_key or not org:
        print("ERROR: Set NGC_API_KEY and NGC_ORG environment variables")
        sys.exit(1)

    deployer = OSWorldDeployer(api_key=api_key, org_name=org)

    if args.list or (not args.cleanup and not args.delete):
        print("Listing all private NVCF functions in org...\n")
        result = deployer.list_functions()
        functions = result.get("functions", [])
        if not functions:
            print("No functions found.")
            return
        for fn in functions:
            fn_id = fn.get("id", "?")
            name = fn.get("name", "?")
            status = fn.get("status", "?")
            ver_id = fn.get("versionId", "?")
            mine = " <-- pool" if name.startswith(POOL_NAME_PREFIX) else ""
            print(f"  {name:40s}  status={status:10s}  fn={fn_id}  ver={ver_id}{mine}")
        print(f"\nTotal: {len(functions)} functions")

    if args.delete:
        fn_id, ver_id = args.delete
        print(f"Undeploying {fn_id}...")
        try:
            deployer.undeploy(fn_id, ver_id, graceful=True)
            print("Undeployed. Deleting...")
        except Exception as e:
            print(f"Undeploy failed (may already be undeployed): {e}")
        try:
            deployer.delete_function(fn_id, ver_id)
            print("Deleted.")
        except Exception as e:
            print(f"Delete failed: {e}")

    if args.cleanup:
        result = deployer.list_functions()
        functions = result.get("functions", [])

        if not args.all:
            # Only clean up pool functions
            functions = [f for f in functions if f.get("name", "").startswith(POOL_NAME_PREFIX)]
            print(f"Cleaning up {len(functions)} pool functions (nvcf-pool-*)...\n")
        else:
            print(f"Cleaning up ALL {len(functions)} functions...\n")

        if not functions:
            print("Nothing to clean up.")
            return

        for fn in functions:
            fn_id = fn.get("id", "?")
            ver_id = fn.get("versionId", "?")
            name = fn.get("name", "?")
            status = fn.get("status", "?")
            print(f"  Cleaning up: {name} ({fn_id}) status={status}")
            try:
                deployer.undeploy(fn_id, ver_id, graceful=True)
                print(f"    Undeployed")
            except Exception as e:
                print(f"    Undeploy skipped: {e}")
            try:
                deployer.delete_function(fn_id, ver_id)
                print(f"    Deleted")
            except Exception as e:
                print(f"    Delete failed: {e}")
        print(f"\nDone. Cleaned up {len(functions)} functions.")


if __name__ == "__main__":
    main()
