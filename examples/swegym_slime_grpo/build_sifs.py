#!/usr/bin/env python3
"""Build per-task SWE-Gym SIF images into $APPTAINER_IMAGE_DIR via skopeo -> apptainer build.

On cw-dfw `apptainer pull docker://...` fails (the MITM container-cache proxy
presents a cert apptainer's Go client rejects). So instead, per instance:

    skopeo --insecure-policy copy --src-tls-verify=false docker://<ref> oci:<tmp>/oci:latest
    apptainer build <APPTAINER_IMAGE_DIR>/<instance_id>.sif oci:<tmp>/oci

The output filename is `<instance_id>.sif`, matching stable polar_config.yaml's
runtime.image = "{args.polar_apptainer_image_dir}/{sample.metadata.instance_id}.sif".
The docker SOURCE ref comes from sample_tasks.registry_image_for_instance_id()
(xingyaoww/sweb.eval.x86_64.<suffix>:latest). This is the cw-dfw replacement for
the stock prepare_apptainer_images.py (which uses the proxy-blocked apptainer pull).

Run INSIDE a container that has apptainer + skopeo (your flappydora image), after
`source env.cwdfw.sh` (sets APPTAINER_IMAGE_DIR, POLAR_APPTAINER_BIN, proxy):

  source examples/swegym_slime_grpo/env.cwdfw.sh
  python3 examples/swegym_slime_grpo/build_sifs.py --jobs 4
  python3 examples/swegym_slime_grpo/build_sifs.py --num-shards 8 --shard-index 0 --jobs 2

"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sample_tasks import registry_image_for_instance_id  # noqa: E402  # docker ref per instance_id

SIF_DIR = (os.environ.get("APPTAINER_IMAGE_DIR") or os.environ.get("POLAR_SIF_DIR") or "").strip()
DEFAULT_PROXY = "http://cw-dfw-cs-001-container-cache:3128"
JSONL_FILES = ("swegym_train_293.jsonl", "swegym_eval_23.jsonl")


def apptainer_bin() -> str:
    b = os.environ.get("POLAR_APPTAINER_BIN")
    if b:
        return b
    for c in ("/usr/bin/apptainer", "/bin/apptainer"):
        if Path(c).is_file():
            return c
    return shutil.which("apptainer") or "apptainer"


def sif_name(instance_id: str) -> str:
    # Must match stable polar_config.yaml: <image_dir>/{instance_id}.sif
    return f"{instance_id}.sif"


def docker_ref(instance_id: str) -> str:
    ref = registry_image_for_instance_id(instance_id)
    if not ref.startswith(("docker://", "oci://", "oci-archive:", "docker-archive:")):
        ref = "docker://" + ref
    return ref


def instance_ids() -> list[str]:
    ids: set[str] = set()
    for fn in JSONL_FILES:
        p = HERE / fn
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            iid = (json.loads(line).get("metadata") or {}).get("instance_id")
            if iid:
                ids.add(str(iid))
    return sorted(ids)


def build_one(
    instance_id: str,
    *,
    force: bool,
    env: dict,
    cache_root: Path,
    mksquashfs_args: str,
) -> tuple[str, str, str]:
    target = Path(SIF_DIR) / sif_name(instance_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if target.is_file() and target.stat().st_size > 0 and not force:
            return ("skip", instance_id, str(target))

        ref = docker_ref(instance_id)
        # OCI layout DIRECTORY (not a single oci-archive tar) — more robust handoff to
        # apptainer. skopeo --retry-times re-pulls blobs truncated by the MITM proxy
        # (the "expected blob size N, but only wrote N-k" error). The .sif.tmp is on the
        # same fs as target (lustre) so os.replace is atomic; the OCI dir + apptainer
        # scratch live under cache_root (point --cache-root at fast local disk if lustre
        # IO causes short writes).
        with tempfile.TemporaryDirectory(dir=str(cache_root), prefix="ocibuild-") as td:
            ocidir = str(Path(td) / "oci")
            subprocess.run(
                [
                    "skopeo",
                    "--insecure-policy",
                    "copy",
                    "--retry-times",
                    "5",
                    "--src-tls-verify=false",
                    ref,
                    f"oci:{ocidir}:latest",
                ],
                check=True,
                env=env,
            )
            tmp_sif = str(target) + ".tmp"
            cmd = [apptainer_bin(), "build", "--disable-cache"]
            if mksquashfs_args:
                cmd.extend(["--mksquashfs-args", mksquashfs_args])
            cmd.extend(["-F", tmp_sif, f"oci:{ocidir}"])
            subprocess.run(cmd, check=True, env=env)
            os.replace(tmp_sif, target)
    return ("built", instance_id, str(target))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4, help="concurrent builds")
    ap.add_argument("--instance-id", action="append", default=[], help="only these instance ids")
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split the selected instance ids into this many deterministic shards",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="build only this 0-based shard index; use with --num-shards",
    )
    ap.add_argument("--force", action="store_true", help="rebuild even if the .sif exists")
    ap.add_argument(
        "--cache-root",
        default=(os.environ.get("POLAR_JOB_CACHE_ROOT")
                 or f"/tmp/polar-sifbuild-{os.environ.get('USER') or os.getuid()}"),
        help="node-local scratch for OCI layout + apptainer tmp/cache (NOT in the repo). "
             "Default: /tmp/polar-sifbuild-<user>. Point at a roomy local disk if /tmp is small.",
    )
    ap.add_argument(
        "--mksquashfs-args",
        default=os.environ.get("POLAR_MKSQUASHFS_ARGS", "-processors 1 -mem 1024M"),
        help="extra arguments passed through to apptainer build --mksquashfs-args. "
             "Default limits Apptainer's bundled mksquashfs, which segfaults on cw-dfw "
             "with its default threaded settings. Override with POLAR_MKSQUASHFS_ARGS.",
    )
    args = ap.parse_args()

    if not SIF_DIR:
        sys.exit("ERROR: APPTAINER_IMAGE_DIR (or POLAR_SIF_DIR) must be set (source env.cwdfw.sh first).")
    if shutil.which("skopeo") is None:
        sys.exit("ERROR: skopeo not found (run inside the flappydora container that has it).")

    cache = Path(args.cache_root)
    (cache / "apptainer-cache").mkdir(parents=True, exist_ok=True)
    (cache / "apptainer-tmp").mkdir(parents=True, exist_ok=True)
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or DEFAULT_PROXY
    env = {
        **os.environ,
        "http_proxy": proxy, "https_proxy": proxy,
        "APPTAINER_CACHEDIR": str(cache / "apptainer-cache"),
        "APPTAINER_TMPDIR": str(cache / "apptainer-tmp"),
    }

    ids = args.instance_id or instance_ids()
    if not ids:
        sys.exit("No instance ids found (need swegym_*.jsonl or --instance-id).")
    if args.num_shards < 1:
        sys.exit("ERROR: --num-shards must be >= 1.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        sys.exit("ERROR: --shard-index must satisfy 0 <= shard_index < num_shards.")
    total_ids = len(ids)
    if args.num_shards > 1:
        ids = [iid for n, iid in enumerate(ids) if n % args.num_shards == args.shard_index]
    print(
        f"Building {len(ids)}/{total_ids} SIF(s) into {SIF_DIR} as <instance_id>.sif "
        f"with {max(args.jobs,1)} job(s); proxy={proxy}; "
        f"shard={args.shard_index}/{args.num_shards}; "
        f"mksquashfs_args={args.mksquashfs_args!r}",
        flush=True,
    )
    if not ids:
        print("Shard is empty; nothing to do.")
        return 0

    failures = []
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as ex:
        futs = {
            ex.submit(
                build_one,
                i,
                force=args.force,
                env=env,
                cache_root=cache,
                mksquashfs_args=args.mksquashfs_args,
            ): i
            for i in ids
        }
        for fut in as_completed(futs):
            iid = futs[fut]
            try:
                status, _, path = fut.result()
                print(f"{status}: {path}", flush=True)
            except subprocess.CalledProcessError as e:
                failures.append(iid)
                print(f"FAILED: {iid}: {e}", file=sys.stderr, flush=True)
    if failures:
        print(f"\n{len(failures)} build(s) FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
