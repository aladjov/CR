#!/usr/bin/env python3
"""Dev deployment: build wheel, upload to UC Volume, local editable install."""

import argparse
import glob
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_version() -> str:
    init = PROJECT_ROOT / "src" / "customer_retention" / "__init__.py"
    match = re.search(r'__version__\s*=\s*"(.+?)"', init.read_text())
    return match.group(1) if match else "unknown"


def _clean():
    print("Cleaning previous builds...")
    for d in ["dist", "build"]:
        p = PROJECT_ROOT / d
        if p.exists():
            shutil.rmtree(p)
    for egg in glob.glob(str(PROJECT_ROOT / "src" / "*.egg-info")):
        shutil.rmtree(egg)


def _build_wheel() -> Path:
    print("Building wheel...")
    if shutil.which("uv"):
        subprocess.run(["uv", "build", "--wheel"], cwd=PROJECT_ROOT, check=True)
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", "dist/"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    wheels = sorted((PROJECT_ROOT / "dist").glob("*.whl"))
    if not wheels:
        print("Error: no wheel found in dist/", file=sys.stderr)
        sys.exit(1)
    return wheels[0]


def _upload(wheel: Path, volume: str):
    print(f"Uploading to {volume} ...")
    if not shutil.which("databricks"):
        print("Error: databricks CLI not found", file=sys.stderr)
        sys.exit(1)
    dest = f"{volume.rstrip('/')}/{wheel.name}"
    subprocess.run(
        ["databricks", "fs", "cp", str(wheel), dest, "--overwrite"],
        check=True,
    )
    print(f"Uploaded: {dest}")


def _local_install():
    print("Installing editable (local)...")
    if shutil.which("uv"):
        subprocess.run(
            ["uv", "pip", "install", "-e", "."],
            cwd=PROJECT_ROOT,
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=PROJECT_ROOT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build wheel, upload to UC Volume, and do local editable install.",
    )
    parser.add_argument(
        "volume",
        nargs="?",
        help="UC Volume path, e.g. /Volumes/catalog/schema/packages",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip local editable install",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip upload to UC Volume",
    )
    args = parser.parse_args()

    if not args.no_upload and not args.volume:
        parser.error("volume is required unless --no-upload is specified")

    _clean()
    wheel = _build_wheel()

    if not args.no_upload:
        _upload(wheel, args.volume)

    if not args.no_install:
        _local_install()

    version = _read_version()
    print("\n--- Summary ---")
    print(f"  Wheel:   {wheel.name}")
    if not args.no_upload:
        print(f"  Volume:  {args.volume.rstrip('/')}/{wheel.name}")
    print(f"  Version: {version}")
    if not args.no_install:
        print("  Local:   editable install OK")


if __name__ == "__main__":
    main()
