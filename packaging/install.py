#!/usr/bin/env python3
"""Install this checkout in a dedicated environment without editing host settings."""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Python 3.12+ is required")
    dest = args.prefix.expanduser().resolve()
    if dest.exists():
        parser.error("Use a new dedicated installation directory")
    venv.EnvBuilder(with_pip=True).create(dest)
    python = dest / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    source = Path(__file__).resolve().parents[1]
    spec = str(source) + ("" if args.core_only else "[desktop]")
    subprocess.run([str(python), "-m", "pip", "install", spec], check=True)
    (dest / ".modelforge-install").write_text("Dedicated ModelForge environment\n")
    print(f"Installed. Run: {python} -m modelforge.cli gui")


if __name__ == "__main__":
    main()
