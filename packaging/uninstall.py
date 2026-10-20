#!/usr/bin/env python3
"""Remove only an explicitly selected, marked dedicated installation."""

import argparse
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument("--prefix", type=Path, required=True)
p.add_argument("--yes", action="store_true")
a = p.parse_args()
target = a.prefix.expanduser().resolve()
if not a.yes or not (target / ".modelforge-install").is_file():
    p.error("Requires --yes and a dedicated marked ModelForge installation")
shutil.rmtree(target)
print("Installation removed. Research projects were retained in MODELFORGE_HOME.")
