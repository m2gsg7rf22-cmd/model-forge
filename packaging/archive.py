"""Archive only the native app built on this platform; hash release assets."""

import hashlib
from pathlib import Path
import platform
import shutil
import sys

root = Path(__file__).resolve().parents[1]
dist = root / "dist"
name = f"ModelForge-0.1.0-{platform.system()}-{platform.machine()}"
source = "ModelForge.app" if sys.platform == "darwin" else "ModelForge"
shutil.make_archive(str(dist / name), "zip", root_dir=dist, base_dir=source)
assets = sorted(
    p
    for p in dist.iterdir()
    if p.is_file() and p.name.endswith((".zip", ".whl", ".tar.gz", ".dmg"))
)
(dist / "SHA256SUMS.txt").write_text(
    "".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
        for p in assets
    )
)
