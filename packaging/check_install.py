"""Exercise real fresh installation, CLI workflow and uninstall on this OS."""

from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="modelforge-install-") as temporary:
    base = Path(temporary)
    prefix = base / "installation"
    subprocess.run(
        [
            sys.executable,
            str(root / "packaging/install.py"),
            "--prefix",
            str(prefix),
            "--core-only",
        ],
        check=True,
    )
    python = prefix / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "modelforge.cli",
            "--home",
            str(base / "data"),
            "example",
            "--destination",
            str(base / "example.zip"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "packaging/uninstall.py"),
            "--prefix",
            str(prefix),
            "--yes",
        ],
        check=True,
    )
    assert not prefix.exists() and (base / "example.zip").is_file()
print("Fresh installation, CPU example and uninstall passed.")
