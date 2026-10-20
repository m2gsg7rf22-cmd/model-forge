"""Build platform-native standalone desktop and preserve third-party notices."""

from pathlib import Path
import os
import importlib.metadata
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "PYINSTALLER_CONFIG_DIR", str(root / "build" / "pyinstaller-cache")
)
notices = root / "build" / "THIRD_PARTY_NOTICES"
notices.mkdir(parents=True, exist_ok=True)
for dist in importlib.metadata.distributions():
    for file in dist.files or []:
        if (
            "license" in str(file).lower() or "copying" in str(file).lower()
        ) and file.suffix.lower() in {".txt", ".md", ".rst", ""}:
            source = Path(dist.locate_file(file))
            if source.is_file():
                dest = notices / str(dist.metadata["Name"]) / file.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
separator = ";" if sys.platform == "win32" else ":"
subprocess.run(
    [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        "ModelForge",
        "--paths",
        str(root / "src"),
        "--add-data",
        str(notices) + separator + "THIRD_PARTY_NOTICES",
        "--exclude-module",
        "PySide6.QtWebEngineCore",
        "--exclude-module",
        "PySide6.QtWebEngineWidgets",
        "--exclude-module",
        "PySide6.QtWebEngineQuick",
        "--exclude-module",
        "onnx",
        "--exclude-module",
        "coremltools",
        "--exclude-module",
        "pytest",
        str(root / "packaging" / "desktop_entry.py"),
    ],
    cwd=root,
    check=True,
)
if sys.platform == "darwin":
    subprocess.run(["xattr", "-cr", str(root / "dist" / "ModelForge.app")], check=True)
    subprocess.run(
        [
            "codesign",
            "--force",
            "--deep",
            "--sign",
            "-",
            str(root / "dist" / "ModelForge.app"),
        ],
        check=True,
    )
print(
    "Built native desktop in dist/. Optional ONNX/Core ML parsers are source-install extras."
)
