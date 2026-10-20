"""Optional model engines live in bounded companion Python processes."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from .execution import run_json


def engine_call(
    root: Path,
    operation: str,
    arguments: dict[str, Any],
    *,
    python: str = "",
    memory_limit_mb: int = 2048,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    """Run one whitelisted engine operation with no shell or remote model code."""
    frozen = bool(getattr(sys, "frozen", False))
    companion = (
        root
        / "engine-venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    executable = (
        python
        or os.environ.get("MODELFORGE_ENGINE_PYTHON", "")
        or (str(companion) if companion.is_file() else "" if frozen else sys.executable)
    )
    if not executable or not Path(executable).is_file():
        raise RuntimeError(
            "Install an optional engine in Settings, or set MODELFORGE_ENGINE_PYTHON to a Python 3.12+ ModelForge installation with [llm] or [mlx]."
        )
    environment = dict(os.environ)
    for key in ("PYTHONHOME", "_MEIPASS2", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"):
        environment.pop(key, None)
    if frozen:
        environment.pop("PYTHONPATH", None)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        if "LD_LIBRARY_PATH_ORIG" in environment:
            environment["LD_LIBRARY_PATH"] = environment["LD_LIBRARY_PATH_ORIG"]
        else:
            environment.pop("LD_LIBRARY_PATH", None)
    else:
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
    )
    return run_json(
        [executable, "-m", "modelforge.runtime_worker"],
        {"operation": operation, "arguments": arguments},
        cwd=root / "engine-jobs",
        memory_limit_mb=memory_limit_mb,
        timeout_seconds=timeout_seconds,
        env=environment,
    )


def install_runtime(
    root: Path, backend: str, authorized: bool = False
) -> dict[str, Any]:
    """Install a dedicated optional environment after explicit download consent.

    Installation never edits system Python, a shell profile or an agent setting.
    A failed newly created environment is removed, permitting a clean retry.
    """
    if not authorized:
        raise PermissionError(
            "Optional dependencies require explicit download/install authorization"
        )
    if backend not in {"llm", "mlx"}:
        raise ValueError("Backend must be llm or mlx")
    if backend == "mlx" and (
        sys.platform != "darwin" or __import__("platform").machine() != "arm64"
    ):
        raise ValueError("MLX requires macOS Apple Silicon")
    if shutil.disk_usage(root).free < 3 * 1024**3:
        raise OSError("Need at least 3 GiB free disk for optional engine installation")
    destination = root / "engine-venv"
    if destination.exists():
        raise FileExistsError(
            "Engine environment exists. Use its installed engines or configure another dedicated interpreter; it will not be overwritten."
        )
    uv = shutil.which("uv")
    if uv is None:
        uv = next(
            (
                str(path)
                for path in (
                    Path.home() / ".local/bin/uv",
                    Path("/opt/homebrew/bin/uv"),
                    Path("/usr/local/bin/uv"),
                )
                if path.is_file() and os.access(path, os.X_OK)
            ),
            None,
        )
    base = None if getattr(sys, "frozen", False) else sys.executable
    if not uv and not base:
        for candidate in ("python3.13", "python3.12", "python3"):
            executable = shutil.which(candidate)
            if executable:
                check = subprocess.run(
                    [
                        executable,
                        "-c",
                        "import sys;sys.exit(sys.version_info < (3,12))",
                    ],
                    capture_output=True,
                    timeout=10,
                )
                if check.returncode == 0:
                    base = executable
                    break
    if not uv and not base:
        raise RuntimeError(
            "Install Python 3.12+ or uv, then retry. No system settings were changed."
        )
    resources = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    wheels = list((resources / "engine-wheel").glob("*.whl"))
    spec = str(wheels[0]) if wheels else str(Path(__file__).resolve().parents[2])
    if getattr(sys, "frozen", False) and not wheels:
        raise RuntimeError(
            "This desktop package lacks its optional-engine wheel; install ModelForge from the source release with [llm] or [mlx]."
        )
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {"PYTHONHOME", "PYTHONPATH", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"}
    }
    if getattr(sys, "frozen", False):
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        if "LD_LIBRARY_PATH_ORIG" in env:
            env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH_ORIG"]
        else:
            env.pop("LD_LIBRARY_PATH", None)
    try:
        if uv:
            subprocess.run(
                [uv, "venv", "--python", "3.12", str(destination)],
                check=True,
                timeout=300,
                env=env,
                capture_output=True,
            )
        else:
            assert base is not None
            subprocess.run(
                [base, "-m", "venv", str(destination)],
                check=True,
                timeout=120,
                env=env,
                capture_output=True,
            )
        python = destination / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        command = (
            [uv, "pip", "install", "--python", str(python)]
            if uv
            else [str(python), "-m", "pip", "install"]
        )
        # Explicitly choose CPU wheels on systems where the default Torch includes CUDA.
        if backend == "llm" and sys.platform != "darwin":
            subprocess.run(
                command
                + [
                    "torch>=2.6,<3",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ],
                check=True,
                timeout=900,
                env=env,
                capture_output=True,
            )
        subprocess.run(
            command + [spec + "[" + ("llm,mlx" if backend == "mlx" else backend) + "]"],
            check=True,
            timeout=900,
            env=env,
            capture_output=True,
        )
        (destination / ".modelforge-engine").write_text(backend, encoding="utf-8")
    except Exception:
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        raise
    return {
        "python": str(python),
        "backend": backend,
        "dedicated": True,
        "status": engine_call(root, "status", {}, python=str(python)),
    }
