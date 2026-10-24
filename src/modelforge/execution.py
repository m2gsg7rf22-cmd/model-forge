"""Bounded subprocess execution for optional heavyweight research engines."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import psutil


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Reap the parent even when child inspection is denied or races with exit."""
    if process.poll() is not None:
        return
    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            try:
                child.kill()
            except psutil.Error:
                pass
    except (psutil.Error, PermissionError):
        pass
    finally:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run_json(
    command: Sequence[str],
    payload: dict[str, Any],
    *,
    cwd: Path,
    memory_limit_mb: int = 2048,
    timeout_seconds: float = 180,
    output_limit_bytes: int = 2_000_000,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute one JSON-in/JSON-out engine with bounded logs, time and process RSS.

    Payloads are private temporary files in the chosen workspace. No shell is used.
    RSS is periodically sampled rather than reserved by the operating system.
    """
    if not 128 <= memory_limit_mb <= 65536 or not 0 < timeout_seconds <= 3600:
        raise ValueError("Engine limits must be 128–65536 MiB and 0–3600 seconds")
    if not command or not 1024 <= output_limit_bytes <= 16_000_000:
        raise ValueError("Supply an executable and a 1KiB–16MB output limit")
    cwd.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
    if len(encoded) > 2_000_000:
        raise ValueError("Engine input exceeds the 2MB request bound")
    with tempfile.TemporaryDirectory(prefix="engine-", dir=cwd) as temporary:
        root = Path(temporary)
        request, output, error = (
            root / "request.json",
            root / "output.json",
            root / "error.log",
        )
        request.write_bytes(encoded)
        start = time.monotonic()
        monitor_scope = "process_and_children"
        peak_rss = 0
        memory_samples = 0
        with (
            request.open("rb") as incoming,
            output.open("wb") as outgoing,
            error.open("wb") as errors,
        ):
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                stdin=incoming,
                stdout=outgoing,
                stderr=errors,
                env=env,
            )
            try:
                while process.poll() is None:
                    if time.monotonic() - start > timeout_seconds:
                        raise TimeoutError(
                            "Engine exceeded its time budget; process terminated"
                        )
                    try:
                        current = psutil.Process(process.pid)
                        rss = current.memory_info().rss
                        try:
                            children = current.children(recursive=True)
                        except (psutil.AccessDenied, PermissionError):
                            children = []
                            monitor_scope = "process_only_children_unavailable"
                        for child in children:
                            try:
                                rss += child.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        continue
                    peak_rss = max(peak_rss, rss)
                    memory_samples += 1
                    if rss > memory_limit_mb * 1024**2:
                        raise MemoryError(
                            "Engine exceeded its process memory budget; process terminated"
                        )
                    if (
                        output.stat().st_size + error.stat().st_size
                        > output_limit_bytes
                    ):
                        raise RuntimeError(
                            "Engine exceeded its output/log budget; process terminated"
                        )
                    time.sleep(0.05)
            finally:
                _stop(process)
        if output.stat().st_size + error.stat().st_size > output_limit_bytes:
            raise RuntimeError("Engine result exceeded the output/log budget")
        try:
            result: Any = json.loads(output.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            # Never return provider credentials or unbounded dependency tracebacks.
            raise RuntimeError(
                f"Engine did not return JSON (exit {process.returncode}); check optional runtime installation"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError("Engine response must be a JSON object")
        if process.returncode or result.get("ok") is False:
            raise RuntimeError(
                str(result.get("error", "Engine execution failed"))[:2000]
            )
        result["process_monitor_scope"] = monitor_scope
        result["peak_sampled_rss_bytes"] = peak_rss
        result["memory_samples"] = memory_samples
        result["memory_limit_mb"] = memory_limit_mb
        return result
