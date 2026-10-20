"""Real short-lived workers exercise bounded execution without actual OOMs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

import modelforge.execution as execution


def command(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_json_roundtrip_and_private_file_cleanup(tmp_path: Path) -> None:
    result = execution.run_json(
        command("import json,sys; print(json.dumps({'echo':json.load(sys.stdin)}))"),
        {"text": "שלום"},
        cwd=tmp_path,
    )
    assert result["echo"] == {"text": "שלום"}
    assert result["process_monitor_scope"] == "process_and_children"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "code,match",
    [
        ("print('not json; private-log-marker')", "did not return JSON"),
        ("print('[]')", "JSON object"),
        ('print(\'{"ok":false,"error":"controlled failure"}\')', "controlled failure"),
        ("print('{}'); raise SystemExit(3)", "execution failed"),
    ],
)
def test_bad_worker_results_are_bounded(tmp_path: Path, code: str, match: str) -> None:
    with pytest.raises(RuntimeError, match=match) as error:
        execution.run_json(command(code), {}, cwd=tmp_path)
    assert "private-log-marker" not in str(error.value)
    assert not list(tmp_path.iterdir())


def test_timeout_kills_and_reaps_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[Any] = []
    original = execution.subprocess.Popen

    def capture(*args: Any, **kwargs: Any) -> Any:
        child = original(*args, **kwargs)
        launched.append(child)
        return child

    monkeypatch.setattr(execution.subprocess, "Popen", capture)
    with pytest.raises(TimeoutError, match="time budget"):
        execution.run_json(
            command("import time; time.sleep(10)"),
            {},
            cwd=tmp_path,
            timeout_seconds=0.15,
        )
    assert len(launched) == 1
    assert launched[0].poll() is not None
    assert not list(tmp_path.iterdir())


def test_memory_budget_by_injected_rss_without_allocating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Oversized:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def memory_info(self) -> Any:
            return SimpleNamespace(rss=129 * 1024**2)

        def children(self, recursive: bool = True) -> list[Any]:
            return []

    monkeypatch.setattr(execution.psutil, "Process", Oversized)
    with pytest.raises(MemoryError, match="memory budget"):
        execution.run_json(
            command("import time; time.sleep(10)"),
            {},
            cwd=tmp_path,
            memory_limit_mb=128,
        )
    assert not list(tmp_path.iterdir())


def test_child_inspection_permission_denial_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = psutil.Process

    class ParentOnly:
        def __init__(self, pid: int) -> None:
            self.process = original(pid)

        def memory_info(self) -> Any:
            return self.process.memory_info()

        def children(self, recursive: bool = True) -> list[Any]:
            raise psutil.AccessDenied(self.process.pid)

    monkeypatch.setattr(execution.psutil, "Process", ParentOnly)
    result = execution.run_json(
        command("import time; time.sleep(.15); print('{}')"), {}, cwd=tmp_path
    )
    assert result["process_monitor_scope"] == "process_only_children_unavailable"


def test_output_limit_is_enforced_even_after_fast_exit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="output/log budget"):
        execution.run_json(
            command("print('x'*4096)"), {}, cwd=tmp_path, output_limit_bytes=1024
        )


@pytest.mark.parametrize(
    "options",
    [{"memory_limit_mb": 1}, {"timeout_seconds": 0}, {"output_limit_bytes": 10}],
)
def test_invalid_limits(tmp_path: Path, options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        execution.run_json(command("print('{}')"), {}, cwd=tmp_path, **options)


def test_payload_bound(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2MB"):
        execution.run_json(
            command("print('{}')"), {"text": "x" * 2_000_000}, cwd=tmp_path
        )
