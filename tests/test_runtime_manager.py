"""Companion runtime selection, isolated environments and install rollback."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import modelforge.runtime_manager as manager


def test_actual_worker_status_without_loading_heavy_models(tmp_path: Path) -> None:
    result = manager.engine_call(
        tmp_path,
        "status",
        {},
        python=sys.executable,
        memory_limit_mb=512,
        timeout_seconds=30,
    )
    assert result["ok"]
    assert "torch" in result["packages"]
    assert result["python"].startswith("3.")
    assert result["process_monitor_scope"] in {
        "process_and_children",
        "process_only_children_unavailable",
    }
    assert list((tmp_path / "engine-jobs").iterdir()) == []


def test_missing_interpreter_and_wrong_program_fail_explicitly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Install an optional engine"):
        manager.engine_call(
            tmp_path, "status", {}, python=str(tmp_path / "missing-python")
        )
    wrong = shutil.which("echo")
    if wrong is not None:
        with pytest.raises(RuntimeError, match="did not return JSON"):
            manager.engine_call(
                tmp_path, "status", {}, python=wrong, memory_limit_mb=512
            )


def test_frozen_worker_restores_external_library_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def run(command: Any, payload: Any, **options: Any) -> dict[str, Any]:
        captured.update(command=command, payload=payload, **options)
        return {"ok": True}

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/bundled/python")
    monkeypatch.setenv("PYTHONHOME", "/bundled/home")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundled/libs")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/system/libs")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/bundled/dyld")
    monkeypatch.setattr(manager, "run_json", run)
    manager.engine_call(tmp_path, "status", {}, python=sys.executable)
    environment = captured["env"]
    assert environment["LD_LIBRARY_PATH"] == "/system/libs"
    assert not {"PYTHONHOME", "PYTHONPATH", "DYLD_LIBRARY_PATH"} & environment.keys()
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["PYINSTALLER_RESET_ENVIRONMENT"] == "1"


@pytest.fixture
def mocked_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        manager.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3)
    )
    monkeypatch.setattr(
        manager.shutil, "which", lambda name: "/tools/uv" if name == "uv" else None
    )

    def run(command: list[str], **options: Any) -> Any:
        calls.append({"command": command, **options})
        if "venv" in command:
            destination = Path(command[-1])
            (destination / "bin").mkdir(parents=True)
            (destination / "bin/python").write_text("fake interpreter; never executed")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(manager.subprocess, "run", run)
    monkeypatch.setattr(
        manager, "engine_call", lambda *args, **kwargs: {"ok": True, "packages": {}}
    )
    return calls


def test_install_requires_consent_and_refuses_existing_environment(
    tmp_path: Path, mocked_install: Any
) -> None:
    with pytest.raises(PermissionError):
        manager.install_runtime(tmp_path, "llm")
    destination = tmp_path / "engine-venv"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("existing data")
    with pytest.raises(FileExistsError):
        manager.install_runtime(tmp_path, "llm", authorized=True)
    assert marker.read_text() == "existing data"
    assert not mocked_install


def test_linux_install_selects_cpu_torch_and_restores_library_path(
    tmp_path: Path, mocked_install: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/frozen/native")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/outside/native")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    resources = tmp_path / "resources"
    wheel = resources / "engine-wheel/modelforge_research-0.2.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"mock wheel; never installed")
    monkeypatch.setattr(sys, "_MEIPASS", str(resources), raising=False)
    result = manager.install_runtime(tmp_path, "llm", authorized=True)
    assert result["dedicated"]
    assert any(
        "https://download.pytorch.org/whl/cpu" in call["command"]
        for call in mocked_install
    )
    assert any(str(wheel) + "[llm]" in call["command"] for call in mocked_install)
    assert all(
        call["env"].get("LD_LIBRARY_PATH") == "/outside/native"
        for call in mocked_install
    )
    assert (tmp_path / "engine-venv/.modelforge-engine").read_text() == "llm"


def test_install_rollback_removes_only_new_environment(
    tmp_path: Path, mocked_install: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = manager.subprocess.run
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep")

    def fail(command: list[str], **options: Any) -> Any:
        if "install" in command:
            raise subprocess.CalledProcessError(1, command)
        return original(command, **options)

    monkeypatch.setattr(manager.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        manager.install_runtime(tmp_path, "llm", authorized=True)
    assert not (tmp_path / "engine-venv").exists()
    assert unrelated.read_text() == "keep"


def test_finder_launch_finds_user_uv_outside_path(
    tmp_path: Path, mocked_install: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    uv = home / ".local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("mock uv; never executed")
    uv.chmod(0o700)
    monkeypatch.setattr(manager.shutil, "which", lambda _: None)
    monkeypatch.setattr(Path, "home", lambda: home)
    manager.install_runtime(tmp_path, "llm", authorized=True)
    assert mocked_install[0]["command"][0] == str(uv)


def test_frozen_package_reports_missing_embedded_wheel(
    tmp_path: Path, mocked_install: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        sys, "_MEIPASS", str(tmp_path / "missing-resources"), raising=False
    )
    with pytest.raises(RuntimeError, match="lacks its optional-engine wheel"):
        manager.install_runtime(tmp_path, "llm", authorized=True)
    assert not mocked_install


def test_source_python_fallback_without_uv(
    tmp_path: Path, mocked_install: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager.shutil, "which", lambda _: None)
    monkeypatch.setattr(manager.os, "access", lambda *args: False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    manager.install_runtime(tmp_path, "llm", authorized=True)
    assert mocked_install[0]["command"][:3] == [sys.executable, "-m", "venv"]
    assert any(
        call["command"][1:4] == ["-m", "pip", "install"] for call in mocked_install
    )
