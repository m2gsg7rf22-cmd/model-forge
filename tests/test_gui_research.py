"""Offscreen controls exercise persistent research service boundaries."""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from modelforge.gui import MainWindow
from modelforge.service import Service


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def drain(window: MainWindow, app: QApplication) -> None:
    deadline = time.monotonic() + 95
    while window.workers and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    app.processEvents()
    assert not window.workers, window.status.text()


@pytest.fixture
def desktop(tmp_path: Path, app: QApplication) -> Iterator[tuple[MainWindow, Service]]:
    service = Service(tmp_path / "workspace")
    service.create_project("Research GUI")
    window = MainWindow(service)
    drain(window, app)
    yield window, service
    drain(window, app)
    window.close()
    service.close()


def test_cache_benchmark_persists_selected_controls(
    desktop: tuple[MainWindow, Service],
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing a selected prompt/budget must change the persisted run configuration."""
    window, service = desktop
    service._save_binding(
        window.project_id(), {"active_path": "/owned/fixture", "adapter_path": ""}
    )

    def gpu_transport(
        operation: str,
        arguments: dict[str, Any],
        memory_limit_mb: int = 2048,
        timeout_seconds: float = 300,
    ) -> dict[str, Any]:
        # Replace only expensive worker transport. Binding, dispatch, result storage
        # and GUI signals remain real; no scientific measurements are synthesized.
        return {
            "scope": "test transport configuration receipt",
            "configuration": arguments,
        }

    monkeypatch.setattr(service, "_engine", gpu_transport)
    window.local_prompt.setText("a selected cache probe")
    window.local_tokens.setValue(6)
    window.local_memory.setValue(768)
    window.local_bits.setCurrentText("4")
    window.benchmark_cache()
    drain(window, app)
    saved = service.findings(window.project_id())
    assert saved[0]["kind"] == "mlx-cache-benchmark"
    config = saved[0]["data"]["configuration"]
    assert config["prompt"] == "a selected cache probe"
    assert config["max_new_tokens"] == 6
    assert config["memory_limit_mb"] == 768
    assert config["quantize_bits"] == 4
    assert (
        json.loads(window.outputs["Reconstruction"].toPlainText()) == saved[0]["data"]
    )


def test_discovery_action_runs_real_cpu_worker_and_persists_report(
    desktop: tuple[MainWindow, Service], app: QApplication
) -> None:
    window, service = desktop
    window.umrre_seed.setValue(31)
    window.discovery()
    drain(window, app)
    report = json.loads(window.outputs["Research"].toPlainText())
    assert report["engine"] == "ModelForge Discovery Lab"
    assert report["preregistration"]["public_sampling_seed"] == 31
    assert report["verification"]["status"] == "completed"
    assert report["verification"]["process_separated"] is True
    assert report["verification"]["sealed_test_used_for_selection"] is False
    assert report["cost"]["remote_requests"] == 0
    saved = service.findings(window.project_id())
    assert saved[0]["kind"] == "discovery"
    assert saved[0]["data"] == report
    public = service.root / window.project_id() / "discovery"
    reports = list(public.glob("*/report.json"))
    assert len(reports) == 1
    artifact = json.loads(reports[0].read_text())
    assert artifact["numerical_sha256"] == report["numerical_sha256"]
    assert artifact["results"] == report["results"]
    assert not list(public.rglob("truth.json"))
    assert list((service.root / "private-research").glob("*/truth.json"))


def test_repository_recovery_action_persists_interruption_without_executing_jobs(
    desktop: tuple[MainWindow, Service], app: QApplication, tmp_path: Path
) -> None:
    import hashlib
    import subprocess

    window, _ = desktop
    root = tmp_path / "checkpoint-repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "base",
        ],
        check=True,
    )
    base = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    control = root / ".modelforge"
    control.mkdir()
    mission = root / "mission.md"
    mission.write_text("Owned immutable GUI recovery fixture", encoding="utf-8")
    (control / "mission_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "immutable_documents": {
                    "mission.md": hashlib.sha256(mission.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    marker = root / "must-not-execute"
    state = {
        "schema_version": 1,
        "checkpoint_base_commit": base,
        "ledger_head": None,
        "private_model_gate": {"status": "closed"},
        "tasks": [
            {
                "id": "interrupted_work",
                "status": "running",
                "priority": 1,
                "dependencies": [],
                "owner": {"pid": 2147483647, "create_time": 1},
                "command": f"touch {marker}",
            }
        ],
    }
    state_path = control / "research_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    window.research_repository.setText(str(root))
    window.resume_repository()
    drain(window, app)
    report = json.loads(window.outputs["Research"].toPlainText())
    assert report["execution"] == "report_only"
    assert report["private_model_gate"] == "closed"
    assert report["eligible_tasks"][0]["id"] == "interrupted_work"
    assert report["eligible_tasks"][0]["status"] == "interrupted"
    assert json.loads(state_path.read_text())["tasks"][0]["status"] == "interrupted"
    assert (root / "experiments/ledger.jsonl").is_file()
    assert not marker.exists()
    window.research_repository.setText(str(tmp_path / "missing-repository"))
    window.resume_repository()
    drain(window, app)
    assert "Operation failed" in window.outputs["Research"].toPlainText()
