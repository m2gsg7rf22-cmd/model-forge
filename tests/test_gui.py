"""Offscreen integration tests against the real persistent desktop service."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from modelforge.gui import MainWindow
from modelforge.service import Service


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def drain(window: MainWindow, app: QApplication) -> None:
    deadline = time.monotonic() + 20
    while window.workers and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    app.processEvents()
    assert not window.workers, "Background GUI operation timed out"


def test_real_desktop_project_workflow(tmp_path: Path, app: QApplication) -> None:
    service = Service(tmp_path / "workspace")
    window = MainWindow(service)
    window.show()
    drain(window, app)
    assert window.stack.count() == 9
    assert '"memory_total"' in window.outputs["Dashboard"].toPlainText()
    window.project_name.setText("Desktop integration")
    window.create_project()
    drain(window, app)
    assert len(service.projects()) == 1
    project_id = window.project_id()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_type": "test", "hidden_size": 32}))
    window.artifact_path.setText(str(config))
    window.inspect()
    drain(window, app)
    assert service.findings(project_id)[0]["kind"] == "inspection"
    assert len(window.graph.graph_scene.items()) >= 2
    window.build()
    drain(window, app)
    assert "Operation failed" not in window.outputs["Reconstruction"].toPlainText()
    window.infer_text.setText("hello")
    window.infer()
    drain(window, app)
    assert '"output"' in window.outputs["Reconstruction"].toPlainText()
    window.training_text.setPlainText("hello world hello world")
    window.steps.setValue(1)
    window.train()
    drain(window, app)
    assert "Operation failed" not in window.outputs["Training"].toPlainText()
    window.export_path.setText(str(tmp_path / "project.zip"))
    window.export()
    drain(window, app)
    assert (tmp_path / "project.zip").is_file()
    window.settings_editor.setPlainText('{"api_model":"local-model"}')
    window.save_settings()
    drain(window, app)
    assert service.settings()["api_model"] == "local-model"
    window.task_kind.setCurrentText("infer")
    window.task_payload.setPlainText('{"text":"hello"}')
    window.enqueue()
    drain(window, app)
    assert service.tasks()[0]["status"] == "queued"
    window.run_pending()
    drain(window, app)
    assert service.tasks()[0]["status"] == "completed"
    for index in range(9):
        window.nav.setCurrentRow(index)
        assert window.stack.currentIndex() == index
    window.close()
    service.close()


def test_visible_failures_and_auth_gate(tmp_path: Path, app: QApplication) -> None:
    service = Service(tmp_path / "workspace")
    window = MainWindow(service)
    drain(window, app)
    window.inspect()
    assert "Create or select a project" in window.outputs["Explorer"].toPlainText()
    window.project_name.setText("Error tests")
    window.create_project()
    drain(window, app)
    window.artifact_path.setText(str(tmp_path / "absent.safetensors"))
    window.inspect()
    drain(window, app)
    assert "Operation failed" in window.outputs["Explorer"].toPlainText()
    window.probe()
    assert "permission" in window.outputs["Research"].toPlainText()
    window.task_payload.setPlainText("[]")
    window.enqueue()
    assert "JSON object" in window.outputs["Agents"].toPlainText()
    window.task_payload.setPlainText("{}")
    window.task_kind.setCurrentText("agent")
    window.enqueue()
    assert "Authorize external agent" in window.outputs["Agents"].toPlainText()
    assert service.tasks() == []
    window.settings_editor.setPlainText("invalid")
    window.save_settings()
    assert "Operation failed" in window.outputs["Settings"].toPlainText()
    window.close()
    service.close()
