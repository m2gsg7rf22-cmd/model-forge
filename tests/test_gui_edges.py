"""Desktop interaction boundaries; dialogs are selected by explicit test fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QFileDialog
from modelforge.gui import GraphView, MainWindow
from modelforge.service import Service
from test_gui import drain
from test_service_edges import http_fixture


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_dialog_selected_paths_load_actual_files(
    tmp_path: Path, app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = Service(tmp_path / "workspace")
    window = MainWindow(service)
    drain(window, app)
    artifact = tmp_path / "config.json"
    artifact.write_text('{"model_type":"dialog-fixture"}')
    corpus = tmp_path / "training.txt"
    corpus.write_text("Actual UTF-8 text: café")
    selected = iter([(str(artifact), ""), (str(corpus), ""), ("", ""), ("", "")])
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: next(selected)
    )
    window.browse_artifact()
    assert window.artifact_path.text() == str(artifact)
    window.browse_training()
    drain(window, app)
    assert window.training_text.toPlainText() == corpus.read_text()
    window.browse_artifact()
    window.browse_training()
    assert window.artifact_path.text() == str(artifact)
    target = str(tmp_path / "archive.zip")
    saves = iter([(target, ""), ("", "")])
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: next(saves)
    )
    window.browse_export()
    window.browse_export()
    assert window.export_path.text() == target
    window.close()
    service.close()


def test_authorized_probe_renders_real_loopback_result(
    tmp_path: Path, app: QApplication
) -> None:
    service = Service(tmp_path / "workspace")
    service.create_project("Probe UI")
    window = MainWindow(service)
    drain(window, app)
    with http_fixture("plain") as (endpoint, requests):
        window.endpoint.setText(endpoint)
        window.model_name.setText("loopback-fixture")
        window.prompts.setPlainText("hello\n\nworld")
        window.budget.setValue(20)
        window.authorized.setChecked(True)
        window.probe()
        drain(window, app)
    rendered = json.loads(window.outputs["Research"].toPlainText())
    assert len(rendered["results"]) == 2
    assert rendered["results"][0]["output"] == "not json"
    assert len(requests) == 2
    window.load_tasks()
    drain(window, app)
    assert json.loads(window.outputs["Agents"].toPlainText()) == []
    window.agent_authorized.setChecked(True)
    window.task_kind.setCurrentText("agent")
    window.task_payload.setPlainText('{"provider":"codex","prompt":"fixture"}')
    window.enqueue()
    drain(window, app)
    assert service.tasks()[0]["payload"]["authorized"] is True
    # Queuing is persisted; this test deliberately does not execute a provider.
    window.settings_editor.setPlainText("[]")
    window.save_settings()
    assert "JSON object" in window.outputs["Settings"].toPlainText()
    window.close()
    service.close()


def test_worker_callback_failure_and_close_during_job(
    tmp_path: Path, app: QApplication
) -> None:
    service = Service(tmp_path / "workspace")
    window = MainWindow(service)
    drain(window, app)
    window.create_project()
    assert "Enter a project name" in window.outputs["Dashboard"].toPlainText()
    window.build()
    assert "select a project" in window.outputs["Reconstruction"].toPlainText()

    def invalid_callback(result: Any) -> None:
        raise ValueError("fixture callback failure")

    window.run("Dashboard", "callback fixture", lambda: {}, invalid_callback)
    drain(window, app)
    assert "fixture callback failure" in window.outputs["Dashboard"].toPlainText()
    window.show()
    release = threading.Event()
    window.run("Dashboard", "blocking fixture", lambda: release.wait(5))
    assert window.close() is False
    assert "still running" in window.status.text()
    release.set()
    drain(window, app)
    assert window.close() is True
    service.close()


def test_graph_actual_edges_fallback_and_zoom(app: QApplication) -> None:
    graph = GraphView()
    graph.set_data(
        {
            "architecture": {
                "nodes": [
                    {"id": "a", "label": "Input"},
                    {"id": "b", "label": "Output"},
                ],
                "edges": [
                    {"source": "a", "target": "b", "evidence": "fixture graph"},
                    "invalid",
                ],
            }
        }
    )
    assert len(graph.graph_scene.items()) == 5
    assert any("fixture graph" in item.toolTip() for item in graph.graph_scene.items())
    before = graph.transform().m11()
    event = QWheelEvent(
        QPointF(20, 20),
        QPointF(20, 20),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    graph.wheelEvent(event)
    assert graph.transform().m11() > before
    graph.set_data({"tensors": [{"name": "embedding", "shape": [10, 4]}]})
    assert len(graph.graph_scene.items()) == 2
    graph.set_data({"config": {"hidden_size": 32}})
    assert len(graph.graph_scene.items()) == 2
    graph.close()
