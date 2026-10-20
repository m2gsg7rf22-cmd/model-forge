"""Qt event-loop checks for engine controls, authorization, and worker errors."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

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
    deadline = time.monotonic() + 30
    while window.workers and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    app.processEvents()
    assert not window.workers


@pytest.fixture
def desktop(tmp_path: Path, app: QApplication):
    service = Service(tmp_path / "workspace")
    service.create_project("Engine UI")
    window = MainWindow(service)
    window.show()
    drain(window, app)
    yield window, service
    drain(window, app)
    window.close()
    service.close()


def test_engine_authorization_prevents_dispatch(desktop: Any) -> None:
    window, service = desktop
    window.attach_local()
    assert "permission" in window.outputs["Reconstruction"].toPlainText()
    window.reconstruct_hf()
    assert "rights statement" in window.outputs["Reconstruction"].toPlainText()
    window.train_hf()
    assert "rights statement" in window.outputs["Training"].toPlainText()
    window.experiment()
    assert "permission" in window.outputs["Research"].toPlainText()
    window.install_runtime()
    assert "Authorize" in window.outputs["Settings"].toPlainText()
    assert service.findings(window.project_id()) == []


def test_dependency_errors_are_visible(
    desktop: Any, app: QApplication, monkeypatch: Any
) -> None:
    window, service = desktop

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Install optional llm engine dependencies")

    monkeypatch.setattr(service, "local_generate", unavailable, raising=False)
    window.generate_local()
    drain(window, app)
    assert "Install optional llm" in window.outputs["Reconstruction"].toPlainText()
    assert window.status.text().startswith("Failed")


def test_durable_experiment_validates_cases_and_preserves_resume(
    desktop: Any, app: QApplication, monkeypatch: Any
) -> None:
    window, service = desktop
    window.authorized.setChecked(True)
    window.experiment_cases.setPlainText("{}")
    window.experiment()
    assert "JSON array" in window.outputs["Research"].toPlainText()
    received: dict[str, Any] = {}

    def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"experiment_id": "ledger-test", "reserved_requests": 1}

    monkeypatch.setattr(service, "run_experiment", run, raising=False)
    window.experiment_cases.setPlainText(
        '[{"kind":"capability","prompt":"hello","expected":"hello"}]'
    )
    window.experiment_resume.setText("ledger-test")
    window.experiment_requests.setValue(2)
    window.budget.setValue(64)
    window.experiment()
    drain(window, app)
    assert received == {
        "authorized": True,
        "max_requests": 2,
        "output_token_budget": 64,
        "resume_id": "ledger-test",
    }
    assert '"reserved_requests": 1' in window.outputs["Research"].toPlainText()


def test_interpreter_path_validation_and_persistence(
    desktop: Any, app: QApplication
) -> None:
    window, service = desktop
    window.runtime_python.setText("relative/python")
    window.save_runtime_python()
    assert "absolute path" in window.outputs["Settings"].toPlainText()
    import sys

    window.runtime_python.setText(sys.executable)
    window.save_runtime_python()
    drain(window, app)
    assert service.settings()["runtime_python"] == sys.executable


def test_real_llama_fixture_and_generation_through_gui(
    desktop: Any, app: QApplication
) -> None:
    """Create actual SafeTensors, then run the subprocess from a Qt action."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    window, service = desktop
    window.local_fixture_type.setCurrentText("llama")
    window.create_hf_fixture()
    drain(window, app)
    assert "Operation failed" not in window.outputs["Reconstruction"].toPlainText()
    window.local_prompt.setText("hello world")
    window.local_tokens.setValue(2)
    window.generate_local()
    drain(window, app)
    assert "Operation failed" not in window.outputs["Reconstruction"].toPlainText()
    import json

    result = json.loads(window.outputs["Reconstruction"].toPlainText())
    assert 0 < result["new_tokens"] <= 2
    assert result["token_ids"]
    assert {"local-build", "local-inference"} <= {
        row["kind"] for row in service.findings(window.project_id())
    }
