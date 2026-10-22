"""Offscreen test of the private chat page using a fake streaming model."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from modelforge import private_runtime as pr
from modelforge.gui_chat import PrivateChatPage


class FakeModel:
    loaded = True
    path = Path("/private/fake")

    def __init__(self) -> None:
        self.seen: list[list[dict[str, Any]]] = []

    def stream(self, messages: list[dict[str, Any]], **settings: Any) -> Iterator[dict[str, Any]]:
        self.seen.append(list(messages))
        for piece in ["שלום ", "world"]:
            yield {"text": piece, "generation_tokens": 2, "generation_tps": 10.0, "prompt_tps": 5.0, "peak_memory_gb": 1.0}

    def status(self) -> dict[str, Any]:
        return {"loaded": True}


def wait(page: PrivateChatPage, app: QApplication) -> None:
    deadline = time.monotonic() + 10
    while page.jobs and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    app.processEvents()
    assert not page.jobs


def test_multi_turn_streaming_chat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("MODELFORGE_PRIVATE_HOME", str(tmp_path))
    manager = pr.PrivateModelManager()
    page = PrivateChatPage(manager)
    assert "No private models" in page.status.text()
    page.input.setPlainText("hello")
    page.send()
    assert "Load a model" in page.status.text()
    fake = FakeModel()
    manager.current = fake  # type: ignore[assignment]
    page.input.setPlainText("מה שלומך?")
    page.send()
    wait(page, app)
    page.input.setPlainText("and again")
    page.send()
    wait(page, app)
    assert [m["role"] for m in fake.seen[-1]] == ["user", "assistant", "user"]
    assert page.history[1]["content"] == "שלום world"
    assert "tok/s" in page.status.text()
    assert "מה שלומך" in page.transcript.toPlainText()
    page.clear()
    assert page.history == []


def test_leaderboard_button(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("MODELFORGE_PRIVATE_HOME", str(tmp_path))
    page = PrivateChatPage(pr.PrivateModelManager())
    page.show_leaderboard()
    assert "No evaluated" in page.status.text()
    runs = tmp_path / "improvement" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "cand--validation.json").write_text(json.dumps({
        "name": "cand", "mean": 0.5, "pooled": {"ci95": [0.4, 0.6], "n": 10}, "strategy": "greedy", "adapter": None,
        "generation_tps": 40.0, "peak_memory_gb": 2.5, "seconds": 60, "scores": {"math": 0.5}}))
    page.show_leaderboard()
    app.processEvents()
    assert "cand: mean 0.500" in page.transcript.toPlainText()
