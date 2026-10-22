"""Private model chat page for the ModelForge workbench (streaming, multi-turn)."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import private_runtime as pr


class _Signals(QObject):
    chunk = Signal(str)
    done = Signal(object)
    failed = Signal(str)


class _Job(QRunnable):
    def __init__(self, fn: Any, signals: _Signals) -> None:
        super().__init__()
        self.fn, self.signals = fn, signals

    @Slot()
    def run(self) -> None:
        try:
            self.signals.done.emit(self.fn(self.signals))
        except Exception as exc:  # shown in the page, never swallowed
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class PrivateChatPage(QWidget):
    """Local chat with a private MLX model; generation runs off the GUI thread."""

    def __init__(self, manager: pr.PrivateModelManager | None = None) -> None:
        super().__init__()
        self.manager = manager or pr.MANAGER
        self.pool = QThreadPool(self)
        self.history: list[dict[str, Any]] = []
        self.jobs: set[_Job] = set()
        self.stop_requested = False
        layout = QVBoxLayout(self)
        title = QLabel("Private Chat")
        title.setObjectName("title")
        layout.addWidget(title)
        hint = QLabel("Runs a private local model on this Mac (MLX/Metal). No hosted inference API is used.")
        hint.setObjectName("description")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(320)
        row.addWidget(self.model_combo, 1)
        self.load_button = self._button("Load", self.load_model, row, primary=True)
        self.unload_button = self._button("Unload", self.unload_model, row)
        self._button("Refresh", self.refresh_models, row)
        layout.addLayout(row)
        settings = QHBoxLayout()
        settings.addWidget(QLabel("Max tokens"))
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(16, 4096)
        self.max_tokens.setValue(512)
        settings.addWidget(self.max_tokens)
        settings.addWidget(QLabel("Temperature"))
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.0)
        settings.addWidget(self.temperature)
        settings.addStretch(1)
        layout.addLayout(settings)
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setMinimumHeight(320)
        layout.addWidget(self.transcript, 1)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Type a message (Hebrew or English)…  הקלד הודעה")
        self.input.setMaximumHeight(110)
        layout.addWidget(self.input)
        actions = QHBoxLayout()
        self.send_button = self._button("Send", self.send, actions, primary=True)
        self._button("Stop", self.stop, actions)
        self._button("New conversation", self.clear, actions)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.status = QLabel("No model loaded")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refresh_models()

    def _button(self, text: str, callback: Any, row: QHBoxLayout, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(callback)
        row.addWidget(button)
        return button

    def refresh_models(self) -> None:
        current = self.model_combo.currentText()
        self.model_combo.clear()
        names = [m["name"] for m in pr.list_private_models()]
        self.model_combo.addItems(names)
        if current in names:
            self.model_combo.setCurrentText(current)
        if not names:
            self.status.setText(f"No private models found in {pr.private_home() / 'models'}")

    def _start(self, fn: Any, on_done: Any, on_chunk: Any = None) -> None:
        signals = _Signals()
        job = _Job(fn, signals)
        self.jobs.add(job)

        def finish(result: Any) -> None:
            self.jobs.discard(job)
            self._busy(False)
            on_done(result)

        def fail(message: str) -> None:
            self.jobs.discard(job)
            self._busy(False)
            self.status.setText("Error: " + message)

        signals.done.connect(finish)
        signals.failed.connect(fail)
        if on_chunk:
            signals.chunk.connect(on_chunk)
        job.signals_ref = signals  # type: ignore[attr-defined]  # keep alive
        self._busy(True)
        self.pool.start(job)

    def _busy(self, busy: bool) -> None:
        for button in (self.load_button, self.unload_button, self.send_button):
            button.setEnabled(not busy)

    def load_model(self) -> None:
        name = self.model_combo.currentText()
        if not name:
            self.status.setText("Select a model first")
            return
        self.status.setText(f"Loading {name}…")
        self._start(lambda _s: self.manager.load(name), self._show_status)

    def unload_model(self) -> None:
        self._start(lambda _s: self.manager.unload(), lambda _r: self.status.setText("Model unloaded"))

    def _show_status(self, status: dict[str, Any]) -> None:
        gb = 1024**3
        self.status.setText(
            f"Loaded {status.get('path', '').split('/')[-1]} in {status.get('load_seconds', 0):.1f}s · "
            f"GPU memory {status.get('active_memory_bytes', 0) / gb:.2f} GB "
            f"(limit {status.get('memory_limit_bytes', 0) / gb:.0f} GB, unified {status.get('unified_memory_bytes') or 0 / gb})"
        )

    def _append(self, role: str, text: str) -> None:
        align = "right" if any("֐" <= ch <= "׿" for ch in text) else "left"
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        color = "#7dece4" if role == "You" else "#dce5f2"
        self.transcript.append(f"<p align='{align}'><b style='color:{color}'>{role}:</b> {safe}</p>")

    def send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        if self.manager.current is None or not self.manager.current.loaded:
            self.status.setText("Load a model before chatting")
            return
        self.input.clear()
        self.history.append({"role": "user", "content": text})
        self._append("You", text)
        self.stop_requested = False
        settings: dict[str, Any] = {"max_tokens": self.max_tokens.value(), "temperature": self.temperature.value()}
        messages = list(self.history)
        parts: list[str] = []

        def work(signals: _Signals) -> dict[str, Any]:
            assert self.manager.current is not None
            last: dict[str, Any] = {}
            for chunk in self.manager.current.stream(messages, stop=lambda: self.stop_requested, **settings):
                signals.chunk.emit(chunk["text"])
                last = chunk
            return last

        def on_chunk(piece: str) -> None:
            parts.append(piece)
            self.status.setText("Generating… " + str(len("".join(parts))) + " chars")

        def on_done(last: dict[str, Any]) -> None:
            reply = "".join(parts)
            self.history.append({"role": "assistant", "content": reply})
            self._append("Model", reply)
            self.status.setText(
                f"{last.get('generation_tokens', 0)} tokens · {last.get('generation_tps', 0):.1f} tok/s · "
                f"prompt {last.get('prompt_tps', 0):.0f} tok/s · peak {last.get('peak_memory_gb', 0):.2f} GB"
            )

        self._start(work, on_done, on_chunk)

    def stop(self) -> None:
        self.stop_requested = True

    def clear(self) -> None:
        self.history.clear()
        self.transcript.clear()
        self.status.setText("New conversation")
