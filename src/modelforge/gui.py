"""Native desktop workbench for local model research.

All service work runs outside the GUI thread. Results, including errors, remain
visible in the relevant page; the workbench never invents progress or metrics.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

STYLE = """
QWidget { background: #0e1420; color: #dce5f2; font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 13px; }
QMainWindow { background: #0e1420; }
QWidget#sidebar { background: #111b2b; border-right: 1px solid #26364d; }
QLabel#brand { color: #6de5de; font-size: 25px; font-weight: 800; padding: 14px 0; }
QLabel#eyebrow { color: #6de5de; font-size: 11px; font-weight: 700; }
QLabel#title { font-size: 29px; font-weight: 700; color: #f3f7ff; }
QLabel#description { color: #8fa1b9; padding: 0 0 15px 0; }
QListWidget { background: transparent; border: 0; outline: 0; }
QListWidget::item { padding: 12px 16px; border-radius: 7px; margin: 3px 0; color: #9daec4; }
QListWidget::item:selected { background: #21364a; color: #7dece4; border-left: 3px solid #67dfd5; }
QPushButton { background: #23344b; color: #dfeaff; padding: 10px 15px; border: 1px solid #354a66; border-radius: 6px; font-weight: 600; }
QPushButton:hover { background: #2e4660; border-color: #67dfd5; }
QPushButton:disabled { color: #65758c; background: #182334; }
QPushButton#primary { background: #63ded2; color: #092127; border: 0; }
QPushButton#primary:hover { background: #8bf1e6; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox { background: #141f30; border: 1px solid #2c405b; border-radius: 6px; padding: 9px; selection-background-color: #386c7c; }
QPlainTextEdit { font-family: 'SF Mono', 'Consolas', monospace; font-size: 12px; }
QComboBox::drop-down { border: 0; width: 24px; }
QLabel#status { background: #152435; padding: 10px; border-radius: 6px; color: #9aafc9; }
QLabel#metric { background: #172536; padding: 22px; border: 1px solid #2b4058; border-radius: 9px; font-size: 15px; }
QGraphicsView { background: #101c2a; border: 1px solid #2c405b; border-radius: 8px; }
QScrollArea { border: 0; }
QCheckBox { padding: 7px 0; }
"""


def pretty(value: Any) -> str:
    """Render serializable service data without losing non-JSON values."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


class WorkerSignals(QObject):
    """Signals retained for the lifetime of one background request."""

    finished = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    """Execute a single service operation in the thread pool."""

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.fn())
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class GraphView(QGraphicsView):
    """Pan, zoom and inspect evidence-backed architecture nodes."""

    def __init__(self) -> None:
        super().__init__()
        self.graph_scene = QGraphicsScene(self)
        self.setScene(self.graph_scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMinimumHeight(270)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.set_data({})

    def wheelEvent(self, event: Any) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        event.accept()

    def set_data(self, data: dict[str, Any]) -> None:
        """Show discovered layers or configuration fields, never assumed layers."""
        self.graph_scene.clear()
        self.resetTransform()
        architecture = data.get("architecture", {})
        layers = data.get("layers") or (
            architecture.get("nodes") or architecture.get("layers")
            if isinstance(architecture, dict)
            else None
        )
        nodes: list[tuple[str, Any]] = []
        if isinstance(layers, list):
            nodes = [
                (
                    (
                        str(
                            v.get(
                                "label",
                                v.get("name", v.get("id", v.get("type", f"Layer {i}"))),
                            )
                        )
                        if isinstance(v, dict)
                        else str(v)
                    ),
                    v,
                )
                for i, v in enumerate(layers)
            ]
        if not nodes:
            tensors = data.get("tensors", [])
            if isinstance(tensors, list):
                nodes = [
                    (
                        (
                            str(v.get("name", f"Tensor {i}"))
                            if isinstance(v, dict)
                            else str(v)
                        ),
                        v,
                    )
                    for i, v in enumerate(tensors)
                ]
        if not nodes:
            config = data.get("config") or (
                architecture if isinstance(architecture, dict) else {}
            )
            if isinstance(config, dict):
                nodes = [(str(k), {k: v}) for k, v in config.items()]
        if not nodes:
            text = self.graph_scene.addText(
                "Inspect an artifact to visualize discovered architecture.\nDrag to pan · Scroll to zoom · Hover for evidence"
            )
            text.setDefaultTextColor(QColor("#8fa1b9"))
            return
        positions = {
            str(detail.get("id", label)) if isinstance(detail, dict) else label: (
                (i % 4) * 250,
                (i // 4) * 95,
            )
            for i, (label, detail) in enumerate(nodes[:256])
        }
        edges = architecture.get("edges", []) if isinstance(architecture, dict) else []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = positions.get(str(edge.get("source", edge.get("from"))))
            target = positions.get(str(edge.get("target", edge.get("to"))))
            if source and target:
                line = self.graph_scene.addLine(
                    source[0] + 112,
                    source[1] + 34,
                    target[0] + 112,
                    target[1] + 34,
                    QPen(QColor("#4d7e8c"), 2),
                )
                line.setToolTip(pretty(edge))
        for i, (label, detail) in enumerate(nodes[:256]):
            x, y = (i % 4) * 250, (i // 4) * 95
            # Configuration and tensor nodes have no asserted execution edges.
            rect = self.graph_scene.addRect(
                x, y, 225, 68, QPen(QColor("#427a89")), QBrush(QColor("#1b3445"))
            )
            rect.setToolTip(pretty(detail))
            text = self.graph_scene.addText(label[:30])
            text.setDefaultTextColor(QColor("#a3f0e7"))
            text.setPos(x + 12, y + 18)
            text.setToolTip(pretty(detail))
        self.graph_scene.setSceneRect(
            self.graph_scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        )


class MainWindow(QMainWindow):
    """The MODELForge research workbench, backed by the supplied service."""

    PAGE_NAMES = [
        "Dashboard",
        "Explorer",
        "Architecture",
        "Research",
        "Reconstruction",
        "Agents",
        "Training",
        "Export",
        "Settings",
    ]

    def __init__(self, service: Any = None) -> None:
        super().__init__()
        if service is None:
            from .service import Service

            service = Service()
        self.service = service
        self.pool = QThreadPool(self)
        self.workers: set[Worker] = set()
        self.outputs: dict[str, QPlainTextEdit] = {}
        self.inspections: dict[str, dict[str, Any]] = {}
        self.setWindowTitle("MODELForge — Model Research Workbench")
        self.resize(1280, 850)
        self.setMinimumSize(1000, 680)
        self.setStyleSheet(STYLE)
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(230)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 15, 20, 20)
        brand = QLabel("MODELForge")
        brand.setObjectName("brand")
        side.addWidget(brand)
        tagline = QLabel("MODEL RESEARCH\nWORKBENCH")
        tagline.setWordWrap(True)
        tagline.setStyleSheet(
            "color: #8fa1b9; background: transparent; font-size: 11px;"
        )
        side.addWidget(tagline)
        self.nav = QListWidget()
        self.nav.addItems(self.PAGE_NAMES)
        side.addWidget(self.nav)
        footer = QLabel("Local evidence.\nReproducible work.")
        footer.setWordWrap(True)
        footer.setStyleSheet(
            "color: #8fa1b9; background: transparent; font-size: 11px;"
        )
        side.addWidget(footer)
        layout.addWidget(sidebar)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(28, 22, 28, 18)
        project_row = QHBoxLayout()
        project_row.addWidget(QLabel("WORKSPACE"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(230)
        self.project_combo.currentIndexChanged.connect(self.project_changed)
        project_row.addWidget(self.project_combo)
        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("New project name")
        project_row.addWidget(self.project_name)
        project_row.addWidget(self.button("Create project", self.create_project))
        project_row.addWidget(self.button("Refresh", self.refresh))
        body_layout.addLayout(project_row)
        self.stack = QStackedWidget()
        body_layout.addWidget(self.stack, 1)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        body_layout.addWidget(self.status)
        layout.addWidget(body, 1)
        self.setCentralWidget(root)
        self.build_pages()
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        self.refresh()

    def button(
        self, text: str, callback: Callable[[], Any], primary: bool = False
    ) -> QPushButton:
        button = QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(callback)
        return button

    def page(self, name: str, description: str) -> QVBoxLayout:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 24, 0, 4)
        eyebrow = QLabel("MODELForge / " + name.upper())
        eyebrow.setObjectName("eyebrow")
        layout.addWidget(eyebrow)
        title = QLabel(name)
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel(description)
        subtitle.setWordWrap(True)
        subtitle.setObjectName("description")
        layout.addWidget(subtitle)
        self.stack.addWidget(widget)
        return layout

    def output(self, name: str, layout: QVBoxLayout) -> QPlainTextEdit:
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlaceholderText("Results will appear here after an operation.")
        layout.addWidget(editor, 1)
        self.outputs[name] = editor
        return editor

    def build_pages(self) -> None:
        dash = self.page(
            "Dashboard",
            "A live view of your projects, local hardware and persisted research tasks.",
        )
        metrics_row = QHBoxLayout()
        self.metrics: dict[str, QLabel] = {}
        for metric_name in ["Projects", "Task ledger", "Available RAM", "Compute"]:
            metric = QLabel(metric_name + "<br><br>Awaiting hardware data")
            metric.setObjectName("metric")
            metric.setTextFormat(Qt.TextFormat.RichText)
            metric.setWordWrap(True)
            metrics_row.addWidget(metric, 1)
            self.metrics[metric_name] = metric
        dash.addLayout(metrics_row)
        dash.addWidget(
            self.button("Refresh workspace and hardware", self.refresh, True)
        )
        self.output("Dashboard", dash)
        explorer = self.page(
            "Explorer",
            "Inspect local JSON, SafeTensors, ONNX or GGUF metadata. PyTorch checkpoints are refused.",
        )
        file_row = QHBoxLayout()
        self.artifact_path = QLineEdit()
        self.artifact_path.setPlaceholderText("Select a local model artifact or config")
        file_row.addWidget(self.artifact_path, 1)
        file_row.addWidget(self.button("Browse…", self.browse_artifact))
        file_row.addWidget(self.button("Inspect artifact", self.inspect, True))
        explorer.addLayout(file_row)
        self.output("Explorer", explorer)
        architecture = self.page(
            "Architecture",
            "Discovered tensors and configuration. Pan, zoom and hover over nodes to read their source evidence.",
        )
        self.graph = GraphView()
        architecture.addWidget(self.graph, 2)
        architecture.addWidget(self.button("Load project findings", self.load_findings))
        self.output("Architecture", architecture)
        research = self.page(
            "Research",
            "Run bounded, authorized behavioral probes against an OpenAI-compatible API. Responses are evidence, not recovered weights.",
        )
        form = QFormLayout()
        self.endpoint = QLineEdit("http://127.0.0.1:8000/v1")
        self.model_name = QLineEdit()
        self.model_name.setPlaceholderText("Model identifier exposed by the endpoint")
        form.addRow("Endpoint", self.endpoint)
        form.addRow("Model", self.model_name)
        self.budget = QSpinBox()
        self.budget.setRange(1, 8192)
        self.budget.setValue(1024)
        form.addRow("Total requested output tokens", self.budget)
        research.addLayout(form)
        self.prompts = QPlainTextEdit()
        self.prompts.setPlaceholderText("One probe prompt per line")
        self.prompts.setMaximumHeight(110)
        research.addWidget(self.prompts)
        self.authorized = QCheckBox("I own this endpoint or have permission to test it")
        research.addWidget(self.authorized)
        research.addWidget(self.button("Run authorized probes", self.probe, True))
        self.output("Research", research)
        reconstruction = self.page(
            "Reconstruction",
            "Build the independent ModelForge tiny causal fixture, then evaluate its actual response.",
        )
        reconstruction.addWidget(
            self.button("Build local reconstruction", self.build, True)
        )
        self.infer_text = QLineEdit()
        self.infer_text.setPlaceholderText("Enter a prompt for the local model")
        reconstruction.addWidget(self.infer_text)
        reconstruction.addWidget(self.button("Run local inference", self.infer))
        self.output("Reconstruction", reconstruction)
        agents = self.page(
            "Agents",
            "Persist bounded jobs in the local task queue. Queued jobs are tracked explicitly.",
        )
        self.task_kind = QComboBox()
        self.task_kind.addItems(["inspect", "build", "train", "infer", "agent"])
        agents.addWidget(self.task_kind)
        self.task_payload = QPlainTextEdit()
        self.task_payload.setPlainText("{}")
        self.task_payload.setMaximumHeight(100)
        agents.addWidget(QLabel("Job payload (JSON object)"))
        agents.addWidget(self.task_payload)
        hints = QLabel(
            'Payload examples: inspect {"path":"/model/config.json"} · train {"text":"your corpus","steps":30} · agent {"provider":"codex","prompt":"Review this project"}'
        )
        hints.setWordWrap(True)
        agents.addWidget(hints)
        self.agent_authorized = QCheckBox(
            "Authorize external agent execution using my installed provider account (provider billing applies)"
        )
        agents.addWidget(self.agent_authorized)
        agents.addWidget(self.button("Queue job", self.enqueue, True))
        agents.addWidget(self.button("Run pending jobs", self.run_pending))
        agents.addWidget(self.button("Refresh task ledger", self.load_tasks))
        self.output("Agents", agents)
        training = self.page(
            "Training",
            "Train the local baseline on your own text. Reported loss and steps come from the training run.",
        )
        self.training_text = QPlainTextEdit()
        self.training_text.setPlaceholderText(
            "Paste your training corpus here, or load a UTF-8 text file"
        )
        self.training_text.setMaximumHeight(170)
        training.addWidget(self.training_text)
        training.addWidget(self.button("Load training text…", self.browse_training))
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("Training steps"))
        self.steps = QSpinBox()
        self.steps.setRange(1, 1000)
        self.steps.setValue(30)
        train_row.addWidget(self.steps)
        train_row.addWidget(self.button("Train baseline", self.train, True))
        training.addLayout(train_row)
        self.output("Training", training)
        export_page = self.page(
            "Export",
            "Export recorded findings and available reconstruction artifacts to a portable ZIP archive.",
        )
        self.export_path = QLineEdit()
        self.export_path.setPlaceholderText("New ZIP archive path")
        export_page.addWidget(self.export_path)
        export_page.addWidget(self.button("Choose destination…", self.browse_export))
        export_page.addWidget(self.button("Export project", self.export, True))
        self.output("Export", export_page)
        settings = self.page(
            "Settings",
            "Inspect and save local service settings. Secrets should be provided through the environment.",
        )
        self.settings_editor = QPlainTextEdit()
        self.settings_editor.setPlaceholderText("Loading settings…")
        settings.addWidget(self.settings_editor, 1)
        settings.addWidget(self.button("Reload settings", self.load_settings))
        settings.addWidget(self.button("Save settings", self.save_settings, True))
        self.output("Settings", settings).setMaximumHeight(110)

    def project_id(self) -> str:
        value = self.project_combo.currentData()
        if value is None:
            raise ValueError("Create or select a project first.")
        return str(value)

    def run(
        self,
        page: str,
        label: str,
        fn: Callable[[], Any],
        callback: Callable[[Any], None] | None = None,
    ) -> None:
        """Dispatch work and retain the worker until its terminal signal."""
        self.status.setText(f"Running · {label}")
        worker = Worker(fn)
        self.workers.add(worker)

        def success(result: Any) -> None:
            self.outputs[page].setPlainText(pretty(result))
            self.status.setText(f"Completed · {label}")
            try:
                if callback:
                    callback(result)
            except Exception as exc:
                self.show_error(page, f"{type(exc).__name__}: {exc}")
            self.workers.discard(worker)

        def failure(message: str) -> None:
            self.show_error(page, message)
            self.workers.discard(worker)

        worker.signals.finished.connect(success)
        worker.signals.failed.connect(failure)
        self.pool.start(worker)

    def show_error(self, page: str, message: str) -> None:
        self.status.setText("Failed · " + message)
        self.outputs[page].setPlainText("Operation failed\n\n" + message)

    def project_action(
        self,
        page: str,
        label: str,
        operation: Callable[[str], Any],
        callback: Callable[[Any], None] | None = None,
    ) -> None:
        try:
            project_id = self.project_id()
            self.run(page, label, lambda: operation(project_id), callback)
        except Exception as exc:
            self.show_error(page, str(exc))

    def refresh(self) -> None:
        def fetch() -> dict[str, Any]:
            return {
                "projects": self.service.projects(),
                "hardware": self.service.hardware(),
                "tasks": self.service.tasks(),
            }

        def apply(data: dict[str, Any]) -> None:
            hardware = data["hardware"]
            values = {
                "Projects": str(len(data["projects"])),
                "Task ledger": str(len(data["tasks"])),
                "Available RAM": f'{hardware.get("memory_available", 0) / 1024**3:.1f} GiB',
                "Compute": str(hardware.get("inference_backend", "Unavailable")),
            }
            for name, value in values.items():
                self.metrics[name].setText(
                    f'<span style="color:#8fa1b9">{name.upper()}</span><br><br><span style="color:#7dece4;font-size:23px;font-weight:700">{value}</span>'
                )
            previous = self.project_combo.currentData()
            self.project_combo.blockSignals(True)
            self.project_combo.clear()
            for project in data["projects"]:
                self.project_combo.addItem(
                    str(project.get("name", project["id"])), project["id"]
                )
            index = self.project_combo.findData(previous)
            if index >= 0:
                self.project_combo.setCurrentIndex(index)
            self.project_combo.blockSignals(False)
            self.project_changed()

        self.run("Dashboard", "Refresh workspace", fetch, apply)
        self.load_settings()

    def create_project(self) -> None:
        name = self.project_name.text().strip()
        if not name:
            self.show_error("Dashboard", "Enter a project name.")
            return

        def apply(result: Any) -> None:
            self.project_name.clear()
            self.project_combo.addItem(str(result.get("name", name)), result["id"])
            self.project_combo.setCurrentIndex(self.project_combo.count() - 1)

        self.run(
            "Dashboard",
            "Create project",
            lambda: self.service.create_project(name),
            apply,
        )

    def project_changed(self) -> None:
        key = self.project_combo.currentData()
        self.graph.set_data(self.inspections.get(str(key), {}))
        self.outputs["Explorer"].clear()
        self.outputs["Architecture"].clear()
        if key is not None:
            self.load_findings()

    def browse_artifact(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Inspect model artifact")
        if path:
            self.artifact_path.setText(path)

    def inspect(self) -> None:
        path = self.artifact_path.text().strip()
        try:
            project_id = self.project_id()
        except ValueError as exc:
            self.show_error("Explorer", str(exc))
            return

        def apply(result: Any) -> None:
            self.inspections[project_id] = result
            if self.project_combo.currentData() == project_id:
                self.graph.set_data(result)

        self.run(
            "Explorer",
            "Inspect artifact",
            lambda: self.service.inspect(project_id, path),
            apply,
        )

    def load_findings(self) -> None:
        try:
            project_id = self.project_id()
        except ValueError as exc:
            self.show_error("Architecture", str(exc))
            return

        def apply(findings: Any) -> None:
            if self.project_combo.currentData() != project_id:
                return
            for finding in findings:
                if finding.get("kind") == "inspection":
                    self.inspections[project_id] = finding["data"]
                    self.graph.set_data(finding["data"])
                    break

        self.run(
            "Architecture",
            "Load findings",
            lambda: self.service.findings(project_id),
            apply,
        )

    def build(self) -> None:
        self.project_action(
            "Reconstruction", "Build reconstruction", self.service.build
        )

    def infer(self) -> None:
        text = self.infer_text.text()
        self.project_action(
            "Reconstruction",
            "Local inference",
            lambda key: self.service.infer(key, text),
        )

    def probe(self) -> None:
        if not self.authorized.isChecked():
            self.show_error(
                "Research",
                "Confirm endpoint ownership or testing permission before probing.",
            )
            return
        endpoint, model = self.endpoint.text().strip(), self.model_name.text().strip()
        prompts = [
            line.strip()
            for line in self.prompts.toPlainText().splitlines()
            if line.strip()
        ]
        budget = self.budget.value()
        self.project_action(
            "Research",
            "Behavioral probe",
            lambda key: self.service.probe(key, endpoint, model, prompts, True, budget),
        )

    def enqueue(self) -> None:
        try:
            payload = json.loads(self.task_payload.toPlainText())
            if not isinstance(payload, dict):
                raise ValueError("Job payload must be a JSON object.")
        except (ValueError, TypeError) as exc:
            self.show_error("Agents", str(exc))
            return
        kind = self.task_kind.currentText()
        if kind == "agent":
            if not self.agent_authorized.isChecked():
                self.show_error(
                    "Agents",
                    "Authorize external agent execution before queuing an agent job.",
                )
                return
            payload["authorized"] = True
        self.project_action(
            "Agents", "Queue job", lambda key: self.service.enqueue(key, kind, payload)
        )

    def run_pending(self) -> None:
        self.run("Agents", "Run pending jobs", self.service.run_pending)

    def load_tasks(self) -> None:
        self.run("Agents", "Refresh task ledger", self.service.tasks)

    def browse_training(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load UTF-8 training corpus",
            filter="Text files (*.txt);;All files (*)",
        )
        if path:
            from pathlib import Path

            self.run(
                "Training",
                "Load corpus",
                lambda: Path(path).read_text(encoding="utf-8"),
                self.training_text.setPlainText,
            )

    def train(self) -> None:
        text, steps = self.training_text.toPlainText(), self.steps.value()
        self.project_action(
            "Training",
            "Train baseline",
            lambda key: self.service.train(key, text, steps),
        )

    def browse_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export project archive", "project.zip", "ZIP archives (*.zip)"
        )
        if path:
            self.export_path.setText(path)

    def export(self) -> None:
        destination = self.export_path.text().strip()
        self.project_action(
            "Export",
            "Export project",
            lambda key: self.service.export(key, destination),
        )

    def load_settings(self) -> None:
        self.run(
            "Settings",
            "Load settings",
            self.service.settings,
            lambda value: self.settings_editor.setPlainText(pretty(value)),
        )

    def save_settings(self) -> None:
        try:
            value = json.loads(self.settings_editor.toPlainText())
            if not isinstance(value, dict):
                raise ValueError("Settings must be a JSON object.")
        except (ValueError, TypeError) as exc:
            self.show_error("Settings", str(exc))
            return
        self.run("Settings", "Save settings", lambda: self.service.save_settings(value))

    def closeEvent(self, event: Any) -> None:
        """Do not destroy a running pool or its signal recipients."""
        if self.workers:
            self.status.setText("Operations are still running. Close after completion.")
            event.ignore()
        else:
            event.accept()


def launch() -> int:
    """Launch the native workbench."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec())
