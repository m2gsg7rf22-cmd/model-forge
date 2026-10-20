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
    QDoubleSpinBox,
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
    QScrollArea,
    QTabWidget,
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        self.stack.addWidget(scroll)
        return layout

    def output(self, name: str, layout: QVBoxLayout) -> QPlainTextEdit:
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setMinimumHeight(150)
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
        explorer.addWidget(
            self.button("Choose model folder…", self.browse_artifact_folder)
        )
        self.output("Explorer", explorer)
        architecture = self.page(
            "Architecture",
            "Discovered tensors and configuration. Pan, zoom and hover over nodes to read their source evidence.",
        )
        self.graph = GraphView()
        architecture.addWidget(self.graph, 2)
        architecture.addWidget(self.button("Load project findings", self.load_findings))
        self.trace_prompt = QLineEdit("hello world")
        self.trace_prompt.setPlaceholderText("Activation probe: at most 64 tokens")
        architecture.addWidget(self.trace_prompt)
        self.trace_ablation = QLineEdit()
        self.trace_ablation.setPlaceholderText(
            "Optional exact MLP name from trace ablation_modules"
        )
        architecture.addWidget(self.trace_ablation)
        architecture.addWidget(
            self.button("Trace local activations / ablate named MLP", self.trace_model)
        )
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
            "Reconstruct a supported ModelForge bundle in an empty project, or build an independent tiny fixture. Other imported architectures remain inspection-only.",
        )
        source_row = QHBoxLayout()
        self.reconstruction_source = QLineEdit()
        self.reconstruction_source.setPlaceholderText(
            "Supported ModelForge source bundle directory"
        )
        source_row.addWidget(self.reconstruction_source, 1)
        source_row.addWidget(
            self.button("Choose source folder…", self.browse_reconstruction_source)
        )
        reconstruction.addLayout(source_row)
        reconstruction.addWidget(
            self.button("Reconstruct supported bundle", self.reconstruct, True)
        )
        reconstruction.addWidget(
            self.button("Build independent tiny fixture", self.build)
        )
        reconstruction.addWidget(self.button("Quantize current fixture", self.quantize))
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
        self.output("Settings", settings).setMaximumHeight(160)
        self.build_engine_panels(reconstruction, training, research, settings)

    def engine_tab(self, layout: QVBoxLayout, name: str) -> QVBoxLayout:
        """Keep legacy controls accessible and share one visible result area."""
        result = layout.takeAt(layout.count() - 1)
        legacy = QWidget()
        legacy_layout = QVBoxLayout(legacy)
        while layout.count() > 3:
            item = layout.takeAt(3)
            if item is not None:
                legacy_layout.addItem(item)
        tabs = QTabWidget()
        tabs.addTab(legacy, "Baseline fixture")
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        tabs.addTab(panel, name)
        layout.addWidget(tabs)
        if result is not None:
            layout.addItem(result)
        return panel_layout

    def build_engine_panels(
        self,
        reconstruction: QVBoxLayout,
        training: QVBoxLayout,
        research: QVBoxLayout,
        settings: QVBoxLayout,
    ) -> None:
        local = self.engine_tab(reconstruction, "Local GPT-2 / Llama / MLX")
        self.local_source = QLineEdit()
        self.local_source.setPlaceholderText(
            "Local SafeTensors model folder; no remote code"
        )
        local.addWidget(self.local_source)
        local.addWidget(
            self.button("Choose local model folder…", self.browse_local_model)
        )
        self.local_authorized = QCheckBox("I am authorized to load and run this model")
        local.addWidget(self.local_authorized)
        options = QFormLayout()
        self.local_device = QComboBox()
        self.local_device.addItems(["cpu", "auto", "mps", "cuda"])
        options.addRow("Torch device", self.local_device)
        self.local_memory = QSpinBox()
        self.local_memory.setRange(256, 65536)
        self.local_memory.setValue(2048)
        options.addRow("Worker memory limit (MiB)", self.local_memory)
        self.local_backend = QComboBox()
        self.local_backend.addItems(["hf", "mlx"])
        options.addRow("Generation engine", self.local_backend)
        self.local_bits = QComboBox()
        self.local_bits.addItems(["0", "4", "8"])
        options.addRow("MLX quantization bits (0 = original)", self.local_bits)
        self.local_tokens = QSpinBox()
        self.local_tokens.setRange(1, 512)
        self.local_tokens.setValue(32)
        options.addRow("Maximum new tokens", self.local_tokens)
        local.addLayout(options)
        local.addWidget(
            self.button("Validate and attach local model", self.attach_local)
        )
        self.local_fixture_type = QComboBox()
        self.local_fixture_type.addItems(["llama", "gpt2"])
        local.addWidget(self.local_fixture_type)
        local.addWidget(
            self.button(
                "Create owned offline language-model fixture", self.create_hf_fixture
            )
        )
        self.local_prompt = QLineEdit("hello world")
        local.addWidget(self.local_prompt)
        local.addWidget(
            self.button("Generate with selected engine", self.generate_local, True)
        )
        local.addWidget(self.button("Run MLX GPU benchmark", self.benchmark_mlx))
        self.local_rights = QLineEdit()
        self.local_rights.setPlaceholderText(
            "License / ownership statement for redistribution"
        )
        local.addWidget(self.local_rights)
        self.local_redistributable = QCheckBox(
            "I have permission to redistribute this model"
        )
        local.addWidget(self.local_redistributable)
        local.addWidget(
            self.button("Reconstruct and check prompt parity", self.reconstruct_hf)
        )

        train = self.engine_tab(training, "Full / LoRA training")
        train.addWidget(
            QLabel(
                "Uses the attached local model. Supply text you have permission to train on."
            )
        )
        self.local_training_text = QPlainTextEdit()
        self.local_training_text.setPlaceholderText("Training corpus (UTF-8 text)")
        self.local_training_text.setMaximumHeight(130)
        train.addWidget(self.local_training_text)
        form = QFormLayout()
        self.local_method = QComboBox()
        self.local_method.addItems(["lora", "full", "qlora"])
        form.addRow("Training method", self.local_method)
        self.local_steps = QSpinBox()
        self.local_steps.setRange(1, 1000)
        self.local_steps.setValue(10)
        form.addRow("Total target steps (including resumed steps)", self.local_steps)
        self.local_rate = QDoubleSpinBox()
        self.local_rate.setDecimals(6)
        self.local_rate.setRange(0.000001, 0.999999)
        self.local_rate.setValue(0.01)
        form.addRow("Learning rate", self.local_rate)
        self.local_sequence = QSpinBox()
        self.local_sequence.setRange(2, 4096)
        self.local_sequence.setValue(64)
        form.addRow("Maximum sequence tokens", self.local_sequence)
        train.addLayout(form)
        self.training_rights = QLineEdit()
        self.training_rights.setPlaceholderText("Dataset and model rights statement")
        train.addWidget(self.training_rights)
        self.training_authorized = QCheckBox(
            "I am authorized to train this model using this corpus"
        )
        self.training_resume = QCheckBox("Resume the saved training checkpoint")
        train.addWidget(self.training_authorized)
        train.addWidget(self.training_resume)
        train.addWidget(self.button("Train attached model", self.train_hf, True))
        train.addWidget(self.button("Train MLX LoRA on Apple GPU", self.train_mlx))

        lab = self.engine_tab(research, "Durable experiments / UMRRE")
        lab.addWidget(
            QLabel(
                "API experiments use the endpoint, model, permission and token cap from the first tab."
            )
        )
        self.experiment_cases = QPlainTextEdit()
        self.experiment_cases.setPlainText(
            '[{"kind":"capability","prompt":"Return the word hello","expected":"hello"}]'
        )
        self.experiment_cases.setMaximumHeight(130)
        lab.addWidget(QLabel("Cases (JSON array)"))
        lab.addWidget(self.experiment_cases)
        self.experiment_requests = QSpinBox()
        self.experiment_requests.setRange(1, 100)
        self.experiment_requests.setValue(10)
        lab.addWidget(QLabel("Maximum API requests"))
        lab.addWidget(self.experiment_requests)
        self.experiment_resume = QLineEdit()
        self.experiment_resume.setPlaceholderText("Optional experiment ID to resume")
        lab.addWidget(self.experiment_resume)
        lab.addWidget(
            self.button("Run authorized API experiment", self.experiment, True)
        )
        self.umrre_seed = QSpinBox()
        self.umrre_seed.setRange(0, 2147483647)
        self.umrre_seed.setValue(7)
        lab.addWidget(QLabel("Local UMRRE random seed"))
        lab.addWidget(self.umrre_seed)
        lab.addWidget(self.button("Run local UMRRE research", self.umrre))

        runtime = self.engine_tab(settings, "Optional local engines")
        note = QLabel(
            "Optional engines run in a separate Python environment. Source install: pip install '.[llm]' or '.[mlx]'. Missing dependencies and unsupported devices appear in the result below."
        )
        note.setWordWrap(True)
        runtime.addWidget(note)
        runtime.addWidget(
            self.button("Check local runtime status", self.runtime_status)
        )
        self.runtime_python = QLineEdit()
        self.runtime_python.setPlaceholderText(
            "Absolute path to a Python interpreter with ModelForge and engine extras"
        )
        runtime.addWidget(self.runtime_python)
        runtime.addWidget(
            self.button("Save runtime interpreter", self.save_runtime_python)
        )
        self.runtime_extra = QComboBox()
        self.runtime_extra.addItems(["llm", "mlx"])
        runtime.addWidget(self.runtime_extra)
        self.runtime_install_authorized = QCheckBox(
            "Authorize downloading and installing the selected optional engine"
        )
        runtime.addWidget(self.runtime_install_authorized)
        runtime.addWidget(self.button("Install optional engine", self.install_runtime))

    def trace_model(self) -> None:
        text, module = self.trace_prompt.text(), self.trace_ablation.text().strip()
        memory, device = self.local_memory.value(), self.local_device.currentText()
        self.project_action(
            "Architecture",
            "Trace local activations",
            lambda key: self.service.trace_local_model(
                key, text, ablate_module=module, memory_limit_mb=memory, device=device
            ),
        )

    def browse_local_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose local SafeTensors model")
        if path:
            self.local_source.setText(path)

    def attach_local(self) -> None:
        if not self.local_authorized.isChecked():
            self.show_error(
                "Reconstruction", "Confirm permission to load this local model."
            )
            return
        source, memory, device = (
            self.local_source.text().strip(),
            self.local_memory.value(),
            self.local_device.currentText(),
        )
        self.project_action(
            "Reconstruction",
            "Attach local model",
            lambda key: self.service.attach_local_model(
                key, source, authorized=True, memory_limit_mb=memory, device=device
            ),
        )

    def create_hf_fixture(self) -> None:
        model_type = self.local_fixture_type.currentText()
        self.project_action(
            "Reconstruction",
            "Create local language-model fixture",
            lambda key: self.service.create_local_fixture(key, model_type=model_type),
        )

    def generate_local(self) -> None:
        text, tokens = self.local_prompt.text(), self.local_tokens.value()
        memory, device = self.local_memory.value(), self.local_device.currentText()
        backend, bits = self.local_backend.currentText(), int(
            self.local_bits.currentText()
        )
        self.project_action(
            "Reconstruction",
            "Generate local tokens",
            lambda key: self.service.local_generate(
                key,
                text,
                max_new_tokens=tokens,
                memory_limit_mb=memory,
                device=device,
                backend=backend,
                quantize_bits=bits,
            ),
        )

    def benchmark_mlx(self) -> None:
        self.project_action(
            "Reconstruction", "MLX GPU benchmark", self.service.mlx_benchmark
        )

    def reconstruct_hf(self) -> None:
        if (
            not self.local_redistributable.isChecked()
            or not self.local_rights.text().strip()
        ):
            self.show_error(
                "Reconstruction",
                "Redistribution permission and a rights statement are required.",
            )
            return
        prompts, rights = [self.local_prompt.text()], self.local_rights.text().strip()
        self.project_action(
            "Reconstruction",
            "Reconstruct local language model",
            lambda key: self.service.reconstruct_local(
                key, prompts, redistributable=True, rights=rights
            ),
        )

    def train_hf(self) -> None:
        if (
            not self.training_authorized.isChecked()
            or not self.training_rights.text().strip()
        ):
            self.show_error(
                "Training", "Training permission and a rights statement are required."
            )
            return
        values = dict(
            text=self.local_training_text.toPlainText(),
            method=self.local_method.currentText(),
            steps=self.local_steps.value(),
            learning_rate=self.local_rate.value(),
            max_sequence_length=self.local_sequence.value(),
            memory_limit_mb=self.local_memory.value(),
            device=self.local_device.currentText(),
            authorized=True,
            rights=self.training_rights.text().strip(),
            resume=self.training_resume.isChecked(),
        )
        self.project_action(
            "Training",
            "Train local language model",
            lambda key: self.service.train_local_model(key, **values),
        )

    def train_mlx(self) -> None:
        if (
            not self.training_authorized.isChecked()
            or not self.training_rights.text().strip()
        ):
            self.show_error(
                "Training", "Training permission and a rights statement are required."
            )
            return
        if self.training_resume.isChecked():
            self.show_error(
                "Training",
                "MLX checkpoint resume is not supported; clear Resume to start a new run.",
            )
            return
        values = dict(
            text=self.local_training_text.toPlainText(),
            steps=self.local_steps.value(),
            learning_rate=self.local_rate.value(),
            memory_limit_mb=self.local_memory.value(),
            authorized=True,
            rights=self.training_rights.text().strip(),
            quantize_bits=int(self.local_bits.currentText()),
        )
        self.project_action(
            "Training",
            "Train MLX LoRA",
            lambda key: self.service.train_mlx_model(key, **values),
        )

    def experiment(self) -> None:
        if not self.authorized.isChecked():
            self.show_error(
                "Research",
                "Confirm endpoint ownership or testing permission before experiments.",
            )
            return
        try:
            cases = json.loads(self.experiment_cases.toPlainText())
            if (
                not isinstance(cases, list)
                or not cases
                or not all(isinstance(case, dict) for case in cases)
            ):
                raise ValueError("Cases must be a nonempty JSON array of objects.")
        except (ValueError, TypeError) as exc:
            self.show_error("Research", str(exc))
            return
        endpoint, model = self.endpoint.text().strip(), self.model_name.text().strip()
        requests, budget = self.experiment_requests.value(), self.budget.value()
        resume = self.experiment_resume.text().strip() or None
        self.project_action(
            "Research",
            "Durable API experiment",
            lambda key: self.service.run_experiment(
                key,
                endpoint,
                model,
                cases,
                authorized=True,
                max_requests=requests,
                output_token_budget=budget,
                resume_id=resume,
            ),
        )

    def umrre(self) -> None:
        seed = self.umrre_seed.value()
        self.project_action(
            "Research",
            "Local UMRRE research",
            lambda key: self.service.run_umrre(key, seed=seed),
        )

    def runtime_status(self) -> None:
        self.run("Settings", "Local runtime status", self.service.local_runtime_status)

    def save_runtime_python(self) -> None:
        path = self.runtime_python.text().strip()
        from pathlib import Path

        if not Path(path).is_absolute():
            self.show_error("Settings", "Runtime interpreter must be an absolute path.")
            return
        self.run(
            "Settings",
            "Save runtime interpreter",
            lambda: self.service.save_settings({"runtime_python": path}),
        )

    def install_runtime(self) -> None:
        if not self.runtime_install_authorized.isChecked():
            self.show_error(
                "Settings",
                "Authorize the optional engine download and installation first.",
            )
            return
        backend = self.runtime_extra.currentText()
        self.run(
            "Settings",
            "Install optional engine",
            lambda: self.service.install_local_runtime(
                backend=backend, authorized=True
            ),
        )

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

    def browse_artifact_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose model folder")
        if path:
            self.artifact_path.setText(path)

    def browse_reconstruction_source(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose supported ModelForge source bundle"
        )
        if path:
            self.reconstruction_source.setText(path)

    def reconstruct(self) -> None:
        source = self.reconstruction_source.text().strip()
        self.project_action(
            "Reconstruction",
            "Reconstruct supported bundle",
            lambda key: self.service.reconstruct(key, source),
        )

    def quantize(self) -> None:
        self.project_action(
            "Reconstruction", "Quantize current fixture", self.service.quantize
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
            "Reconstruction", "Build independent tiny fixture", self.service.build
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
