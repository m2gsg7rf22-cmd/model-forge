# Developer guide

Python 3.12+ source lives in `src/modelforge`. The core depends on NumPy, psutil
and the official MCP Python SDK. PySide6 and ONNX are optional extras; Core ML
metadata additionally requires a separately installed `coremltools` package.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[desktop,dev,onnx]'
.venv/bin/python -m pytest -q
.venv/bin/python -m mypy src/modelforge
.venv/bin/python -m ruff check src tests packaging
.venv/bin/python -m black --check src tests packaging
```

Use the corresponding `.venv\Scripts\python.exe` on Windows. Tests create their
own temporary artifacts. Headless GUI tests set `QT_QPA_PLATFORM=offscreen`; this
checks real Qt widgets and service workflows, not a substitute for every native
window-system or installer check. Real optional-format tests skip when their
parser is absent. Never report a skipped integration as validated. Consult
[validation](validation.md) for executed commands and platform limitations.

## Components

| Module | Responsibility |
| --- | --- |
| `adapters.py` | Uniform modality interfaces, declared-config routing, explicit trusted entry-point registry |
| `inspector.py` | Bounded metadata parsers and evidence-linked structure reports |
| `builder.py` | Project-owned causal fixture, bounded loading, inference, output-head training, quantization, reconstruction/parity |
| `research.py` | Explicitly authorized OpenAI-compatible text probes with bounded responses |
| `service.py` | Shared SQLite projects/findings/settings/jobs, bounded scheduler and external-agent subprocesses |
| `gui.py` | Native nine-page UI with background dispatch and interactive graph |
| `cli.py` | JSON CLI output and integrated example |
| `mcp_server.py` | Stdio FastMCP tools calling the same service |

The GUI, CLI and MCP do not maintain competing business logic. New capabilities
should enter a tested engine, then the service, then a thin surface adapter.
Always close `Service` after use. SQLite WAL connections are scoped per operation.
Tasks persist as `queued`, `running`, `completed`, or `failed`. On reopening,
process-owned interrupted jobs are requeued if their process no longer exists;
this is not mid-computation checkpoint recovery. `run_pending` processes one
batch of up to two jobs; call it again for remaining jobs.

New parsers must preserve shape/count/read limits, refuse model code execution,
and distinguish observed descriptors, artifact declarations, and inference.
Unknown formats should return explicit unsupported reports. Avoid assuming a
complete model from loose shard filenames. Add malformed-file and allocation
boundary tests as well as real generated artifacts. Keep numerical comparison
scope finite and distinguish storage-preserving reconstruction from lossy export.

`adapters.py` supplies separate language, multimodal, image, video, and audio
classes with uniform inspection/inference/capability methods. Metadata-only
adapters explicitly refuse inference; the language runtime accepts only the tiny
project format. Plugin distributions may declare `modelforge.adapters` entry
points whose factories return a `ModelAdapter` subclass. `list_plugins()` reads
only declarations; `load_plugin(name, trusted=True)` imports code on explicit
trust and rejects collisions with built-ins. Do not call it automatically from
artifact inspection. This trusted extension mechanism is not an execution
sandbox. General LLM runtimes, modality-specific generation, LoRA/QLoRA, and
arbitrary training remain future work. Review [ecosystem](ecosystem.md) before introducing a heavy
dependency. Preserve separate framework/model/dataset license records. Do not
vendor third-party code without its required notices.

Packaging scripts are under `packaging/`; CI definitions are in
`.github/workflows/`. Build/release publication must use only artifacts whose
acceptance checks passed. A supplied cross-platform workflow does not establish
that its jobs or resulting platform binaries have run successfully.
