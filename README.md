# ModelForge

An open-source desktop workbench for evidence-based AI model research. Inspect local artifacts without executing model code, record authorized API observations, and build a small independent causal model whose weights and computations can be verified.

**Version 0.1.0 is an early research release.** It implements an integrated CPU workflow. It does not reconstruct arbitrary LLMs or recover closed-model weights. Read the [capability matrix](docs/capabilities.md) before choosing a workflow.

![Desktop workbench](docs/desktop.png)

## Install and run

Requires Python 3.12+ and a desktop display for the GUI. The CLI and MCP work headlessly.

```sh
git clone https://github.com/m2gsg7rf22-cmd/model-forge.git
cd model-forge
python3.12 packaging/install.py --prefix "$PWD/.venv"
.venv/bin/modelforge gui
```

Windows: use `py -3.12 packaging/install.py --prefix .venv`, then `.venv\Scripts\modelforge.exe gui`. Terminal installation downloads dependencies from PyPI. Use `--core-only` to omit the GUI. See [installation and removal](docs/installation.md) and [validation status](docs/validation.md) for the tested platforms and artifacts.

## End-to-end local example

```sh
.venv/bin/modelforge --home ./example-data example --destination ./example-export.zip
```

Creates a persistent project, builds project-owned weights and a UTF-8 byte tokenizer, inspects tensor metadata, generates text, trains the output head, compares reloaded logits and exports code, weights, model card, provenance and findings. The fixture is for correctness research; it has no claim to useful general language capabilities. It uses no downloaded training data or proprietary weights.

## What works

- Nine native PySide6 desktop pages, with background operations and visible errors.
- Bounded metadata readers for SafeTensors and GGUF; Hugging Face config/shard inspection; optional ONNX and Core ML metadata.
- Evidence-linked architecture graph with explicit inferred/config-declared scope.
- Authorized OpenAI-compatible text API probes: latency, repeated-prompt consistency, JSON validity and retained observations.
- Independent single-head causal attention model, NumPy inference, output-head training, float32 reload parity, lossy INT8 storage and standalone runtime export.
- SQLite projects, findings, settings and jobs; deduplication, two concurrent workers and interrupted-process recovery.
- Claude Code and Codex subprocess adapters with explicit authorization, time/memory/log bounds; provider account billing remains external.
- Official MCP Python SDK stdio server with real service operations, Claude plugin and Codex skill.

MLX/CUDA inference, arbitrary imported-model execution, LoRA/QLoRA, diffusion/video generation, mechanistic interventions and closed-model weight extraction are **not implemented**. Optional formats report unsupported when their parser is unavailable. PyTorch pickle checkpoints are rejected, including trusted ones in this release.

## Integrations

Activate the environment so `modelforge-mcp` is available. Prepare a portable integration with an absolute executable path:

```sh
. .venv/bin/activate
python packaging/connect.py claude --destination ./claude-integration
python packaging/connect.py codex --destination ./codex-integration
```

These prepare files without changing existing host settings. Explicit host registration uses `--apply` with a new destination. See [Claude Code](docs/claude-code.md), [Codex](docs/codex.md) and [MCP reference](docs/mcp.md).

## Development

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[desktop,dev,onnx]'
.venv/bin/python -m pytest -q
.venv/bin/python -m mypy src/modelforge
.venv/bin/python -m ruff check src tests packaging
```

[User guide](docs/user-guide.md) · [Developer guide](docs/development.md) · [Reconstruction](docs/reconstruction.md) · [Research basis](docs/research.md) · [Third-party evaluation](docs/ecosystem.md) · [Release notes](docs/release-notes.md)

## Evidence and rights

Config claims, tensor shapes and behavioral observations are distinct forms of evidence. A finite prompt set cannot certify structural or weight equivalence. Even complete weights require a compatible graph, tokenizer and numerical verification. This release verifies its own tiny-model format only.

Do not use endpoints or files without permission. Provider outputs are never automatically used for training. Exports include generated assets and reports, never imported model files. Reports may contain local paths and authorized API outputs; review before sharing. API keys are read only from `MODELFORGE_API_KEY` and are not saved in settings.

MIT licensed original implementation. Dependency licenses remain their own; no code or weights from the researched projects are vendored. Bundled dependency notices are supplied with binary builds.
