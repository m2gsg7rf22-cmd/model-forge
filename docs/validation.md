# Validation record

This file is finalized with measured results before release. A passing unit test does not prove universal model equivalence.

Initial macOS Apple Silicon run: Python 3.12.15; official MCP SDK 1.30.0; PySide6 6.12.0; NumPy 2.5.4. 105 pytest tests and 28 subtests passed with 95% aggregate statement coverage before final integration additions. Final results are recorded below when available.

Real checks include generated SafeTensors and GGUF binary files, generated ONNX/Core ML protobufs, NumPy inference and training, independent scalar attention comparison, export/reload and reconstructed-fixture parity, SQLite persistence/concurrent claims, real SDK stdio subprocess transport and Qt offscreen service workflows.

Local HTTP tests are **fixture endpoints**, not third-party paid model calls. External agent adapter tests execute **provider-named temporary fixtures**, not authenticated Claude/Codex inference. Low-memory and disk-full conditions include explicitly injected tests; they are not claims that the host ran out of physical memory. Qt tests use offscreen rendering; screenshots are real Qt renders, not a human usability study.

Claude Code 2.1.304 was installed in an isolated workspace directory. Strict marketplace validation, plugin installation into isolated CLAUDE_CONFIG_DIR, and `claude mcp list` returning Connected passed. Codex CLI 0.155.1 registered and listed the actual stdio executable using isolated CODEX_HOME. This does not yet establish an authenticated agent research turn or the separate desktop app's plugin discovery UX.

Windows-specific provider executable fixtures and privileged symlink tests are skipped where POSIX assumptions do not apply. Optional Core ML tests skip when coremltools is absent. CI smoke tests validate boot and basic service startup, not signed-installation UX on every OS version.
