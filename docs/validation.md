# Validation record

This file is finalized with measured results before release. A passing unit test does not prove universal model equivalence.

Initial macOS Apple Silicon run: Python 3.12.15; official MCP SDK 1.30.0; PySide6 6.12.0; NumPy 2.5.4. 105 pytest tests and 28 subtests passed with 95% aggregate statement coverage before final integration additions. Final results are recorded below when available.

Real checks include generated SafeTensors and GGUF binary files, generated ONNX/Core ML protobufs, NumPy inference and training, independent scalar attention comparison, export/reload and reconstructed-fixture parity, SQLite persistence/concurrent claims, real SDK stdio subprocess transport and Qt offscreen service workflows.

Local HTTP tests are **fixture endpoints**, not third-party paid model calls. External agent adapter tests execute **provider-named temporary fixtures**, not authenticated Claude/Codex inference. Low-memory and disk-full conditions include explicitly injected tests; they are not claims that the host ran out of physical memory. Qt tests use offscreen rendering; screenshots are real Qt renders, not a human usability study.

Claude Code 2.1.304 was installed in an isolated workspace directory. Strict marketplace validation, plugin installation into isolated CLAUDE_CONFIG_DIR, and `claude mcp list` returning Connected passed. Codex CLI 0.155.1 registered and listed the actual stdio executable using isolated CODEX_HOME. This does not yet establish an authenticated agent research turn or the separate desktop app's plugin discovery UX.

Windows-specific provider executable fixtures and privileged symlink tests are skipped where POSIX assumptions do not apply. Optional Core ML tests skip when coremltools is absent. CI smoke tests validate boot and basic service startup, not signed-installation UX on every OS version.

## Additional verified integration results

- Final local suite after reconstruction/quantization and evaluation integration: **155 tests passed, 28 subtests passed, 96% aggregate statement coverage**. Strict mypy and Ruff are checked separately.
- Fresh dedicated source installation with desktop dependencies succeeded. Installed CLI executed the complete CPU example; installed GUI entry passed its timed offscreen launch. Dedicated uninstall removed the environment and retained research data.
- Frozen macOS app built, ad-hoc signed and passed a timed offscreen launch.
- Authenticated Codex CLI turn successfully called the real `hardware_status` MCP tool and reported observed NumPy CPU / arm64. First attempt was blocked by the noninteractive approval policy; a retry explicitly approved only that read-only tool. No blanket tool approval or persistent host-settings edit was used.
- GitHub Actions run [39433001579](https://github.com/m2gsg7rf22-cmd/model-forge/actions/runs/39433001579) passed Linux x64 and macOS ARM test, type/lint, fresh installation/uninstallation, package and frozen-app smoke stages. Its Windows test failed and is being corrected; it is not a passing Windows release record.

No authenticated Claude model turn or local Claude desktop application UI test has
been completed. Claude CLI plugin install and real MCP connection were verified.
No third-party production-model extraction, high-memory training, actual GPU
backend, notarization or signed Windows installation was tested.
