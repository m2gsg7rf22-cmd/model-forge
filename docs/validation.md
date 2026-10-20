# Validation record — 2026-10-20

## Executed checks

| Environment | Pytest result | Installation, package and frozen launch |
| --- | --- | --- |
| Local macOS ARM / Python 3.12.15 | **155 passed + 28 subtests**, no skips; **96% statement coverage** | Dedicated desktop install, full example, uninstall, frozen app launch passed |
| GitHub macOS ARM runner | **154 passed, 1 skipped + 28 subtests** | Fresh core install, full CPU example, uninstall, native package and frozen launch passed |
| GitHub Linux x64 runner | **154 passed, 1 skipped + 28 subtests** | Fresh core install, full CPU example, uninstall, native package and frozen launch passed |
| GitHub Windows x64 runner | **145 passed, 10 skipped + 28 subtests** | Fresh core install, full CPU example, uninstall, native package and frozen launch passed |

[Successful cross-platform run 39433685862](https://github.com/m2gsg7rf22-cmd/model-forge/actions/runs/39433685862) verifies executable source commit `cfedbfa784a70c0d20a653b80f63bf99f7ca8baa`. Windows skips comprise POSIX executable/symlink fixtures and optional Core ML. The other CI platforms skip optional Core ML; local acceptance installed coremltools and ran that test. Strict mypy and Ruff passed. Version details: [tested-dependencies.json](tested-dependencies.json).

The initial Windows failure was a test corpus written with the default Windows encoding. The test now explicitly writes UTF-8, matching the product's documented UTF-8 corpus loader. Corrected CI passed without changing the loader to accept incorrect encodings.

## Real execution versus fixtures

Real generated-file tests cover SafeTensors/GGUF, ONNX and Core ML protobufs, malformed inputs and shard consistency. Real NumPy tests cover causal masking, an independent scalar attention reference, loss reduction with a frozen backbone, exact float32 export/reconstruction parity and lossy quantized storage. SQLite tests exercise concurrent deduplication, atomic job claims and recovery. Official MCP SDK tests use an actual subprocess stdio connection. Qt tests use the real service and persistence, not fabricated service results.

Local HTTP servers are **fixture endpoints**, not paid production-model calls. Temporary provider-named executables are **agent adapter fixtures**, not authenticated Claude/Codex research. Disk-full/low-memory and some process error cases are explicitly **injected failures**. Dialog selections are patched in GUI tests while the underlying file/service operations actually execute. Qt tests and screenshots use **offscreen rendering**, not a human usability study.

## Host integrations

Claude Code 2.1.304 was installed in a workspace-local environment. Strict marketplace validation, plugin installation into isolated `CLAUDE_CONFIG_DIR`, and `claude mcp list` reporting **Connected** passed. A separate `CODEX_HOME` registered/listed the real stdio server using Codex CLI 0.155.1.

An **authenticated Codex turn** successfully invoked ModelForge's real `hardware_status` tool and reported observed `NumPy CPU` / `arm64`. The first attempt was blocked by noninteractive tool approval; the successful retry explicitly approved only that read-only tool. Existing host settings were not edited.

Claude reports no authenticated account on this host. No authenticated Claude model turn or Claude desktop application plugin-discovery UI test is claimed. The user must log in with `claude auth login` to complete authenticated Claude-agent testing. The existing host Claude executable also lacks its native binary; the isolated acceptance installation works.

## Distribution

A fresh clone of the public repository was installed, executed the full CPU example, and uninstalled. Research data remained intact. Local macOS ZIP creation uses `ditto` to preserve framework symlinks; the archive was unpacked to a fresh temporary directory, its ad-hoc signature verified, and the extracted application launched successfully. This catches archive failures that an in-place executable smoke test cannot catch. CI's archive stage now repeats extraction/launch before uploading assets.

macOS builds are ad-hoc signed, **not Developer-ID signed or notarized**. Windows artifacts are portable executable folders, **not signed MSI installers**. Source installation scripts are supplied for all three systems and were exercised on the CI runners. macOS Intel, specific Windows 10/11 client versions, GPU backends and high-memory training were not directly tested. No production-model extraction or proprietary weight reconstruction was run.
