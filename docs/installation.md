# Installation and removal

Source installation requires Python 3.12+, internet access to PyPI and a compatible operating system. The standalone desktop archive includes Python, NumPy and Qt; optional ONNX/Core ML parsers are available only in source installations.

From a checkout:

```sh
python3.12 packaging/install.py --prefix "$PWD/.venv"
.venv/bin/modelforge gui
```

On Windows use `py -3.12` and `.venv\Scripts\modelforge.exe`. Installation creates a new virtual environment. An existing prefix is refused. It does not edit your shell, system Python, agent settings or model files. `--core-only` omits Qt. Optional parsers: run the environment's Python with `-m pip install '.[onnx]' coremltools` from the checkout.

Uninstall only the marked dedicated installation:

```sh
python3.12 packaging/uninstall.py --prefix "$PWD/.venv" --yes
```

Research data remains in `~/.modelforge` or `MODELFORGE_HOME`. Choose an explicit data directory with `modelforge --home ./research-data ...`. Delete research data separately only when intended.

## Standalone desktop

Extract the OS-specific ZIP and run ModelForge (`ModelForge.app` on macOS, `ModelForge.exe` inside its folder on Windows, `ModelForge/ModelForge` on Linux). Keep the whole folder on Windows/Linux. macOS builds are ad-hoc signed by the packager, not Developer-ID signed or notarized; Gatekeeper may require the standard user approval flow. Do not disable system security.

CI builds and smoke-tests each native artifact before upload. GitHub Actions artifacts are verification outputs; only assets attached to an actual Release are published downloads. See validation.md for exactly which platforms have been tested. Installer scripts for other operating systems are not claimed tested until their jobs succeed.

## Python wheel

Install the wheel using `python -m pip install 'path/to/modelforge_research-0.2.0-py3-none-any.whl[desktop]'`. The wheel contains CLI, engines and MCP. Plugin manifests and installer scripts are in the source archive/repository, not the wheel.

## Checksums

Compare `SHA256SUMS.txt` with `shasum -a 256 FILE` on macOS, `sha256sum FILE` on Linux, or `Get-FileHash FILE -Algorithm SHA256` on Windows. Hashes establish file integrity relative to that manifest, not publisher authenticity.


## Optional language engines

The native base keeps heavyweight Torch/Transformers/PEFT and MLX separate. Open
**Settings → Optional local engines** to inspect dependencies, install `llm` or
`mlx` after explicit consent, or save an absolute compatible Python interpreter.
A new companion environment requires at least 3 GiB free disk, network access,
and either `uv` or Python 3.12+. Existing engine environments are not overwritten.
MLX requires macOS Apple Silicon. Libraries are downloaded; model weights are not.

From source, install `'.[llm]'` into the dedicated environment, or
`'.[llm,mlx]'` on Apple Silicon. Missing packages or unsupported devices fail
explicitly. Source/companion extras are not evidence that every backend has been
hardware-verified; see [capabilities](capabilities.md) and [MLX](mlx.md).

For a dedicated Claude Code/Codex installation directly from the repository,
including actual MCP verification, scoped connection, update and removal, use
[packaging/bootstrap.py instructions](agent-install.md). Host registration is
applied only when explicitly requested with `--apply`.

## macOS Apple Silicon: v0.2.0

Download `ModelForge-0.2.0-Darwin-arm64.zip` and `SHA256SUMS.txt` from the [v0.2.0 release](https://github.com/m2gsg7rf22-cmd/model-forge/releases/tag/v0.2.0). In Terminal, change into the download directory and run `shasum -a 256 ModelForge-0.2.0-Darwin-arm64.zip`; compare the entire hash with its line in the manifest. Double-click the ZIP in Finder, drag `ModelForge.app` to Applications and open it. If Gatekeeper blocks this ad-hoc-signed app, attempt opening once, then use System Settings → Privacy & Security → Open Anyway and confirm the specific application. Do not disable Gatekeeper globally. This first-open approval path follows [Apple’s documented app-opening procedure](https://support.apple.com/en-il/102445), but has not been manually verified on a clean Mac.

The base application bundles Python; pretrained weights are not included. For MLX, install `uv` or Python 3.12+ if absent, ensure at least 3 GiB free disk, then choose **Settings → Optional local engines → mlx** for the first companion installation and explicitly consent to library downloads. The `mlx` extra includes the HF/Torch dependencies used for validation. Existing companion environments are refused by the new-install action; use a compatible external interpreter or the source bootstrap update workflow instead.

For an immediate offline pipeline demonstration, use the owned random Llama fixture in the builder; it is not a pretrained assistant. The release example ZIP includes that tiny owned model after LoRA and merge. For useful inference, supply a compatible authorized GPT-2/Llama SafeTensors bundle and tokenizer. MLX runs Llama on Apple Metal with a 2048 MiB per-worker configured limit. Model admission can reject larger bundles; 3 GiB installer disk space is not a promise that every model will fit.
