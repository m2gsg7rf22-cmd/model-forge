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

Install the wheel using `python -m pip install 'path/to/modelforge_research-0.1.0-py3-none-any.whl[desktop]'`. The wheel contains CLI, engines and MCP. Plugin manifests and installer scripts are in the source archive/repository, not the wheel.

## Checksums

Compare `SHA256SUMS.txt` with `shasum -a 256 FILE` on macOS, `sha256sum FILE` on Linux, or `Get-FileHash FILE -Algorithm SHA256` on Windows. Hashes establish file integrity relative to that manifest, not publisher authenticity.
