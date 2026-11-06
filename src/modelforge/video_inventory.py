"""Read-only inventory for an optional local FastMetal video research runtime.

Presence of these known files is not integrity, compatibility, or quality proof.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import platform
import sys
from typing import Any

KNOWN_FILES = (
    "mlx_dit.json",
    "mlx_dit.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
REQUIRED_MODULES = ("mlx", "torch", "transformers", "safetensors", "av", "imageio", "imageio_ffmpeg", "fastvideo")


def inspect_fastmetal(model_root: str | Path) -> dict[str, Any]:
    """Inspect known files and the running Python only; never load or download weights."""
    root = Path(model_root).expanduser().resolve()
    missing = [name for name in KNOWN_FILES if not (root / name).is_file() or (root / name).stat().st_size == 0]
    modules = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}
    return {
        "model_root": str(root),
        "python": sys.executable,
        "platform": platform.system(),
        "machine": platform.machine(),
        "known_files_present": not missing,
        "missing_or_empty_files": missing,
        "modules_discoverable": modules,
        "model_integrity_verified": False,
        "generation_verified": False,
        "quality_or_equivalence_verified": False,
        "limitations": "Known filename and import-spec checks only. VAE, decoder, versions, integrity, memory, offline operation, generation and quality are not verified.",
    }
