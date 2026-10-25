"""Small bounded JSON metadata with hash-bound numerical evidence outside stdout."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge.glu32_io import raw

Array = NDArray[np.float64]


def save_evidence(
    output: Path, arrays: dict[str, Array], metadata: dict[str, Any]
) -> dict[str, Any]:
    if any(p.is_symlink() for p in [output, *output.parents]):
        raise ValueError("Evidence output must not traverse symlinks")
    output.mkdir(exist_ok=False)
    (output / "attempt.json").write_text('{"automatic_retry":false}\n')
    if not isinstance(metadata, dict):
        raise ValueError("Compact metadata object required")
    encoded = json.dumps(metadata, allow_nan=False).encode()
    if len(encoded) > 16_384:
        raise ValueError("Metadata exceeds16KiB compact bound")
    if not 1 <= len(arrays) <= 64 or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name)
        or name in {"file", "allow_pickle"}
        for name in arrays
    ):
        raise ValueError("Invalid bounded evidence array inventory")
    if any(
        not isinstance(a, np.ndarray)
        or a.dtype != np.float64
        or not np.isfinite(a).all()
        for a in arrays.values()
    ):
        raise ValueError("Only finite FP64 numerical evidence supported")
    if sum(a.nbytes for a in arrays.values()) > 32 * 1024**2:
        raise ValueError("Numerical evidence exceeds32MiB prototype bound")
    numerical: dict[str, Any] = dict(arrays)
    np.savez(output / "numeric-evidence.npz", **numerical)
    data = raw(output / "numeric-evidence.npz")
    summary = {
        "schema_version": 1,
        "evidence_file": "numeric-evidence.npz",
        "evidence_sha256": hashlib.sha256(data).hexdigest(),
        "evidence_bytes": len(data),
        "arrays": {
            name: {"shape": list(a.shape), "dtype": str(a.dtype)}
            for name, a in arrays.items()
        },
        "metadata": metadata,
    }
    compact = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if len(compact.encode()) > 32_768:
        raise ValueError("Summary exceeds32KiB stdout/reader bound")
    (output / "summary.json").write_text(compact)
    return summary
