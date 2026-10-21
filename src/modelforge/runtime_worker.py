"""Private JSON worker entry point for optional installed engines.

This process never interprets arbitrary function names or executes supplied code.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from typing import Any


def dispatch(payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch an explicitly supported engine operation."""
    operation = payload.get("operation")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("Engine arguments must be an object")
    if operation == "status":
        packages: dict[str, Any] = {}
        for name in (
            "torch",
            "transformers",
            "peft",
            "safetensors",
            "mlx_lm",
            "onnxruntime",
        ):
            found = importlib.util.find_spec(name) is not None
            try:
                version = (
                    importlib.metadata.version(name.replace("_", "-"))
                    if found
                    else None
                )
            except importlib.metadata.PackageNotFoundError:
                version = None
            packages[name] = {"available": found, "version": version}
        return {"ok": True, "python": sys.version.split()[0], "packages": packages}
    if operation in {"hf_preflight", "hf_generate", "hf_reconstruct", "hf_fixture"}:
        from . import runtime_hf

        name = {
            "hf_preflight": "preflight",
            "hf_generate": "generate",
            "hf_reconstruct": "reconstruct",
            "hf_fixture": "create_owned_fixture",
        }[str(operation)]
        result = getattr(runtime_hf, name)(**arguments)
    elif operation in {"train_local", "merge_adapter", "generate_adapter"}:
        from . import training

        result = getattr(training, str(operation))(**arguments)
    elif operation in {
        "mlx_generate",
        "mlx_benchmark",
        "mlx_cache_benchmark",
        "mlx_train_lora",
        "mlx_generate_adapter",
    }:
        from . import runtime_mlx

        name = {
            "mlx_generate": "generate",
            "mlx_benchmark": "benchmark",
            "mlx_cache_benchmark": "benchmark_cache",
            "mlx_train_lora": "train_lora",
            "mlx_generate_adapter": "generate_adapter",
        }[str(operation)]
        result = getattr(runtime_mlx, name)(**arguments)
    elif operation == "trace_local":
        from .interpretability import trace_local

        result = trace_local(**arguments)
    elif operation == "attention_research":
        from .attention_research import run_attention_research

        result = run_attention_research(**arguments)
    elif operation == "discovery":
        from .discovery import run_discovery

        result = run_discovery(**arguments)
    elif operation == "umrre":
        from .umrre import run_research

        result = run_research(**arguments)
    else:
        raise ValueError("Unsupported optional engine operation")
    if not isinstance(result, dict):
        raise TypeError("Engine operation must return an object")
    return {"ok": True, **result}


def main() -> None:
    """Read at most 2MB; report failures as bounded structured errors."""
    try:
        raw = sys.stdin.buffer.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Engine request exceeds 2MB")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Engine request must be an object")
        result = dispatch(payload)
    except Exception as exc:
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
        print(json.dumps(result, allow_nan=False))
        raise SystemExit(1)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
