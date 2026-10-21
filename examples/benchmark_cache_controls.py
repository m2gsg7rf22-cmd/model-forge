"""Reproduce bounded native/float32 KV diagnostics on an authorized local Llama."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from modelforge.execution import run_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="operator-supplied local Llama")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new evidence file; existing results are never overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(repo / "src"))
    results: list[dict[str, Any]] = []
    for precision in ("native", "float32"):
        result = run_json(
            [sys.executable, "-m", "modelforge.runtime_worker"],
            {
                "operation": "mlx_cache_benchmark",
                "arguments": {
                    "path": str(args.model.resolve()),
                    "prompt": "Research findings require careful independent verification. "
                    * 8,
                    "max_new_tokens": 32,
                    "repeats": 2,
                    "memory_limit_mb": 2048,
                    "precision": precision,
                    "compare_logits": True,
                },
            },
            cwd=args.output.parent / "cache-jobs",
            env=environment,
            memory_limit_mb=2048,
            timeout_seconds=180,
        )
        result["preflight"].pop("path", None)
        results.append(result)
    fingerprints = {}
    for item in sorted(args.model.glob("*.safetensors")):
        with item.open("rb") as handle:
            fingerprints[item.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "operator_model_label": args.label,
        "weights_sha256": fingerprints,
        "source_sha256": hashlib.sha256(
            (repo / "src/modelforge/runtime_mlx.py").read_bytes()
        ).hexdigest(),
        "hypothesis": "Increased numeric precision may remove observed argmax divergence between cached and full-prefix kernels.",
        "design": "Diagnostic control after initial native divergence; alternating warmed paths; fixed prompt and 32-token maximum, two repetitions per path; not a blind preregistered quality experiment.",
        "comparisons": results,
        "default_generation_changed": False,
        "heldout_quality_claim": False,
        "scope": "Local finite numerical and timing observations only; neither original parameter recovery nor frontier model parity.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "saved": str(args.output),
                "tested_token_parity": {
                    r["precision"]: r["tested_token_parity"] for r in results
                },
            }
        )
    )


if __name__ == "__main__":
    main()
