"""Finite owned-Llama Q/K rescaling witness; no original-parameter identity claim.

Use only a project-owned local fixture. Does not fit/train or save weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite evidence")
    started = time.monotonic()
    import torch
    from modelforge.runtime_hf import load_bundle

    # Built-in architecture, SafeTensors only, no external model/provider access.
    model, _, preflight = load_bundle(
        str(args.fixture), memory_limit_mb=2048, device="cpu"
    )
    if preflight["model_type"] != "llama":
        parser.error("Requires an owned Llama fixture")
    layer = model.model.layers[0].self_attn
    if layer.q_proj.bias is not None or layer.k_proj.bias is not None:
        parser.error(
            "The weight-only scaling control requires bias-free Q/K projections"
        )
    model.eval()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261021)
    tokens = torch.randint(0, model.config.vocab_size, (64, 8), generator=generator)
    with torch.inference_mode():
        before = model(tokens).logits.clone()
        old_q = layer.q_proj.weight.clone()
        old_k = layer.k_proj.weight.clone()
        layer.q_proj.weight.mul_(2.0)
        layer.k_proj.weight.mul_(0.5)
        after = model(tokens).logits
        result = {
            "experiment": "Owned Llama first-layer Q/K reciprocal scaling",
            "public_input_seed": 20261021,
            "shape": list(tokens.shape),
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "q_weight_l2_change": float(
                torch.linalg.vector_norm(layer.q_proj.weight - old_q)
            ),
            "k_weight_l2_change": float(
                torch.linalg.vector_norm(layer.k_proj.weight - old_k)
            ),
            "max_absolute_logit_difference": float(
                torch.max(torch.abs(before - after))
            ),
            "argmax_disagreements": int((before.argmax(-1) != after.argmax(-1)).sum()),
            "finite_logit_exact_value_equality": bool(torch.equal(before, after)),
            "assumptions": "Known bias-free project-owned Llama eval; reciprocal global Q/K scaling commutes with linear RoPE; no overflow/underflow in the ideal-arithmetic identity",
            "identity": "(R_i (c q_i))^T (R_j (k_j/c)) = (R_i q_i)^T (R_j k_j), c != 0",
            "scope": "Explicit original-parameter ambiguity witness; finite numerical check is not universal float equivalence, model extraction or useful-language capability",
            "elapsed_seconds": time.monotonic() - started,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_steps": 0,
            "external_calls": 0,
            "weights_written": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
