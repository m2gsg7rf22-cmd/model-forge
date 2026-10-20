"""Bounded forward measurements and reversible MLP ablation of local models.

These are numeric observations of a supplied model, not explanations of what a
neuron means or evidence about an inaccessible model's internal architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _summary(tensor: Any) -> dict[str, Any]:
    """Reduce one activation without retaining its values in the report."""
    import torch

    values = tensor.detach().float()
    if not torch.isfinite(values).all().item():
        raise ValueError("Model produced non-finite activations")
    return {
        "shape": list(values.shape),
        "elements": values.numel(),
        "mean": float(values.mean().item()),
        "standard_deviation": float(values.std(unbiased=False).item()),
        "rms": float(values.square().mean().sqrt().item()),
        "maximum_absolute": float(values.abs().max().item()),
    }


def trace_local(
    path: str | Path,
    prompt: str,
    ablate_module: str = "",
    memory_limit_mb: int = 2048,
    device: str = "cpu",
    adapter: str = "",
) -> dict[str, Any]:
    """Trace at most 128 leaf modules over at most 64 input tokens.

    Optional ablation replaces one named built-in GPT-2/Llama MLP output with
    zeros for a second forward pass. Hooks are removed even when a pass fails;
    parameters are never edited. KL is for the final next-token distribution.
    """
    import torch
    from .runtime_hf import _inputs, load_bundle

    if adapter:
        from .training import load_adapter_bundle

        model, tokenizer, preflight = load_adapter_bundle(
            str(path), adapter, memory_limit_mb=memory_limit_mb, device=device
        )
    else:
        model, tokenizer, preflight = load_bundle(path, memory_limit_mb, device)
    inputs = _inputs(model, tokenizer, prompt)
    count = int(inputs["input_ids"].shape[-1])
    if count > 64:
        raise ValueError("Activation tracing is limited to 64 input tokens")
    modules = list(model.named_modules())
    eligible = {
        name: module
        for name, module in modules
        if name.endswith(".mlp") and type(module).__name__ in {"GPT2MLP", "LlamaMLP"}
    }
    if ablate_module and ablate_module not in eligible:
        raise ValueError(
            "Ablation requires an exact supported MLP module name from ablation_modules"
        )
    leaves = [
        (name, module)
        for name, module in modules
        if name and not list(module.children())
    ]
    observations: list[dict[str, Any]] = []
    handles: list[Any] = []

    def observe(name: str) -> Any:
        def hook(module: Any, args: Any, output: Any) -> None:
            value = (
                output[0] if isinstance(output, (tuple, list)) and output else output
            )
            if (
                isinstance(value, torch.Tensor)
                and value.numel()
                and len(observations) < 128
            ):
                observations.append(
                    {"module": name, "class": type(module).__name__, **_summary(value)}
                )

        return hook

    try:
        for name, module in leaves[:128]:
            handles.append(module.register_forward_hook(observe(name)))
        with torch.inference_mode():
            baseline = model(**inputs, use_cache=False).logits.detach().float()
        if not torch.isfinite(baseline).all().item():
            raise ValueError("Model produced non-finite logits")
    finally:
        for handle in handles:
            handle.remove()
    result: dict[str, Any] = {
        "scope": "Numeric forward activations; no semantic interpretation or remote-model inference",
        "input_tokens": count,
        "activations": observations[:128],
        "module_limit": 128,
        "module_selection_truncated": len(leaves) > 128,
        "ablation_modules": list(eligible)[:128],
        "baseline_logits": _summary(baseline),
        "preflight": preflight,
    }
    if ablate_module:

        def zero_output(module: Any, args: Any, output: Any) -> Any:
            if not isinstance(output, torch.Tensor):
                raise ValueError("Supported MLP unexpectedly returned a non-tensor")
            return torch.zeros_like(output)

        handle = eligible[ablate_module].register_forward_hook(zero_output)
        try:
            with torch.inference_mode():
                changed = model(**inputs, use_cache=False).logits.detach().float()
            if not torch.isfinite(changed).all().item():
                raise ValueError("Ablation produced non-finite logits")
            left = torch.log_softmax(baseline[0, -1], dim=-1)
            right = torch.log_softmax(changed[0, -1], dim=-1)
            kl = (left.exp() * (left - right)).sum().item()
            delta = changed - baseline
            result["ablation"] = {
                "module": ablate_module,
                "intervention": "Replace this MLP output with zeros for one forward pass",
                "maximum_absolute_logit_change": float(delta.abs().max().item()),
                "rms_logit_change": float(delta.square().mean().sqrt().item()),
                "last_token_kl_baseline_to_ablated": max(0.0, float(kl)),
                "baseline_next_token_id": int(baseline[0, -1].argmax().item()),
                "ablated_next_token_id": int(changed[0, -1].argmax().item()),
                "parameters_modified": False,
            }
        finally:
            handle.remove()
    return result
