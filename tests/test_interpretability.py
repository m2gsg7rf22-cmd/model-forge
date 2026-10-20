"""Real bounded activation observations on owned GPT-2 and Llama fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from modelforge.interpretability import trace_local
from modelforge.runtime_hf import create_owned_fixture, load_bundle


@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
def test_activation_trace_and_reversible_ablation(
    tmp_path: Path, architecture: str, monkeypatch: Any
) -> None:
    import torch
    from modelforge import runtime_hf

    folder = tmp_path / architecture
    create_owned_fixture(folder, model_type=architecture)
    model, tokenizer, report = load_bundle(folder, device="cpu")
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    monkeypatch.setattr(
        runtime_hf, "load_bundle", lambda *args: (model, tokenizer, report)
    )
    module = "transformer.h.0.mlp" if architecture == "gpt2" else "model.layers.0.mlp"
    result = trace_local(folder, "hello world", ablate_module=module)
    assert 0 < len(result["activations"]) <= 128
    assert result["input_tokens"] <= 64
    assert result["ablation"]["maximum_absolute_logit_change"] > 0
    assert result["ablation"]["last_token_kl_baseline_to_ablated"] >= 0
    assert result["ablation"]["parameters_modified"] is False
    json.dumps(result, allow_nan=False)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    assert all(not module._forward_hooks for module in model.modules())
    with pytest.raises(ValueError, match="exact supported MLP"):
        trace_local(folder, "hello", ablate_module="lm_head")
    with pytest.raises(ValueError, match="64 input tokens"):
        trace_local(folder, "hello " * 70)


def test_hooks_removed_when_ablation_forward_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from modelforge import runtime_hf

    folder = tmp_path / "gpt2"
    create_owned_fixture(folder, model_type="gpt2")
    model, tokenizer, report = load_bundle(folder, device="cpu")
    monkeypatch.setattr(
        runtime_hf, "load_bundle", lambda *args: (model, tokenizer, report)
    )
    original_forward = model.forward
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Injected forward interruption")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", fail_second)
    with pytest.raises(RuntimeError, match="interruption"):
        trace_local(folder, "hello world", ablate_module="transformer.h.0.mlp")
    assert all(not module._forward_hooks for module in model.modules())
