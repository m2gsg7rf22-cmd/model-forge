"""Real offline HF runtime, hostile bundle rejection and reconstruction tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from modelforge.runtime_hf import (
    create_tiny_fixture,
    generate,
    load_bundle,
    preflight,
    reconstruct,
)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return Path(create_tiny_fixture(tmp_path / "owned"))


def test_real_offline_generation_and_exact_reconstruction(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    import torch

    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Network is forbidden")

    monkeypatch.setattr(socket, "create_connection", no_network)
    result = generate(bundle, "hello world", max_new_tokens=3, device="cpu")
    assert result["new_tokens"] > 0
    assert result["preflight"]["complete_checkpoint"]
    exported = tmp_path / "reconstructed"
    evidence = reconstruct(
        bundle,
        exported,
        ["hello world", "the model is small"],
        redistributable=True,
        rights="Project-owned random fixture; MIT",
    )
    assert all(
        row["exact"] and row["max_absolute_logit_error"] == 0
        for row in evidence["parity"]
    )
    assert (exported / "modelforge-evidence.json").is_file()
    left, _, _ = load_bundle(bundle, device="cpu")
    right, _, _ = load_bundle(exported, device="cpu")
    assert left.state_dict().keys() == right.state_dict().keys()
    for key in left.state_dict():
        assert torch.equal(left.state_dict()[key], right.state_dict()[key])
    assert (
        generate(exported, "hello world", 3, device="cpu")["token_ids"]
        == result["token_ids"]
    )


def test_budget_refusal_before_weight_load(bundle: Path) -> None:
    with pytest.raises(ValueError, match="Estimated RAM"):
        preflight(bundle, memory_limit_mb=1, device="cpu")


@pytest.mark.parametrize("filename", ["pytorch_model.bin", "remote.py", "data.pkl"])
def test_reject_unsafe_files(bundle: Path, filename: str) -> None:
    (bundle / filename).write_bytes(b"malicious")
    with pytest.raises(ValueError, match="Unsafe"):
        preflight(bundle, device="cpu")


def test_missing_tensor_is_not_randomly_initialized(bundle: Path) -> None:
    from safetensors.torch import load_file, save_file

    file = bundle / "model.safetensors"
    tensors = load_file(str(file))
    del tensors["transformer.h.0.mlp.c_fc.weight"]
    save_file(tensors, str(file), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="Incomplete"):
        load_bundle(bundle, device="cpu")


def test_shape_mismatch(bundle: Path) -> None:
    from safetensors.torch import load_file, save_file

    file = bundle / "model.safetensors"
    tensors = load_file(str(file))
    tensors["transformer.h.0.mlp.c_fc.weight"] = tensors[
        "transformer.h.0.mlp.c_fc.weight"
    ][:1]
    save_file(tensors, str(file), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="shape mismatch"):
        preflight(bundle, device="cpu")


def test_rights_required(bundle: Path, tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="authorization"):
        reconstruct(bundle, tmp_path / "no", ["hello"])
    assert not (tmp_path / "no").exists()


def test_remote_code_refused(bundle: Path) -> None:
    file = bundle / "config.json"
    config = json.loads(file.read_text())
    config["auto_map"] = {"AutoModel": "evil.Custom"}
    file.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="built-in"):
        preflight(bundle, device="cpu")


def test_truncated_checkpoint(bundle: Path) -> None:
    file = bundle / "model.safetensors"
    file.write_bytes(file.read_bytes()[:-4])
    with pytest.raises(ValueError, match="payload"):
        preflight(bundle, device="cpu")


def test_missing_shard(bundle: Path) -> None:
    (bundle / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="Missing SafeTensors shard"):
        preflight(bundle, device="cpu")


def test_shard_path_traversal(bundle: Path) -> None:
    (bundle / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="Unsafe shard"):
        preflight(bundle, device="cpu")


def test_prompt_limits(bundle: Path) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(bundle, "hello", 9999, device="cpu")
    with pytest.raises(ValueError, match="nonempty"):
        generate(bundle, "", device="cpu")


def test_real_llama_fixture(tmp_path: Path) -> None:
    from modelforge.runtime_hf import create_owned_fixture

    report = create_owned_fixture(tmp_path / "llama", model_type="llama", seed=9)
    assert report["preflight"]["model_type"] == "llama"
    model, tokenizer, loaded = load_bundle(report["path"], device="cpu")
    assert model.config.model_type == "llama"
    assert tokenizer("hello world")["input_ids"] == [4, 5]
    assert loaded["module_tree"][0]["parent"] is None
    assert "not an execution graph" in loaded["module_tree_scope"]
    output = generate(report["path"], "hello world", max_new_tokens=2, device="cpu")
    assert output["new_tokens"] > 0
