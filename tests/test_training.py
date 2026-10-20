"""Train actual, offline project-owned HF weights; never contact a model hub."""

from pathlib import Path
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from safetensors.torch import load_file  # noqa: E402
import torch  # noqa: E402
from modelforge.adapters import UnsupportedModelOperation  # noqa: E402
from modelforge.training import train_local, merge_adapter  # noqa: E402
from modelforge.runtime_hf import create_tiny_fixture, load_bundle  # noqa: E402


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "base"
    create_tiny_fixture(str(path))
    return path


TEXT = "hello world hello world hello world hello world"
RIGHTS = "  My own text.\nAll rights retained. אבג  "


@pytest.mark.parametrize("method", ["full", "lora"])
def test_real_training_resume_and_merge(
    source: Path, tmp_path: Path, method: str
) -> None:
    original = (source / "model.safetensors").read_bytes()
    out = tmp_path / method
    first = train_local(
        str(source),
        str(out),
        TEXT,
        method=method,
        steps=2,
        device="cpu",
        authorized=True,
        rights=RIGHTS,
        learning_rate=0.1,
    )
    assert first["completed_steps"] == 2
    assert first["final_loss"] < first["initial_loss"]
    assert first["rights"] == RIGHTS
    assert (source / "model.safetensors").read_bytes() == original
    assert not list(out.rglob("*.bin"))
    assert not list(out.rglob("*.pt"))
    if method == "lora":
        weights = load_file(
            str(Path(first["artifact_path"]) / "adapter_model.safetensors")
        )
        assert any(
            "lora_B" in k and bool(torch.count_nonzero(v)) for k, v in weights.items()
        )
        assert first["trainable_parameters"] < first["total_parameters"]
        merged = tmp_path / "merged"
        result = merge_adapter(
            str(source),
            first["artifact_path"],
            str(merged),
            redistributable=True,
            rights="Merge permitted",
        )
        assert result["training_rights"] == RIGHTS
        model, tokenizer, _ = load_bundle(str(merged), device="cpu")
        assert model(**tokenizer(TEXT, return_tensors="pt")).logits.shape[-1] > 0
    resumed = train_local(
        str(source),
        str(out),
        TEXT,
        method=method,
        steps=2,
        device="cpu",
        authorized=True,
        rights=RIGHTS,
        learning_rate=0.1,
        resume=True,
    )
    assert resumed["completed_steps"] == 4
    assert resumed["final_loss"] < first["final_loss"]
    assert json.loads((out / "training-state.json").read_text())["rights"] == RIGHTS
    with pytest.raises(ValueError, match="Resume settings"):
        train_local(
            str(source),
            str(out),
            TEXT,
            method=method,
            device="cpu",
            authorized=True,
            rights="different rights",
            learning_rate=0.1,
            resume=True,
        )


def test_training_guards(source: Path, tmp_path: Path) -> None:
    destination = tmp_path / "new"
    with pytest.raises(ValueError, match="authorization"):
        train_local(str(source), str(destination), TEXT)
    assert not destination.exists()
    with pytest.raises(UnsupportedModelOperation, match="QLoRA"):
        train_local(
            str(source),
            str(destination),
            TEXT,
            method="qlora",
            device="cpu",
            authorized=True,
            rights=RIGHTS,
        )
    with pytest.raises(ValueError, match="separate"):
        train_local(
            str(source), str(source / "new"), TEXT, authorized=True, rights=RIGHTS
        )
    destination.mkdir()
    with pytest.raises(FileExistsError):
        train_local(str(source), str(destination), TEXT, authorized=True, rights=RIGHTS)
    with pytest.raises(ValueError, match="redistribution"):
        merge_adapter(str(source), str(destination), str(tmp_path / "merge"))


def test_tampered_checkpoint_refused(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "training"
    first = train_local(
        str(source),
        str(out),
        TEXT,
        steps=1,
        device="cpu",
        authorized=True,
        rights=RIGHTS,
    )
    (Path(first["artifact_path"]) / "extra.txt").write_text("tampering")
    with pytest.raises(ValueError, match="integrity"):
        train_local(
            str(source),
            str(out),
            TEXT,
            device="cpu",
            authorized=True,
            rights=RIGHTS,
            resume=True,
        )


def test_full_resume_matches_uninterrupted(source: Path, tmp_path: Path) -> None:
    common = dict(
        method="full", device="cpu", authorized=True, rights=RIGHTS, learning_rate=0.1
    )
    uninterrupted = train_local(
        str(source), str(tmp_path / "straight"), TEXT, steps=4, **common
    )
    train_local(str(source), str(tmp_path / "split"), TEXT, steps=2, **common)
    resumed = train_local(
        str(source), str(tmp_path / "split"), TEXT, steps=2, resume=True, **common
    )
    expected = load_file(
        str(Path(uninterrupted["artifact_path"]) / "model.safetensors")
    )
    actual = load_file(str(Path(resumed["artifact_path"]) / "model.safetensors"))
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_lora_frozen_base(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peft
    from typing import Any

    original = peft.get_peft_model
    captured: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        captured["model"] = model
        captured["frozen"] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if not param.requires_grad
        }
        assert all(
            "lora_" in name
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        return model

    monkeypatch.setattr(peft, "get_peft_model", capture)
    train_local(
        str(source),
        str(tmp_path / "frozen"),
        TEXT,
        steps=2,
        device="cpu",
        authorized=True,
        rights=RIGHTS,
    )
    for name, param in captured["model"].named_parameters():
        if name in captured["frozen"]:
            torch.testing.assert_close(param, captured["frozen"][name], rtol=0, atol=0)
