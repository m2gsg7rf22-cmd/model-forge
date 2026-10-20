"""Real Apple Metal MLX tests using project-owned random Llama weights."""

from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path

import pytest

from modelforge.runtime_hf import create_owned_fixture
from modelforge.runtime_mlx import benchmark, generate, generate_adapter, train_lora

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or platform.machine() != "arm64"
    or importlib.util.find_spec("mlx_lm") is None,
    reason="Optional MLX tests require Apple Silicon and the mlx extra",
)


@pytest.fixture
def llama(tmp_path: Path) -> Path:
    path = tmp_path / "llama"
    create_owned_fixture(path, model_type="llama")
    return path


def test_actual_gpu_generation_and_quantization(llama: Path) -> None:
    import mlx.core as mx

    old_limit = mx.get_memory_limit()
    full = generate(llama, "hello world", max_new_tokens=3)
    assert full["device"] == "gpu"
    assert full["hardware"]["device_name"].startswith("Apple")
    assert full["new_tokens"] == len(full["token_ids"]) == 3
    assert 0 < full["peak_mlx_bytes"] < 2048 * 1024**2
    assert mx.get_memory_limit() == old_limit
    quant = generate(llama, "hello world", 3, quantize_bits=4)
    assert quant["preflight"]["quantize_bits"] == 4
    assert quant["preflight"]["quantization_group_size"] == 32
    assert quant["new_tokens"] == 3


def test_real_cpu_gpu_benchmark(llama: Path) -> None:
    report = benchmark(llama)
    assert len(report["measurements"]) == 4
    assert report["unquantized_cpu_gpu_token_parity"]
    for row in report["measurements"]:
        assert row["seconds"] > 0
        assert row["tokens_per_second"] > 0
        assert row["complete_checkpoint"]


@pytest.mark.parametrize("bits", [0, 4])
def test_actual_lora_gradients_and_safe_artifact(
    llama: Path, tmp_path: Path, bits: int
) -> None:
    import mlx.core as mx

    report = train_lora(
        llama,
        tmp_path / "adapter",
        "hello world hello world hello world",
        steps=5,
        learning_rate=0.1,
        authorized=True,
        rights="Project-owned random fixture and text; MIT",
        quantize_bits=bits,
    )
    assert report["final_loss"] < report["initial_loss"]
    assert report["trainable_parameters"] > 0
    tensors = mx.load(str(tmp_path / "adapter" / "adapters.safetensors"))
    assert tensors
    assert all("lora_" in key for key in tensors)
    assert any(
        float(mx.abs(value).sum().item()) > 0
        for key, value in tensors.items()
        if "lora_b" in key
    )
    stored = json.loads((tmp_path / "adapter" / "modelforge-mlx-lora.json").read_text())
    assert stored["corpus_sha256"]
    assert not list((tmp_path / "adapter").glob("*.bin"))
    inferred = generate_adapter(llama, tmp_path / "adapter", "hello world", 3)
    assert inferred["adapter_verified"]
    assert inferred["new_tokens"] == 3
    file = tmp_path / "adapter" / "adapters.safetensors"
    data = file.read_bytes()
    file.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="integrity"):
        generate_adapter(llama, tmp_path / "adapter", "hello", 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "cuda"},
        {"memory_limit_mb": 1},
        {"memory_limit_mb": 4096},
        {"quantize_bits": 3},
        {"max_new_tokens": 0},
        {"prompt": ""},
        {"max_new_tokens": 256},
    ],
)
def test_invalid_budgets_and_context(llama: Path, kwargs: dict[str, object]) -> None:
    options = {"prompt": "hello", "max_new_tokens": 3, **kwargs}
    with pytest.raises(ValueError):
        generate(llama, **options)


def test_reject_gpt2_and_code(tmp_path: Path, llama: Path) -> None:
    gpt = tmp_path / "gpt"
    create_owned_fixture(gpt)
    with pytest.raises(ValueError, match="Llama"):
        generate(gpt, "hello", 1)
    (llama / "evil.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="Unsafe"):
        generate(llama, "hello", 1)


def test_training_rights_and_paths(llama: Path, tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        train_lora(llama, tmp_path / "no", "hello world")
    with pytest.raises(ValueError, match="outside"):
        train_lora(
            llama, llama / "nested", "hello world", authorized=True, rights="owned"
        )
    with pytest.raises(ValueError, match="steps"):
        train_lora(
            llama,
            tmp_path / "bad",
            "hello world",
            steps=0,
            authorized=True,
            rights="owned",
        )


def test_actual_generation_stops_at_configured_eos(llama: Path) -> None:
    first = generate(llama, "hello world", max_new_tokens=1)
    config = json.loads((llama / "generation_config.json").read_text())
    config["eos_token_id"] = [first["token_ids"][0]]
    (llama / "generation_config.json").write_text(json.dumps(config))
    ended = generate(llama, "hello world", max_new_tokens=10)
    assert ended["new_tokens"] == 1
    assert ended["finish_reason"] == "eos"
    assert ended["output"] == ""
