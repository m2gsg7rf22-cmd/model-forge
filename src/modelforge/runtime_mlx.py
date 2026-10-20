"""Offline Llama execution using MLX-LM's built-in Apple Silicon model.

Only validated, unquantized local HF SafeTensors are accepted as inputs.
Quantization happens in MLX before inference; no custom model code is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast

from .runtime_hf import _hashes, _headers, _json, preflight


@contextmanager
def _limits(memory_limit_mb: int, device: str) -> Iterator[Any]:
    import mlx.core as mx

    if device not in {"cpu", "gpu"}:
        raise ValueError("MLX device must be cpu or gpu")
    if not isinstance(memory_limit_mb, int) or not 256 <= memory_limit_mb <= 2048:
        raise ValueError("MLX memory limit must be between 256 and 2048 MiB")
    if device == "gpu" and not mx.metal.is_available():
        raise ValueError("MLX Metal GPU is unavailable")
    old_memory = mx.set_memory_limit(memory_limit_mb * 1024**2)
    old_cache = mx.set_cache_limit(min(64, memory_limit_mb // 8) * 1024**2)
    mx.reset_peak_memory()
    try:
        with mx.stream(mx.cpu if device == "cpu" else mx.gpu):
            yield mx
    finally:
        mx.synchronize()
        mx.clear_cache()
        mx.set_memory_limit(old_memory)
        mx.set_cache_limit(old_cache)


def _load(
    path: str | Path, memory_limit_mb: int, quantize_bits: int
) -> tuple[Any, Any, dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.llama import Model, ModelArgs
    from transformers import PreTrainedTokenizerFast

    if quantize_bits not in {0, 4, 8}:
        raise ValueError("quantize_bits must be 0, 4, or 8")
    report = preflight(path, memory_limit_mb=memory_limit_mb, device="cpu")
    root = Path(report["path"])
    config = _json(root / "config.json")
    if config["model_type"] != "llama":
        raise ValueError("MLX currently supports validated built-in Llama bundles only")
    model = Model(ModelArgs.from_dict(config))
    index = root / "model.safetensors.index.json"
    names = (
        sorted(set(_json(index)["weight_map"].values()))
        if index.exists()
        else ["model.safetensors"]
    )
    weights: dict[str, Any] = {}
    for name in names:
        loaded = mx.load(str(root / name))
        if not isinstance(loaded, dict):
            raise ValueError("Expected SafeTensors named tensor mapping")
        weights.update(loaded)
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    if quantize_bits:
        nn.quantize(
            model,
            group_size=32,
            bits=quantize_bits,
            class_predicate=lambda _, module: isinstance(module, nn.Linear),
        )
    mx.eval(model.parameters())
    model.eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json"))
    return (
        model,
        tokenizer,
        {
            **report,
            "runtime": "mlx-lm",
            "quantize_bits": quantize_bits,
            "quantization_group_size": 32 if quantize_bits else None,
        },
    )


def _tokens(tokenizer: Any, model: Any, prompt: str, extra: int) -> list[int]:
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt.encode("utf-8")) > 16000
    ):
        raise ValueError("Prompt must be nonempty and at most 16000 UTF-8 bytes")
    if not isinstance(extra, int) or not 0 <= extra <= 256:
        raise ValueError("Token budget must be between 0 and 256")
    ids: list[int] = tokenizer.encode(prompt, add_special_tokens=False)
    context = min(int(model.args.max_position_embeddings or 4096), 4096)
    if not ids or len(ids) + extra > context:
        raise ValueError("Prompt and generation exceed the local context budget")
    return ids


def _generate(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    import mlx.core as mx

    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256")
    ids = _tokens(tokenizer, model, prompt, max_new_tokens)
    generated: list[int] = []
    started = time.perf_counter()
    # Bounded full-prefix decoding avoids a backend-specific unbounded KV cache.
    for _ in range(max_new_tokens):
        logits = model(mx.array([ids + generated], dtype=mx.int32))[:, -1, :]
        token = int(cast(Any, mx.argmax(logits, axis=-1).item()))
        generated.append(token)
    elapsed = time.perf_counter() - started
    return {
        "output": tokenizer.decode(generated),
        "token_ids": generated,
        "prompt_tokens": len(ids),
        "new_tokens": len(generated),
        "seconds": elapsed,
        "tokens_per_second": len(generated) / elapsed,
        "decoding": "greedy fixed-budget full-prefix",
    }


def generate(
    path: str | Path,
    prompt: str,
    max_new_tokens: int = 32,
    memory_limit_mb: int = 2048,
    quantize_bits: int = 0,
    device: str = "gpu",
) -> dict[str, Any]:
    """Generate actual greedy tokens on MLX CPU or GPU within a bounded job.

    The MLX allocator limit covers MLX allocations. The service worker separately
    watches whole-process RSS, wall time, and logs; neither is an OS reservation.
    """
    with _limits(memory_limit_mb, device) as mx:
        model, tokenizer, report = _load(path, memory_limit_mb, quantize_bits)
        result = _generate(model, tokenizer, prompt, max_new_tokens)
        return {
            **result,
            "device": device,
            "preflight": report,
            "peak_mlx_bytes": mx.get_peak_memory(),
            "allocator_limit_bytes": memory_limit_mb * 1024**2,
            "hardware": (
                mx.device_info() if mx.metal.is_available() else {"device_name": "CPU"}
            ),
        }


def benchmark(path: str | Path, memory_limit_mb: int = 2048) -> dict[str, Any]:
    """Measure warm CPU/GPU inference and quantized GPU inference on one input.

    Tiny untrained fixtures measure execution only, not language quality or
    representative production throughput. Timing includes full-prefix decode.
    """
    rows = []
    for device, bits in [("cpu", 0), ("gpu", 0), ("gpu", 4), ("gpu", 8)]:
        with _limits(memory_limit_mb, device) as mx:
            model, tokenizer, report = _load(path, memory_limit_mb, bits)
            _generate(model, tokenizer, "hello world", 2)
            mx.synchronize()
            mx.reset_peak_memory()
            result = _generate(model, tokenizer, "hello world", 8)
            rows.append(
                {
                    "device": device,
                    "bits": bits or 32,
                    **result,
                    "peak_mlx_bytes": mx.get_peak_memory(),
                    "complete_checkpoint": report["complete_checkpoint"],
                }
            )
    return {
        "runtime": "mlx-lm",
        "measurements": rows,
        "unquantized_cpu_gpu_token_parity": rows[0]["token_ids"]
        == rows[1]["token_ids"],
        "scope": "Single warm run per device and precision; no quality or general speedup claim",
    }


def train_lora(
    path: str | Path,
    destination: str | Path,
    text: str,
    steps: int = 5,
    learning_rate: float = 0.01,
    memory_limit_mb: int = 2048,
    authorized: bool = False,
    rights: str = "",
    quantize_bits: int = 0,
) -> dict[str, Any]:
    """Train genuine MLX LoRA (or quantized-base LoRA) on authorized user text.

    Checkpoints contain only SafeTensors adapter weights and provenance. No text
    corpus is written. This bounded local training does not assert generalization.
    """
    if not authorized or not rights.strip():
        raise PermissionError("Training requires authorization and a rights statement")
    if not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("steps must be between 1 and 100")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be finite and in (0, 1]")
    target = Path(destination).expanduser().resolve()
    source = Path(path).expanduser().resolve()
    if (
        target.exists()
        or target.is_relative_to(source)
        or source.is_relative_to(target)
    ):
        raise ValueError("Destination must be new and outside the source bundle")
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < 64 * 1024**2:
        raise ValueError("At least 64 MiB free disk is required")
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    with _limits(memory_limit_mb, "gpu") as mx:
        model, tokenizer, report = _load(source, memory_limit_mb, quantize_bits)
        ids = _tokens(tokenizer, model, text, 0)
        if not 2 <= len(ids) <= 64:
            raise ValueError("Training text must contain between 2 and 64 tokens")
        model.freeze()
        config = {
            "rank": 4,
            "scale": 8.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
        }
        linear_to_lora_layers(model, len(model.layers), config)
        optimizer = optim.SGD(learning_rate=learning_rate)
        batch = mx.array([ids], dtype=mx.int32)

        def loss_fn(current: Any) -> Any:
            return nn.losses.cross_entropy(
                current(batch[:, :-1]), batch[:, 1:], reduction="mean"
            )

        value_and_grad = nn.value_and_grad(model, loss_fn)
        initial = float(loss_fn(model).item())
        losses = []
        target.mkdir()
        for _ in range(steps):
            loss, gradients = value_and_grad(model)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(float(loss.item()))
        final = float(loss_fn(model).item())
        adapters = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(target / "adapters.safetensors"), adapters)
        evidence = {
            "format": "modelforge-mlx-lora-v1",
            "runtime": "mlx-lm",
            "device": "gpu",
            "base_path": str(source),
            "rights": rights,
            "authorized": True,
            "steps": steps,
            "quantize_bits": quantize_bits,
            "lora_config": config,
            "initial_loss": initial,
            "final_loss": final,
            "losses": losses,
            "trainable_parameters": sum(value.size for value in adapters.values()),
            "adapter_sha256": hashlib.sha256(
                (target / "adapters.safetensors").read_bytes()
            ).hexdigest(),
            "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_sha256": _hashes(source),
            "peak_mlx_bytes": mx.get_peak_memory(),
            "preflight": report,
            "scope": "Training-set loss only; no held-out quality claim",
        }
        (target / "modelforge-mlx-lora.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        return {"path": str(target), **evidence}


def generate_adapter(
    path: str | Path,
    adapter_path: str | Path,
    prompt: str,
    max_new_tokens: int = 32,
    memory_limit_mb: int = 2048,
) -> dict[str, Any]:
    """Reload an integrity-checked ModelForge MLX LoRA adapter for GPU inference."""
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    root = Path(adapter_path).expanduser().resolve()
    if not root.is_dir() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Adapter must be a local data-only directory without symlinks")
    if set(p.name for p in root.iterdir()) != {
        "adapters.safetensors",
        "modelforge-mlx-lora.json",
    }:
        raise ValueError("Unexpected adapter files")
    evidence = _json(root / "modelforge-mlx-lora.json")
    weights = root / "adapters.safetensors"
    if weights.stat().st_size > 16 * 1024**2:
        raise ValueError("Adapter exceeds the 16 MiB safety budget")
    _headers(weights)
    if evidence.get("format") != "modelforge-mlx-lora-v1" or hashlib.sha256(
        weights.read_bytes()
    ).hexdigest() != evidence.get("adapter_sha256"):
        raise ValueError("Adapter integrity verification failed")
    config = {
        "rank": 4,
        "scale": 8.0,
        "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    }
    if evidence.get("lora_config") != config:
        raise ValueError("Unsupported LoRA topology")
    with _limits(memory_limit_mb, "gpu") as mx:
        model, tokenizer, report = _load(
            path, memory_limit_mb, evidence.get("quantize_bits", 0)
        )
        if _hashes(Path(report["path"])) != evidence.get("source_sha256"):
            raise ValueError("Adapter base fingerprint mismatch")
        model.freeze()
        linear_to_lora_layers(model, len(model.layers), config)
        stored = mx.load(str(weights))
        expected = dict(tree_flatten(model.trainable_parameters()))
        if set(stored) != set(expected) or any(
            stored[key].shape != expected[key].shape for key in stored
        ):
            raise ValueError("Adapter tensor inventory or shapes mismatch")
        model.load_weights(list(stored.items()), strict=False)
        mx.eval(model.parameters())
        result = _generate(model, tokenizer, prompt, max_new_tokens)
        return {
            **result,
            "runtime": "mlx-lm",
            "device": "gpu",
            "adapter_path": str(root),
            "adapter_verified": True,
            "preflight": report,
            "peak_mlx_bytes": mx.get_peak_memory(),
        }
