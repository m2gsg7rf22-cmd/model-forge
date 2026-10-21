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
    from transformers import AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    generation_config = (
        _json(root / "generation_config.json")
        if (root / "generation_config.json").is_file()
        else {}
    )
    eos = generation_config.get(
        "eos_token_id", config.get("eos_token_id", tokenizer.eos_token_id)
    )
    eos_ids = [] if eos is None else eos if isinstance(eos, list) else [eos]
    if len(eos_ids) > 16 or any(
        type(value) is not int or not 0 <= value < config["vocab_size"]
        for value in eos_ids
    ):
        raise ValueError("Invalid configured EOS token IDs")
    tokenizer._modelforge_eos_ids = frozenset(eos_ids)
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
    ids: list[int] = tokenizer.encode(prompt, add_special_tokens=True)
    context = min(int(model.args.max_position_embeddings or 4096), 4096)
    if not ids or len(ids) + extra > context:
        raise ValueError("Prompt and generation exceed the local context budget")
    return ids


def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    *,
    use_cache: bool = False,
) -> dict[str, Any]:
    import mlx.core as mx

    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256")
    ids = _tokens(tokenizer, model, prompt, max_new_tokens)
    generated: list[int] = []
    eos_ids: frozenset[int] = getattr(tokenizer, "_modelforge_eos_ids", frozenset())
    finished = False
    started = time.perf_counter()
    # A fresh per-call cache is bounded by _tokens' 4096-token admission limit.
    # The reference full-prefix implementation remains available for comparison.
    if use_cache:
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(model)
    else:
        cache = None
    for _ in range(max_new_tokens):
        current = (
            (ids if not generated else [generated[-1]])
            if use_cache
            else ids + generated
        )
        inputs = mx.array([current], dtype=mx.int32)
        logits = (model(inputs, cache=cache) if use_cache else model(inputs))[:, -1, :]
        token = int(cast(Any, mx.argmax(logits, axis=-1).item()))
        generated.append(token)
        if token in eos_ids:
            finished = True
            break
    elapsed = time.perf_counter() - started
    return {
        "output": tokenizer.decode(
            generated[:-1] if finished else generated, skip_special_tokens=True
        ),
        "token_ids": generated,
        "prompt_tokens": len(ids),
        "new_tokens": len(generated),
        "seconds": elapsed,
        "tokens_per_second": len(generated) / elapsed,
        "decoding": (
            "greedy bounded KV cache with configured EOS"
            if use_cache
            else "greedy bounded full-prefix with configured EOS"
        ),
        "cache_logical_bytes": (
            sum(value.nbytes for layer in cache for value in layer.state)
            if cache
            else 0
        ),
        "finish_reason": "eos" if finished else "length",
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


def benchmark_cache(
    path: str | Path,
    prompt: str = "hello world",
    max_new_tokens: int = 16,
    memory_limit_mb: int = 2048,
    repeats: int = 3,
    quantize_bits: int = 0,
    precision: str = "native",
    compare_logits: bool = False,
) -> dict[str, Any]:
    """Compare actual fresh bounded KV decode with the existing full-prefix path.

    Both paths use the same loaded model, tokenizer, prompt and greedy budget.
    Order alternates; one warmup per path is excluded. Finite token parity is
    measured, never described as universal or original-weight reconstruction.
    """
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("repeats must be between 1 and 5")
    if precision not in {"native", "float32"} or type(compare_logits) is not bool:
        raise ValueError("Use native/float32 precision and a boolean logit comparison")
    if precision == "float32" and quantize_bits:
        raise ValueError("Float32 control requires an unquantized base")
    with _limits(memory_limit_mb, "gpu") as mx:
        model, tokenizer, report = _load(path, memory_limit_mb, quantize_bits)
        if precision == "float32":
            model.set_dtype(mx.float32)
            mx.eval(model.parameters())
        _tokens(tokenizer, model, prompt, max_new_tokens)
        for cached in (False, True):
            _generate(
                model, tokenizer, prompt, min(2, max_new_tokens), use_cache=cached
            )
        rows = []
        for repeat in range(repeats):
            for cached in ((False, True) if repeat % 2 == 0 else (True, False)):
                mx.reset_peak_memory()
                result = _generate(
                    model, tokenizer, prompt, max_new_tokens, use_cache=cached
                )
                rows.append(
                    {
                        "repeat": repeat,
                        "cached": cached,
                        **result,
                        "peak_mlx_bytes": mx.get_peak_memory(),
                    }
                )
        pairs = [
            [row for row in rows if row["repeat"] == repeat]
            for repeat in range(repeats)
        ]
        parity = all(pair[0]["token_ids"] == pair[1]["token_ids"] for pair in pairs)
        diagnostic = (
            _compare_cache_logits(model, tokenizer, prompt, min(max_new_tokens, 32))
            if compare_logits
            else None
        )
        return {
            "configuration": {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "memory_limit_mb": memory_limit_mb,
                "repeats": repeats,
                "quantize_bits": quantize_bits,
                "precision": precision,
                "compare_logits": compare_logits,
            },
            "precision": precision,
            "reference_prefix_logit_comparison": diagnostic,
            "runtime": "mlx-lm",
            "device": "gpu",
            "preflight": report,
            "measurements": rows,
            "tested_token_parity": parity,
            "cache_scope": "Fresh per-call full KV; no rotation, quantization or cross-conversation reuse; maximum admitted total context 4096 tokens",
            "scope": "Alternating warm same-model full generation including prefill; sampled outer RSS remains separate from MLX allocator peak. Timings are local measurements, not a general speedup or quality claim.",
        }


def _compare_cache_logits(
    model: Any, tokenizer: Any, prompt: str, steps: int
) -> dict[str, Any]:
    """Teacher-force both algorithms on the same prefix to isolate numeric drift."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    ids = _tokens(tokenizer, model, prompt, steps)
    cache = make_prompt_cache(model)
    selected: list[int] = []
    errors: list[float] = []
    margins: list[float] = []
    disagreements: list[int] = []
    eos_ids: frozenset[int] = getattr(tokenizer, "_modelforge_eos_ids", frozenset())
    for index in range(steps):
        reference = model(mx.array([ids + selected], dtype=mx.int32))[:, -1, :]
        inputs = ids if not selected else [selected[-1]]
        cached = model(mx.array([inputs], dtype=mx.int32), cache=cache)[:, -1, :]
        difference = mx.abs(reference.astype(mx.float32) - cached.astype(mx.float32))
        errors.append(float(cast(Any, mx.max(difference).item())))
        ordered = mx.sort(reference.astype(mx.float32), axis=-1)
        margins.append(float(cast(Any, (ordered[0, -1] - ordered[0, -2]).item())))
        token = int(cast(Any, mx.argmax(reference, axis=-1).item()))
        if token != int(cast(Any, mx.argmax(cached, axis=-1).item())):
            disagreements.append(index)
        selected.append(token)
        if token in eos_ids:
            break
    return {
        "steps": len(selected),
        "max_absolute_logit_error": max(errors, default=0.0),
        "per_step_max_absolute_logit_errors": errors,
        "reference_top_two_logit_margins": margins,
        "argmax_disagreement_steps": disagreements,
        "reference_chosen_token_ids": selected,
        "scope": "Same reference-chosen prefix at each step; numerical observations, not universal equivalence proof",
    }
