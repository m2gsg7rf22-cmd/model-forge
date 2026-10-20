"""A bounded, project-owned byte-level causal attention language model.

This runtime reconstructs only its own documented architecture. It never infers
unknown weights from API responses. Training updates the output projection only.
"""

from __future__ import annotations

import json
import hashlib
import math
import shutil
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np

FORMAT = "modelforge-tiny-causal-v1"
WEIGHTS = (
    "embedding",
    "position",
    "query",
    "key",
    "value",
    "attention_out",
    "output",
    "bias",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_file(path: Path, max_bytes: int) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{path.name} exceeds size limit")
    result = json.loads(path.read_text(), object_pairs_hook=_json_object)
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} must contain an object")
    return result


def _offsets(entry: Any) -> tuple[int, int]:
    if not isinstance(entry, dict):
        raise ValueError("Invalid tensor descriptor")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(i) is not int for i in offsets)
    ):
        raise ValueError("Tensor offsets must be two integers")
    return offsets[0], offsets[1]


def _config(d_model: int, context_length: int) -> dict[str, Any]:
    if (
        isinstance(d_model, bool)
        or not isinstance(d_model, int)
        or not 4 <= d_model <= 256
    ):
        raise ValueError("d_model must be an integer between 4 and 256")
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or not 2 <= context_length <= 512
    ):
        raise ValueError("context_length must be an integer between 2 and 512")
    return {
        "model_type": FORMAT,
        "architectures": ["TinyCausalLM"],
        "vocab_size": 256,
        "hidden_size": d_model,
        "max_position_embeddings": context_length,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "tokenizer": "utf8-bytes",
        "activation": "tanh",
        "training_scope": "output_projection_only",
    }


def _shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    d, c = int(config["hidden_size"]), int(config["max_position_embeddings"])
    return {
        "embedding": (256, d),
        "position": (c, d),
        "query": (d, d),
        "key": (d, d),
        "value": (d, d),
        "attention_out": (d, d),
        "output": (d, 256),
        "bias": (256,),
    }


def _guard(config: dict[str, Any], memory_limit_mb: float) -> int:
    if not math.isfinite(memory_limit_mb) or memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive and finite")
    d, c = int(config["hidden_size"]), int(config["max_position_embeddings"])
    # Includes float64 NumPy temporaries, gradients, attention matrices, and logits.
    estimate = (
        sum(math.prod(s) for s in _shapes(config).values()) * 32
        + c * c * 32
        + c * (d + 256) * 32
    )
    if estimate > memory_limit_mb * 1024**2:
        raise MemoryError(
            f"Estimated working memory {estimate / 1024**2:.2f} MiB exceeds limit {memory_limit_mb} MiB"
        )
    return estimate


def _write_tensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, Any] = {}
    chunks = []
    offset = 0
    for name, arr in sorted(tensors.items()):
        arr = np.ascontiguousarray(arr)
        dtype = "I8" if arr.dtype == np.int8 else "F32"
        arr = arr.astype(np.int8 if dtype == "I8" else "<f4", copy=False)
        payload = arr.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        chunks.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for chunk in chunks:
            stream.write(chunk)


def _read_tensors(
    path: Path, expected: dict[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    # Strict bounded reader for this runtime's own bundles, not a generic importer.
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Tiny model weight file exceeds 64 MiB hard limit")
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("Truncated SafeTensors file")
    size = struct.unpack("<Q", data[:8])[0]
    if size > 1024**2 or size + 8 > len(data):
        raise ValueError("Invalid SafeTensors header size")
    header = json.loads(data[8 : 8 + size], object_pairs_hook=_json_object)
    if not isinstance(header, dict):
        raise ValueError("SafeTensors header must be an object")
    payload = memoryview(data)[8 + size :]
    allowed = set(expected) | {name + ".scale" for name in expected}
    if set(header) - allowed or not set(expected).issubset(header):
        raise ValueError("Unexpected or missing model tensors")
    result = {}
    ranges = []
    for name, shape in expected.items():
        entry = header[name]
        dtype_map: dict[str, np.dtype[Any]] = {
            "F32": np.dtype("<f4"),
            "I8": np.dtype("int8"),
        }
        start, stop = _offsets(entry)
        dtype = dtype_map.get(str(entry.get("dtype")))
        if (
            dtype is None
            or entry.get("shape") != list(shape)
            or not 0 <= start <= stop <= len(payload)
        ):
            raise ValueError(f"Invalid tensor {name}")
        if stop - start != math.prod(shape) * dtype.itemsize:
            raise ValueError(f"Invalid tensor size for {name}")
        ranges.append((start, stop))
        arr = (
            np.frombuffer(payload[start:stop], dtype=dtype)
            .reshape(shape)
            .astype(np.float32)
        )
        if dtype == np.dtype("int8"):
            scale = header.get(name + ".scale", {})
            a, b = _offsets(scale)
            if (
                scale.get("dtype") != "F32"
                or scale.get("shape") != [1]
                or not 0 <= a < b <= len(payload)
                or b - a != 4
            ):
                raise ValueError("Invalid quantization scale")
            ranges.append((a, b))
            value = float(np.frombuffer(payload[a:b], dtype="<f4")[0])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid quantization scale value")
            arr *= value
        elif name + ".scale" in header:
            raise ValueError("Unexpected scale for float tensor")
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite model weights")
        result[name] = arr
    ranges.sort()
    if (
        not ranges
        or ranges[0][0] != 0
        or ranges[-1][1] != len(payload)
        or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:]))
    ):
        raise ValueError("Overlapping, missing, or extra tensor data")
    return result


class TinyCausalLM:
    def __init__(
        self,
        config: dict[str, Any],
        weights: dict[str, np.ndarray],
        provenance: dict[str, Any],
    ):
        self.config, self.weights, self.provenance = config, weights, provenance

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    def _features(self, token_ids: list[int]) -> np.ndarray:
        ids = np.asarray(token_ids)
        if (
            ids.ndim != 1
            or not len(ids)
            or len(ids) > self.config["max_position_embeddings"]
        ):
            raise ValueError(
                "Expected a nonempty token sequence within the context limit"
            )
        if (
            not np.issubdtype(ids.dtype, np.integer)
            or (ids < 0).any()
            or (ids > 255).any()
        ):
            raise ValueError("Byte tokens must be integers in [0, 255]")
        w = self.weights
        x = w["embedding"][ids] + w["position"][: len(ids)]
        q, k, v = (x @ w[name] for name in ("query", "key", "value"))
        scores = (q @ k.T) / math.sqrt(x.shape[1])
        scores[np.triu_indices(len(ids), 1)] = -np.inf
        scores -= scores.max(axis=1, keepdims=True)
        attention = np.exp(scores)
        attention /= attention.sum(axis=1, keepdims=True)
        return cast(np.ndarray, np.tanh(x + attention @ v @ w["attention_out"]))

    def logits(self, token_ids: list[int]) -> np.ndarray:
        return cast(
            np.ndarray,
            self._features(token_ids) @ self.weights["output"] + self.weights["bias"],
        )

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 0 <= max_new_tokens <= 4096
        ):
            raise ValueError("max_new_tokens must be an integer in [0, 4096]")
        if len(prompt) > 1024**2:
            raise ValueError("Prompt exceeds 1 MiB character limit")
        ids = self.encode(prompt) or [10]
        generated = []
        for _ in range(max_new_tokens):
            token = int(
                self.logits(ids[-self.config["max_position_embeddings"] :])[-1].argmax()
            )
            ids.append(token)
            generated.append(token)
        return self.decode(generated)

    def train(
        self, text: str, steps: int = 100, learning_rate: float = 0.2
    ) -> dict[str, Any]:
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or not 1 <= steps <= 10000
        ):
            raise ValueError("steps must be an integer in [1, 10000]")
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 10:
            raise ValueError("learning_rate must be finite in (0, 10]")
        if len(text) > 1024**2:
            raise ValueError("Training corpus exceeds 1 MiB limit")
        ids = self.encode(text)
        if len(ids) < 2:
            raise ValueError("Training needs at least two UTF-8 bytes")
        # A bounded context window is intentional; train each selected window repeatedly.
        ids = ids[: self.config["max_position_embeddings"] + 1]
        features, targets = self._features(ids[:-1]), np.asarray(ids[1:])
        losses = []
        for step in range(steps + 1):
            logits = features @ self.weights["output"] + self.weights["bias"]
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            loss = float(
                -np.log(
                    np.maximum(probs[np.arange(len(targets)), targets], 1e-30)
                ).mean()
            )
            if step in (0, steps):
                losses.append(loss)
            if step == steps:
                break
            probs[np.arange(len(targets)), targets] -= 1
            probs /= len(targets)
            self.weights["output"] -= learning_rate * (features.T @ probs)
            self.weights["bias"] -= learning_rate * probs.sum(axis=0)
        report = {
            "steps": steps,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "trained_parameters": "output,bias",
            "training_bytes": len(ids),
            "corpus_truncated": len(self.encode(text)) > len(ids),
            "learning_rate": learning_rate,
        }
        self.provenance.setdefault("training", []).append(report)
        return report

    def save(self, output_dir: str | Path, quantize: bool = False) -> dict[str, Any]:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        required = sum(a.nbytes for a in self.weights.values()) + 2 * 1024**2
        if shutil.disk_usage(path).free < required:
            raise OSError("Insufficient disk space to export model")
        tensors = {}
        for name, arr in self.weights.items():
            if quantize:
                scale = max(float(np.abs(arr).max()) / 127, 1e-12)
                tensors[name] = np.rint(arr / scale).clip(-127, 127).astype(np.int8)
                tensors[name + ".scale"] = np.asarray([scale], dtype=np.float32)
            else:
                tensors[name] = arr.astype(np.float32)
        _write_tensors(path / "model.safetensors", tensors)
        config = {
            **self.config,
            "quantization": "symmetric-int8-per-tensor" if quantize else None,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))
        (path / "tokenizer.json").write_text(
            json.dumps({"type": "utf8-bytes", "vocab_size": 256, "normalization": None})
        )
        provenance = {
            **self.provenance,
            "export_quantized": quantize,
            "fidelity": (
                "lossy quantized project-owned model"
                if quantize
                else "exact stored project-owned weights; validate numerical parity separately"
            ),
        }
        (path / "README.md").write_text(
            "# ModelForge TinyCausalLM\n\nIndependent project-owned byte language model. "
            "One causal self-attention layer, residual tanh, learned positions and output projection. "
            "This is not a reconstruction of any proprietary model.\n\n"
            "Training updates only the output projection and bias on the first context window. "
            "Suitable for runtime verification and small experiments, not general language tasks. "
            "No LoRA, QLoRA, CUDA or MLX backend is implemented. INT8 export dequantizes to CPU float32 for inference.\n\n"
            f"Rights/source: {provenance.get('rights', 'Unspecified; review before redistribution')}\n\n"
            f"Fidelity: {provenance['fidelity']}\n\n"
            "Runtime: `from runtime import load_model; model = load_model('.'); print(model.generate('hello'))`\n"
            "Requires Python 3.11+ and NumPy. See provenance.json for training history.\n"
        )
        if Path(__file__).resolve() != (path / "runtime.py").resolve():
            shutil.copyfile(Path(__file__), path / "runtime.py")
        provenance["artifact_sha256"] = {
            name: _sha256(path / name)
            for name in (
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "runtime.py",
            )
        }
        (path / "provenance.json").write_text(json.dumps(provenance, indent=2))
        return {
            "path": str(path.resolve()),
            "format": FORMAT,
            "parameters": sum(a.size for a in self.weights.values()),
            "quantized": quantize,
            "provenance": provenance,
        }


def create_model(
    output_dir: str | Path,
    *,
    seed: int = 7,
    d_model: int = 32,
    context_length: int = 64,
    memory_limit_mb: float = 256,
    rights: str = "Project-owned synthetic weights, MIT",
) -> dict[str, Any]:
    config = _config(d_model, context_length)
    estimate = _guard(config, memory_limit_mb)
    rng = np.random.default_rng(seed)
    weights = {
        name: (
            rng.normal(
                0,
                0.15 if name in ("embedding", "position") else 1 / math.sqrt(d_model),
                shape,
            )
        ).astype(np.float32)
        for name, shape in _shapes(config).items()
    }
    weights["bias"].fill(0)
    model = TinyCausalLM(
        config,
        weights,
        {
            "origin": "independent seeded initialization",
            "seed": seed,
            "rights": rights,
            "source_model": None,
        },
    )
    return {**model.save(output_dir), "estimated_working_memory_bytes": estimate}


def load_model(path: str | Path, memory_limit_mb: float = 256) -> TinyCausalLM:
    path = Path(path)
    config_path = path / "config.json"
    config = _json_file(config_path, 65536)
    if config.get("model_type") != FORMAT:
        raise ValueError(
            "Unsupported architecture; only ModelForge TinyCausalLM can be loaded"
        )
    canonical = _config(
        cast(int, config.get("hidden_size")),
        cast(int, config.get("max_position_embeddings")),
    )
    if any(config.get(key) != value for key, value in canonical.items()):
        raise ValueError("Config contradicts the supported architecture")
    _guard(canonical, memory_limit_mb)
    provenance_path = path / "provenance.json"
    provenance = _json_file(provenance_path, 1024**2)
    tokenizer = _json_file(path / "tokenizer.json", 65536)
    if tokenizer != {"type": "utf8-bytes", "vocab_size": 256, "normalization": None}:
        raise ValueError("Unsupported tokenizer")
    weights = _read_tensors(path / "model.safetensors", _shapes(config))
    hashes = provenance.get("artifact_sha256", {})
    if not isinstance(hashes, dict):
        raise ValueError("Invalid artifact hashes")
    # Check only fixed bundle paths, never provenance-provided paths. Runtime is hashed,
    # never imported by the loader. These hashes detect corruption, not authenticity.
    for name in ("config.json", "model.safetensors", "tokenizer.json", "runtime.py"):
        if name in hashes and _sha256(path / name) != hashes[name]:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return TinyCausalLM(
        config,
        weights,
        provenance,
    )


def train_model(
    path: str | Path,
    text: str,
    *,
    steps: int = 100,
    learning_rate: float = 0.2,
    output_dir: str | Path | None = None,
    memory_limit_mb: float = 256,
) -> dict[str, Any]:
    model = load_model(path, memory_limit_mb)
    report = model.train(text, steps, learning_rate)
    return {"training": report, "export": model.save(output_dir or path)}


def quantize_model(path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    if Path(path).resolve() == Path(output_dir).resolve():
        raise ValueError("Quantized export must use a separate directory")
    return load_model(path).save(output_dir, quantize=True)


def reconstruct_model(
    path: str | Path, output_dir: str | Path, memory_limit_mb: float = 256
) -> dict[str, Any]:
    """Rebuild a supported own-architecture bundle and test finite CPU parity.

    Quantized inputs are reconstructed as their dequantized float32 tensors.
    Original pre-quantization weights cannot be recovered from those inputs.
    """
    if Path(path).resolve() == Path(output_dir).resolve():
        raise ValueError("Reconstruction must use a separate output directory")
    model = load_model(path, memory_limit_mb)
    model.provenance["reconstruction"] = {
        "scope": "supported project fixture architecture and accessible tensors",
        "source_weights_sha256": _sha256(Path(path) / "model.safetensors"),
        "source_config_sha256": _sha256(Path(path) / "config.json"),
        "source_was_quantized": bool(model.config.get("quantization")),
    }
    result = model.save(output_dir)
    return {"export": result, "parity": compare_models(path, output_dir)}


def compare_models(
    path_a: str | Path, path_b: str | Path, prompts: list[str] | None = None
) -> dict[str, Any]:
    a, b = load_model(path_a), load_model(path_b)
    prompts = prompts or ["ModelForge", "attention", "test"]
    rows = []
    for prompt in prompts:
        ids = a.encode(prompt) or [10]
        ids = ids[
            -min(
                a.config["max_position_embeddings"], b.config["max_position_embeddings"]
            ) :
        ]
        x, y = a.logits(ids), b.logits(ids)
        rows.append(
            {
                "prompt": prompt,
                "max_abs_logit_error": float(np.max(np.abs(x - y))),
                "exact_logits": bool(np.array_equal(x, y)),
                "same_greedy_tokens": bool(np.array_equal(x.argmax(-1), y.argmax(-1))),
            }
        )
    return {
        "scope": "finite prompt numerical comparison, not universal behavioral equivalence",
        "exact_on_all_tested_prompts": all(r["exact_logits"] for r in rows),
        "results": rows,
    }
