"""Offline, SafeTensors-only execution of validated GPT-2 and Llama bundles.

These are architecture reimplementations from installed Transformers, not recovery
of source code or a training corpus. No Hub identifier or remote Python is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
import time
from pathlib import Path
from typing import Any

SUPPORTED = {"gpt2", "llama"}
_DTYPES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"Missing or oversized JSON: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def resolve_device(device: str = "auto") -> str:
    """Resolve an installed Torch device, refusing unavailable accelerators."""
    import torch

    if device == "auto":
        for candidate, available in (
            ("cuda", torch.cuda.is_available()),
            ("mps", torch.backends.mps.is_available()),
        ):
            if available:
                try:
                    # Some virtualized hosts advertise MPS but reject even tiny allocations.
                    value = torch.ones((2, 2), device=candidate)
                    float((value @ value).sum().item())
                    return candidate
                except RuntimeError:
                    continue
        return "cpu"
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is unavailable")
    return device


def _headers(path: Path) -> dict[str, list[int]]:
    """Validate offsets and shapes without mapping weight payloads."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated SafeTensors header")
        size = struct.unpack("<Q", prefix)[0]
        if size > 16 * 1024 * 1024 or size > path.stat().st_size - 8:
            raise ValueError("Invalid SafeTensors header size")
        header = json.loads(handle.read(size))
    if not isinstance(header, dict):
        raise ValueError("Invalid SafeTensors header object")
    shapes: dict[str, list[int]] = {}
    intervals: list[tuple[int, int]] = []
    payload = path.stat().st_size - 8 - size
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(spec, dict):
            raise ValueError("Invalid SafeTensors tensor metadata")
        shape, dtype, offsets = (
            spec.get("shape"),
            spec.get("dtype"),
            spec.get("data_offsets"),
        )
        if (
            not isinstance(shape, list)
            or not all(isinstance(x, int) and x >= 0 for x in shape)
            or dtype not in _DTYPES
        ):
            raise ValueError("Invalid SafeTensors shape/dtype")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("Invalid SafeTensors offsets")
        begin, end = offsets
        if (
            not isinstance(begin, int)
            or not isinstance(end, int)
            or begin < 0
            or end > payload
            or end - begin != math.prod(shape) * _DTYPES[dtype]
        ):
            raise ValueError("SafeTensors payload size mismatch")
        intervals.append((begin, end))
        shapes[key] = shape
    cursor = 0
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValueError("SafeTensors offsets overlap or have gaps")
        cursor = end
    if cursor != payload or not shapes:
        raise ValueError("Incomplete SafeTensors payload")
    return shapes


def preflight(
    path: str | Path, memory_limit_mb: int = 2048, device: str = "auto"
) -> dict[str, Any]:
    """Validate a complete local checkpoint and conservatively estimate load RAM.

    The estimate includes float32 weights, staging and 256 MiB inference reserve;
    it is a refusal threshold, not an OS-enforced memory quota.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Expected an existing local model directory")
    if memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive")
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError("Symlinks are unsupported in model bundles")
        if item.suffix.lower() in {".bin", ".pt", ".pth", ".pkl", ".pickle", ".py"}:
            raise ValueError(
                "Unsafe pickle/Python files are unsupported; use a data-only SafeTensors bundle"
            )
    config = _json(root / "config.json")
    if (
        config.get("model_type") not in SUPPORTED
        or config.get("auto_map")
        or config.get("quantization_config")
    ):
        raise ValueError(
            "Only unquantized built-in GPT-2/Llama causal models are supported"
        )
    for key, limit in {
        "num_hidden_layers": 128,
        "n_layer": 128,
        "hidden_size": 16384,
        "n_embd": 16384,
        "vocab_size": 300000,
        "max_position_embeddings": 1048576,
        "n_positions": 1048576,
    }.items():
        if key in config and (
            not isinstance(config[key], int) or not 0 < config[key] <= limit
        ):
            raise ValueError(f"Invalid or excessive config field {key}")
    token_config = _json(root / "tokenizer_config.json")
    if token_config.get("auto_map") or not (root / "tokenizer.json").is_file():
        raise ValueError(
            "A local fast tokenizer.json is required; custom code is unsupported"
        )
    if (root / "tokenizer.json").stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Tokenizer JSON exceeds 16 MiB safety budget")
    index = root / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    if index.exists():
        raw_map = _json(index).get("weight_map")
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("Invalid SafeTensors index")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_map.items()
        ):
            raise ValueError("Invalid SafeTensors index entries")
        weight_map = raw_map
        names = sorted(set(weight_map.values()))
    else:
        names = ["model.safetensors"]
    shapes: dict[str, list[int]] = {}
    disk_bytes = 0
    for name in names:
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Unsafe shard path")
        file = root / name
        if not file.is_file():
            raise ValueError(f"Missing SafeTensors shard: {name}")
        shard_shapes = _headers(file)
        if shapes.keys() & shard_shapes.keys():
            raise ValueError("Duplicate tensors across shards")
        if weight_map and any(weight_map.get(key) != name for key in shard_shapes):
            raise ValueError("Shard/index mismatch")
        shapes.update(shard_shapes)
        disk_bytes += file.stat().st_size
    if weight_map and set(shapes) != set(weight_map):
        raise ValueError("Incomplete checkpoint index")
    parameters = sum(math.prod(shape) for shape in shapes.values())
    estimate = parameters * 4 * 3 + 256 * 1024**2
    if estimate > memory_limit_mb * 1024**2:
        raise ValueError(
            f"Estimated RAM {math.ceil(estimate / 1024**2)} MiB exceeds limit {memory_limit_mb} MiB"
        )
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    hidden = int(cfg.hidden_size)
    layers = int(cfg.num_hidden_layers)
    vocab = int(cfg.vocab_size)
    if config["model_type"] == "gpt2":
        inner = int(cfg.n_inner or 4 * hidden)
        if not 0 < inner <= 131072:
            raise ValueError("Invalid n_inner")
        per_layer = 4 * hidden**2 + 2 * hidden * inner + 9 * hidden + inner
        if cfg.add_cross_attention:
            per_layer += 4 * hidden**2 + 6 * hidden
        declared_parameters = (
            layers * per_layer
            + vocab * hidden
            + int(cfg.n_positions) * hidden
            + 2 * hidden
        )
        if not cfg.tie_word_embeddings:
            declared_parameters += vocab * hidden
    else:
        inner = int(cfg.intermediate_size)
        heads = int(cfg.num_attention_heads)
        kv_heads = int(cfg.num_key_value_heads)
        head_dim = int(getattr(cfg, "head_dim", None) or hidden // heads)
        if (
            not 0 < inner <= 131072
            or not 0 < head_dim <= 65536
            or not 0 < kv_heads <= heads
        ):
            raise ValueError(
                "Invalid intermediate_size, head dimension or KV head count"
            )
        attention = 2 * hidden * head_dim * (heads + kv_heads)
        if getattr(cfg, "attention_bias", False):
            attention += head_dim * (heads + 2 * kv_heads) + hidden
        mlp = 3 * hidden * inner
        if getattr(cfg, "mlp_bias", False):
            mlp += 2 * inner + hidden
        declared_parameters = (
            layers * (attention + mlp + 2 * hidden) + vocab * hidden + hidden
        )
        if not cfg.tie_word_embeddings:
            declared_parameters += vocab * hidden
    declared_estimate = declared_parameters * 12 + 256 * 1024**2
    if declared_estimate > memory_limit_mb * 1024**2:
        raise ValueError(
            "Declared architecture exceeds memory budget before model creation"
        )
    with torch.device("meta"):
        expected = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)
    required = expected.state_dict()
    tied: dict[int, set[str]] = {}
    for name, tensor in expected.named_parameters(remove_duplicate=False):
        tied.setdefault(id(tensor), set()).add(name)
    missing = set(required) - set(shapes)
    for aliases in tied.values():
        if aliases & shapes.keys():
            missing -= aliases
    if missing or set(shapes) - set(required):
        raise ValueError(
            f"Incomplete/incompatible checkpoint: missing={sorted(missing)[:5]}, unexpected={sorted(set(shapes) - set(required))[:5]}"
        )
    for name, shape in shapes.items():
        if list(required[name].shape) != shape:
            raise ValueError(f"Tensor shape mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    token_ids = list(tokenizer.get_vocab().values())
    if not token_ids or min(token_ids) < 0 or max(token_ids) >= cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds model embeddings")
    return {
        "runtime": "transformers",
        "path": str(root),
        "model_type": config["model_type"],
        "device": resolve_device(device),
        "complete_checkpoint": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "weights_format": "safetensors",
        "parameters_stored": parameters,
        "declared_parameters": declared_parameters,
        "weights_bytes": disk_bytes,
        "estimated_memory_mb": math.ceil(estimate / 1024**2),
        "memory_limit_mb": memory_limit_mb,
        "estimated_export_disk_bytes": disk_bytes * 2 + 16 * 1024**2,
        "memory_estimate_is_hard_limit": False,
    }


def load_bundle(
    path: str | Path, memory_limit_mb: int = 2048, device: str = "auto"
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a preflighted complete float32 model and tokenizer entirely offline."""
    report = preflight(path, memory_limit_mb, device)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, loading = AutoModelForCausalLM.from_pretrained(
        report["path"],
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
        attn_implementation="eager",
    )
    if (
        loading.get("missing_keys")
        or loading.get("unexpected_keys")
        or loading.get("mismatched_keys")
        or loading.get("error_msgs")
    ):
        raise ValueError(f"Incomplete checkpoint loading: {loading}")
    try:
        model.to(report["device"]).eval()
    except RuntimeError:
        if device != "auto" or report["device"] == "cpu":
            raise
        model.to("cpu").eval()
        report["device_fallback"] = "Automatic accelerator transfer failed; using CPU"
        report["device"] = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        report["path"], local_files_only=True, trust_remote_code=False, use_fast=True
    )
    modules = list(model.named_modules())
    report["module_tree"] = [
        {
            "name": name or "model",
            "class": type(module).__name__,
            "parent": name.rpartition(".")[0] or "model" if name else None,
        }
        for name, module in modules[:512]
    ]
    report["module_tree_truncated"] = len(modules) > 512
    report["module_tree_scope"] = "Built-in module containment, not an execution graph"
    return model, tokenizer, report


def _inputs(model: Any, tokenizer: Any, prompt: str, extra: int = 0) -> Any:
    if not prompt.strip() or len(prompt.encode()) > 16000:
        raise ValueError("Prompt must be nonempty and at most 16000 UTF-8 bytes")
    # Causal architectures such as Llama do not accept segment/token type IDs.
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
    context = getattr(model.config, "max_position_embeddings", 2048)
    if inputs["input_ids"].shape[-1] + extra > min(context, 4096):
        raise ValueError("Prompt and generation exceed the supported context budget")
    return inputs.to(model.device)


def generate(
    path: str | Path,
    prompt: str,
    max_new_tokens: int = 32,
    memory_limit_mb: int = 2048,
    device: str = "auto",
) -> dict[str, Any]:
    """Greedily generate real token IDs from a validated local causal model."""
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256")
    model, tokenizer, report = load_bundle(path, memory_limit_mb, device)
    import torch

    inputs = _inputs(model, tokenizer, prompt, max_new_tokens)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    ids = output[0].tolist()
    return {
        "output": tokenizer.decode(ids, skip_special_tokens=True),
        "token_ids": ids,
        "new_tokens": len(ids) - inputs["input_ids"].shape[-1],
        "elapsed_seconds": time.perf_counter() - start,
        "preflight": report,
        "scope": "local built-in HF causal language model",
    }


def _hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[str(file.relative_to(path))] = digest.hexdigest()
    return result


def reconstruct(
    source: str | Path,
    destination: str | Path,
    prompts: list[str],
    redistributable: bool = False,
    rights: str = "",
) -> dict[str, Any]:
    """Export authorized safe weights and demonstrate CPU float32 logit parity."""
    if redistributable is not True or not rights.strip():
        raise PermissionError(
            "Export copies imported weights: explicit redistribution authorization and rights statement are required"
        )
    if not prompts or len(prompts) > 32:
        raise ValueError("Provide 1 to 32 parity prompts")
    target = Path(destination).expanduser().absolute()
    if target.exists():
        raise ValueError("Destination must not exist")
    if target.resolve().is_relative_to(Path(source).expanduser().resolve()):
        raise ValueError("Destination must be outside the source bundle")
    report = preflight(source, device="cpu")
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < report["estimated_export_disk_bytes"]:
        raise ValueError("Insufficient free disk for export")
    model, tokenizer, report = load_bundle(source, device="cpu")
    for prompt in prompts:
        _inputs(model, tokenizer, prompt)
    import torch

    with tempfile.TemporaryDirectory(
        prefix=".modelforge-export-", dir=target.parent
    ) as directory:
        staging = Path(directory) / "bundle"
        model.save_pretrained(str(staging), safe_serialization=True)
        tokenizer.save_pretrained(str(staging))
        reference, ref_tokenizer, _ = load_bundle(staging, device="cpu")
        comparisons = []
        with torch.inference_mode():
            for prompt in prompts:
                first = model(**_inputs(model, tokenizer, prompt)).logits.cpu()
                second = reference(
                    **_inputs(reference, ref_tokenizer, prompt)
                ).logits.cpu()
                error = float((first - second).abs().max().item())
                if not torch.equal(first, second):
                    raise ValueError(
                        f"Reconstruction failed exact numeric parity: {error}"
                    )
                comparisons.append(
                    {"prompt": prompt, "max_absolute_logit_error": error, "exact": True}
                )
        evidence = {
            "format": "modelforge-hf-reconstruction-v1",
            "redistribution_authorized": True,
            "rights": rights,
            "source_sha256": _hashes(Path(report["path"])),
            "export_sha256": _hashes(staging),
            "parity": comparisons,
            "parity_device": "cpu",
            "parity_dtype": "float32",
            "scope": "Exact weight serialization and tested-prompt logit parity; no source code or dataset recovery",
        }
        (staging / "modelforge-evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(target)
    return {"path": str(target), **evidence}


def create_tiny_fixture(path: str | Path) -> str:
    """Create an untrained project-owned GPT-2 + real fast tokenizer offline."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    root = Path(path)
    if root.exists():
        raise ValueError("Fixture destination must not exist")
    vocab = {
        word: index
        for index, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[BOS]",
                "[EOS]",
                "hello",
                "world",
                "model",
                "forge",
                "the",
                "is",
                "small",
                "safe",
                "local",
                "test",
                "one",
                "two",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(vocab),
                n_positions=128,
                n_ctx=128,
                n_embd=32,
                n_layer=2,
                n_head=4,
                bos_token_id=2,
                eos_token_id=3,
                pad_token_id=0,
            )
        )
    model.save_pretrained(str(root), safe_serialization=True)
    tokenizer.save_pretrained(str(root))
    return str(root)


def create_owned_fixture(
    destination: str | Path, model_type: str = "gpt2", seed: int = 7
) -> dict[str, Any]:
    """Create project-owned real GPT-2 or Llama weights, without downloads."""
    if model_type not in SUPPORTED:
        raise ValueError("Fixture model_type must be gpt2 or llama")
    import torch
    from transformers import AutoModelForCausalLM, GPT2Config, LlamaConfig

    root = Path(destination)
    create_tiny_fixture(root)
    cfg: Any
    if model_type == "gpt2":
        cfg = GPT2Config(
            vocab_size=16,
            n_positions=128,
            n_ctx=128,
            n_embd=32,
            n_layer=2,
            n_head=4,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )
    else:
        cfg = LlamaConfig(
            vocab_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)
    model.save_pretrained(str(root), safe_serialization=True)
    return {
        "path": str(root),
        "project_owned": True,
        "rights": "Project-owned randomly initialized fixture; MIT",
        "seed": seed,
        "preflight": preflight(root, device="cpu"),
    }
