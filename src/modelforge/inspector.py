"""Bounded, metadata-only model inspection; never execute model-provided code."""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path
from typing import Any, BinaryIO

MAX_HEADER = 32 * 1024 * 1024
MAX_TENSORS = 100_000
MAX_DIMENSIONS = 16
MAX_ELEMENTS = 1 << 64
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "I16": 16,
    "U16": 16,
    "I32": 32,
    "U32": 32,
    "I64": 64,
    "U64": 64,
    "F16": 16,
    "BF16": 16,
    "F32": 32,
    "F64": 64,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E8M0": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U4": 4,
    "I4": 4,
}
# (name, elements per block, bytes per block). Unknown future types remain explicit.
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
}


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid JSON metadata") from exc
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _count(shape: Any) -> int:
    if not isinstance(shape, list) or len(shape) > MAX_DIMENSIONS:
        raise ValueError("Invalid tensor shape or too many dimensions")
    if any(type(n) is not int or n < 0 or n > MAX_ELEMENTS for n in shape):
        raise ValueError("Invalid tensor dimension")
    count = math.prod(shape)
    if count > MAX_ELEMENTS:
        raise ValueError("Tensor exceeds element limit")
    return count


def _evidence(
    report: dict[str, Any],
    claim: str,
    value: Any,
    source: str,
    certainty: str = "observed",
) -> str:
    identifier = f"e{len(report['evidence']) + 1}"
    report["evidence"].append(
        {
            "id": identifier,
            "claim": claim,
            "value": value,
            "source": source,
            "certainty": certainty,
        }
    )
    return identifier


def _base(path: Path, fmt: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": str(path),
        "format": fmt,
        "status": "inspected",
        "metadata": {},
        "tensors": [],
        "evidence": [],
        "architecture": {
            "nodes": [],
            "edges": [],
            "properties": {},
            "summary": "Unknown",
        },
        "parameters": {"total": None, "kind": "unknown"},
        "limitations": [
            "Metadata is a claim of the artifact; numerical behavior and provenance are not verified.",
            "Tensor inventories include buffers and may count tied storage more than once.",
        ],
        "reconstruction": {
            "level": "metadata_only",
            "exact": False,
            "reason": "Inspection does not prove computational equivalence.",
        },
    }


def _safe(path: Path) -> dict[str, Any]:
    report = _base(path, "safetensors")
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated SafeTensors length")
        length = struct.unpack("<Q", prefix)[0]
        if length < 2 or length > MAX_HEADER or length > size - 8:
            raise ValueError("Invalid or oversized SafeTensors header")
        raw = handle.read(length)
    if not raw.startswith(b"{"):
        raise ValueError("SafeTensors header must start with an object")
    header = _json(raw)
    if len(header) > MAX_TENSORS + 1:
        raise ValueError("Too many tensors")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(v, str) for v in metadata.values()
    ):
        raise ValueError("SafeTensors metadata must contain strings")
    report["metadata"] = metadata
    ranges = []
    data_size = size - 8 - length
    for name, entry in header.items():
        if not isinstance(entry, dict):
            raise ValueError("Invalid tensor descriptor")
        shape, dtype, offsets = (
            entry.get("shape"),
            entry.get("dtype"),
            entry.get("data_offsets"),
        )
        count = _count(shape)
        if not isinstance(dtype, str) or dtype not in DTYPE_BITS:
            raise ValueError(f"Unsupported SafeTensors dtype: {dtype}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
        ):
            raise ValueError("Invalid tensor offsets")
        start, end = offsets
        bits = count * DTYPE_BITS[dtype]
        if bits % 8 or not 0 <= start <= end <= data_size or end - start != bits // 8:
            raise ValueError(f"Invalid tensor byte range: {name}")
        ranges.append((start, end))
        eid = _evidence(
            report, f"Tensor descriptor: {name}", entry, f"{path.name}:header/{name}"
        )
        report["tensors"].append(
            {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "parameters": count,
                "data_offsets": offsets,
                "evidence_ids": [eid],
            }
        )
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("Tensor data contains overlapping ranges or gaps")
        cursor = end
    if cursor != data_size:
        raise ValueError("Unindexed SafeTensors data")
    report["parameters"] = {
        "total": sum(t["parameters"] for t in report["tensors"]),
        "kind": "stored_tensor_elements",
    }
    return report


class _GGUFReader:
    """Read GGUF descriptors with a global metadata allocation budget."""

    def __init__(self, handle: BinaryIO, size: int) -> None:
        self.handle = handle
        self.size = size

    def read(self, length: int) -> bytes:
        if length < 0 or self.handle.tell() + length > min(self.size, MAX_HEADER):
            raise ValueError("GGUF metadata is truncated or exceeds the 32 MiB budget")
        data = self.handle.read(length)
        if len(data) != length:
            raise ValueError("Truncated GGUF")
        return data

    def number(self, fmt: str) -> Any:
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self) -> str:
        length = self.number("Q")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Invalid UTF-8 in GGUF") from exc

    def value(self, kind: int, depth: int = 0) -> Any:
        formats = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }
        if kind in formats:
            value = self.number(formats[kind])
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite GGUF metadata")
            return value
        if kind == 8:
            return self.string()
        if kind == 9:
            if depth:
                raise ValueError("Nested GGUF metadata arrays are not supported")
            element_kind, count = self.number("I"), self.number("Q")
            if count > 1_000_000:
                raise ValueError("GGUF array exceeds element limit")
            values = []
            for i in range(count):
                value = self.value(element_kind, depth + 1)
                if i < 64:
                    values.append(value)
            return (
                values
                if count <= 64
                else {"count": count, "preview": values, "truncated": True}
            )
        raise ValueError(f"Unknown GGUF metadata type {kind}")


def _gguf(path: Path) -> dict[str, Any]:
    report = _base(path, "gguf")
    size = path.stat().st_size
    with path.open("rb") as handle:
        reader = _GGUFReader(handle, size)
        if reader.read(4) != b"GGUF":
            raise ValueError("Invalid GGUF magic")
        version = reader.number("I")
        if version not in (2, 3):
            raise ValueError("Only little-endian GGUF versions 2 and 3 are supported")
        nt, nkv = reader.number("Q"), reader.number("Q")
        if nt > MAX_TENSORS or nkv > MAX_TENSORS:
            raise ValueError("GGUF descriptor count exceeds limit")
        metadata: dict[str, Any] = {}
        for _ in range(nkv):
            key, kind = reader.string(), reader.number("I")
            if key in metadata:
                raise ValueError("Duplicate GGUF metadata key")
            metadata[key] = reader.value(kind)
        report["metadata"] = metadata
        report["metadata"]["gguf_version"] = version
        seen = set()
        for _ in range(nt):
            name, nd = reader.string(), reader.number("I")
            if name in seen or nd > MAX_DIMENSIONS or nd == 0:
                raise ValueError("Duplicate tensor name or invalid GGUF rank")
            seen.add(name)
            shape = [reader.number("Q") for _ in range(nd)]
            count = _count(shape)
            if not count:
                raise ValueError("Empty GGUF tensor")
            kind, offset = reader.number("I"), reader.number("Q")
            dtype, block, block_bytes = GGML_TYPES.get(
                kind, (f"GGML_TYPE_{kind}", 0, 0)
            )
            if block and shape[0] % block:
                raise ValueError("GGUF quantized row does not align to block size")
            length = count // block * block_bytes if block else None
            eid = _evidence(
                report,
                f"Tensor descriptor: {name}",
                {"shape": shape, "dtype": dtype, "offset": offset},
                f"{path.name}:tensor_info/{name}",
            )
            report["tensors"].append(
                {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "parameters": count,
                    "offset": offset,
                    "nbytes": length,
                    "evidence_ids": [eid],
                }
            )
        alignment = metadata.get("general.alignment", 32)
        if (
            type(alignment) is not int
            or not 1 <= alignment <= 65536
            or alignment & (alignment - 1)
        ):
            raise ValueError("Invalid GGUF alignment")
        start = (handle.tell() + alignment - 1) // alignment * alignment
        intervals = []
        for tensor in report["tensors"]:
            offset, length = tensor["offset"], tensor["nbytes"]
            if offset % alignment or start + offset >= size:
                raise ValueError("GGUF tensor offset is misaligned or outside the file")
            if length is not None:
                if start + offset + length > size:
                    raise ValueError("Truncated GGUF tensor data")
                intervals.append((offset, offset + length))
            else:
                report["limitations"].append(
                    f"Byte extent not validated for {tensor['dtype']}."
                )
        end = 0
        for lo, hi in sorted(intervals):
            if lo < end:
                raise ValueError("Overlapping GGUF tensors")
            end = hi
    report["parameters"] = {
        "total": sum(t["parameters"] for t in report["tensors"]),
        "kind": "stored_tensor_elements",
    }
    report["limitations"].append(
        "GGUF dimensions are reported in GGML order (fastest dimension first)."
    )
    return report


def _config(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_HEADER:
        raise ValueError("Config exceeds metadata size limit")
    report = _base(path, "huggingface_config")
    report["metadata"] = _json(path.read_bytes())
    report["limitations"].append(
        "Configuration alone contains no weights; parameter count is unknown."
    )
    for key, value in report["metadata"].items():
        _evidence(report, f"Configuration field: {key}", value, f"{path.name}:/{key}")
    return report


def _onnx(path: Path) -> dict[str, Any]:
    report = _base(path, "onnx")
    if path.stat().st_size > MAX_HEADER:
        report["status"] = "unsupported"
        report["limitations"].append(
            "ONNX protobuf exceeds the 32 MiB metadata inspection limit."
        )
        return report
    try:
        import onnx
    except ImportError:
        report["status"] = "unsupported"
        report["limitations"].append(
            "Install the optional onnx dependency for protobuf inspection."
        )
        return report
    model = onnx.load(str(path), load_external_data=False)
    if (
        len(model.graph.node) > MAX_TENSORS
        or len(model.graph.initializer) > MAX_TENSORS
    ):
        raise ValueError("ONNX graph exceeds inspection limit")
    report["metadata"] = {
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "opsets": [
            {"domain": x.domain, "version": x.version} for x in model.opset_import
        ],
    }
    for tensor in model.graph.initializer:
        shape = list(tensor.dims)
        eid = _evidence(
            report,
            f"Initializer: {tensor.name}",
            shape,
            f"{path.name}:graph/initializer",
        )
        report["tensors"].append(
            {
                "name": tensor.name,
                "shape": shape,
                "dtype": onnx.TensorProto.DataType.Name(tensor.data_type),
                "parameters": _count(shape),
                "evidence_ids": [eid],
            }
        )
    producer: dict[str, str] = {}
    for i, node in enumerate(model.graph.node):
        node_id = f"op{i}"
        eid = _evidence(
            report, "ONNX graph operator", node.op_type, f"{path.name}:graph/node/{i}"
        )
        report["architecture"]["nodes"].append(
            {
                "id": node_id,
                "label": node.name or node.op_type,
                "type": node.op_type,
                "certainty": "observed",
                "evidence_ids": [eid],
            }
        )
        for output in node.output:
            producer[output] = node_id
    for i, node in enumerate(model.graph.node):
        for inp in node.input:
            if inp in producer:
                report["architecture"]["edges"].append(
                    {
                        "source": producer[inp],
                        "target": f"op{i}",
                        "label": inp,
                        "certainty": "observed",
                    }
                )
    report["architecture"]["summary"] = "Declared ONNX computation graph"
    report["parameters"] = {
        "total": sum(t["parameters"] for t in report["tensors"]),
        "kind": "dense_initializer_elements",
    }
    report["limitations"].append(
        "External data is not opened; sparse initializers and subgraphs are not inventoried."
    )
    return report


def _coreml(path: Path) -> dict[str, Any]:
    report = _base(path, "coreml")
    if path.suffix.lower() != ".mlmodel" or path.stat().st_size > MAX_HEADER:
        report["status"] = "unsupported"
        report["limitations"].append(
            "Only standalone .mlmodel protobufs up to 32 MiB are inspected; packages are unsupported."
        )
        return report
    try:
        from coremltools.models.utils import load_spec
    except ImportError:
        report["status"] = "unsupported"
        report["limitations"].append(
            "Install optional coremltools for Core ML spec inspection."
        )
        return report
    spec = load_spec(str(path))
    kind = spec.WhichOneof("Type")
    report["metadata"] = {
        "specification_version": spec.specificationVersion,
        "model_type": kind,
        "inputs": [x.name for x in spec.description.input],
        "outputs": [x.name for x in spec.description.output],
    }
    eid = _evidence(report, "Declared Core ML model type", kind, f"{path.name}:Type")
    report["architecture"]["nodes"] = [
        {
            "id": "model",
            "label": kind,
            "type": kind,
            "certainty": "observed",
            "evidence_ids": [eid],
        }
    ]
    report["architecture"]["summary"] = "Core ML model interface"
    report["limitations"].append(
        "Core ML tensor inventory and inference are not implemented; no MLModel runtime is created."
    )
    return report


def _architecture(report: dict[str, Any]) -> None:
    if report["architecture"]["nodes"]:
        return
    metadata = report["metadata"]
    family = metadata.get(
        "model_type",
        metadata.get("general.architecture", metadata.get("_class_name", "unknown")),
    )
    report["architecture"][
        "summary"
    ] = f"Artifact declares {family}; graph is a structural inventory, not execution order."
    properties = report["architecture"]["properties"]
    field_names = {
        "num_hidden_layers": "layers",
        "n_layer": "layers",
        "hidden_size": "hidden_size",
        "n_embd": "hidden_size",
        "num_attention_heads": "attention_heads",
        "num_key_value_heads": "key_value_heads",
        "intermediate_size": "feed_forward_size",
        "hidden_act": "activation",
        "activation_function": "activation",
        "vocab_size": "vocabulary_size",
        "max_position_embeddings": "context_length",
        "rope_theta": "rotary_base",
        "rope_scaling": "rotary_scaling",
        "num_local_experts": "experts",
        "num_experts_per_tok": "experts_per_token",
        "vision_config": "vision_config",
        "text_config": "text_config",
        "audio_config": "audio_config",
        "architectures": "architectures",
    }
    if isinstance(family, str):
        field_names.update(
            {
                f"{family}.block_count": "layers",
                f"{family}.embedding_length": "hidden_size",
                f"{family}.attention.head_count": "attention_heads",
                f"{family}.context_length": "context_length",
                f"{family}.feed_forward_length": "feed_forward_size",
            }
        )
    for key, label in field_names.items():
        if key in metadata:
            eid = _evidence(
                report,
                f"Declared {label}",
                metadata[key],
                f"artifact metadata:/{key}",
                "declared",
            )
            properties[label] = {
                "value": metadata[key],
                "certainty": "declared",
                "evidence_ids": [eid],
            }
    nodes = report["architecture"]["nodes"]
    edges = report["architecture"]["edges"]
    nodes.append(
        {
            "id": "model",
            "label": str(family),
            "type": "model",
            "certainty": "declared",
            "evidence_ids": [
                _evidence(
                    report,
                    "Declared model family",
                    family,
                    "artifact metadata",
                    "declared",
                )
            ],
        }
    )
    for tensor in report["tensors"]:
        name = tensor["name"]
        lower = name.lower()
        category = "tensor"
        for pattern, label in [
            (r"embed|token_embd|wte", "embedding"),
            (r"attn|attention|q_proj|k_proj|v_proj", "attention"),
            (r"expert", "mixture_of_experts"),
            (r"mlp|feed_forward|ffn", "feed_forward"),
            (r"norm", "normalization"),
            (r"vision|patch_embed", "vision"),
            (r"lm_head|output", "output"),
        ]:
            if re.search(pattern, lower):
                category = label
                break
        nid = f"tensor{len(nodes)}"
        nodes.append(
            {
                "id": nid,
                "label": name,
                "type": category,
                "shape": tensor["shape"],
                "certainty": "inferred" if category != "tensor" else "observed",
                "evidence_ids": tensor["evidence_ids"],
            }
        )
        edges.append(
            {
                "source": "model",
                "target": nid,
                "label": "contains",
                "certainty": "observed",
            }
        )
    report["limitations"].append(
        "Layer categories inferred from tensor names do not prove connectivity, attention mechanism, or activation function."
    )


def inspect_model(path: str) -> dict[str, Any]:
    """Inspect an accessible local model without executing it or loading weight arrays.

    Directories combine config.json with all top-level SafeTensors shards. Missing
    shards listed in an index are rejected. Unsafe checkpoints are always refused.
    """
    target = Path(path).expanduser().resolve(strict=True)
    if target.is_dir() and target.suffix.lower() != ".mlpackage":
        config = target / "config.json"
        shards = sorted(target.glob("*.safetensors"))
        if len(shards) > 1024:
            raise ValueError("Too many model shards")
        index = target / "model.safetensors.index.json"
        expected: dict[str, Any] = {}
        if index.exists():
            if index.stat().st_size > MAX_HEADER:
                raise ValueError("Shard index exceeds metadata budget")
            expected = _json(index.read_bytes()).get("weight_map", {})
            if not isinstance(expected, dict) or any(
                not isinstance(x, str) for x in expected.values()
            ):
                raise ValueError("Invalid SafeTensors shard index")
            names = {p.name for p in shards}
            if any(value not in names for value in expected.values()):
                raise ValueError("Missing or nonlocal shard in SafeTensors index")
        header_budget = 0
        for shard in shards:
            if not shard.is_file():
                raise ValueError("Shard must be a regular file")
            with shard.open("rb") as handle:
                prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated SafeTensors shard")
            header_budget += struct.unpack("<Q", prefix)[0]
            if header_budget > MAX_HEADER:
                raise ValueError("Combined shard headers exceed metadata budget")
        if not config.is_file() and not shards:
            raise ValueError("Directory has no config.json or SafeTensors models")
        report = (
            _config(config)
            if config.is_file()
            else _base(target, "safetensors_directory")
        )
        report["source"] = str(target)
        report["format"] = "huggingface_directory"
        seen: set[str] = set()
        for shard in shards:
            child = _safe(shard)
            mapping = {}
            for ev in child["evidence"]:
                old = ev["id"]
                ev["id"] = f"e{len(report['evidence']) + 1}"
                mapping[old] = ev["id"]
                report["evidence"].append(ev)
            for tensor in child["tensors"]:
                name = tensor["name"]
                if name in seen:
                    raise ValueError(f"Duplicate tensor across shards: {name}")
                if expected and expected.get(name) != shard.name:
                    raise ValueError(f"Tensor does not match shard index: {name}")
                seen.add(name)
                tensor["evidence_ids"] = [mapping[x] for x in tensor["evidence_ids"]]
                tensor["shard"] = shard.name
                report["tensors"].append(tensor)
            if len(seen) > MAX_TENSORS:
                raise ValueError("Combined tensor inventory exceeds limit")
        if expected and seen != set(expected):
            raise ValueError("Shard index references missing tensors")
        tokenizer_summary: dict[str, Any] = {}
        for filename in ("tokenizer_config.json", "tokenizer.json"):
            tokenizer_path = target / filename
            if tokenizer_path.is_file():
                if tokenizer_path.stat().st_size > MAX_HEADER:
                    report["limitations"].append(
                        f"{filename} exceeds tokenizer metadata budget; skipped."
                    )
                    continue
                token_data = _json(tokenizer_path.read_bytes())
                if filename == "tokenizer_config.json":
                    for key in (
                        "tokenizer_class",
                        "model_max_length",
                        "bos_token",
                        "eos_token",
                        "unk_token",
                        "pad_token",
                    ):
                        if key in token_data:
                            tokenizer_summary[key] = token_data[key]
                else:
                    token_model = token_data.get("model", {})
                    if isinstance(token_model, dict):
                        tokenizer_summary["algorithm"] = token_model.get(
                            "type", "unknown"
                        )
                        vocab = token_model.get("vocab")
                        if isinstance(vocab, (dict, list)):
                            tokenizer_summary["vocabulary_entries"] = len(vocab)
                _evidence(
                    report,
                    "Tokenizer metadata summary",
                    dict(tokenizer_summary),
                    filename,
                    "declared",
                )
        if tokenizer_summary:
            report["metadata"]["tokenizer"] = tokenizer_summary
            report["limitations"].append(
                "Tokenizer metadata is inspected; tokenization equivalence is not verified."
            )
        if shards:
            report["parameters"] = {
                "total": sum(t["parameters"] for t in report["tensors"]),
                "kind": "stored_tensor_elements",
            }
            report["limitations"] = [
                x
                for x in report["limitations"]
                if not x.startswith("Configuration alone")
            ]
        _architecture(report)
        return report
    if not target.is_file() and target.suffix.lower() != ".mlpackage":
        raise ValueError("Model source must be a regular file or supported directory")
    suffix = target.suffix.lower()
    if suffix in {".pt", ".pth", ".bin", ".ckpt", ".pkl", ".pickle"}:
        raise ValueError(
            "Pickle/PyTorch checkpoints are refused. Convert trusted checkpoints to SafeTensors in your own isolated environment."
        )
    parsers = {
        ".safetensors": _safe,
        ".gguf": _gguf,
        ".json": _config,
        ".onnx": _onnx,
        ".mlmodel": _coreml,
        ".mlpackage": _coreml,
    }
    if suffix not in parsers:
        report = _base(target, "unknown")
        report["status"] = "unsupported"
        report["limitations"].append(f"No metadata adapter for extension {suffix!r}.")
        return report
    report = parsers[suffix](target)
    if report["status"] == "inspected":
        _architecture(report)
    return report
