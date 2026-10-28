"""Bounded stage IO and public policy; no teacher/RNG/width metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from modelforge.glu32_io import raw, read_arrays as read_arrays
from scaling_domain_store import (
    ShardStore as ShardStore,
    _ancestry,
    _unique_object,
    sha as sha,
    NAME,
    verify_manifest as verify_manifest,
)

MARKER = "paired-row-acquisition-v1-attempt.json"
POLICY = dict(
    id="paired-row-acquisition-v1",
    teachers=3,
    base_cases=6,
    cases=12,
    n=32,
    outputs=32,
    probes=17,
    budgets=[16, 17],
    steps=[0.3, 0.6],
    vector_calls=430848,
    callback_batches=204,
    weight_std=0.2,
    probe_std=4.0,
    gap_ratio=10.0,
    stability=0.01,
    reference_cutoffs=[1e-11, 1e-10, 1e-9],
    endpoint_tolerance=0.01,
    memory_limit_mb=512,
    timeout_seconds=180,
    blas_threads=1,
    output_limit_bytes=2048,
    private_gate="closed",
    certificate=False,
)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def save_json(path: Path, value: dict[str, Any]) -> None:
    _ancestry(path)
    data = canonical(value)
    if len(data) > 128 * 1024:
        raise ValueError("JSON exceeds128KiB")
    temporary = path.with_suffix(".tmp")
    _ancestry(temporary)
    if temporary.write_bytes(data) != len(data):
        raise OSError("Short JSON write")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(raw(path, 128 * 1024), object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("JSON object required")
    canonical(result)
    return result


def records(root: Path) -> dict[str, dict[str, Any]]:
    data = raw(root / "manifest.jsonl", 1024**2)
    if not data.endswith(b"\n") or not 1 <= len(data.splitlines()) <= 8192:
        raise ValueError("Complete bounded manifest required")
    result: dict[str, dict[str, Any]] = {}
    for line in data.splitlines():
        v = json.loads(line, object_pairs_hook=_unique_object)
        if (
            not isinstance(v, dict)
            or set(v) != {"name", "bytes", "sha256"}
            or not isinstance(v["name"], str)
            or not NAME.fullmatch(v["name"])
            or v["name"] in result
            or type(v["bytes"]) is not int
            or not 0 < v["bytes"] <= 8 * 1024**2
            or not isinstance(v["sha256"], str)
            or len(v["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in v["sha256"])
        ):
            raise ValueError("Invalid manifest record")
        result[v["name"]] = v
    return result


def create_store(root: Path, limits: dict[str, Any]) -> ShardStore:
    return ShardStore(
        root,
        shard_limit_bytes=limits["shard_limit_bytes"],
        total_limit_bytes=limits["bytes_upper_bound"],
    )


def tree_ok(receipt: dict[str, Any]) -> bool:
    return (
        receipt.get("process_monitor_scope") == "process_and_children"
        and type(receipt.get("memory_samples")) is int
        and receipt["memory_samples"] > 0
    )


def collection_gate(request: dict[str, Any]) -> Path:
    if (
        request.get("policy") != POLICY
        or request.get("full_source_protocol_approved") is not True
    ):
        raise ValueError("Exact reviewed full-source approval required")
    for key in ("scope_preflight", "transport_preflight"):
        if not tree_ok(request.get(key, {})):
            raise ValueError("Pre-data TREE receipt required")
    if request["transport_preflight"].get("status") != "passed":
        raise ValueError("Pre-data transport required")
    root = Path(request["root"]).absolute()
    repo = Path(request["repo"]).absolute()
    marker = repo.parent / "experiments" / MARKER
    v = read_json(marker)
    if (
        v.get("policy") != POLICY
        or v.get("source_commit") != request.get("source_commit")
        or v.get("config_sha256") != request.get("config_sha256")
        or v.get("output") != str(root)
        or sha(marker) != request.get("marker_sha256")
    ):
        raise ValueError("Canonical global marker source/config/output mismatch")
    return root
