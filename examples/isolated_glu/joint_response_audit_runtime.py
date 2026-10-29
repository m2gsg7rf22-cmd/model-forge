"""Numeric-audit runtime gates: bounded literal transport, preflight and layout.

Large allocation is ONLY for the future fully approved mandatory default route.
Fixture helpers exercise tiny profiles and never prove actual resource bounds.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from joint_response_arrays import _schema, decode_typed
from joint_response_audit_inventory import inventory, memory_plan, transport_specs
from joint_response_inventory import Shard
from joint_response_runtime import _engine_bytes
from variable_response_manifest import _read
from variable_response_protocol import ancestry, canonical, digest, read, save, tree_ok

MARKER = "joint-response-numeric-audit-v1-attempt.json"
ROOT_METADATA = (
    "controller-report",
    "outer-monitor",
    "preflight-receipt",
    "input-binding",
    "progress",
    "terminal-binding",
)


def allocation_plan() -> dict[str, int]:
    return {
        k: v
        for k, v in memory_plan().items()
        if k != "runtime_allocator_monitor_allowance"
    }


def _specs(specs: tuple[Shard, ...]) -> None:
    if len(specs) != 2 or tuple(s.name for s in specs) != ("parameters", "predictions"):
        raise ValueError("exact two transport shard names required")
    for spec in specs:
        _schema(spec)
        if any(f.dtype != "float64" for f in spec.fields):
            raise ValueError("native FP64 literal fields required")
    if tuple(f.name for f in specs[0].fields) != ("g", "u", "d") or tuple(
        f.name for f in specs[1].fields
    ) != ("predictions",):
        raise ValueError("exact literal transport fields required")


def _value(shard: str, index: int) -> float:
    return (index + 1) / 4 if shard == "parameters" else -0.25


def _transport(root: Path, specs: tuple[Shard, ...]) -> dict[str, Any]:
    _specs(specs)
    ancestry(root)
    root.mkdir(exist_ok=False)
    records = []
    for spec in specs:
        arrays = {
            f.name: np.full(f.shape, _value(spec.name, i), dtype=np.float64)
            for i, f in enumerate(spec.fields)
        }
        stream = io.BytesIO()
        np.savez(stream, allow_pickle=False, **arrays)
        blob = stream.getvalue()
        if len(blob) > spec.encoded_bound:
            raise ValueError("literal encoded cap exceeded")
        path = root / (spec.name + ".npz")
        with path.open("xb") as f:
            if f.write(blob) != len(blob):
                raise OSError("short literal transport write; no retry")
            f.flush()
            os.fsync(f.fileno())
        records.append(
            dict(
                name=path.name,
                bytes=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),
                fields=[asdict(f) for f in spec.fields],
            )
        )
    save(
        root / "manifest.json",
        dict(
            schema="joint-response-numeric-audit-transport-v1",
            records=records,
            memory_exercise=False,
        ),
        4096,
    )
    return dict(
        status="passed", shards=2, manifest_sha256=digest(root / "manifest.json", 4096)
    )


def _preflight(
    root: Path, specs: tuple[Shard, ...], plan: dict[str, int], *, hold_seconds: float
) -> dict[str, Any]:
    if (
        not plan
        or any(type(v) is not int or not 0 < v <= 256360448 for v in plan.values())
        or sum(plan.values()) > 256360448
        or not 0 <= hold_seconds <= 0.2
    ):
        raise ValueError("bounded native literal allocation plan required")
    receipt = _transport(root, specs)
    buffers = []
    for size in plan.values():
        a = np.empty(size, dtype=np.uint8)
        a.fill(1)
        buffers.append(a)
    time.sleep(hold_seconds)
    del buffers, a
    manifest = read(root / "manifest.json", 4096)
    manifest.update(
        memory_exercise=True,
        allocation_plan=plan,
        allocated_literal_bytes=sum(plan.values()),
        memory_planning_bytes=sum(memory_plan().values()),
    )
    save(root / "manifest.json", manifest, 4096)
    receipt["manifest_sha256"] = digest(root / "manifest.json", 4096)
    return receipt


def preflight(root: Path) -> dict[str, Any]:
    return _preflight(root, transport_specs(), allocation_plan(), hold_seconds=0.2)


def _validate_transport(
    root: Path,
    receipt: dict[str, Any],
    specs: tuple[Shard, ...],
    *,
    require_memory: bool,
) -> None:
    _specs(specs)
    ancestry(root)
    expected_names = {s.name + ".npz" for s in specs} | {"manifest.json"}
    if (
        receipt.get("status") != "passed"
        or type(receipt.get("shards")) is not int
        or receipt["shards"] != 2
        or {p.name for p in root.iterdir()} != expected_names
        or digest(root / "manifest.json", 4096) != receipt.get("manifest_sha256")
    ):
        raise ValueError("exact completed transport receipt and layout required")
    manifest = read(root / "manifest.json", 4096)
    keys = {"schema", "records", "memory_exercise"}
    if manifest.get("memory_exercise") is True:
        keys |= {"allocation_plan", "allocated_literal_bytes", "memory_planning_bytes"}
        if (
            canonical(manifest.get("allocation_plan")) != canonical(allocation_plan())
            or type(manifest.get("allocated_literal_bytes")) is not int
            or manifest["allocated_literal_bytes"] != sum(allocation_plan().values())
            or type(manifest.get("memory_planning_bytes")) is not int
            or manifest["memory_planning_bytes"] != sum(memory_plan().values())
        ):
            raise ValueError("registered actual memory exercise plan mismatch")
    elif require_memory or manifest.get("memory_exercise") is not False:
        raise ValueError("registered actual memory exercise required")
    records = manifest.get("records")
    if (
        set(manifest) != keys
        or manifest.get("schema") != "joint-response-numeric-audit-transport-v1"
        or not isinstance(records, list)
        or len(records) != 2
    ):
        raise ValueError("exact transport manifest required")
    for spec, record in zip(specs, records, strict=True):
        if (
            not isinstance(record, dict)
            or set(record) != {"name", "bytes", "sha256", "fields"}
            or record["name"] != spec.name + ".npz"
            or type(record["bytes"]) is not int
            or not 0 < record["bytes"] <= spec.encoded_bound
            or canonical(record["fields"])
            != canonical([asdict(f) for f in spec.fields])
        ):
            raise ValueError("exact typed literal descriptor required")
        blob = _read(root / record["name"], spec.encoded_bound)
        if (
            len(blob) != record["bytes"]
            or hashlib.sha256(blob).hexdigest() != record["sha256"]
        ):
            raise ValueError("transport size/hash changed before decode")
        arrays = decode_typed(blob, spec)
        for index, field in enumerate(spec.fields):
            if np.any(arrays[field.name] != _value(spec.name, index)):
                raise ValueError("independent literal values differ")


def validate_transport(root: Path, receipt: dict[str, Any]) -> None:
    if not tree_ok(receipt):
        raise ValueError(
            "actual sampled process-and-children receipt required before transport read"
        )
    _validate_transport(root, receipt, transport_specs(), require_memory=True)


def layout(output: Path) -> dict[str, int]:
    """Actual output caps, including live engine files within root metadata budget."""
    ancestry(output)
    total = 0
    files = 0
    root_meta = 0
    root_temp = 0
    caps = {
        name + suffix: 128 * 1024
        for name in ROOT_METADATA
        for suffix in (".json", ".json.tmp")
    }
    transport_caps = {s.name + ".npz": s.encoded_bound for s in transport_specs()} | {
        "manifest.json": 4096,
        "manifest.json.tmp": 4096,
    }
    for path in output.iterdir():
        ancestry(path)
        if path.is_file():
            size = path.stat().st_size
            if path.name not in caps or size > caps[path.name]:
                raise ValueError("unregistered/oversize root metadata")
            total += size
            files += 1
            if path.name.endswith(".tmp"):
                root_temp += size
            else:
                root_meta += size
        elif path.is_dir() and path.name == "cases":
            for case in path.iterdir():
                ancestry(case)
                if not case.is_dir() or re.fullmatch("job0[0-5]", case.name) is None:
                    raise ValueError("canonical case directory required")
                meta = 0
                temp = 0
                for item in case.iterdir():
                    ancestry(item)
                    name = item.name
                    size = item.stat().st_size
                    allowed = {
                        "result.json": 128 * 1024,
                        "result.json.tmp": 128 * 1024,
                        "checkpoint-metrics.jsonl": 17 * 4096,
                        "checkpoint-metrics.jsonl.tmp": 17 * 4096,
                    }
                    if (
                        not item.is_file()
                        or name not in allowed
                        or size > allowed[name]
                    ):
                        raise ValueError("unregistered/oversize case artifact")
                    total += size
                    files += 1
                    if name.endswith(".tmp"):
                        temp += size
                    else:
                        meta += size
                if max(meta, temp) > 128 * 1024 + 17 * 4096:
                    raise ValueError("case metadata aggregate exceeded")
        elif path.is_dir() and path.name == "transport":
            for item in path.iterdir():
                ancestry(item)
                size = item.stat().st_size
                if (
                    not item.is_file()
                    or item.name not in transport_caps
                    or size > transport_caps[item.name]
                ):
                    raise ValueError("unregistered/oversize transport output")
                total += size
                files += 1
        elif path.is_dir() and path.name == "preflight-engine":
            size, _, count = _engine_bytes(path)
            total += size
            root_meta += size
            files += count
        else:
            raise ValueError("unregistered output layout")
    size, _, count = _engine_bytes(output.parent / (output.name + "-launch-engine"))
    total += size
    root_meta += size
    files += count
    inv = inventory()
    # Engine requests/logs share this cap; no uncounted extra reserve is assumed.
    if root_meta > int(inv["root_metadata_bytes"]) or root_temp > 6 * 128 * 1024:
        raise ValueError("root metadata plus engines aggregate exceeded")
    marker_bytes = 0
    for name in (MARKER, MARKER + ".tmp"):
        path = output.parent / name
        ancestry(path)
        if path.exists():
            if not path.is_file() or path.stat().st_size > 128 * 1024:
                raise ValueError("audit marker cap exceeded")
            marker_bytes += path.stat().st_size
            files += 1
    combined = 857668616 + 393216 + total + marker_bytes
    if (
        files > 64
        or combined > int(inv["combined_artifacts_upper_bound"])
        or combined >= 1024**3
    ):
        raise ValueError("combined actual retained artifacts exceeded")
    return dict(
        retained_bytes=total,
        audit_marker_bytes=marker_bytes,
        combined_retained_bytes=combined,
        files=files,
        root_metadata_including_engines_bytes=root_meta,
        root_temporary_metadata_bytes=root_temp,
    )
