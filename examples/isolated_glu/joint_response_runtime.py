"""Literal transport/resource exercise and strict output accounting; no target/RNG.

Only the complete approved default controller may run the large memory exercise.
Small transport/layout tests do not prove the actual experiment resource bound.
"""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import re
import time
from typing import Any, cast
import zipfile

import numpy as np
from numpy.typing import NDArray
from joint_response_inventory import inventory, memory_plan, trace_specs, Shard
from variable_response_protocol import ancestry, canonical, digest, read, save

ROOT_METADATA = (
    "controller-report",
    "outer-monitor",
    "preflight-receipt",
    "input-binding",
    "progress",
    "terminal-binding",
)
CASE_METADATA = {
    "rng.json": 128 * 1024,
    "case.json": 128 * 1024,
    "metrics.json": 128 * 1024,
    "terminal.json": 128 * 1024,
    "manifest.jsonl": 31 * 4096,
    "checkpoint-intents.jsonl": 17 * 4096,
    "checkpoint-metrics.jsonl": 17 * 4096,
}


def allocation_plan() -> dict[str, int]:
    return {
        k: v
        for k, v in memory_plan().items()
        if k != "runtime_allocator_and_monitor_allowance"
    }


def _transport_specs() -> tuple[Shard, ...]:
    all_specs = trace_specs(96)
    return tuple(
        next(s for s in all_specs if s.name == name)
        for name in ("schedule", "checkpoint-0000")
    )


def transport(root: Path) -> dict[str, Any]:
    ancestry(root)
    root.mkdir(exist_ok=False)
    records = []
    for name, spec in zip(("schedule", "state"), _transport_specs(), strict=True):
        arrays = {f.name: np.zeros(f.shape, dtype=f.dtype) for f in spec.fields}
        stream = io.BytesIO()
        np.savez(stream, allow_pickle=False, **cast(dict[str, Any], arrays))
        blob = stream.getvalue()
        if len(blob) > spec.encoded_bound:
            raise ValueError("literal transport exceeds encoded cap")
        path = root / (name + ".npz")
        with path.open("xb") as target:
            if target.write(blob) != len(blob):
                raise OSError("short literal transport write")
            target.flush()
        records.append(
            dict(
                name=path.name,
                bytes=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),
                fields=[asdict(f) for f in spec.fields],
            )
        )
    value = dict(
        schema="joint-response-transport-v1", records=records, memory_exercise=False
    )
    save(root / "manifest.json", value, 4096)
    return dict(
        status="passed", manifest_sha256=digest(root / "manifest.json", 4096), shards=2
    )


def preflight(root: Path) -> dict[str, Any]:
    """Run only from approved default worker, under inner AND shared outer TREE caps."""
    receipt = transport(root)
    buffers: list[NDArray[np.uint8]] = []
    plan = allocation_plan()
    for size in plan.values():
        array = np.empty(size, dtype=np.uint8)
        array.fill(1)  # Touch pages; virtual-only allocation is not RSS evidence.
        buffers.append(array)
    time.sleep(0.2)  # Leave a bounded sampling window for both50ms monitors.
    del buffers, array
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


def validate_transport(
    root: Path, receipt: dict[str, Any], *, require_memory: bool = True
) -> None:
    ancestry(root)
    if (
        receipt.get("status") != "passed"
        or type(receipt.get("shards")) is not int
        or receipt["shards"] != 2
        or {p.name for p in root.iterdir()}
        != {"schedule.npz", "state.npz", "manifest.json"}
        or digest(root / "manifest.json", 4096) != receipt.get("manifest_sha256")
    ):
        raise ValueError("exact completed transport receipt/layout required")
    manifest = read(root / "manifest.json", 4096)
    expected_keys = {"schema", "records", "memory_exercise"}
    if manifest.get("memory_exercise") is True:
        expected_keys |= {
            "allocation_plan",
            "allocated_literal_bytes",
            "memory_planning_bytes",
        }
        if (
            canonical(manifest.get("allocation_plan")) != canonical(allocation_plan())
            or type(manifest.get("allocated_literal_bytes")) is not int
            or manifest["allocated_literal_bytes"] != sum(allocation_plan().values())
            or manifest.get("memory_planning_bytes") != sum(memory_plan().values())
        ):
            raise ValueError("literal memory exercise plan mismatch")
    elif require_memory or manifest.get("memory_exercise") is not False:
        raise ValueError("actual literal memory exercise required before data")
    records = manifest.get("records")
    if (
        set(manifest) != expected_keys
        or manifest.get("schema") != "joint-response-transport-v1"
        or not isinstance(records, list)
        or len(records) != 2
    ):
        raise ValueError("exact transport manifest required")
    for name, spec, record in zip(
        ("schedule", "state"), _transport_specs(), records, strict=True
    ):
        if (
            not isinstance(record, dict)
            or set(record) != {"name", "bytes", "sha256", "fields"}
            or record["name"] != name + ".npz"
            or type(record["bytes"]) is not int
            or not 0 < record["bytes"] <= spec.encoded_bound
            or canonical(record["fields"])
            != canonical([asdict(f) for f in spec.fields])
        ):
            raise ValueError("transport schema/size descriptor mismatch")
        path = root / record["name"]
        ancestry(path)
        if path.stat().st_size != record["bytes"]:
            raise ValueError("transport size changed")
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError("transport hash changed")
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            infos = archive.infolist()
            if (
                len(infos) != len(spec.fields)
                or {i.filename for i in infos} != {f.name + ".npy" for f in spec.fields}
                or any(
                    i.compress_type != zipfile.ZIP_STORED or i.flag_bits & 1
                    for i in infos
                )
            ):
                raise ValueError("uncompressed exact literal members required")
            for field in spec.fields:
                info = archive.getinfo(field.name + ".npy")
                if not 0 < info.file_size <= int(np.prod(field.shape)) * 8 + 4096:
                    raise ValueError("literal member size cap exceeded")
        with np.load(
            io.BytesIO(blob), allow_pickle=False, max_header_size=4096
        ) as restored:
            for field in spec.fields:
                a = restored[field.name]
                if (
                    a.dtype != np.dtype(field.dtype)
                    or not a.dtype.isnative
                    or a.shape != field.shape
                    or np.any(a != 0)
                ):
                    raise ValueError("typed literal values changed")


def _engine_bytes(root: Path) -> tuple[int, int, int]:
    """One live run_json temporary directory, or its empty retained cwd."""
    ancestry(root)
    if not root.exists():
        return 0, 0, 0
    if not root.is_dir():
        raise ValueError("engine cwd must be a directory")
    children = list(root.iterdir())
    if len(children) > 1:
        raise ValueError("multiple engine temporary directories forbidden")
    if not children:
        return 0, 0, 0
    child = children[0]
    ancestry(child)
    if not child.is_dir() or re.fullmatch(r"engine-[A-Za-z0-9_-]+", child.name) is None:
        raise ValueError("unexpected engine directory")
    total, logs, count = 0, 0, 0
    for path in child.iterdir():
        ancestry(path)
        if not path.is_file() or path.name not in (
            "request.json",
            "output.json",
            "error.log",
        ):
            raise ValueError("unregistered engine file")
        size = path.stat().st_size
        if size > (128 * 1024 if path.name == "request.json" else 1024):
            raise ValueError("engine temporary file cap exceeded")
        total += size
        count += 1
        if path.name != "request.json":
            logs += size
    if logs > 1024:
        raise ValueError("combined inner engine output/log cap exceeded")
    return total, logs, count


def layout(output: Path) -> dict[str, int]:
    """Allow only registered partial artifacts; inspect outputs, never source data."""
    ancestry(output)
    root_cap = {
        name + suffix: 128 * 1024
        for name in ROOT_METADATA
        for suffix in (".json", ".json.tmp")
    }
    specs = {s.name + ".npz": s.encoded_bound for s in trace_specs(96)}
    total, files, global_meta = 0, 0, 0
    for path in output.iterdir():
        ancestry(path)
        if path.is_file():
            if path.name not in root_cap or path.stat().st_size > root_cap[path.name]:
                raise ValueError("unregistered/oversize global report")
            size = path.stat().st_size
            total += size
            global_meta += size
            files += 1
        elif path.is_dir() and path.name == "cases":
            for case in path.iterdir():
                ancestry(case)
                if not case.is_dir() or re.fullmatch("job0[0-5]", case.name) is None:
                    raise ValueError("canonical case directory required")
                case_meta = 0
                for item in case.iterdir():
                    ancestry(item)
                    if not item.is_file():
                        raise ValueError("nested/nonfile trace artifact")
                    size = item.stat().st_size
                    cap = specs.get(item.name, CASE_METADATA.get(item.name))
                    if item.name == "progress.jsonl":
                        cap = 4000 * 512
                    if cap is None or size > cap:
                        raise ValueError("unregistered/oversize case trace")
                    total += size
                    files += 1
                    if item.name in CASE_METADATA:
                        case_meta += size
                if case_meta > 1024**2:
                    raise ValueError("case metadata aggregate exceeded")
        elif path.is_dir() and path.name == "transport":
            caps = dict(
                zip(
                    ("schedule.npz", "state.npz"),
                    (s.encoded_bound for s in _transport_specs()),
                    strict=True,
                )
            )
            caps.update({"manifest.json": 4096, "manifest.json.tmp": 4096})
            for item in path.iterdir():
                ancestry(item)
                if (
                    not item.is_file()
                    or item.name not in caps
                    or item.stat().st_size > caps[item.name]
                ):
                    raise ValueError("unregistered/oversize transport file")
                size = item.stat().st_size
                total += size
                files += 1
                if item.suffix != ".npz":
                    global_meta += size
        elif path.is_dir() and path.name == "preflight-engine":
            size, _, count = _engine_bytes(path)
            total += size
            global_meta += size
            files += count
        else:
            raise ValueError("unregistered output layout")
    sibling = output.parent / (output.name + "-launch-engine")
    size, _, count = _engine_bytes(sibling)
    total += size
    global_meta += size
    files += count
    inv = inventory()
    combined = 671428083 + total
    if (
        files > 320
        or global_meta > 2 * 1024**2
        or combined > int(inv["combined_artifacts_upper_bound"])
        or combined >= 1024**3
    ):
        raise ValueError("global artifact/count/metadata bound exceeded")
    return dict(
        retained_bytes=total,
        combined_retained_bytes=combined,
        files=files,
        global_metadata_bytes=global_meta,
    )
