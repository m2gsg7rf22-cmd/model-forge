"""Bounded immutable NPZ shards for future public dimension-domain diagnostics.

No scientific runner, RNG, teacher/query generation or protocol approval here.
Reservations overestimate bytes after failed writes; partial evidence is retained.
No fsync/crash-durability, adversarial concurrent writer or hard RSS guarantee.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from numpy.typing import NDArray

SHARD_LIMIT = 8 * 1024**2
REPORT_RESERVE = 1024**2
TOTAL_LIMIT = 1024**3 - REPORT_RESERVE
MIN_FREE_BYTES = 2 * 1024**3
MAX_RECORDS = 8192
MANIFEST_LIMIT = REPORT_RESERVE
NAME = re.compile(r"[a-zA-Z0-9_-]{1,80}\.npz")
FIELD = re.compile(r"[a-zA-Z0-9_]{1,64}")


def _ancestry(path: Path) -> None:
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("Symlink artifact ancestry forbidden")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


class ShardStore:
    """Exclusive root/filenames; bounded serialization; terminal on write failure."""

    def __init__(
        self,
        root: Path,
        *,
        shard_limit_bytes: int = SHARD_LIMIT,
        total_limit_bytes: int = TOTAL_LIMIT,
    ) -> None:
        if (
            type(shard_limit_bytes) is not int
            or type(total_limit_bytes) is not int
            or not 512 <= shard_limit_bytes <= SHARD_LIMIT
            or not shard_limit_bytes <= total_limit_bytes <= TOTAL_LIMIT
        ):
            raise ValueError("Bounded positive shard/total limits required")
        self.root = root.absolute()
        _ancestry(self.root)
        self.root.mkdir(exist_ok=False)
        with (self.root / "manifest.jsonl").open("xb"):
            pass
        self.shard_limit_bytes = shard_limit_bytes
        self.total_limit_bytes = total_limit_bytes
        self.bytes_reserved = 0
        self.manifest_bytes = 0
        self.names: set[str] = set()
        self.terminal = False

    def save(self, name: str, **arrays: NDArray[np.float64]) -> dict[str, Any]:
        if self.terminal:
            raise ValueError("Store terminal after failed write; no retry")
        _ancestry(self.root)
        _ancestry(self.root / "manifest.jsonl")
        if (
            not (self.root / "manifest.jsonl").is_file()
            or (self.root / "manifest.jsonl").stat().st_size != self.manifest_bytes
        ):
            raise ValueError("Manifest changed before append")
        if not NAME.fullmatch(name):
            raise ValueError("Simple bounded NPZ basename required")
        if (
            name in self.names
            or (self.root / name).exists()
            or (self.root / name).is_symlink()
        ):
            raise FileExistsError(name)
        if not 1 <= len(arrays) <= 16 or any(
            not FIELD.fullmatch(k) or k in {"file", "allow_pickle"} for k in arrays
        ):
            raise ValueError("One to16 bounded array field names required")
        if any(
            not isinstance(a, np.ndarray)
            or a.dtype != np.dtype(np.float64)
            or a.size == 0
            for a in arrays.values()
        ):
            raise ValueError("Nonempty finite native float64 arrays required")
        if sum(a.nbytes for a in arrays.values()) > self.shard_limit_bytes:
            raise ValueError("Array payload exceeds shard limit before serialization")
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Finite array fields required")
        with io.BytesIO() as buffer:
            fields: dict[str, Any] = dict(arrays)
            np.savez(buffer, allow_pickle=False, **fields)
            encoded = buffer.getvalue()
        if len(encoded) > self.shard_limit_bytes:
            raise ValueError("Serialized NPZ exceeds shard limit")
        item = {
            "name": name,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
        line = (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode()
        reservation = len(encoded) + len(line)
        if (
            self.bytes_reserved + reservation > self.total_limit_bytes
            or self.manifest_bytes + len(line) > MANIFEST_LIMIT
            or len(self.names) >= MAX_RECORDS
        ):
            raise ValueError("Total artifact/manifest budget exhausted before write")
        # Reserve before side effects. Failures never release name/bytes or retry.
        self.bytes_reserved += reservation
        self.manifest_bytes += len(line)
        self.names.add(name)
        try:
            with (self.root / name).open("xb") as f:
                if f.write(encoded) != len(encoded):
                    raise OSError("Short shard write")
            with (self.root / "manifest.jsonl").open("ab") as f:
                if f.write(line) != len(line):
                    raise OSError("Short manifest append")
        except Exception:
            self.terminal = True
            raise
        return item


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest JSON key")
        result[key] = value
    return result


def verify_manifest(root: Path) -> dict[str, int]:
    """Stream raw hashes and exact inventory; not an independent numerical audit.

    No NPZ decompression occurs here. Artifact hashes do not imply semantic truth.
    Any partial/missing/extra/duplicate artifact makes this fail.
    """
    try:
        _ancestry(root)
        manifest = root / "manifest.jsonl"
        if manifest.is_symlink() or not 0 < manifest.stat().st_size <= MANIFEST_LIMIT:
            raise ValueError("Missing/empty/oversized manifest")
        raw = manifest.read_bytes()
        if not raw.endswith(b"\n"):
            raise ValueError("Truncated manifest")
        lines = raw.splitlines()
        if not 1 <= len(lines) <= MAX_RECORDS:
            raise ValueError("Bounded nonempty manifest inventory required")
        names: set[str] = set()
        total = len(raw)
        largest = 0
        for line in lines:
            item = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(item, dict) or set(item) != {"name", "bytes", "sha256"}:
                raise ValueError("Invalid manifest record")
            name, size, digest = item["name"], item["bytes"], item["sha256"]
            if (
                not isinstance(name, str)
                or not NAME.fullmatch(name)
                or name in names
                or type(size) is not int
                or not 0 < size <= SHARD_LIMIT
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError("Invalid/duplicate artifact commitment")
            p = root / name
            if (
                p.is_symlink()
                or not p.is_file()
                or p.stat().st_size != size
                or sha(p) != digest
            ):
                raise ValueError("Artifact missing/changed/symlink")
            names.add(name)
            total += size
            largest = max(largest, size)
            if total > TOTAL_LIMIT:
                raise ValueError("Artifact budget exceeded")
        if {p.name for p in root.iterdir()} != names | {"manifest.jsonl"}:
            raise ValueError("Uncommitted extra artifact")
        return {
            "shards": len(names),
            "bytes_reserved": total,
            "largest_shard_bytes": largest,
        }
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid/incomplete artifact store") from exc


def synthetic_preflight(root: Path, *, free_bytes: int | None = None) -> dict[str, Any]:
    """Explicit worst proposed profile-shape transport only; no data/RNG/oracle.

    Controller must run this under tree monitoring BEFORE registering any teacher.
    Injected free_bytes is a testing seam; a real controller must not supply it.
    """
    available = (
        shutil.disk_usage(root.parent).free if free_bytes is None else free_bytes
    )
    if type(available) is not int or available < MIN_FREE_BYTES:
        raise ValueError("At least2GiB free disk required before transport/data")
    store = ShardStore(root)
    # Worst raw (one probe, all4signs), Hessian chunk8, full symmetric right basis.
    arrays: dict[str, dict[str, NDArray[np.float64]]] = {
        "raw": {"inputs": np.zeros((4, 528, 32)), "responses": np.ones((4, 528, 32))},
        "hessian": {"hessian": np.zeros((8, 32, 32, 32))},
        "basis": {"basis": np.eye(528), "singular_values": np.ones(528)},
    }
    for name, fields in arrays.items():
        store.save(name + ".npz", **fields)
        with np.load(root / (name + ".npz"), allow_pickle=False) as loaded:
            if set(loaded.files) != set(fields) or any(
                not np.array_equal(loaded[k], a) for k, a in fields.items()
            ):
                raise ValueError("Synthetic transport roundtrip mismatch")
    result = verify_manifest(root)
    return {
        "status": "passed",
        **result,
        "registered_teachers_created": 0,
        "reference_calls": 0,
        "free_disk_bytes_before": available,
    }
