"""Bounded trace IO and fixed policy; no teacher or experiment startup."""

from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, cast
import zipfile
import numpy as np
from numpy.typing import NDArray
from fresh_rank_common import (
    canonical as canonical,
    read_json as read_json,
    save_json as save_json,
    records as records,
)
from modelforge.glu32_io import raw, read_arrays
from scaling_domain_store import ShardStore, _ancestry
from variable_gate_inventory import inventory, shape_caps, SHARD_LIMIT

Array = NDArray[np.float64]
MARKER = "variable-width-gate-development-v1-attempt.json"
POLICY = dict(
    id="variable-width-gate-development-v1",
    jobs=6,
    n=32,
    max_rank=192,
    initial_starts=256,
    adam_steps=1600,
    initial_refine_steps=12,
    rescue_rounds=4,
    rescue_starts=512,
    rescue_refine_steps=24,
    progress_chunk=8,
    loss_limit=1e-8,
    cosine_limit=0.9995,
    timeout_seconds=180,
    memory_limit_mb=512,
    blas_threads=1,
    output_limit_bytes=2048,
    new_queries=0,
    certificate=False,
    private_gate="closed",
    development_only=True,
)


def digest(path: Path, maximum: int = SHARD_LIMIT) -> str:
    return hashlib.sha256(raw(path, maximum)).hexdigest()


def seed(job: str, stream: str) -> int:
    if (
        not isinstance(job, str)
        or re.fullmatch(r"job0[0-5]", job) is None
        or stream not in ("initial", "rescue")
    ):
        raise ValueError("Exact seed domain required")
    return int.from_bytes(
        hashlib.sha256(
            f"modelforge-variable-gate-development-v1/{job}/{stream}".encode()
        ).digest(),
        "big",
    )


def seed_map() -> dict[str, dict[str, int]]:
    return {
        f"job{i:02d}": {k: seed(f"job{i:02d}", k) for k in ("initial", "rescue")}
        for i in range(6)
    }


def integer(a: Array, low: int, high: int) -> int:
    if (
        a.shape != (1,)
        or not np.isfinite(a).all()
        or a[0] != int(a[0])
        or not low <= a[0] <= high
    ):
        raise ValueError("Bounded integer scalar required")
    return int(a[0])


def decode_state(data: Array, length: Array) -> dict[str, Any]:
    size = integer(length, 1, 512)
    if (
        np.any(data[size:] != 0)
        or np.any(data[:size] != np.floor(data[:size]))
        or np.any(data[:size] < 0)
        or np.any(data[:size] > 127)
    ):
        raise ValueError("Invalid ASCII state encoding")
    from scaling_domain_store import _unique_object

    value = json.loads(
        bytes(data[:size].astype(np.uint8)).decode("ascii"),
        object_pairs_hook=_unique_object,
    )
    if (
        not isinstance(value, dict)
        or set(value) != {"bit_generator", "state", "has_uint32", "uinteger"}
        or value["bit_generator"] != "PCG64"
        or not isinstance(value["state"], dict)
        or set(value["state"]) != {"state", "inc"}
    ):
        raise ValueError("Exact PCG64 state required")
    for key, maximum in [("has_uint32", 1), ("uinteger", 2**32 - 1)]:
        if type(value[key]) is not int or not 0 <= value[key] <= maximum:
            raise ValueError("Native PCG64 state integer required")
    if (
        any(type(x) is not int or not 0 <= x < 2**128 for x in value["state"].values())
        or value["state"]["inc"] % 2 != 1
    ):
        raise ValueError("Invalid PCG64 state/inc")
    return value


def validate_arrays(name: str, arrays: dict[str, Array], n: int, rank: int) -> None:
    caps = shape_caps(name, n, rank)
    if set(arrays) != set(caps):
        raise ValueError("Exact trace fields required")
    for key, bounds in caps.items():
        a = arrays[key]
        if (
            not isinstance(a, np.ndarray)
            or a.dtype != np.float64
            or a.ndim != len(bounds)
            or any(
                not lo <= x <= hi for x, (lo, hi) in zip(a.shape, bounds, strict=True)
            )
            or not np.isfinite(a).all()
        ):
            raise ValueError("Trace shape/dtype/finite cap mismatch")
    if name.endswith("-rng-before") or name.endswith("-rng-after"):
        for stream in ("initial", "rescue"):
            decode_state(arrays[stream], arrays[stream + "_length"])
    chunk = re.search(
        r"(refine-intent|mix-intent|refine-chunk|eigen-chunk)([0-9]{3})$", name
    )
    if chunk:
        if integer(arrays["index"], 0, 511) != int(chunk[2]) * 8:
            raise ValueError("Chunk index mismatch")
        if "intent" in chunk[1]:
            if integer(arrays["requested"], 8, 8) != 8:
                raise ValueError("Chunk request mismatch")
        else:
            count = integer(arrays["count"], 0, 8)
            attempted = integer(arrays["attempted"], 0, 8)
            complete = integer(arrays["complete"], 0, 1)
            if (complete and (count != 8 or attempted != 8)) or (
                not complete and attempted != count + 1
            ):
                raise ValueError("Chunk completion contradicts counters")
            keys = (
                ("refined", "losses")
                if chunk[1] == "refine-chunk"
                else ("values", "vectors", "indices")
            )
            if any(np.any(arrays[k][count:] != 0) for k in keys):
                raise ValueError("Partial chunk padding must be zero")
            if chunk[1] == "eigen-chunk" and np.any(arrays["starts"][2 * count :] != 0):
                raise ValueError("Partial eigen start padding must be zero")
    if name.endswith("-final"):
        k = integer(arrays["count"], 0, 607)
        rows = max(1, k)
        if (
            arrays["directions"].shape != (rows, n)
            or arrays["losses"].shape != (rows,)
            or arrays["cosines"].shape != (rows, rows)
        ):
            raise ValueError("Final count/shape mismatch")
        if not k and any(
            np.any(arrays[x] != 0) for x in ("directions", "losses", "cosines")
        ):
            raise ValueError("Empty final padding must be zero")
    if name.endswith("-view-residuals"):
        k = integer(arrays["count"], 0, 607)
        if arrays["losses"].shape != (3, max(1, k)):
            raise ValueError("View count/shape mismatch")
    if (
        "raw" in arrays
        and "normalized" in arrays
        and arrays["raw"].shape != arrays["normalized"].shape
    ):
        raise ValueError("Mixture shapes differ")
    if name.endswith("-geometry"):
        g = arrays["geometry"]
        component = integer(g[:1], 0, rank)
        dim = integer(g[1:2], 0, rank)
        known = len(arrays["known"])
        if (
            dim != rank - component
            or arrays["residual"].shape != (max(1, dim), n * (n + 1) // 2)
            or len(arrays["tangent_singular"]) != known
            or g[2] < 0
        ):
            raise ValueError("Geometry dimensions disagree")
    if name.endswith("-merge"):
        k = len(arrays["order"])
        if (
            not np.array_equal(np.sort(arrays["order"]), np.arange(k))
            or not np.isin(arrays["decision"], [0.0, 1.0, 2.0]).all()
            or np.any(arrays["losses"] < 0)
        ):
            raise ValueError("Invalid merge permutation/decision/loss")


class TraceStore:
    def __init__(
        self, root: Path, progress: Path, jobs: dict[str, tuple[int, int]]
    ) -> None:
        self.jobs = dict(jobs)
        self.progress = progress
        p = inventory()
        self.store = ShardStore(
            root,
            shard_limit_bytes=p["largest_shard_upper_bound"],
            total_limit_bytes=p["trace_bytes_upper_bound"],
        )

    def write(self, name: str, arrays: dict[str, Array]) -> None:
        if self.store.terminal:
            raise ValueError("Trace store terminal; no retry")
        try:
            job = name.split("-", 1)[0]
            if job not in self.jobs:
                raise ValueError("Unregistered job trace")
            validate_arrays(name, arrays, *self.jobs[job])
            record = self.store.save(name + ".npz", **arrays)
            save_json(
                self.progress,
                dict(
                    status="committed",
                    active_job=job,
                    last_record=record,
                    committed_shards=len(self.store.names),
                    bytes_reserved=self.store.bytes_reserved,
                    uncommitted_current_chunk="unknown_on_external_stop",
                    certificate=False,
                ),
            )
        except Exception:
            self.store.terminal = True
            raise


def read_trace(
    root: Path,
    name: str,
    n: int,
    rank: int,
    *,
    descriptor: dict[str, Any] | None = None,
) -> dict[str, Array]:
    if descriptor is None:
        descriptor = records(root)[name + ".npz"]
    if (
        set(descriptor) != {"name", "bytes", "sha256"}
        or descriptor["name"] != name + ".npz"
        or type(descriptor["bytes"]) is not int
        or not 0 < descriptor["bytes"] <= SHARD_LIMIT
        or not isinstance(descriptor["sha256"], str)
        or re.fullmatch("[0-9a-f]{64}", descriptor["sha256"]) is None
    ):
        raise ValueError("Invalid explicit trace descriptor")
    path = root / descriptor["name"]
    data = raw(path, SHARD_LIMIT)
    if (
        len(data) != descriptor["bytes"]
        or hashlib.sha256(data).hexdigest() != descriptor["sha256"]
    ):
        raise ValueError("Trace commitment mismatch")
    caps = shape_caps(name, n, rank)
    shapes = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        expected = {k + ".npy" for k in caps}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(
            expected
        ):
            raise ValueError("Unexpected trace fields/duplicates")
        for key, bounds in caps.items():
            with archive.open(key + ".npy") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError("Unsupported trace header")
            if (
                dtype != np.dtype(np.float64)
                or len(shape) != len(bounds)
                or any(
                    not lo <= x <= hi for x, (lo, hi) in zip(shape, bounds, strict=True)
                )
            ):
                raise ValueError("Trace header exceeds shape cap")
            shapes[key] = shape
    result = read_arrays(path, shapes, descriptor["sha256"])
    validate_arrays(name, result, n, rank)
    return cast(dict[str, Array], result)


def tree_ok(receipt: dict[str, Any]) -> bool:
    return (
        receipt.get("process_monitor_scope") == "process_and_children"
        and type(receipt.get("memory_samples")) is int
        and receipt["memory_samples"] > 0
        and type(receipt.get("peak_sampled_rss_bytes")) is int
        and 0 < receipt["peak_sampled_rss_bytes"] <= 512 * 1024**2
        and receipt.get("memory_limit_mb") == 512
    )


def fsynced_marker(path: Path, value: dict[str, Any]) -> str:
    _ancestry(path)
    with path.open("xb") as f:
        data = canonical(value)
        if f.write(data) != len(data):
            raise OSError("Short marker; no retry")
        f.flush()
        import os

        os.fsync(f.fileno())
    return digest(path, 128 * 1024)
