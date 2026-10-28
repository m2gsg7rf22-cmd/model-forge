"""Pure integer/shape planning for candidate traces; no arrays, RNG or IO."""

from __future__ import annotations
import json
from math import prod
import re
from typing import Any

Bounds = tuple[tuple[int, int], ...]
SHARD_LIMIT = 8 * 1024**2
PARENT_BOUND = 534980568
METADATA_RESERVE = 8 * 1024**2
HEADER_PER_FIELD = 4096


def trace_names(job: str) -> tuple[str, ...]:
    if re.fullmatch(r"job0[0-5]", job) is None:
        raise ValueError("Opaque job ID required")
    names = [
        job + "-" + k
        for k in (
            "basis",
            "rng-before",
            "initial-raw",
            "initial-starts",
            "initial-pre-refine",
            "initial-refine",
            "initial-merge",
            "final",
            "rng-after",
            "view-residuals",
        )
    ]
    names += [
        f"{job}-initial-refine-{kind}{i:03d}"
        for i in range(32)
        for kind in ("intent", "chunk")
    ]
    for r in range(4):
        prefix = f"{job}-r{r}"
        names += [
            prefix + "-" + k
            for k in (
                "input",
                "geometry",
                "mixtures",
                "eigen",
                "pre-refine",
                "refine",
                "merge",
            )
        ]
        names += [
            f"{prefix}-{kind}{i:03d}"
            for i in range(32)
            for kind in ("mix-intent", "mix-pre", "eigen-chunk")
        ]
        names += [
            f"{prefix}-refine-{kind}{i:03d}"
            for i in range(64)
            for kind in ("intent", "chunk")
        ]
    return tuple(names)


def shape_caps(name: str, n: int = 32, rank: int = 192) -> dict[str, Bounds]:
    if (
        type(n) is not int
        or not 2 <= n <= 32
        or type(rank) is not int
        or not 0 < rank < n * (n + 1) // 2
        or rank % 2
        or rank > 192
    ):
        raise ValueError("Proper bounded symmetric dimension/even rank required")
    m = re.fullmatch(r"(job0[0-5])-(.+)", name)
    if m is None or name not in trace_names(m[1]):
        raise ValueError("Exact registered trace name required")
    suffix = m[2]
    p = n * (n + 1) // 2
    target = rank // 2

    def fixed(*dims: int) -> Bounds:
        return tuple((d, d) for d in dims)

    def rows(maximum: int, *tail: int) -> Bounds:
        return ((1, maximum), *fixed(*tail))

    def merge(count: int) -> dict[str, Bounds]:
        return {
            k: fixed(count)
            for k in ("order", "decision", "max_cosine", "nearest_index", "losses")
        }

    if suffix == "basis":
        return dict(basis=fixed(rank, p))
    if suffix in ("rng-before", "rng-after"):
        return {
            k: fixed(512 if not k.endswith("_length") else 1)
            for k in ("initial", "rescue", "initial_length", "rescue_length")
        }
    if suffix == "initial-raw":
        return dict(raw=fixed(256, n))
    if suffix == "initial-starts":
        return dict(starts=fixed(256, n))
    if suffix in ("initial-pre-refine", "initial-refine"):
        a = dict(
            starts=fixed(256, n), optimized=fixed(256, n), before_losses=fixed(256)
        )
        if suffix == "initial-refine":
            a.update(refined=fixed(256, n), losses=fixed(256))
        return a
    if suffix == "initial-merge":
        return merge(256)
    if suffix == "final":
        return dict(
            directions=rows(607, n),
            losses=rows(607),
            cosines=((1, 607), (1, 607)),
            count=fixed(1),
        )
    if suffix == "view-residuals":
        return dict(losses=(fixed(3)[0], (1, 607)), count=fixed(1))
    chunk = re.fullmatch(r"(?:initial|r[0-3])-refine-(intent|chunk)([0-9]{3})", suffix)
    if chunk:
        if chunk[1] == "intent":
            return dict(index=fixed(1), requested=fixed(1))
        return dict(
            refined=fixed(8, n),
            losses=fixed(8),
            count=fixed(1),
            attempted=fixed(1),
            complete=fixed(1),
            index=fixed(1),
        )
    if target < 2:
        raise ValueError("No rescue domain for target1")
    kind = suffix[3:]
    known = target - 1
    residual = rank - 2
    if kind == "input":
        return dict(known=rows(known, n))
    if kind == "geometry":
        return dict(
            known=rows(known, n),
            residual=rows(rank, p),
            geometry=fixed(3),
            tangent_singular=rows(known, min(3, n)),
        )
    if kind.startswith("mix-intent"):
        return dict(index=fixed(1), requested=fixed(1))
    if kind.startswith("mix-pre"):
        return dict(
            raw=(fixed(8)[0], (1, residual)),
            normalized=(fixed(8)[0], (1, residual)),
            index=fixed(1),
        )
    if kind.startswith("eigen-chunk"):
        return dict(
            values=fixed(8, n),
            vectors=fixed(8, n, n),
            indices=fixed(8, 2),
            starts=fixed(16, n),
            count=fixed(1),
            attempted=fixed(1),
            complete=fixed(1),
            index=fixed(1),
        )
    if kind == "mixtures":
        return {k: (fixed(256)[0], (1, residual)) for k in ("raw", "normalized")}
    if kind == "eigen":
        return dict(
            values=fixed(256, n), vectors=fixed(256, n, n), indices=fixed(256, 2)
        )
    if kind in ("pre-refine", "refine"):
        a = dict(starts=fixed(512, n), before_losses=fixed(512))
        if kind == "refine":
            a.update(refined=fixed(512, n), losses=fixed(512))
        return a
    if kind == "merge":
        return merge(512)
    raise ValueError("Unrecognized trace family")


def upper_fields(name: str, n: int = 32, rank: int = 192) -> dict[str, tuple[int, ...]]:
    return {
        k: tuple(high for _, high in bounds)
        for k, bounds in shape_caps(name, n, rank).items()
    }


def shard_bound(fields: dict[str, tuple[int, ...]]) -> int:
    value = 8 * sum(prod(shape) for shape in fields.values()) + HEADER_PER_FIELD * len(
        fields
    )
    if value > SHARD_LIMIT:
        raise ValueError("Trace shard exceeds8MiB")
    return value


def record_bound(name: str, byte_bound: int) -> int:
    return len(
        (
            json.dumps(
                dict(name=name + ".npz", bytes=byte_bound, sha256="0" * 64),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )


def transport_schemas() -> dict[str, dict[str, tuple[int, ...]]]:
    unique = {}
    for name in trace_names("job00"):
        fields = upper_fields(name)
        key = json.dumps(fields, sort_keys=True)
        if key not in unique:
            unique[key] = fields
    return {f"family{i:02d}": v for i, v in enumerate(unique.values())}


def inventory() -> dict[str, Any]:
    names = tuple(name for j in range(6) for name in trace_names(f"job{j:02d}"))
    sizes = {name: shard_bound(upper_fields(name)) for name in names}
    manifest = sum(record_bound(name, size) for name, size in sizes.items())
    trace = sum(sizes.values()) + manifest
    transport = transport_schemas()
    preflight = sum(
        shard_bound(s) + record_bound(name, shard_bound(s))
        for name, s in transport.items()
    )
    total = trace + preflight + METADATA_RESERVE + PARENT_BOUND
    if len(names) > 8192 or manifest > 1024**2 or total >= 1024**3:
        raise ValueError("Combined inventory exceeds registered limits")
    return dict(
        trace_shards=len(names),
        trace_bytes_upper_bound=trace,
        manifest_bytes_upper_bound=manifest,
        largest_shard_upper_bound=max(sizes.values()),
        preflight_shards=len(transport),
        preflight_bytes_upper_bound=preflight,
        metadata_reserve_bytes=METADATA_RESERVE,
        parent_retained_bytes_upper_bound=PARENT_BOUND,
        bytes_upper_bound=total,
        minimum_free_disk_bytes=2 * 1024**3,
    )
