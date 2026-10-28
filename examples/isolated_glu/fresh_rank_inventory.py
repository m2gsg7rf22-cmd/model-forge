"""Allocation-free planning contract; no teacher, array, RNG or filesystem access.

Known width multiset is privileged planning metadata, never an estimator input.
Runtime combined source/caps and monitored preflight remain future requirements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from math import prod
import re
from types import MappingProxyType
from typing import Any

PROFILES = ((4, (2, 4, 8)), (8, (4, 12, 20)), (16, (8, 24, 48)), (32, (16, 48, 96)))
STORES = ("truth", "public", "estimator", "evaluator")
PROBES = 16
SHARD_LIMIT = 8 * 1024**2
METADATA_RESERVE = 8 * 1024**2
HEADER_PER_FIELD = 4096
MIN_FREE_BYTES = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class ShardSpec:
    store: str
    kind: str
    fields: Mapping[str, tuple[int, ...]]
    count: int

    def __post_init__(self) -> None:
        if (
            self.store not in STORES
            or type(self.count) is not int
            or not 1 <= self.count <= 32
        ):
            raise ValueError("Bounded store/count required")
        fields = dict(self.fields)
        if not 1 <= len(fields) <= 16 or any(
            not isinstance(key, str)
            or not re.fullmatch(r"[a-zA-Z0-9_]{1,64}", key)
            or key in {"file", "allow_pickle"}
            for key in fields
        ):
            raise ValueError("One to16 bounded nonreserved ASCII field names required")
        if any(
            type(shape) is not tuple
            or not 1 <= len(shape) <= 4
            or any(type(x) is not int or x <= 0 for x in shape)
            for shape in fields.values()
        ):
            raise ValueError("Immutable positive-integer shape tuples required")
        object.__setattr__(self, "fields", MappingProxyType(fields))
        if self.bytes_upper_bound > SHARD_LIMIT:
            raise ValueError("Planned NPZ exceeds8MiB shard cap")

    @property
    def payload_bytes(self) -> int:
        return 8 * sum(prod(shape) for shape in self.fields.values())

    @property
    def bytes_upper_bound(self) -> int:
        return self.payload_bytes + HEADER_PER_FIELD * len(self.fields)


def profile_schemas(n: int, width: int) -> tuple[ShardSpec, ...]:
    if (
        type(n) is not int
        or n not in (4, 8, 16, 32)
        or type(width) is not int
        or not 1 <= width <= 96
    ):
        raise ValueError("Bounded registered dimension and width1..96 required")
    p = n * (n + 1) // 2
    q = min(PROBES * n, p)
    count = min(q, p - 1)
    context = dict(
        g=(width, n),
        u=(width, n),
        d=(n, width),
        reference_g=(width, n),
        reference_u=(width, n),
        reference_d=(n, width),
        probes=(PROBES, n),
        random_directions=(64, n),
        atoms=(2 * width, p),
        preactivations=(PROBES, width),
    )
    bases: dict[str, tuple[int, ...]] = {
        kind + "_basis": (q, p) for kind in ("fine", "coarse", "richardson")
    }
    bases.update(
        {kind + "_spectrum": (p,) for kind in ("fine", "coarse", "richardson")}
    )
    choices: dict[str, tuple[int, ...]] = {
        kind + "_projector": (p, p) for kind in ("strict", "gap", "stable")
    }
    choices.update(
        log_gaps=(3, count),
        gap_ratios=(3, count),
        eligible=(3, count),
        resolution_clipped=(3, count),
        stability=(count,),
        resolution_floor=(1,),
        ranks=(3,),
    )
    evaluation = dict(
        reference_spectrum=(p,),
        reference_ranks=(3,),
        ranks=(6,),
        gate_residuals=(6, width),
        random_residuals=(6, 64),
        lost_subspace=(6,),
        extra_subspace=(6,),
        reference_energy_loss=(6,),
        observed_energy_loss=(6,),
        qualifies=(6,),
        reference_sensitive=(1,),
    )
    return (
        ShardSpec("truth", "context", context, 1),
        ShardSpec("truth", "inputs", dict(inputs=(4 * p, n)), 2 * PROBES),
        ShardSpec("truth", "responses", dict(responses=(4 * p, n)), 2 * PROBES),
        ShardSpec("truth", "reference", dict(hessian=(8, n, n, n)), 2),
        ShardSpec("public", "observations", dict(hessian=(8, n, n, n)), 6),
        ShardSpec("estimator", "bases", bases, 1),
        ShardSpec("estimator", "choices", choices, 1),
        ShardSpec("evaluator", "evaluation", evaluation, 1),
    )


def manifest_record_upper_bound() -> int:
    return len(
        (
            json.dumps(
                dict(name="x" * 80 + ".npz", bytes=SHARD_LIMIT, sha256="0" * 64),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )


def worst_schemas() -> tuple[ShardSpec, ...]:
    return tuple(
        replace(s, count=1, kind="hessian" if s.kind == "reference" else s.kind)
        for s in profile_schemas(32, 96)
        if s.kind != "observations"
    )


def inventory_plan() -> dict[str, Any]:
    stores = {
        name: dict(
            shards=0,
            shard_limit_bytes=0,
            payload_bytes=0,
            array_bytes_upper_bound=0,
            manifest_bytes_upper_bound=0,
            bytes_upper_bound=0,
        )
        for name in STORES
    }
    vectors = batches = largest = 0
    for n, widths in PROFILES:
        for width in widths:
            vectors += 2 * PROBES * 2 * 4 * (n * (n + 1) // 2)
            batches += 2 * PROBES * 2
            for s in profile_schemas(n, width):
                entry = stores[s.store]
                entry["shard_limit_bytes"] = max(
                    entry["shard_limit_bytes"], s.bytes_upper_bound
                )
                entry["shards"] += 2 * s.count
                entry["payload_bytes"] += 2 * s.count * s.payload_bytes
                entry["array_bytes_upper_bound"] += 2 * s.count * s.bytes_upper_bound
                largest = max(largest, s.bytes_upper_bound)
    for name, entry in stores.items():
        entry["manifest_bytes_upper_bound"] = (
            entry["shards"] * manifest_record_upper_bound()
        )
        entry["bytes_upper_bound"] = (
            entry["array_bytes_upper_bound"] + entry["manifest_bytes_upper_bound"]
        )
        if entry["shards"] > 8192 or entry["manifest_bytes_upper_bound"] > (
            128 * 1024 if name == "public" else 1024**2
        ):
            raise ValueError("Planned store manifest exceeds shared/public cap")
    preflight = (
        sum(s.bytes_upper_bound for s in worst_schemas())
        + 7 * manifest_record_upper_bound()
    )
    total = (
        sum(s["bytes_upper_bound"] for s in stores.values())
        + preflight
        + METADATA_RESERVE
    )
    if largest > SHARD_LIMIT or preflight > 24 * 1024**2 or total >= 1024**3:
        raise ValueError("Combined planned artifacts exceed preregistered ceiling")
    return dict(
        teachers=sum(len(widths) for _, widths in PROFILES),
        cases=2 * sum(len(widths) for _, widths in PROFILES),
        vector_calls=vectors,
        callback_batches=batches,
        scientific_shards=sum(s["shards"] for s in stores.values()),
        stores=stores,
        largest_shard_upper_bound=largest,
        preflight_shards=7,
        preflight_bytes_upper_bound=preflight,
        metadata_reserve_bytes=METADATA_RESERVE,
        bytes_upper_bound=total,
        minimum_free_disk_bytes=MIN_FREE_BYTES,
        planning_only=True,
        data_generation_approved=False,
        private_gate="closed",
    )
