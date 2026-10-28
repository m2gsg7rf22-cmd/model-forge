"""Allocation-free privileged inventory for distinct paired16/17 acquisition."""

from __future__ import annotations
from dataclasses import replace
from typing import Any
from fresh_rank_inventory import ShardSpec, manifest_record_upper_bound
from fresh_rank_protocol_inventory import profile_schemas as base_schemas

WIDTHS = (16, 48, 96)
STORES = ("truth", "public", "estimator", "evaluator")


def profile_schemas(width: int) -> tuple[ShardSpec, ...]:
    base = {s.kind: s for s in base_schemas(32, width)}
    ctx = dict(base["context"].fields)
    ctx.update(probes=(17, 32), preactivations=(17, width))
    result = [replace(base["context"], fields=ctx)]
    for kind in ("inputs", "responses"):
        result.extend(
            replace(base[kind], kind=kind + str(step), count=17) for step in range(2)
        )
    result.extend(
        (
            base["reference"],
            replace(
                base["reference"],
                kind="reference_tail",
                fields=dict(hessian=(1, 32, 32, 32)),
                count=1,
            ),
            base["observations"],
            replace(
                base["observations"],
                kind="observations_tail",
                fields=dict(hessian=(1, 32, 32, 32)),
                count=3,
            ),
        )
    )
    for budget in (16, 17):
        q = min(budget * 32, 528)
        count = min(q, 527)
        fields = dict(base["bases"].fields)
        for key in ("fine_basis", "coarse_basis", "richardson_basis"):
            fields[key] = (q, 528)
        result.append(replace(base["bases"], kind=f"bases{budget}", fields=fields))
        fields = dict(base["choices"].fields)
        for key in ("log_gaps", "gap_ratios", "eligible", "resolution_clipped"):
            fields[key] = (3, count)
        fields["stability"] = (count,)
        result.append(replace(base["choices"], kind=f"choices{budget}", fields=fields))
        result.append(replace(base["evaluation"], kind=f"evaluation{budget}"))
    return tuple(result)


def worst_schemas() -> tuple[ShardSpec, ...]:
    source = {s.kind: s for s in profile_schemas(96)}
    names = (
        "context",
        "inputs0",
        "responses0",
        "reference",
        "reference_tail",
        "bases16",
        "choices16",
        "bases17",
        "choices17",
        "evaluation16",
    )
    return tuple(replace(source[k], count=1) for k in names)


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
    for width in WIDTHS:
        for spec in profile_schemas(width):
            entry = stores[spec.store]
            entry["shards"] += 2 * spec.count
            entry["payload_bytes"] += 2 * spec.count * spec.payload_bytes
            entry["array_bytes_upper_bound"] += 2 * spec.count * spec.bytes_upper_bound
            entry["shard_limit_bytes"] = max(
                entry["shard_limit_bytes"], spec.bytes_upper_bound
            )
    for entry in stores.values():
        entry["manifest_bytes_upper_bound"] = (
            entry["shards"] * manifest_record_upper_bound()
        )
        entry["bytes_upper_bound"] = (
            entry["array_bytes_upper_bound"] + entry["manifest_bytes_upper_bound"]
        )
    preflight = (
        sum(s.bytes_upper_bound for s in worst_schemas())
        + 10 * manifest_record_upper_bound()
    )
    total = (
        sum(e["bytes_upper_bound"] for e in stores.values()) + preflight + 8 * 1024**2
    )
    largest = max(e["shard_limit_bytes"] for e in stores.values())
    if largest > 8 * 1024**2 or preflight > 40 * 1024**2 or total >= 1024**3:
        raise ValueError("Combined row-acquisition inventory exceeds ceiling")
    return dict(
        teachers=3,
        base_cases=6,
        cases=12,
        probes=17,
        budgets=[16, 17],
        vector_calls=430848,
        callback_batches=204,
        scientific_shards=sum(e["shards"] for e in stores.values()),
        stores=stores,
        largest_shard_upper_bound=largest,
        preflight_shards=10,
        preflight_bytes_upper_bound=preflight,
        metadata_reserve_bytes=8 * 1024**2,
        bytes_upper_bound=total,
        minimum_free_disk_bytes=2 * 1024**3,
        planning_only=True,
        data_generation_approved=False,
        private_gate="closed",
    )
