"""Full-protocol inventory extension; the reviewed base inventory stays frozen.

Allocation-free privileged planning metadata, never imported by estimator.
"""

from __future__ import annotations
from dataclasses import replace
from typing import Any
import fresh_rank_inventory as base
from fresh_rank_inventory import PROFILES as PROFILES, ShardSpec as ShardSpec


def profile_schemas(n: int, width: int) -> tuple[ShardSpec, ...]:
    original = base.profile_schemas(n, width)
    fields = dict(original[-1].fields)
    fields["reference_relative_residual"] = fields.pop("reference_energy_loss")
    fields["observed_relative_residual"] = fields.pop("observed_energy_loss")
    fields["reference_component_residuals"] = (6, n * (n + 1) // 2)
    return (*original[:-1], replace(original[-1], fields=fields))


def worst_schemas() -> tuple[ShardSpec, ...]:
    return tuple(
        replace(s, count=1, kind="hessian" if s.kind == "reference" else s.kind)
        for s in profile_schemas(32, 96)
        if s.kind != "observations"
    )


def inventory_plan() -> dict[str, Any]:
    plan = base.inventory_plan()
    entry = plan["stores"]["evaluator"]
    payload = upper = largest = 0
    for n, widths in PROFILES:
        for width in widths:
            spec = profile_schemas(n, width)[-1]
            payload += 2 * spec.payload_bytes
            upper += 2 * spec.bytes_upper_bound
            largest = max(largest, spec.bytes_upper_bound)
    entry.update(
        payload_bytes=payload,
        array_bytes_upper_bound=upper,
        shard_limit_bytes=largest,
        bytes_upper_bound=upper + entry["manifest_bytes_upper_bound"],
    )
    preflight = (
        sum(s.bytes_upper_bound for s in worst_schemas())
        + 7 * base.manifest_record_upper_bound()
    )
    plan.update(
        preflight_bytes_upper_bound=preflight,
        bytes_upper_bound=sum(s["bytes_upper_bound"] for s in plan["stores"].values())
        + preflight
        + plan["metadata_reserve_bytes"],
    )
    if plan["bytes_upper_bound"] >= 1024**3 or largest > base.SHARD_LIMIT:
        raise ValueError("Expanded protocol inventory exceeds ceiling")
    return plan
