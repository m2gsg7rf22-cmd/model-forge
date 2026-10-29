"""Prospective scalar caps only; no arrays/IO/RNG and no resource proof.

Later controller must enforce all listed actual size/layout/partial-write bounds.
Prior known bytes include the original global development marker separately.
"""

from __future__ import annotations
from joint_response_inventory import Field, Shard


def transport_specs() -> tuple[Shard, Shard]:
    return (
        Shard(
            "parameters",
            tuple(
                Field(name, "float64", shape)
                for name, shape in [("g", (96, 32)), ("u", (96, 32)), ("d", (32, 96))]
            ),
        ),
        Shard("predictions", (Field("predictions", "float64", (8192, 32)),)),
    )


def memory_plan() -> dict[str, int]:
    largest = max(s.encoded_bound for s in transport_specs())
    return dict(
        original_and_immutable_assembly_overlap=4 * 71808 * 32 * 8,
        index_and_mask_work=8 * 71808 * 8,
        response_archive_decode_allowance=4 * 544768,
        checkpoint_decode_allowance=3 * (9 * 96 * 32 * 8 + 10 * 4096),
        parameter_and_conversion_allowance=16 * 3 * 96 * 32 * 8,
        forward_workspace=24 * 4096 * 96 * 8,
        prediction_and_comparison_allowance=8 * 8192 * 32 * 8,
        transport_serialization_allowance=3 * largest,
        rank_loader_allowance=64 * 1024**2,
        metadata_hash_and_report_allowance=8 * 1024**2,
        runtime_allocator_monitor_allowance=96 * 1024**2,
    )


def inventory() -> dict[str, int | bool]:
    specs = transport_specs()
    largest = max(s.encoded_bound for s in specs)
    prior = 857667407 + 1209
    older_markers = 3 * 128 * 1024
    root = 6 * 128 * 1024
    cases = 6 * (128 * 1024 + 17 * 4096)
    metadata_temp = root + cases
    marker = 2 * 128 * 1024
    # Transport includes two NPZs and one bounded manifest; temporary/partial
    # reserve is two largest NPZs plus one manifest. No data/checkpoint copies.
    transport = sum(s.encoded_bound for s in specs) + 4096
    partial = 2 * largest + 4096
    combined = (
        prior
        + older_markers
        + root
        + cases
        + metadata_temp
        + marker
        + transport
        + partial
    )
    memory = sum(memory_plan().values())
    return dict(
        prior_output_and_ancestor_bytes=857667407,
        prior_development_marker_bytes=1209,
        prior_known_bytes=prior,
        older_marker_reserve=older_markers,
        root_metadata_bytes=root,
        case_metadata_bytes=cases,
        temporary_metadata_bytes=metadata_temp,
        audit_marker_and_temp_bytes=marker,
        transport_shards=len(specs),
        transport_encoded_upper_bound=transport,
        transport_partial_reserve=partial,
        largest_shard_upper_bound=largest,
        combined_artifacts_upper_bound=combined,
        memory_planning_bytes=memory,
        preflight_literal_bytes=memory - 96 * 1024**2,
        resource_proof=False,
        execution_approved=False,
    )
