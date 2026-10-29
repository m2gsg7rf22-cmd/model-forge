"""Scalar prospective trace caps only; no arrays, IO, RNG or execution approval.

Storage bounds hold only if the future writer enforces every schema, header,
record, count and reserve cap. Memory is a planning model, never measured RSS.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Shard:
    name: str
    fields: tuple[Field, ...]

    @property
    def encoded_bound(self) -> int:
        # ZIP_STORED, bounded short member names, <=4096 per NPY/ZIP field
        # plus one4096byte archive reserve; future writer must enforce this cap.
        return sum(prod(f.shape) * 8 for f in self.fields) + 4096 * (
            len(self.fields) + 1
        )


def trace_specs(width: int = 96) -> tuple[Shard, ...]:
    if type(width) is not int or not 1 <= width <= 96:
        raise ValueError("native width hypothesis in1..96 required")
    params = tuple(
        Field(name, "float64", shape)
        for name, shape in (("g", (width, 32)), ("u", (width, 32)), ("d", (32, width)))
    )
    state = tuple(
        Field(prefix + f.name, f.dtype, f.shape)
        for prefix in ("", "m_", "v_")
        for f in params
    )
    specs = [
        Shard("initial", params),
        Shard("schedule", (Field("indices", "int64", (4000, 256)),)),
        Shard(
            "split-indices",
            (Field("train", "int64", (54912,)), Field("guard", "int64", (16896,))),
        ),
        Shard("selected-physical", params),
        Shard("failure-state", state),
    ]
    specs += [
        Shard(f"checkpoint-{step:04d}", state)
        for step in (0, *range(256, 4000, 256), 4000)
    ]
    specs += [
        Shard(
            f"prediction-{start // 8192:02d}",
            (Field("predictions", "float64", (min(8192, 71808 - start), 32)),),
        )
        for start in range(0, 71808, 8192)
    ]
    return tuple(specs)


def memory_plan() -> dict[str, int]:
    """Explicit single-case live-array model plus runtime/allocator allowance.

    Conservative overlaps are counted even when phases should release buffers.
    This excludes a guarantee for native libraries/allocators/children: actual
    full preflight and sampled process+children monitoring remain mandatory.
    """
    full_pair = 2 * 71808 * 32 * 8
    maximum_shard = max(s.encoded_bound for s in trace_specs())
    return dict(
        original_normalized_kernel_and_assembly_copies=4 * full_pair,
        three_schedule_copies=3 * 4000 * 256 * 8,
        seventeen_parameter_moment_states=17 * 9 * 96 * 32 * 8,
        bounded_forward_gradient_workspace=16 * 4096 * 96 * 8,
        parameter_temporary_allowance=32 * 3 * 96 * 32 * 8,
        serialization_copies=3 * maximum_shard,
        frozen_rank_loader_allowance=64 * 1024**2,
        runtime_allocator_and_monitor_allowance=96 * 1024**2,
    )


def inventory() -> dict[str, int | bool]:
    specs = trace_specs()
    largest = max(s.encoded_bound for s in specs)
    prior = 671428083  # parent534980568 + gate135203053 + readpostouter1244462
    traces = 6 * sum(s.encoded_bound for s in specs)
    progress = 6 * 4000 * 512  # one bounded JSONL step/loss event per complete update
    metadata = (
        8 * 1024**2
    )  # RNGstates/scales/17intents/metrics/reports/source+shard manifests
    transport = 2 * largest
    interruption = 2 * largest  # bounded failed-write/serialization tail, no retry
    return dict(
        prior_retained_upper_bound=prior,
        trace_shards=6 * len(specs),
        trace_encoded_upper_bound=traces,
        largest_shard_upper_bound=largest,
        progress_upper_bound=progress,
        metadata_reserve=metadata,
        transport_upper_bound=transport,
        interruption_reserve=interruption,
        combined_artifacts_upper_bound=prior
        + traces
        + progress
        + metadata
        + transport
        + interruption,
        minimum_free_disk_bytes=2 * 1024**3,
        memory_planning_bytes=sum(memory_plan().values()),
        memory_is_measured=False,
    )
