"""Injected optimizer streams: code only, no default generator or execution.

Production factory may wrap PCG64 ONLY after the future complete protocol gates,
including case normalization. Tests inject literal streams, never registered RNG.
"""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from joint_response_case import CaseArrays
from joint_response_join import JointJob
from joint_response_kernel import Parameters

Array = NDArray[np.float64]
Indices = NDArray[np.int64]


class Stream(Protocol):
    def state(self) -> dict[str, Any]: ...
    def normal(self, shape: tuple[int, int]) -> Array: ...
    def indices(self, high: int, shape: tuple[int, int]) -> Indices: ...


@dataclass(frozen=True, slots=True)
class OptimizerInputs:
    initial: Parameters
    schedule: Indices
    initial_state_before: str
    initial_state_after: str
    batch_state_before: str
    batch_state_after: str


def stream_seed(job_id: str, role: str) -> int:
    if (
        not isinstance(job_id, str)
        or re.fullmatch("job0[0-5]", job_id) is None
        or role not in ("initial", "batches")
    ):
        raise ValueError("exact opaque paired seed domain required")
    group = int(job_id[-2:]) // 2
    domain = f"modelforge-joint-response-development-v1/group{group:02d}/{role}"
    return int.from_bytes(hashlib.sha256(domain.encode("ascii")).digest(), "big")


def _snapshot(stream: Stream) -> str:
    state = stream.state()
    if (
        not isinstance(state, dict)
        or set(state) != {"bit_generator", "state", "has_uint32", "uinteger"}
        or state["bit_generator"] != "PCG64"
        or not isinstance(state["state"], dict)
        or set(state["state"]) != {"state", "inc"}
    ):
        raise ValueError("exact PCG64 state schema required")
    for key, maximum in (("has_uint32", 1), ("uinteger", 2**32 - 1)):
        if type(state[key]) is not int or not 0 <= state[key] <= maximum:
            raise ValueError("native PCG64 state integer required")
    if (
        any(type(v) is not int or not 0 <= v < 2**128 for v in state["state"].values())
        or state["state"]["inc"] % 2 != 1
    ):
        raise ValueError("valid native PCG64 state/inc required")
    result = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(result.encode()) > 4096:
        raise ValueError("PCG64 state cap exceeded")
    return result


def _prepare(
    job_id: str,
    n: int,
    outputs: int,
    width: int,
    train_rows: int,
    factory: Callable[[int], Stream],
) -> OptimizerInputs:
    initial_seed, batch_seed = (
        stream_seed(job_id, "initial"),
        stream_seed(job_id, "batches"),
    )
    if (
        any(type(v) is not int for v in (n, outputs, width, train_rows))
        or not 1 <= n <= 32
        or not 1 <= outputs <= 32
        or not 1 <= width <= 96
        or not 1 <= train_rows <= 54912
    ):
        raise ValueError("bounded native optimizer input dimensions required")
    initial_stream = factory(initial_seed)
    batch_stream = factory(batch_seed)
    if initial_stream is batch_stream:
        raise ValueError("initial and batch streams must be distinct objects")
    initial_before, batch_before = _snapshot(initial_stream), _snapshot(batch_stream)

    def normal(shape: tuple[int, int], divisor: int) -> Array:
        a = initial_stream.normal(shape)
        if (
            not isinstance(a, np.ndarray)
            or a.shape != shape
            or a.dtype != np.dtype("float64")
            or not a.dtype.isnative
            or not np.isfinite(a).all()
        ):
            raise ValueError("exact finite native standard-normal array required")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            return a / math.sqrt(divisor)

    initial = Parameters(
        normal((width, n), n), normal((width, n), n), normal((outputs, width), width)
    )
    indices = batch_stream.indices(train_rows, (4000, 256))
    if (
        not isinstance(indices, np.ndarray)
        or indices.dtype != np.dtype("int64")
        or not indices.dtype.isnative
        or indices.shape != (4000, 256)
        or np.any(indices < 0)
        or np.any(indices >= train_rows)
    ):
        raise ValueError("exact native bounded4000x256 schedule required")
    schedule = np.frombuffer(indices.tobytes(), dtype=np.int64).reshape(indices.shape)
    return OptimizerInputs(
        initial,
        schedule,
        initial_before,
        _snapshot(initial_stream),
        batch_before,
        _snapshot(batch_stream),
    )


def prepare_inputs(
    job: JointJob, case: CaseArrays, factory: Callable[[int], Stream]
) -> OptimizerInputs:
    """Registered dimensions/scales first; global source/attempt gates belong to caller.

    The future trace must commit all states/initial/schedule before the first fit
    update. Equal paired seeds do not themselves verify the factory or its draws.
    """
    if (
        job.n != 32
        or job.outputs != 32
        or job.batch_count != 34
        or not isinstance(case, CaseArrays)
        or case.inputs.shape != (71808, 32)
        or case.responses.shape != (71808, 32)
        or case.train_x.shape != (54912, 32)
        or case.train_y.shape != (54912, 32)
        or case.guard_x.shape != (16896, 32)
        or case.guard_y.shape != (16896, 32)
        or not all(math.isfinite(v) and v > 1e-12 for v in (case.sx, case.sy))
    ):
        raise ValueError(
            "registered prevalidated normalized case required before streams"
        )
    return _prepare(job.job_id, 32, 32, job.width_hypothesis, 54912, factory)


def _prepare_fixture(
    job_id: str,
    n: int,
    outputs: int,
    width: int,
    train_rows: int,
    factory: Callable[[int], Stream],
) -> OptimizerInputs:
    """Literal injected streams only; future registered controller must not use."""
    return _prepare(job_id, n, outputs, width, train_rows, factory)
