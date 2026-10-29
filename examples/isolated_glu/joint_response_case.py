"""Bounded one-case assembly of authorized response batches; no IO/RNG/fit.

assemble_case consumes actual observations only inside a future approved protocol.
The private fixture seam uses the same arithmetic on small synthetic batches.
"""

from __future__ import annotations
from dataclasses import dataclass
import re

import numpy as np
from numpy.typing import NDArray
from joint_response_join import JointJob
from joint_response_kernel import training_scales

Array = NDArray[np.float64]
Indices = NDArray[np.int64]
GUARD_CENTERS = (3, 7, 11, 15)


def _float_copy(a: Array) -> Array:
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def _index_copy(a: Indices) -> Indices:
    return np.frombuffer(a.tobytes(), dtype=np.int64).reshape(a.shape)


@dataclass(frozen=True, slots=True)
class CaseArrays:
    inputs: Array
    responses: Array
    train_indices: Indices
    guard_indices: Indices
    train_x: Array
    train_y: Array
    guard_x: Array
    guard_y: Array
    sx: float
    sy: float


def _assemble(job: JointJob, probes: int, rows: int) -> CaseArrays:
    if (
        not isinstance(job.job_id, str)
        or re.fullmatch("job0[0-5]", job.job_id) is None
        or any(
            type(v) is not int
            for v in (job.n, job.outputs, job.width_hypothesis, job.batch_count)
        )
        or not 1 <= job.n <= 32
        or not 1 <= job.outputs <= 32
        or not 1 <= job.width_hypothesis <= 96
        or job.batch_count != 2 * probes
        or type(probes) is not int
        or not 4 <= probes <= 17
        or type(rows) is not int
        or not 1 <= rows <= 2112
    ):
        raise ValueError("bounded canonical case profile required")
    total = rows * probes * 2
    x = np.empty((total, job.n), dtype=np.float64)
    y = np.empty((total, job.outputs), dtype=np.float64)
    iterator = iter(job.batches)
    for index in range(job.batch_count):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise ValueError("partial response case; no retry") from exc
        for array, columns in ((batch.inputs, job.n), (batch.responses, job.outputs)):
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.dtype("float64")
                or not array.dtype.isnative
                or array.shape != (rows, columns)
                or not np.isfinite(array).all()
            ):
                raise ValueError("exact native finite response batch required")
        x[index * rows : (index + 1) * rows] = batch.inputs
        y[index * rows : (index + 1) * rows] = batch.responses
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("extra response batch; no retry")
    all_indices = np.arange(total, dtype=np.int64)
    guard_mask = np.isin((all_indices // rows) % probes, GUARD_CENTERS)
    train_indices = all_indices[~guard_mask]
    guard_indices = all_indices[guard_mask]
    train_raw_x, train_raw_y = x[train_indices], y[train_indices]
    sx, sy = training_scales(train_raw_x, train_raw_y)
    # Scaling must not silently erase a component through underflow.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        tx, ty = _float_copy(train_raw_x / sx), _float_copy(train_raw_y / sy)
        del train_raw_x, train_raw_y
        gx, gy = _float_copy(x[guard_indices] / sx), _float_copy(y[guard_indices] / sy)
    if any(not np.isfinite(a).all() for a in (tx, ty, gx, gy)):
        raise FloatingPointError("nonfinite normalized case")
    return CaseArrays(
        _float_copy(x),
        _float_copy(y),
        _index_copy(train_indices),
        _index_copy(guard_indices),
        tx,
        ty,
        gx,
        gy,
        sx,
        sy,
    )


def assemble_case(job: JointJob) -> CaseArrays:
    """Fixed actual profile; call only after all new attempt/input/resource gates."""
    if job.n != 32 or job.outputs != 32 or job.batch_count != 34:
        raise ValueError("registered32/32/34 response job required")
    return _assemble(job, 17, 2112)


def _assemble_fixture(job: JointJob, *, probes: int, rows: int) -> CaseArrays:
    """Small literal tests only; never used by the future registered controller."""
    return _assemble(job, probes, rows)
