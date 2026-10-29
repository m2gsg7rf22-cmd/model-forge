"""Pure candidate SwiGLU arithmetic; no observation access, RNG or persistence.

All inputs are normalized coordinates except physical_parameters. Fixture-level
validation is not reconstruction evidence. The caller owns resource/trace gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CHUNK = 4096


def _matrix(a: Array, columns: int | None = None, rows: int = 71808) -> Array:
    if (
        not isinstance(a, np.ndarray)
        or a.dtype != np.dtype("float64")
        or not a.dtype.isnative
        or a.ndim != 2
        or not 1 <= a.shape[0] <= rows
        or not 1 <= a.shape[1] <= 96
        or not np.isfinite(a).all()
        or (columns is not None and a.shape[1] != columns)
    ):
        raise ValueError("expected bounded finite native FP64 matrix")
    return a


def _freeze(a: Array) -> Array:
    return np.frombuffer(a.tobytes(order="C"), dtype=np.float64).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class Parameters:
    g: Array
    u: Array
    d: Array

    def __post_init__(self) -> None:
        g = _matrix(self.g, rows=96)
        u = _matrix(self.u, rows=96)
        d = _matrix(self.d, rows=32)
        if g.shape != u.shape or g.shape[1] > 32 or d.shape[1] != g.shape[0]:
            raise ValueError("incompatible G/U/D dimensions")
        for name, value in (("g", g), ("u", u), ("d", d)):
            object.__setattr__(self, name, _freeze(value))


def _forward(p: Parameters, x: Array) -> tuple[Array, Array, Array, Array]:
    a, b = x @ p.g.T, x @ p.u.T
    # Every exponential argument is nonpositive; saturation underflow is natural.
    e = np.exp(-np.abs(a))
    s = np.where(a >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e))
    silu = a * s
    h = silu * b
    return h @ p.d.T, b, silu, s * (1.0 + a * (1.0 - s))


def predict(p: Parameters, x: Array) -> Array:
    _matrix(x, p.g.shape[1])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        result = np.empty((len(x), p.d.shape[0]), dtype=np.float64)
        for start in range(0, len(x), CHUNK):
            result[start : start + CHUNK] = _forward(p, x[start : start + CHUNK])[0]
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite prediction")
        return result


def _xy(p: Parameters, x: Array, y: Array, rows: int = 71808) -> None:
    _matrix(x, p.g.shape[1], rows)
    _matrix(y, p.d.shape[0], rows)
    if len(x) != len(y):
        raise ValueError("input/response row mismatch")


def loss_gradient(p: Parameters, x: Array, y: Array) -> tuple[float, Parameters]:
    _xy(p, x, y, CHUNK)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        prediction, b, silu, derivative = _forward(p, x)
        residual = prediction - y
        loss = float(0.5 * np.mean(residual * residual))
        if loss == 0.0 and np.any(residual != 0.0):
            raise FloatingPointError("nonzero sampled loss underflowed to zero")
        error = residual / residual.size
        v = error @ p.d
        grad = Parameters(
            (v * b * derivative).T @ x, (v * silu).T @ x, error.T @ (silu * b)
        )
        if not math.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        return loss, grad


def _rms(a: Array) -> float:
    _matrix(a)
    scale = float(np.max(np.abs(a)))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        value = float(scale * np.sqrt(np.mean((a / scale) ** 2)))
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite RMS")
    return value


def training_scales(train_x: Array, train_y: Array) -> tuple[float, float]:
    if len(train_x) != len(train_y):
        raise ValueError("training rows differ")
    sx, sy = _rms(train_x), _rms(train_y)
    if min(sx, sy) <= 1e-12:
        raise ValueError("degenerate training RMS")
    return sx, sy


def physical_parameters(p: Parameters, sx: float, sy: float) -> Parameters:
    if not all(math.isfinite(v) and v > 1e-12 for v in (sx, sy)):
        raise ValueError("invalid physical scales")
    # Physical folding must not silently erase a nonzero parameter by underflow.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        return Parameters(p.g / sx, p.u / sx, p.d * sy)


@dataclass(frozen=True)
class Step:
    step: int
    sampled_loss_before_update: float
    parameters: Parameters


@dataclass(frozen=True)
class Checkpoint:
    step: int
    parameters: Parameters
    first_moment: Parameters
    second_moment: Parameters
    train_loss: float
    guard_mse: float
    guard_relative_error: float | None


@dataclass(frozen=True)
class FitResult:
    final: Parameters
    selected: Parameters
    selected_step: int
    updates_completed: int
    checkpoints: tuple[Checkpoint, ...]


class NumericalFailure(FloatingPointError):
    """Failed case, with its last finite parameter/moment state; never success."""

    def __init__(self, step: int, p: Parameters, m: Parameters, v: Parameters):
        super().__init__(f"numerical case failure after {step} completed updates")
        self.updates_completed = step
        self.parameters = p
        self.first_moment = m
        self.second_moment = v


def learning_rate(index: int) -> float:
    if type(index) is not int or not 0 <= index < 4000:
        raise ValueError("update index outside frozen schedule")
    return 0.0001 + 0.5 * (0.01 - 0.0001) * (1.0 + math.cos(math.pi * index / 3999))


def _ssq_update(scale: float, total: float, values: Array) -> tuple[float, float]:
    peak = float(np.max(np.abs(values)))
    if peak == 0.0:
        return scale, total
    next_scale = max(scale, peak)
    return next_scale, total * (scale / next_scale) ** 2 + float(
        np.sum((values / next_scale) ** 2)
    )


def _metric(p: Parameters, x: Array, y: Array) -> tuple[float, float | None]:
    error_scale, error_sum, reference_scale, reference_sum = 0.0, 0.0, 0.0, 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for start in range(0, len(x), CHUNK):
            target = y[start : start + CHUNK]
            residual = _forward(p, x[start : start + CHUNK])[0] - target
            error_scale, error_sum = _ssq_update(error_scale, error_sum, residual)
            reference_scale, reference_sum = _ssq_update(
                reference_scale, reference_sum, target
            )
        rms = error_scale * math.sqrt(error_sum / y.size)
        # Do not silently report exact-zero MSE after loss arithmetic underflow.
        with np.errstate(under="raise"):
            mse = float(np.multiply(rms, rms))
        relative = (
            (error_scale / reference_scale) * math.sqrt(error_sum / reference_sum)
            if reference_scale > 0.0
            else None
        )
    if error_scale > 0.0 and (mse == 0.0 or relative == 0.0):
        raise FloatingPointError("nonzero calibration metric underflowed to zero")
    if not math.isfinite(mse) or (relative is not None and not math.isfinite(relative)):
        raise FloatingPointError("nonfinite calibration metric")
    # Only a genuinely zero target has undefined Frobenius relative error.
    return mse, relative


def fit(
    initial: Parameters,
    train_x: Array,
    train_y: Array,
    guard_x: Array,
    guard_y: Array,
    schedule: NDArray[np.int64],
    observer: Callable[[Step | Checkpoint], None],
) -> FitResult:
    """4000 fixed Adam steps. No RNG, restarts, clipping, persistence or oracle.

    Initial checkpoint precedes updates; step events follow complete updates and
    periodic checkpoint events follow their metrics. Callback exceptions propagate
    unchanged. Caller must save intent/arrays/receipts and enforce global budgets.
    NumericalFailure carries finite current state but does not certify convergence.
    Own immutable copies prevent callbacks from modifying subsequent inputs.
    """
    _xy(initial, train_x, train_y)
    _xy(initial, guard_x, guard_y)
    if (
        not isinstance(schedule, np.ndarray)
        or schedule.dtype != np.dtype("int64")
        or not schedule.dtype.isnative
        or schedule.shape != (4000, 256)
        or np.any(schedule < 0)
        or np.any(schedule >= len(train_x))
    ):
        raise ValueError("expected native int64 frozen 4000x256 training schedule")
    tx, ty, gx, gy = (_freeze(a) for a in (train_x, train_y, guard_x, guard_y))
    indices = np.frombuffer(schedule.tobytes(), dtype=np.int64).reshape(4000, 256)
    p = initial
    m = Parameters(np.zeros_like(p.g), np.zeros_like(p.u), np.zeros_like(p.d))
    v = Parameters(m.g, m.u, m.d)
    completed = 0
    records: list[Checkpoint] = []

    def checkpoint() -> Checkpoint:
        train_mse, _ = _metric(p, tx, ty)
        guard_mse, relative = _metric(p, gx, gy)
        return Checkpoint(completed, p, m, v, 0.5 * train_mse, guard_mse, relative)

    try:
        first = checkpoint()
    except (FloatingPointError, OverflowError) as exc:
        raise NumericalFailure(completed, p, m, v) from exc
    records.append(first)
    selected = first
    observer(first)
    for index in range(4000):
        try:
            with np.errstate(
                over="raise", invalid="raise", divide="raise", under="ignore"
            ):
                loss, gradient = loss_gradient(
                    p, tx[indices[index]], ty[indices[index]]
                )
                nm = Parameters(
                    *(
                        0.9 * a + 0.1 * b
                        for a, b in zip(
                            (m.g, m.u, m.d), (gradient.g, gradient.u, gradient.d)
                        )
                    )
                )
                nv = Parameters(
                    *(
                        0.999 * a + 0.001 * b * b
                        for a, b in zip(
                            (v.g, v.u, v.d), (gradient.g, gradient.u, gradient.d)
                        )
                    )
                )
                updated = Parameters(
                    *(
                        a
                        - learning_rate(index)
                        * (b / (1.0 - 0.9 ** (index + 1)))
                        / (np.sqrt(c / (1.0 - 0.999 ** (index + 1))) + 1e-8)
                        for a, b, c in zip(
                            (p.g, p.u, p.d), (nm.g, nm.u, nm.d), (nv.g, nv.u, nv.d)
                        )
                    )
                )
        except (FloatingPointError, OverflowError, ValueError) as exc:
            raise NumericalFailure(completed, p, m, v) from exc
        p, m, v = updated, nm, nv
        completed = index + 1
        observer(Step(completed, loss, p))
        if completed % 256 == 0 or completed == 4000:
            try:
                current = checkpoint()
            except (FloatingPointError, OverflowError) as exc:
                raise NumericalFailure(completed, p, m, v) from exc
            records.append(current)
            if current.guard_mse < selected.guard_mse:
                selected = current
            observer(current)
    return FitResult(p, selected.parameters, selected.step, completed, tuple(records))
