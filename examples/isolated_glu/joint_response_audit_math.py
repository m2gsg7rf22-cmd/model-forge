"""Independent candidate arithmetic; pure bounded arrays, no IO/RNG/fit imports.

SiLU uses signed log-magnitude arithmetic, independent of the fitted kernel.
Scaled square reductions may discard individually unrepresentable normalized
terms, but always retain a unit maximum, so nonzero residuals cannot become zero.
Every numerical outcome here is calibration consistency, never a certificate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
ABS = 1e-12
REL = 1e-10
MAX_COMPONENTS = 71808 * 32


def _array(a: Array, rows: int = 4096, cols: int = 32) -> None:
    if (
        not isinstance(a, np.ndarray)
        or a.dtype != np.dtype("float64")
        or not a.dtype.isnative
        or a.ndim != 2
        or not 1 <= a.shape[0] <= rows
        or not 1 <= a.shape[1] <= cols
        or not np.isfinite(a).all()
    ):
        raise ValueError("bounded native finite FP64 matrix required")


def forward(x: Array, g: Array, u: Array, d: Array) -> Array:
    """Independent log-magnitude SiLU; at most4096rows,32in/out,96hidden."""
    _array(x)
    _array(g, 96)
    _array(u, 96)
    _array(d, 32, 96)
    if g.shape != u.shape or g.shape[1] != x.shape[1] or d.shape[1] != g.shape[0]:
        raise ValueError("incompatible candidate dimensions")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        z = x @ g.T
        silu = np.zeros_like(z)
        nonzero = z != 0
        v = z[nonzero]
        # logaddexp can underflow only its negligible exp correction at saturation.
        # Permit that local rounding; the final nonzero SiLU exp stays guarded.
        with np.errstate(under="ignore"):
            denominator = np.logaddexp(0.0, -v)
        silu[nonzero] = np.sign(v) * np.exp(np.log(np.abs(v)) - denominator)
        result = (silu * (x @ u.T)) @ d.T
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite independent forward")
    return result


@dataclass
class _Squares:
    scale: float = 0.0
    squares: float = 0.0
    count: int = 0

    def add(self, a: Array) -> None:
        _array(a)
        if self.count + a.size > MAX_COMPONENTS:
            raise ValueError("single case component cap exceeded")
        self.count += a.size
        maximum = float(np.max(np.abs(a)))
        if maximum == 0:
            return
        new_scale = max(self.scale, maximum)
        # Underflow in negligible scaled terms cannot erase the retained maximum.
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            prior = self.squares * (self.scale / new_scale) ** 2
            block = float(np.sum(np.square(a / new_scale), dtype=np.float64))
        self.squares = math.fsum((prior, block))
        self.scale = new_scale
        if not math.isfinite(self.squares) or self.squares < 1:
            raise FloatingPointError("invalid scaled square sum")

    def root_mean(self) -> float:
        if self.count == 0:
            raise ValueError("empty reduction")
        value = self.scale * math.sqrt(self.squares / self.count)
        if not math.isfinite(value) or (self.scale != 0 and value == 0):
            raise FloatingPointError("invalid or underflowed RMS")
        return value


def rms(blocks: Iterable[Array]) -> float:
    stat = _Squares()
    columns = None
    for block in blocks:
        _array(block)
        if columns is not None and columns != block.shape[1]:
            raise ValueError("inconsistent reduction columns")
        columns = block.shape[1]
        stat.add(block)
    return stat.root_mean()


def metrics(blocks: Iterable[tuple[Array, Array]]) -> dict[str, float | int | None]:
    error, target = _Squares(), _Squares()
    columns = None
    for prediction, reference in blocks:
        _array(prediction)
        _array(reference)
        if prediction.shape != reference.shape or (
            columns is not None and columns != reference.shape[1]
        ):
            raise ValueError("metric shape mismatch")
        columns = reference.shape[1]
        with np.errstate(over="raise", invalid="raise", under="raise"):
            residual = prediction - reference
        error.add(residual)
        target.add(reference)
    rmse = error.root_mean()
    mse = rmse * rmse
    half = mse / 2
    if not math.isfinite(mse) or (error.scale != 0 and (mse == 0 or half == 0)):
        raise FloatingPointError("nonzero loss underflow or overflow")
    relative = None
    if target.scale != 0:
        if error.scale == 0:
            relative = 0.0
        else:
            # Logs avoid an overflowing intermediate norm or scale ratio.
            try:
                relative = math.exp(
                    math.log(error.scale)
                    - math.log(target.scale)
                    + 0.5 * (math.log(error.squares) - math.log(target.squares))
                )
            except OverflowError as exc:
                raise FloatingPointError("relative metric overflow") from exc
            if not math.isfinite(relative) or relative == 0:
                raise FloatingPointError("nonzero relative metric underflow")
    return {
        "mse": mse,
        "half_mse": half,
        "relative_error": relative,
        "components": error.count,
    }


def scalar_agrees(value: float, reference: float) -> bool:
    if not all(math.isfinite(v) for v in (value, reference)):
        raise ValueError("finite scalar comparison required")
    return abs(value - reference) <= ABS + REL * abs(reference)


def array_agrees(value: Array, reference: Array) -> bool:
    _array(value, 71808, 96)
    _array(reference, 71808, 96)
    if value.size > MAX_COMPONENTS or reference.size > MAX_COMPONENTS:
        raise ValueError("comparison component cap exceeded")
    if value.shape != reference.shape:
        return False
    # One bounded chunk at a time; include both maximum and Frobenius criteria.
    e, t = _Squares(), _Squares()
    maximum = 0.0
    refmax = 0.0
    with np.errstate(over="raise", invalid="raise", under="raise"):
        for start in range(0, len(value), 4096):
            a = value[start : start + 4096]
            b = reference[start : start + 4096]
            delta = a - b
            # _Squares accepts at most32columns; flatten bounded parameter blocks.
            flat = delta.reshape(-1, 1)
            ref = b.reshape(-1, 1)
            for k in range(0, len(flat), 4096):
                e.add(flat[k : k + 4096])
                t.add(ref[k : k + 4096])
            maximum = max(maximum, float(np.max(np.abs(delta))))
            refmax = max(refmax, float(np.max(np.abs(b))))
    # Divide the Frobenius inequality by sqrt(component count): RMS equivalent.
    return maximum <= ABS + REL * refmax and e.root_mean() <= ABS + REL * t.root_mean()


def selection(
    values: Sequence[float], saved: Sequence[float]
) -> dict[str, int | bool | list[int]]:
    if (
        not 1 <= len(values) <= 17
        or len(values) != len(saved)
        or any(not math.isfinite(v) or v < 0 for v in (*values, *saved))
    ):
        raise ValueError("bounded finite nonnegative checkpoint losses required")
    index = min(range(len(values)), key=lambda k: values[k])
    tolerance = [ABS + REL * abs(v) for v in saved]
    ambiguous = [
        i
        for i, v in enumerate(values)
        if i != index and abs(v - values[index]) <= tolerance[i] + tolerance[index]
    ]
    return {"index": index, "ambiguous": bool(ambiguous), "ambiguous_with": ambiguous}
