"""Pure privileged diagnostic mathematics: no RNG, queries, fitting or file I/O.

Floating spans are not exact-rank, gate-recovery or reconstruction certificates.
A separately reviewed bounded data runner/controller is still required.
"""

from __future__ import annotations

from typing import TypedDict
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
RANK_RTOL = 1e-10


class Spectrum(TypedDict):
    singular_values: Array
    basis: Array
    rank: int
    gap: float | None
    gap_reason: str | None


def _finite(a: Array, name: str) -> Array:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"Empty/nonfinite {name}")
    return a


def svec(h: Array) -> Array:
    """Upper triangular row-major, sqrt2 offdiagonal; Frobenius isometry."""
    h = _finite(h, "symmetric tensor")
    if h.ndim < 2 or h.shape[-1] != h.shape[-2]:
        raise ValueError("Square symmetric tensor required")
    if not np.array_equal(h, h.swapaxes(-1, -2)):
        raise ValueError("Asymmetric tensor")
    i, j = np.triu_indices(h.shape[-1])
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            result = h[..., i, j] * np.where(i == j, 1.0, np.sqrt(2.0))
    except FloatingPointError as exc:
        raise ValueError("svec overflow/underflow") from exc
    return np.asarray(result, dtype=np.float64)


def _factors(g: Array, u: Array) -> tuple[Array, Array]:
    g, u = _finite(g, "gate"), _finite(u, "up")
    if g.ndim != 2 or u.shape != g.shape:
        raise ValueError("Matching width-by-input factors required")
    return g, u


def symmetric_atoms(g: Array, u: Array) -> Array:
    """Natural UNNORMALIZED interleaved gg^T, gu^T+ug^T svec rows."""
    g, u = _factors(g, u)
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            gg = g[:, :, None] * g[:, None, :]
            gu = g[:, :, None] * u[:, None, :]
            cross = gu + gu.swapaxes(-1, -2)
    except FloatingPointError as exc:
        raise ValueError("Atom products overflow/underflow") from exc
    return svec(np.stack((gg, cross), axis=1)).reshape(2 * len(g), -1)


def analytic_hessian(g: Array, u: Array, d: Array, x: Array) -> Array:
    """Smooth FP64 diagnostic, never derivative of rounded FP32 execution.

    Explicit caller-supplied weights may be FP32-rounded then promoted. Computed
    zeros (including sigmoid saturation) are not ideal mathematical certificates.
    """
    g, u = _factors(g, u)
    d, x = _finite(d, "down"), _finite(x, "probes")
    if d.ndim != 2 or d.shape[1] != len(g) or x.ndim != 2 or x.shape[1] != g.shape[1]:
        raise ValueError("Incompatible down/probe shapes")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            z, value = x @ g.T, x @ u.T
            if not np.isfinite(z).all() or not np.isfinite(value).all():
                raise ValueError("Nonfinite preactivations")
            t = np.exp(-np.abs(z))
            sig = np.where(z >= 0, 1 / (1 + t), t / (1 + t))
            deriv = t / (1 + t) ** 2
            first = sig + z * deriv
            second = deriv * (2 + z * (1 - 2 * sig))
            result = np.empty((len(x), len(d), g.shape[1], g.shape[1]))
            i, j = np.triu_indices(g.shape[1])
            for k in range(len(x)):
                gg = np.einsum("ow,wi,wj->oij", d * (second[k] * value[k]), g, g)
                gu = np.einsum("ow,wi,wj->oij", d * first[k], g, u)
                h = gg + gu + gu.swapaxes(-1, -2)
                result[k, :, i, j] = h[:, i, j].T
                result[k, :, j, i] = h[:, i, j].T
    except FloatingPointError as exc:
        raise ValueError("Analytic reference arithmetic failure") from exc
    return _finite(result, "analytic Hessian")


def spectrum(rows: Array) -> Spectrum:
    """Row span via right singular vectors, STRICT relative threshold1e-10.

    One global scale protects SVD; no row normalization. Full p spectrum includes
    structural zero padding when rows<p. Gap null at zero/full rank or zero next.
    """
    rows = _finite(rows, "span matrix")
    if rows.ndim != 2:
        raise ValueError("Two dimensional span matrix required")
    p = rows.shape[1]
    scale = float(np.max(np.abs(rows)))
    if scale == 0:
        return {
            "singular_values": np.zeros(p),
            "basis": np.zeros((0, p)),
            "rank": 0,
            "gap": None,
            "gap_reason": "zero_rank",
        }
    normalized = rows / scale
    if np.any((rows != 0) & (normalized == 0)):
        raise ValueError("Span normalization underflow")
    _, singular, right = np.linalg.svd(normalized, full_matrices=False)
    if not np.isfinite(singular).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite SVD")
    rank = int(np.count_nonzero(singular > RANK_RTOL * singular[0]))
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            values = np.pad(singular * scale, (0, p - len(singular)))
    except FloatingPointError as exc:
        raise ValueError("Spectrum rescaling overflow/underflow") from exc
    gap, reason = None, None
    if rank == 0:
        reason = "zero_rank"
    elif rank == p:
        reason = "full_space"
    elif values[rank] == 0:
        reason = "zero_next_singular"
    else:
        gap = float(singular[rank - 1] / singular[rank])
        if not np.isfinite(gap):
            raise ValueError("Nonfinite gap")
    return {
        "singular_values": values,
        "basis": right[:rank].copy(),
        "rank": rank,
        "gap": gap,
        "gap_reason": reason,
    }


def direction_residuals(basis: Array, directions: Array) -> Array:
    """Unit direction rank-one distances to a numerical row span; no fitting."""
    directions = _finite(directions, "directions")
    basis = np.asarray(basis, dtype=np.float64)
    if directions.ndim != 2 or basis.ndim != 2 or not np.isfinite(basis).all():
        raise ValueError("Finite matrix directions/basis required")
    n = directions.shape[1]
    if basis.shape[1] != n * (n + 1) // 2 or basis.shape[0] > basis.shape[1]:
        raise ValueError("Incompatible symmetric-space basis")
    if not np.allclose(basis @ basis.T, np.eye(len(basis)), rtol=0, atol=1e-10):
        raise ValueError("Orthonormal row basis required")
    scale = np.max(np.abs(directions), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Nonzero direction required")
    v = directions / scale
    if np.any((directions != 0) & (v == 0)):
        raise ValueError("Direction normalization underflow")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            atoms = svec(v[:, :, None] * v[:, None, :])
            residual = atoms - (atoms @ basis.T) @ basis
    except FloatingPointError as exc:
        raise ValueError("Rank-one projection products overflow/underflow") from exc
    # Scaling avoids false zero when squaring a representable tiny residual.
    magnitude = np.max(np.abs(residual), axis=1)
    divisor = np.where(magnitude == 0, 1.0, magnitude)
    normalized = residual / divisor[:, None]
    if np.any((residual != 0) & (normalized == 0)):
        raise ValueError("Residual normalization underflow")
    return _finite(
        magnitude * np.linalg.norm(normalized, axis=1), "direction residuals"
    )


def four_corner_inputs(x: Array, step: float) -> Array:
    """(sign,pair,input); diagonal effective step2h, all4positions retained."""
    x = _finite(x, "probe")
    if x.ndim != 1 or not np.isfinite(step) or step <= 0:
        raise ValueError("Vector probe/positive finite step required")
    i, j = np.triu_indices(len(x))
    ei, ej = np.eye(len(x))[i] * step, np.eye(len(x))[j] * step
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            q = np.stack((x + ei + ej, x + ei - ej, x - ei + ej, x - ei - ej))
    except FloatingPointError as exc:
        raise ValueError("Stencil overflow/underflow") from exc
    return _finite(q, "stencil")


def curvature_from_responses(responses: Array, dimension: int, step: float) -> Array:
    """(4,p,output) saved responses -> (output,n,n); no oracle callback."""
    y = _finite(responses, "responses")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("Positive integer dimension required")
    p = dimension * (dimension + 1) // 2
    if y.ndim != 3 or y.shape[:2] != (4, p) or not np.isfinite(step) or step <= 0:
        raise ValueError("Four-corner response shape/step required")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            denom = np.float64(4) * np.float64(step) * np.float64(step)
            value = (y[0] - y[1] - y[2] + y[3]) / denom
    except FloatingPointError as exc:
        raise ValueError("Curvature arithmetic overflow/underflow") from exc
    result = np.empty((y.shape[2], dimension, dimension))
    i, j = np.triu_indices(dimension)
    result[:, i, j] = value.T
    result[:, j, i] = value.T
    return _finite(result, "curvature")


def richardson(fine: Array, coarse: Array) -> Array:
    fine, coarse = _finite(fine, "fine"), _finite(coarse, "coarse")
    if fine.shape != coarse.shape:
        raise ValueError("Matching Richardson arrays required")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            result = (4 * fine - coarse) / 3
    except FloatingPointError as exc:
        raise ValueError("Richardson arithmetic failure") from exc
    return _finite(result, "Richardson")
