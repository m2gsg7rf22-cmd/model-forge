"""Bounded algebra for the even part of owned, bias-free SwiGLU/GeGLU maps.

For f(x)=sum_l silu(g_l.x) (C_l x), the identity silu(t)-silu(-t)=t
implies (f(x)+f(-x))/2 = x.T Q x, output by output. Exact GELU has
this identity too. This reconstructs neither the odd component nor the gates.
Orientation recovery is conditional on externally supplied candidate gates and
couplings with independent provenance; this module does not discover them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
_MAX_DIMENSION = 64
_MAX_WIDTH = 128
_MAX_ELEMENTS = 2_000_000
_MAX_CONDITION = 1e8
_SIGN_TOLERANCE = 1e-7
_RESIDUAL_TOLERANCE = 1e-9


def _dimension(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_DIMENSION
    ):
        raise ValueError("dimension must be an integer in 1..64")
    return value


def _array(value: ArrayLike, name: str) -> Array:
    raw = np.asarray(value)
    if raw.size > _MAX_ELEMENTS:
        raise ValueError(f"{name} exceeds the resource element limit")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _quadratic(value: ArrayLike) -> Array:
    q = _array(value, "Q")
    if q.ndim != 3 or q.shape[1] != q.shape[2]:
        raise ValueError("Q must have shape [outputs, dimension, dimension]")
    _dimension(q.shape[1])
    if not 1 <= q.shape[0] <= _MAX_DIMENSION:
        raise ValueError("Q output count must be in 1..64")
    if not np.allclose(q, q.swapaxes(1, 2), rtol=1e-12, atol=1e-14):
        raise ValueError("Q must be symmetric")
    return q


def canonical_probe_points(dimension: int) -> Array:
    """Return e_i, then e_i+e_j in lexicographic i<j order.

    There are d(d+1)/2 points; querying each point and its negative uses
    d(d+1) signed evaluations. Input dimension is bounded to 1..64.
    """
    d = _dimension(dimension)
    points = np.zeros((d * (d + 1) // 2, d), dtype=np.float64)
    points[:d] = np.eye(d)
    row = d
    for i in range(d):
        for j in range(i + 1, d):
            points[row, i] = points[row, j] = 1.0
            row += 1
    return points


def recover_even_quadratic(
    positive_outputs: ArrayLike, negative_outputs: ArrayLike, dimension: int
) -> Array:
    """Interpolate symmetric Q[output,d,d] from ordered antipodal outputs.

    Outputs must both have shape [d(d+1)/2, outputs]. This only interpolates
    the quadratic under the stated bias-free GLU assumption. The probes alone
    cannot detect bias or other departures; independent heldout checks are
    required before claiming that Q describes the teacher's even component.
    """
    d = _dimension(dimension)
    pos = _array(positive_outputs, "positive_outputs")
    neg = _array(negative_outputs, "negative_outputs")
    n = d * (d + 1) // 2
    if pos.ndim != 2 or pos.shape != neg.shape or pos.shape[0] != n:
        raise ValueError("outputs must have identical [d(d+1)/2, outputs] shapes")
    if not 1 <= pos.shape[1] <= _MAX_DIMENSION:
        raise ValueError("output count must be in 1..64")
    # Halve before adding to avoid an unnecessary intermediate overflow.
    even = pos * 0.5 + neg * 0.5
    q = np.zeros((pos.shape[1], d, d), dtype=np.float64)
    diagonal = even[:d].T
    indices = np.arange(d)
    q[:, indices, indices] = diagonal
    row = d
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(d):
            for j in range(i + 1, d):
                cross = even[row] * 0.5 - diagonal[:, i] * 0.5 - diagonal[:, j] * 0.5
                q[:, i, j] = q[:, j, i] = cross
                row += 1
    if not np.all(np.isfinite(q)):
        raise ValueError(
            "recovered Q is not finite; output scale exceeds numeric range"
        )
    return q


def predict_even(quadratic: ArrayLike, inputs: ArrayLike) -> Array:
    """Evaluate x.T Q x for x[d] or a batch x[n,d], preserving batch rank."""
    q = _quadratic(quadratic)
    x = _array(inputs, "inputs")
    if x.ndim not in (1, 2) or x.shape[-1] != q.shape[1]:
        raise ValueError("inputs must have shape [dimension] or [n, dimension]")
    if x.ndim == 2 and (x.shape[0] < 1 or x.shape[0] * q.shape[0] > _MAX_ELEMENTS):
        raise ValueError("prediction exceeds the resource limit or has an empty batch")
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = np.einsum("...i,oij,...j->...o", x, q, x, optimize=False)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction is not finite; input scale exceeds numeric range")
    return np.asarray(prediction, dtype=np.float64)


def recover_orientations(
    unsigned_gates: ArrayLike, couplings: ArrayLike, quadratic: ArrayLike
) -> Array:
    """Recover signs conditional on independently obtained gates and couplings.

    gates has shape [width,d], C has shape [width,outputs,d]. Each C_l is
    the owned neuron's down-column/up-row outer product. A candidate sign s_l
    contributes s_l*(C_l g_l.T + g_l C_l.T)/4 to Q. No provenance or gate
    discovery is inferred from solving this linear system.

    Reject zero/invalid candidates, couplings with a second singular value
    above 1e-10 of the first, deficient rank, normalized condition >1e8,
    relative least-squares residual >1e-9, coefficients >1e-7 from +/-1,
    or a rounded-sign residual >1e-9. The design is limited to two million
    elements before allocation. Tolerances are fixed numerical checks, not
    evidence of identifiability outside this conditional candidate family.
    """
    q = _quadratic(quadratic)
    gates = _array(unsigned_gates, "unsigned_gates")
    c = _array(couplings, "couplings")
    d, outputs = q.shape[1], q.shape[0]
    if gates.ndim != 2 or gates.shape[1] != d or not 1 <= gates.shape[0] <= _MAX_WIDTH:
        raise ValueError("unsigned_gates must have shape [width in 1..128, dimension]")
    width = gates.shape[0]
    if c.shape != (width, outputs, d):
        raise ValueError("couplings must have shape [width, outputs, dimension]")
    rows = outputs * d * (d + 1) // 2
    if rows * width > _MAX_ELEMENTS:
        raise ValueError("orientation design exceeds the resource element limit")
    if np.any(np.max(np.abs(gates), axis=1) == 0) or np.any(
        np.max(np.abs(c), axis=(1, 2)) == 0
    ):
        raise ValueError("zero gate or coupling candidate")
    for unit in range(width):
        singular_c = np.linalg.svd(c[unit] / np.max(np.abs(c[unit])), compute_uv=False)
        if len(singular_c) > 1 and singular_c[1] > 1e-10 * singular_c[0]:
            raise ValueError("each GLU coupling candidate must be rank-one")
    # Build only upper triangles, avoiding width*outputs*d*d tensors.
    i, j = np.triu_indices(d)
    design = np.empty((outputs * len(i), width), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for unit in range(width):
            design[:, unit] = (
                c[unit, :, i].T * gates[unit, j][None, :] * 0.25
                + c[unit, :, j].T * gates[unit, i][None, :] * 0.25
            ).ravel()
    if not np.all(np.isfinite(design)):
        raise ValueError("candidate design must be finite at float64 scale")
    scale = np.max(np.abs(design), axis=0)
    if np.any(scale == 0):
        raise ValueError("zero candidate contribution")
    normalized = design / scale
    target = q[:, i, j].ravel()
    # Bound the target scale too, retaining original coefficient magnitudes.
    target_scale = float(np.max(np.abs(target)))
    if target_scale == 0:
        raise ValueError("zero Q cannot identify signs with full column rank")
    solution, _, rank, singular = np.linalg.lstsq(
        normalized, target / target_scale, rcond=None
    )
    if rank != width:
        raise ValueError("candidate design has deficient rank")
    if singular[0] / singular[-1] > _MAX_CONDITION:
        raise ValueError("candidate design condition exceeds 1e8")
    relative_target = target / target_scale
    denominator = float(np.linalg.norm(relative_target))
    relative_residual = (
        float(np.linalg.norm(normalized @ solution - relative_target)) / denominator
    )
    if relative_residual > _RESIDUAL_TOLERANCE:
        raise ValueError("candidate design residual exceeds tolerance")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coefficients = solution * (target_scale / scale)
    if not np.all(np.isfinite(coefficients)) or np.any(
        np.abs(np.abs(coefficients) - 1) > _SIGN_TOLERANCE
    ):
        raise ValueError("candidate coefficients are not signs (+/-1)")
    signs = np.where(coefficients > 0, 1.0, -1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        signed_target = design @ signs / target_scale
    if not np.all(np.isfinite(signed_target)) or (
        float(np.linalg.norm(signed_target - relative_target)) / denominator
        > _RESIDUAL_TOLERANCE
    ):
        raise ValueError("rounded sign residual exceeds tolerance")
    return signs
