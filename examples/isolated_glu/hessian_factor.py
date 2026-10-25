"""Synthetic-stage fixed-gate SwiGLU Hessian factor primitive.

No target queries, stored experiment reads, optimizer loops or candidate promotion.
For one fixed gate g, its coupling C maps linearly to a stack of Hessians.
A positive-definite input Gram permits an exact rank-at-most-one local solve.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _base(g: Array, x: Array) -> tuple[Array, Array, float]:
    if (
        g.ndim != 1
        or not 1 <= len(g) <= 32
        or x.ndim != 2
        or not 1 <= len(x) <= 64
        or x.shape[1] != len(g)
        or g.dtype != np.float64
        or x.dtype != np.float64
        or not np.isfinite(g).all()
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite FP64 gate[d<=32] and probes[1..64,d]")
    t = x @ g
    sigmoid = 0.5 * (1 + np.tanh(0.5 * t))
    curvature = sigmoid * (1 - sigmoid)
    first = sigmoid + t * curvature
    second = curvature * (2 + t * (1 - 2 * sigmoid))
    norm = float(g @ g)
    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or not np.isfinite(norm)
    ):
        raise ValueError("Nonfinite gate derivatives")
    return first, second, norm


def _residual(y: Array, x: Array) -> None:
    if (
        y.ndim != 4
        or y.shape[0] != len(x)
        or not 1 <= y.shape[1] <= 32
        or y.shape[2:] != (x.shape[1], x.shape[1])
        or y.dtype != np.float64
        or not np.isfinite(y).all()
    ):
        raise ValueError("Expected finite FP64 residual[probes,outputs<=32,d,d]")


def _finite(value: Array) -> Array:
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite Hessian factor result")
    return value


def operator(g: Array, c: Array, x: Array) -> Array:
    """H[p,o]=silu''(t)(C_o.x)gg^T+silu'(t)(gC_o^T+C_og^T)."""
    first, second, _ = _base(g, x)
    if (
        c.ndim != 2
        or not 1 <= len(c) <= 32
        or c.shape[1] != len(g)
        or c.dtype != np.float64
        or not np.isfinite(c).all()
    ):
        raise ValueError("Expected finite FP64 coupling[outputs<=32,d]")
    value = x @ c.T
    h: Array = (second[:, None] * value)[:, :, None, None] * np.outer(g, g)[
        None, None, :, :
    ]
    h += first[:, None, None, None] * (
        g[None, None, :, None] * c[None, :, None, :]
        + c[None, :, :, None] * g[None, None, None, :]
    )
    return _finite(h)


def adjoint(g: Array, y: Array, x: Array) -> Array:
    """Adjoint for the FULL Hessian Frobenius inner product, not raw vech."""
    first, second, _ = _base(g, x)
    _residual(y, x)
    quadratic = np.einsum("i,poij,j->po", g, y, g)
    mixed = np.einsum("poij,j->poi", y + y.swapaxes(2, 3), g)
    result: Array = np.einsum("p,po,pj->oj", second, quadratic, x) + np.einsum(
        "p,poj->oj", first, mixed
    )
    return _finite(result)


def gram(g: Array, x: Array) -> Array:
    """K=L_g^*L_g, with no output-axis mixing and no explicit large design."""
    first, second, norm = _base(g, x)
    weighted_x = np.sum((2 * second * first * norm)[:, None] * x, axis=0)
    result: Array = (x.T * (second**2 * norm**2)) @ x
    result += np.outer(weighted_x, g) + np.outer(g, weighted_x)
    result += float(first @ first) * (2 * norm * np.eye(len(g)) + 2 * np.outer(g, g))
    return _finite((result + result.T) * 0.5)


def rankone_update(g: Array, partial: Array, x: Array) -> tuple[Array, dict[str, Any]]:
    """Globally minimize this ONE fixed-gate rank<=1 local Hessian objective.

    If T=L*partial and M=K^-1/2, the variable Z=C K^1/2 has
    objective constant+||Z-TM||_F^2. Invertible whitening preserves rank.
    The leading singular triplet gives the minimizer, transformed back by M.
    This does not prove descent of another objective or unseen forward accuracy.
    """
    k = gram(g, x)
    _residual(partial, x)
    eigen, vectors = np.linalg.eigh(k)
    if not np.isfinite(eigen).all() or eigen[-1] <= 0 or eigen[0] <= eigen[-1] * 1e-12:
        raise ValueError("Singular or ill-conditioned whitening; no ridge substitution")
    inverse: Array = (vectors / np.sqrt(eigen)[None, :]) @ vectors.T
    white = adjoint(g, partial, x) @ inverse
    left, singular, right = np.linalg.svd(white, full_matrices=False)
    result: Array = (left[:, 0] * singular[0])[:, None] * (inverse @ right[0])[None, :]
    _finite(result)
    difference = partial - operator(g, result, x)
    objective = float(np.sum(difference * difference))
    if not np.isfinite(objective):
        raise ValueError("Nonfinite local objective")
    return result, {
        "objective": objective,
        "gram_condition": float(eigen[-1] / eigen[0]),
        "rank_at_most_one": True,
        "ridge_used": False,
        "fixed_gate": True,
    }
