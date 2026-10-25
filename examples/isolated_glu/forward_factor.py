"""Synthetic-stage weighted full-forward fixed-gate rank-one primitive.

All response tensors passed to operator/adjoint/rankone_update use the same
residual multipliers w. A physical response must be multiplied by w ONCE.
No actual experiment data, queries, model loading or optimization loops here.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _finite(value: Array) -> Array:
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite forward factor result")
    return value


def features(g: Array, x: Array, weights: Array) -> Array:
    """Phi[p,:]=w[p]*silu(g.x[p])*x[p]; w are residual multipliers."""
    if (
        g.ndim != 1
        or not 1 <= len(g) <= 32
        or g.dtype != np.float64
        or x.ndim != 2
        or not 1 <= len(x) <= 4096
        or x.shape[1] != len(g)
        or x.dtype != np.float64
        or weights.shape != (len(x),)
        or weights.dtype != np.float64
        or not np.isfinite(g).all()
        or not np.isfinite(x).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "Expected finite FP64 gate/probes and positive residual multipliers"
        )
    t = x @ g
    value = weights * t * (0.5 * (1 + np.tanh(0.5 * t)))
    result: Array = value[:, None] * x
    return _finite(result)


def _target(y: Array, rows: int) -> None:
    if (
        y.ndim != 2
        or y.shape[0] != rows
        or not 1 <= y.shape[1] <= 32
        or y.dtype != np.float64
        or not np.isfinite(y).all()
    ):
        raise ValueError("Expected finite FP64 WEIGHTED responses[probes,outputs<=32]")


def operator(g: Array, c: Array, x: Array, weights: Array) -> Array:
    """Return WEIGHTED unit outputs, never raw target observations."""
    phi = features(g, x, weights)
    if (
        c.ndim != 2
        or not 1 <= len(c) <= 32
        or c.shape[1] != len(g)
        or c.dtype != np.float64
        or not np.isfinite(c).all()
    ):
        raise ValueError("Expected finite FP64 coupling[outputs<=32,d]")
    result: Array = phi @ c.T
    return _finite(result)


def adjoint(g: Array, weighted_y: Array, x: Array, weights: Array) -> Array:
    """L*(Y)=Y.T@Phi, where Y has already been multiplied by w ONCE."""
    phi = features(g, x, weights)
    _target(weighted_y, len(x))
    result: Array = weighted_y.T @ phi
    return _finite(result)


def gram(g: Array, x: Array, weights: Array) -> Array:
    phi = features(g, x, weights)
    result: Array = phi.T @ phi
    return _finite((result + result.T) * 0.5)


def rankone_update(
    g: Array, weighted_partial: Array, x: Array, weights: Array
) -> tuple[Array, dict[str, Any]]:
    """Compute a rank<=1 minimizer of ||weighted_partial-Phi C.T||_F².

    Exact-arithmetic proof uses positive-definite K=Phi.T Phi and M=K^-1/2:
    set Z=C K^1/2, T=weighted_partial.T Phi. The variable part is ||Z-TM||².
    Whitening preserves rank, so a leading singular triplet solves the local
    quadratic problem. No floating-point optimality certificate or guard-set
    improvement is implied; fixed gates cannot repair direction errors.
    """
    k = gram(g, x, weights)
    _target(weighted_partial, len(x))
    eigen, vectors = np.linalg.eigh(k)
    if not np.isfinite(eigen).all() or eigen[-1] <= 0 or eigen[0] <= eigen[-1] * 1e-12:
        raise ValueError("Singular or ill-conditioned whitening; no ridge substitution")
    inverse: Array = (vectors / np.sqrt(eigen)[None, :]) @ vectors.T
    white = adjoint(g, weighted_partial, x, weights) @ inverse
    left, singular, right = np.linalg.svd(white, full_matrices=False)
    result: Array = (left[:, 0] * singular[0])[:, None] * (inverse @ right[0])[None, :]
    _finite(result)
    delta = weighted_partial - operator(g, result, x, weights)
    objective = float(np.sum(delta * delta))
    if not np.isfinite(objective):
        raise ValueError("Nonfinite weighted local objective")
    return result, {
        "weighted_objective": objective,
        "gram_condition": float(eigen[-1] / eigen[0]),
        "rank_at_most_one": True,
        "ridge_used": False,
        "fixed_gate": True,
        "weights_semantics": "residual_multipliers_not_squared_loss_weights",
        "target_semantics": "already_weighted_once",
    }
