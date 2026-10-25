"""Synthetic-stage gradients of a known candidate, never target gradients.

Physical targets Y enter unweighted. Residual multipliers w are applied once
in E=w*(candidate-Y). The squared objective has NO one-half factor.
The diagnostic C gradient is unconstrained and does not preserve rank one.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _finite(value: Array) -> Array:
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite candidate-gradient arithmetic")
    return value


def _profile(g: Array, c: Array, x: Array) -> None:
    if (
        g.ndim != 2
        or not 1 <= len(g) <= 64
        or not 1 <= g.shape[1] <= 32
        or c.ndim != 3
        or c.shape[0] != len(g)
        or not 1 <= c.shape[1] <= 32
        or c.shape[2] != g.shape[1]
        or x.ndim != 2
        or not 1 <= len(x) <= 4096
        or x.shape[1] != g.shape[1]
        or any(v.dtype != np.float64 or not np.isfinite(v).all() for v in [g, c, x])
    ):
        raise ValueError(
            "Expected finite FP64 candidate/probes within64units/32dims/32outputs/4096rows"
        )


def forward(g: Array, c: Array, x: Array) -> Array:
    """Physical candidate response; no target call or candidate update."""
    _profile(g, c, x)
    prediction = np.zeros((len(x), c.shape[1]), dtype=np.float64)
    for gate, coupling in zip(g, c):
        t = x @ gate
        sigmoid = 0.5 * (1 + np.tanh(0.5 * t))
        activation = t * sigmoid
        prediction += activation[:, None] * (x @ coupling.T)
    return _finite(prediction)


def loss_gradient(
    g: Array, c: Array, x: Array, y: Array, weights: Array
) -> tuple[float, Array, Array]:
    """Return sum(E²), dL/dg and unconstrained dL/dC, with E=w*(f-Y).

    w is fixed, not differentiated; Y MUST be physical rather than preweighted.
    Results concern the known smooth candidate's real-arithmetic objective.
    FP64 implementation is not an optimality or unseen-data certificate.
    """
    _profile(g, c, x)
    if (
        y.shape != (len(x), c.shape[1])
        or y.dtype != np.float64
        or not np.isfinite(y).all()
        or weights.shape != (len(x),)
        or weights.dtype != np.float64
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "Expected physical FP64 targets and positive residual multipliers"
        )
    error = (forward(g, c, x) - y) * weights[:, None]
    value = float(np.sum(error * error))
    if not np.isfinite(value):
        raise ValueError("Nonfinite candidate objective")
    dg = np.empty_like(g)
    dc = np.empty_like(c)
    for index, (gate, coupling) in enumerate(zip(g, c)):
        t = x @ gate
        sigmoid = 0.5 * (1 + np.tanh(0.5 * t))
        activation = t * sigmoid
        prime = sigmoid + t * sigmoid * (1 - sigmoid)
        branch = x @ coupling.T
        aligned = np.sum(error * branch, axis=1)
        dg[index] = (2 * weights * prime * aligned) @ x
        dc[index] = 2 * (error.T @ ((weights * activation)[:, None] * x))
    return value, _finite(dg), _finite(dc)


def _single_gate(g: Array, c: Array, x: Array, weights: Array) -> tuple[Array, Array]:
    if g.ndim != 1 or c.ndim != 2:
        raise ValueError("Expected one gate and coupling matrix")
    _profile(g[None, :], c[None, :, :], x)
    if (
        weights.shape != (len(x),)
        or weights.dtype != np.float64
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("Expected positive FP64 residual multipliers")
    t = x @ g
    sigmoid = 0.5 * (1 + np.tanh(0.5 * t))
    prime = sigmoid + t * sigmoid * (1 - sigmoid)
    scaled_prime: Array = weights * prime
    branch: Array = x @ c.T
    return _finite(scaled_prime), _finite(branch)


def gate_jvp(g: Array, c: Array, x: Array, weights: Array, direction: Array) -> Array:
    """Weighted candidate output derivative for one gate-vector direction."""
    scaled_prime, branch = _single_gate(g, c, x, weights)
    if (
        direction.shape != g.shape
        or direction.dtype != np.float64
        or not np.isfinite(direction).all()
    ):
        raise ValueError("Invalid finite gate direction")
    result: Array = (scaled_prime * (x @ direction))[:, None] * branch
    return _finite(result)


def gate_gram(g: Array, c: Array, x: Array, weights: Array) -> Array:
    """J*J=sum_p(w*silu')²||C x||² x x.T; no full Jacobian allocation.

    A positive-definite Gram may define a LOCAL LINEARIZED Gauss-Newton
    direction. Its minimizer is not a guarantee for the nonlinear objective.
    This function performs no inversion, ridge, step selection or update.
    """
    scaled_prime, branch = _single_gate(g, c, x, weights)
    scale = scaled_prime * scaled_prime * np.sum(branch * branch, axis=1)
    result: Array = (x.T * scale) @ x
    return _finite((result + result.T) * 0.5)
