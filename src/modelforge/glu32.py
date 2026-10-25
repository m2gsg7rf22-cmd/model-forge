"""Fixed isolated SwiGLU32/64/32 candidate kernel; port of registered v4.

No oracle/teacher/filesystem access. Results are candidates, not certificates.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray
from .antipodal import recover_even_quadratic

Array = NDArray[np.float64]


def pack(matrix: Array) -> Array:
    d = matrix.shape[-1]
    i, j = np.triu_indices(d)
    return np.asarray(matrix[..., i, j] * np.where(i == j, 1.0, np.sqrt(2.0)))


def unpack(vector: Array, d: int) -> Array:
    i, j = np.triu_indices(d)
    result = np.zeros((d, d))
    result[i, j] = vector / np.where(i == j, 1.0, np.sqrt(2.0))
    result[j, i] = result[i, j]
    return result


def tangent(x: Array) -> Array:
    """Orthonormal Frobenius coordinates for {xa'+ax'} (not sphere tangent)."""
    eye = np.eye(len(x))
    linear = np.stack([pack(np.outer(x, a) + np.outer(a, x)) for a in eye], axis=1)
    return np.asarray(np.linalg.qr(linear, mode="reduced")[0])


def residual_subspace(directions: Array, basis: Array) -> tuple[Array, dict[str, Any]]:
    components: list[Array] = []
    gaps = []
    for x in directions:
        t = tangent(x)
        _, singular, vt = np.linalg.svd(t - basis.T @ (basis @ t), full_matrices=False)
        # Two least-angle components, not just the degenerate smallest value.
        components.extend((basis @ (t @ vt[-1]), basis @ (t @ vt[-2])))
        gaps.append([float(v) for v in singular[-3:][::-1]])
    matrix = np.stack(components)
    _, s, vt = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(s > 1e-8 * s[0]))
    residual = vt[rank:] @ basis
    return residual, {
        "component_rank": rank,
        "smallest_component_singular": float(s[-1]),
        "known_tangent_smallest_three": gaps,
        "residual_dimension": len(residual),
        "coverage_certified": False,
        "warning": "Residual dimension/energy after projection into V is structural, not truth evidence",
    }


def refine(x: Array, basis: Array, steps: int = 24) -> tuple[Array, float, float]:
    """24 tangent Gauss-Newton steps against original V, never deflated V."""
    d = len(x)
    i, j = np.triu_indices(d)
    scale = np.where(i == j, 1.0, np.sqrt(2.0))
    eye = np.eye(d)

    def residual(y: Array) -> Array:
        q = pack(np.outer(y, y))
        return q - (q @ basis.T) @ basis

    r = residual(x)
    before = float(r @ r)
    for _ in range(steps):
        jac = (x[i, None] * eye[j] + x[j, None] * eye[i]) * scale[:, None]
        jac -= basis.T @ (basis @ jac)
        _, _, vt = np.linalg.svd(x.reshape(1, -1), full_matrices=True)
        t = vt[1:].T
        delta = t @ np.linalg.lstsq(jac @ t, -r, rcond=1e-10)[0]
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            y = x + alpha * delta
            y /= np.linalg.norm(y)
            rr = residual(y)
            if rr @ rr < r @ r:
                x, r, accepted = y, rr, True
                break
        if not accepted:
            break
    return x, before, float(r @ r)


def merge(known: Array, candidates: Array, losses: Array) -> Array:
    selected = list(known)
    for k in np.argsort(losses):
        if losses[k] <= 1e-8 and all(abs(candidates[k] @ x) < 0.9995 for x in selected):
            selected.append(candidates[k])
    return np.stack(selected)  # No target-count truncation.


def search(
    known: Array,
    basis: Array,
    *,
    target: int,
    mixtures: int = 256,
) -> tuple[Array, list[dict[str, Any]]]:
    rng = np.random.default_rng(20261025)
    rounds: list[dict[str, Any]] = []
    d = known.shape[1]
    for iteration in range(4):
        if len(known) >= target:
            break
        before = known.copy()
        residual, geometry = residual_subspace(known, basis)
        if geometry["component_rank"] != 2 * len(known) or len(residual) != len(
            basis
        ) - 2 * len(known):
            rounds.append(
                {
                    "round": iteration,
                    "stop": "unexpected component rank",
                    "geometry": geometry,
                }
            )
            break
        starts: list[Array] = []
        for _ in range(mixtures):
            coeff = rng.normal(size=len(residual))
            coeff /= np.linalg.norm(coeff)
            matrix = unpack(coeff @ residual, d)
            eig, vectors = np.linalg.eigh(matrix)
            starts.extend(vectors[:, k] for k in np.argsort(np.abs(eig))[-2:])
        candidates = []
        losses = []
        for x in starts:
            y, _, loss = refine(x.copy(), basis)
            candidates.append(y)
            losses.append(loss)
        all_d = np.stack(candidates)
        errors = np.array(losses)
        known = merge(known, all_d, errors)
        row: dict[str, Any] = {
            "round": iteration,
            "input_count": len(before),
            "retained_count": len(known),
            "starts": len(starts),
            "loss_min": float(errors.min()),
            "geometry": geometry,
        }
        rounds.append(row)
    return known, rounds


def directions(h: Array) -> tuple[Array, dict[str, Any]]:
    if h.shape != (64, 32, 32, 32) or h.dtype != np.float64 or not np.isfinite(h).all():
        raise ValueError("Requires finite FP64 Hessians64x32x32x32")
    d = 32
    rank = 128
    i, j = np.triu_indices(d)
    scale = np.where(i == j, 1.0, np.sqrt(2.0))
    _, sv, vt = np.linalg.svd(
        h[:, :, i, j].reshape(-1, len(i)) * scale, full_matrices=False
    )
    if sv[127] <= 1e-10 * sv[0] or sv[127] < 100 * sv[128]:
        raise ValueError("Inadequate Hessian rank or spectral separation")
    b = vt[:128]
    B = np.zeros((rank, d, d))
    B[:, i, j] = b / scale
    B[:, j, i] = b / scale
    rng = np.random.default_rng(20261024)
    v = rng.normal(size=(256, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    adam_m = np.zeros_like(v)
    adam_v = np.zeros_like(v)
    for t in range(1, 1601):
        vv = v[:, :, None] * v[:, None, :]
        coef = vv.reshape(len(v), -1) @ B.reshape(rank, -1).T
        proj = (coef @ B.reshape(rank, -1)).reshape(-1, d, d)
        grad = -4 * np.einsum("bij,bj->bi", proj, v)
        grad -= np.sum(grad * v, axis=1, keepdims=True) * v
        adam_m = 0.9 * adam_m + 0.1 * grad
        adam_v = 0.999 * adam_v + 0.001 * grad * grad
        v -= 0.02 * (adam_m / (1 - 0.9**t)) / (np.sqrt(adam_v / (1 - 0.999**t)) + 1e-08)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    vv = v[:, :, None] * v[:, None, :]
    coef = vv.reshape(len(v), -1) @ B.reshape(rank, -1).T
    residual = np.sum(
        (vv.reshape(len(v), -1) - coef @ B.reshape(rank, -1)) ** 2, axis=1
    )
    initial_loss = residual.copy()
    for k, x in enumerate(v):
        v[k], _, residual[k] = refine(x, b, steps=12)
    known = merge(np.empty((0, 32)), v, residual)
    before = len(known)
    if not 1 <= before <= 64:
        raise ValueError("Initial refinement requires1..64directions; no truncation")
    rounds: list[dict[str, Any]] = []
    if before < 64:
        known, rounds = search(known, b, target=64)
    if known.shape != (64, 32):
        raise ValueError("Multi-rescue failed exactly64candidate requirement")
    q = pack(known[:, :, None] * known[:, None, :])
    rr = q - (q @ b.T) @ b
    if np.max(np.sum(rr * rr, axis=1)) > 1e-8:
        raise ValueError("Candidate residual too large")
    cos = np.abs(known @ known.T)
    np.fill_diagonal(cos, 0)
    if cos.max() >= 0.9995 or not np.allclose(
        np.linalg.norm(known, axis=1), 1, atol=1e-10, rtol=0
    ):
        raise ValueError("Duplicate or nonunit candidates")
    return known, {
        "subspace_rank": 128,
        "singular_128": float(sv[127]),
        "singular_129": float(sv[128]),
        "initial_min_loss": float(initial_loss.min()),
        "base_candidates": before,
        "final_candidates": 64,
        "max_retained_loss": float(np.max(np.sum(rr * rr, axis=1))),
        "rescue_rounds": rounds,
    }


def multiscale(v: Array, data: dict[str, Array]) -> tuple[Array, Array, dict[str, Any]]:
    w, d = v.shape
    small = data["small_x"]
    y = data["small_odd"]
    scale = 0.001
    # Cubic asymptotic: O(rho*x) / (.25*rho^3) = sum (v.x)^2 B_l x.
    a = ((small @ v.T) ** 2)[:, :, None] * small[:, None, :]
    a = a.reshape(len(small), w * d)
    b, _, rank, svals = np.linalg.lstsq(a, y / (0.25 * scale**3), rcond=None)
    B = b.reshape(w, d, -1).transpose(0, 2, 1)
    large = data["large_x"]
    lv = large @ v.T
    Bvals = np.einsum("nd,wod->now", large, B)
    design = (0.5 * np.abs(lv)[:, None, :] * Bvals).reshape(-1, w)
    inv, _, r, sv = np.linalg.lstsq(
        design, data["large_odd"].reshape(-1) / 1000**2, rcond=None
    )
    if np.any(inv <= 0):
        raise ValueError("Nonpositive inferred magnitude")
    c = 1 / inv
    C = B / c[:, None, None] ** 2
    # Rank-one projection is a public structural constraint, not access to branch truth.
    ss = np.linalg.svd(C, full_matrices=False)
    C1 = (ss[0][:, :, 0] * ss[1][:, 0, None])[:, :, None] * ss[2][:, 0, None, :]
    Q = recover_even_quadratic(data["even_pos"], data["even_neg"], d)
    i, j = np.triu_indices(d)
    g = v * c[:, None]
    design = np.stack(
        [
            (0.25 * (np.outer(C1[k, o], g[k]) + np.outer(g[k], C1[k, o])))[i, j]
            for o in range(Q.shape[0])
            for k in range(w)
        ]
    )
    design = design.reshape(Q.shape[0], w, -1).transpose(0, 2, 1).reshape(-1, w)
    coeff, _, sr, svs = np.linalg.lstsq(design, Q[:, i, j].reshape(-1), rcond=None)
    sign = np.where(coeff > 0, 1.0, -1.0)
    if (
        rank != 2048
        or r != 64
        or sr != 64
        or not np.isfinite(C1).all()
        or not np.isfinite(g).all()
    ):
        raise ValueError("Multiscale rank/finite guard failed")
    denominator = float(np.linalg.norm(Q[:, i, j]))
    if denominator <= 0:
        raise ValueError("Undefined even calibration norm")
    diagnostics = {
        "small_design_rank": int(rank),
        "magnitude_design_rank": int(r),
        "orientation_rank": int(sr),
        "magnitude_min": float(c.min()),
        "magnitude_max": float(c.max()),
        "coupling_rank1_max_relative_tail_before_projection": float(
            (ss[1][:, 1] / ss[1][:, 0]).max()
        ),
        "orientation_rounded_relative_residual": float(
            np.linalg.norm(design @ sign - Q[:, i, j].reshape(-1)) / denominator
        ),
    }
    return g * sign[:, None], C1, diagnostics


def fit(
    hessian: Array, observations: dict[str, Array]
) -> tuple[Array, Array, dict[str, Any]]:
    v, first = directions(hessian)
    g, c, second = multiscale(v, observations)
    return g, c, {"directions": first, "multiscale": second}


def predict(gates: Array, couplings: Array, inputs: Any) -> Array:
    x = np.asarray(inputs, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != 32
        or not 1 <= len(x) <= 4096
        or not np.isfinite(x).all()
    ):
        raise ValueError("Requires1..4096finite input rows of32coordinates")
    result = np.zeros((len(x), 32), dtype=np.float64)
    for unit in range(64):
        z = x @ gates[unit]
        result += (z * 0.5 * (1 + np.tanh(z / 2)))[:, None] * (x @ couplings[unit].T)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite candidate prediction")
    return result
