"""Calibration-only residual starts; never a recovery/coverage certificate.

Separate development experiment: does not modify or rescue the formal v2 result.
Call under the existing process-tree resource monitor. Only saved public Hessians
and retained directions are accepted; no teacher/heldout paths or new queries.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

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


def refine(x: Array, basis: Array) -> tuple[Array, float, float]:
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
    for _ in range(24):
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


def run(public: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    paths = [public / 'hessians.npz', public / 'refined-solution.npz']
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with np.load(paths[0], allow_pickle=False) as z:
        h = z['hessian']
    with np.load(paths[1], allow_pickle=False) as z:
        known = z['directions'].copy()
    if h.shape != (64, 32, 32, 32) or known.shape != (63, 32):
        raise ValueError('This development hypothesis requires fixed d32/width64 and63 retained inputs')
    if not np.isfinite(h).all() or not np.isfinite(known).all():
        raise ValueError('Nonfinite calibration arrays')
    if np.max(np.abs(np.linalg.norm(known, axis=1) - 1)) > 1e-10:
        raise ValueError('Input directions must be normalized')
    a = pack(h).reshape(-1, 528)
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    if s[127] < 1e-10 * s[0] or s[127] < 100 * s[128]:
        raise ValueError('Original Hessian subspace spectral guard failed')
    basis = vt[:128]
    residual, diagnostics = residual_subspace(known, basis)
    if residual.shape != (2, 528):
        raise ValueError('Expected rank126 components; do not silently change method')
    starts: list[Array] = []
    for angle in np.arange(64) * np.pi / 64:
        matrix = unpack(np.cos(angle) * residual[0] + np.sin(angle) * residual[1], 32)
        eig, vectors = np.linalg.eigh(matrix)
        starts.extend(vectors[:, k] for k in np.argsort(np.abs(eig))[-2:])
    candidates = []
    losses = []
    before_losses = []
    for x in starts:
        y, before, after = refine(x.copy(), basis)
        candidates.append(y)
        before_losses.append(before)
        losses.append(after)
    vectors = np.stack(candidates)
    retained = list(known)
    added = []
    for k in np.argsort(losses):
        if losses[k] <= 1e-8 and all(abs(vectors[k] @ v) < 0.9995 for v in retained):
            retained.append(vectors[k])
            added.append(int(k))
    diagnostic_new = []
    for k in added:
        t = tangent(vectors[k])
        ss = np.linalg.svd(t - basis.T @ (basis @ t), compute_uv=False)
        diagnostic_new.append({
            'start_index': k, 'original_subspace_loss': losses[k],
            'max_abs_cosine_to_input': float(np.max(np.abs(known @ vectors[k]))),
            'tangent_smallest_three': [float(v) for v in ss[-3:][::-1]],
        })
    np.savez(output / 'coverage-solution.npz', directions=np.stack(retained),
             starts=np.stack(starts), all_directions=vectors, residual=np.array(losses))
    report = {
        'experiment': 'isolated-glu-coverage-development-v1',
        'sealed': False, 'formal_v2_overall_passed': False,
        'private_teacher_read': False, 'heldout_read': False, 'new_queries': 0,
        'input_sha256': hashes, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method': '63 tangent intersections; projected component deflation;64 angles x2 extremal eigenvectors;24 original-subspace GN steps',
        'starts': len(starts), 'input_count': len(known), 'retained_count': len(retained),
        'new_candidates': diagnostic_new, 'minimum_loss': min(losses),
        'median_before_loss': float(np.median(before_losses)),
        'median_after_loss': float(np.median(losses)), 'geometry': diagnostics,
        'seconds': time.perf_counter() - start,
        'original_weights_recovered': False, 'full_glu_recovered': False,
        'coverage_certified': False,
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2])), allow_nan=False))
