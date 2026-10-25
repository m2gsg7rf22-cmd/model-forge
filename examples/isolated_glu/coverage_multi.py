"""Multi-direction calibration-only development. No teacher/heldout access.

Geometry functions below are frozen copies from coverage.py to preserve v3's
source commitments. Residual mixtures initialize optimization on ORIGINAL V.
No residual dimension, candidate count, or tiny loss is a truth certificate.
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
    output: Path | None = None,
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
        if output is not None:
            name = output / f"round-{iteration}.npz"
            np.savez(
                name,
                input_directions=before,
                starts=np.stack(starts),
                all_directions=all_d,
                residual=errors,
                directions=known,
            )
            row["artifact_sha256"] = hashlib.sha256(name.read_bytes()).hexdigest()
        rounds.append(row)
    return known, rounds


def run(public: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    paths = [public / "hessians.npz", public / "refined-solution.npz"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with np.load(paths[0], allow_pickle=False) as z:
        h = z["hessian"]
    with np.load(paths[1], allow_pickle=False) as z:
        known = z["directions"].copy()
    if (
        h.shape != (64, 32, 32, 32)
        or known.ndim != 2
        or known.shape[1] != 32
        or not 1 <= len(known) < 64
    ):
        raise ValueError(
            "Requires fixed d32/width64 and1..63retained public directions"
        )
    if not np.isfinite(h).all() or not np.isfinite(known).all():
        raise ValueError("Nonfinite inputs")
    if not np.allclose(np.linalg.norm(known, axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError("Nonunit inputs")
    _, s, vt = np.linalg.svd(pack(h).reshape(-1, 528), full_matrices=False)
    if s[127] < 1e-10 * s[0] or s[127] < 100 * s[128]:
        raise ValueError("Original Hessian subspace spectral guard failed")
    directions, rounds = search(known, vt[:128], target=64, output=output)
    np.savez(output / "multi-coverage-solution.npz", directions=directions)
    report = {
        "experiment": "isolated-glu-multi-coverage-development-v1",
        "sealed": False,
        "input_sha256": hashes,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_count": len(known),
        "retained_count": len(directions),
        "rounds": rounds,
        "configuration": {
            "public_seed": 20261025,
            "max_rounds": 4,
            "mixtures_per_round": 256,
            "eigenvectors_per_mixture": 2,
            "gn_steps": 24,
        },
        "seconds": time.perf_counter() - start,
        "new_queries": 0,
        "private_teacher_read": False,
        "heldout_read": False,
        "coverage_certified": False,
        "full_glu_recovered": False,
        "formal_v3_outcome_changed": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2])), allow_nan=False))
