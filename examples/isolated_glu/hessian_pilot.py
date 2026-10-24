# ruff: noqa: E402
"""Development pilot only: forward-query Hessian gate candidates, no full GLU."""

from pathlib import Path
import os

for env_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[env_key] = "1"
import sys
import json
import time
import subprocess
import hashlib
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

root = Path(sys.argv[2])
stage = sys.argv[1]
if stage == "solve":
    h = np.load(root / "hessians.npz", allow_pickle=False)["hessian"]
    d = h.shape[-1]
    i, j = np.triu_indices(d)
    scale = np.where(i == j, 1.0, np.sqrt(2.0))
    a = h[:, :, i, j].reshape(-1, len(i)) * scale
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    rank = 128
    if (
        not np.all(np.isfinite(h))
        or sv[0] == 0
        or sv[rank - 1] <= 1e-10 * sv[0]
        or (sv[rank - 1] < 100 * sv[rank])
    ):
        raise ValueError(
            "Hessian subspace rank or spectral separation is inadequate for this rank128 pilot"
        )
    b = vt[:rank]
    B = np.zeros((rank, d, d))
    B[:, i, j] = b / scale
    B[:, j, i] = b / scale
    rng = np.random.default_rng(20261024)
    v = rng.normal(size=(256, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    adam_m = np.zeros_like(v)
    adam_v = np.zeros_like(v)
    start = time.perf_counter()
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
    kept: list[int] = []
    for k in np.argsort(residual):
        if residual[k] <= 1e-08 and all((abs(v[k] @ v[q]) < 0.9995 for q in kept)):
            kept.append(int(k))
    np.savez(
        root / "solution.npz", directions=v[kept], all_directions=v, residual=residual
    )
    (root / "solve.json").write_text(
        json.dumps(
            {
                "candidates": len(kept),
                "residual_min": float(residual.min()),
                "residual_median": float(np.median(residual)),
                "singular_128": float(sv[127]),
                "singular_129": float(sv[128]),
                "seconds": time.perf_counter() - start,
                "iterations": 1600,
                "starts": 256,
                "subspace_rank": rank,
            },
            indent=2,
        )
    )
else:
    root.mkdir(exist_ok=False, parents=True)
    rng = np.random.default_rng(572041)
    d = 32
    w = 64
    o = 32
    G = rng.normal(size=(w, d)) * 0.2
    U = rng.normal(size=(w, d)) * 0.2
    D = rng.normal(size=(o, w)) * 0.2

    def f(x: Array) -> Array:
        z = x @ G.T
        return np.asarray(z / (1 + np.exp(-z)) * (x @ U.T) @ D.T, dtype=np.float64)

    x = rng.normal(size=(64, d)) * 4
    step = 0.003
    i, j = np.triu_indices(d)
    ei = np.eye(d)[i] * step
    ej = np.eye(d)[j] * step
    H = np.zeros((len(x), o, d, d))
    calls = 0
    for n, point in enumerate(x):
        value = (
            f(point + ei + ej)
            - f(point + ei - ej)
            - f(point - ei + ej)
            + f(point - ei - ej)
        ) / (4 * step**2)
        calls += 4 * len(i)
        H[n, :, i, j] = value
        H[n, :, j, i] = value
    np.savez(root / "hessians.npz", hessian=H)
    start = time.perf_counter()
    p = subprocess.run(
        [sys.executable, __file__, "solve", str(root)],
        timeout=150,
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise RuntimeError(p.stderr)
    sol = np.load(root / "solution.npz", allow_pickle=False)
    v = sol["directions"]
    truth = G / np.linalg.norm(G, axis=1, keepdims=True)
    similarity = np.abs(truth @ v.T) if len(v) else np.zeros((w, 1))
    match = similarity.max(axis=1)
    allmatch = np.abs(truth @ sol["all_directions"].T).max(axis=1)
    report = json.loads((root / "solve.json").read_text())
    report.update(
        {
            "development_pilot_not_sealed": True,
            "dimension": d,
            "width": w,
            "outputs": o,
            "probe_count": len(x),
            "probe_scale": 4,
            "finite_difference_step": step,
            "forward_vector_queries": calls,
            "gate_cosine_above_0_9995": int((match > 0.9995).sum()),
            "all_candidates_gate_cosine_above_0_9995": int((allmatch > 0.9995).sum()),
            "min_gate_cosine": float(match.min()),
            "solution_sha256": hashlib.sha256(
                (root / "solution.npz").read_bytes()
            ).hexdigest(),
            "elapsed_parent_seconds": time.perf_counter() - start,
            "full_glu_recovered": False,
            "original_weights_recovered": False,
            "private_model_gate": "closed",
            "known_development_seed": 572041,
        }
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
