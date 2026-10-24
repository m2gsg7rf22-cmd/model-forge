# ruff: noqa: E402
"""Development pilot. Separate public-data solve; no full-model equivalence claim."""

from pathlib import Path
import os

for env_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[env_key] = "1"
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
import sys
import json
import time
import subprocess
from modelforge.antipodal import canonical_probe_points, recover_even_quadratic

p = Path(sys.argv[2])
stage = sys.argv[1]
if stage == "solve":
    start = time.perf_counter()
    data = np.load(p / "multiscale-observations.npz", allow_pickle=False)
    v = np.load(p / "refined-solution.npz", allow_pickle=False)["directions"]
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
    np.savez(p / "multiscale-solution.npz", gates=g * sign[:, None], couplings=C1)
    report = {
        "development_pilot_not_sealed": True,
        "small_design_rank": int(rank),
        "small_design_columns": w * d,
        "small_design_condition": float(svals[0] / svals[-1]),
        "magnitude_design_rank": int(r),
        "magnitude_min": float(c.min()),
        "magnitude_max": float(c.max()),
        "coupling_rank1_max_relative_tail": float((ss[1][:, 1] / ss[1][:, 0]).max()),
        "orientation_rank": int(sr),
        "orientation_max_distance_from_sign": float(np.max(np.abs(coeff - sign))),
        "orientation_rounded_relative_residual": float(
            np.linalg.norm(design @ sign - Q[:, i, j].reshape(-1))
            / np.linalg.norm(Q[:, i, j])
        ),
        "seconds": time.perf_counter() - start,
        "public_only_fit": True,
    }
    (p / "multiscale-solve.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
else:
    rng = np.random.default_rng(572041)
    G = rng.normal(size=(64, 32)) * 0.2
    U = rng.normal(size=(64, 32)) * 0.2
    D = rng.normal(size=(32, 64)) * 0.2

    def f(x: Array) -> Array:
        z = x @ G.T
        sig = 0.5 * (1 + np.tanh(z / 2))
        return (z * sig * (x @ U.T)) @ D.T

    def odd(x: Array) -> Array:
        return 0.5 * (f(x) - f(-x))

    rng = np.random.default_rng(931082)
    small = rng.normal(size=(4096, 32))
    large = rng.normal(size=(512, 32))
    even = canonical_probe_points(32)
    np.savez(
        p / "multiscale-observations.npz",
        small_x=small,
        small_odd=odd(small * 0.001),
        large_x=large,
        large_odd=odd(large * 1000),
        even_pos=f(even),
        even_neg=f(-even),
    )
    run = subprocess.run(
        [sys.executable, __file__, "solve", str(p)],
        timeout=150,
        capture_output=True,
        text=True,
    )
    if run.returncode:
        raise RuntimeError(run.stderr)
    sol = np.load(p / "multiscale-solution.npz", allow_pickle=False)
    g = sol["gates"]
    C = sol["couplings"]
    x = rng.normal(size=(256, 32))
    z = x @ g.T
    student = np.einsum("nw,wod,nd->no", z * 0.5 * (1 + np.tanh(z / 2)), C, x)
    truth = f(x)
    report = json.loads((p / "multiscale-solve.json").read_text())
    report.update(
        {
            "full_glu_validation_relative_error": float(
                np.linalg.norm(student - truth) / np.linalg.norm(truth)
            ),
            "full_glu_validation_max_abs_error": float(np.max(np.abs(student - truth))),
            "validation_rows": len(x),
            "sealed": False,
            "new_calibration_queries": 2 * 4096 + 2 * 512 + 1056,
            "new_validation_queries": len(x),
            "private_model_gate": "closed",
            "original_weights_recovered": False,
            "all_inputs_equivalence_proven": False,
        }
    )
    (p / "multiscale-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
