# ruff: noqa: E402
from pathlib import Path
import os

for env_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[env_key] = "1"
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
import json
import time
import hashlib
import sys

p = Path(sys.argv[1])
h = np.load(p / "hessians.npz", allow_pickle=False)["hessian"]
v = np.load(p / "solution.npz", allow_pickle=False)["all_directions"].copy()
n, d = v.shape
i, j = np.triu_indices(d)
weights = np.where(i == j, 1.0, np.sqrt(2.0))
a = h[:, :, i, j].reshape(-1, len(i)) * weights
_, s, vt = np.linalg.svd(a, full_matrices=False)
basis = vt[:128]


def calc(x: Array) -> Array:
    q = x[i] * x[j] * weights
    return q - (q @ basis.T) @ basis


start = time.perf_counter()
before = []
after = []
iterations = []
eye = np.eye(d)
for k, x in enumerate(v):
    r = calc(x)
    before.append(float(r @ r))
    for t in range(12):
        jac = (x[i, None] * eye[j] + x[j, None] * eye[i]) * weights[:, None]
        jac -= basis.T @ (basis @ jac)
        # SVD produces an orthonormal tangent basis, avoiding a redundant radial column.
        _, _, tangent = np.linalg.svd(x.reshape(1, -1), full_matrices=True)
        T = tangent[1:].T
        delta = T @ np.linalg.lstsq(jac @ T, -r, rcond=1e-10)[0]
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            y = x + alpha * delta
            y /= np.linalg.norm(y)
            rr = calc(y)
            if rr @ rr < r @ r:
                x, r = y, rr
                accepted = True
                break
        if not accepted:
            break
    v[k] = x
    after.append(float(r @ r))
    iterations.append(t + 1)
after_array = np.array(after)
kept: list[int] = []
for k in np.argsort(after_array):
    if after_array[k] <= 1e-8 and all(abs(v[k] @ v[q]) < 0.9995 for q in kept):
        kept.append(int(k))
np.savez(
    p / "refined-solution.npz",
    directions=v[kept],
    all_directions=v,
    residual=after_array,
)
report = {
    "development_pilot_not_sealed": True,
    "input_solution_sha256": hashlib.sha256(
        (p / "solution.npz").read_bytes()
    ).hexdigest(),
    "algorithm": "public-residual tangent Gauss-Newton,12steps,backtracking only on calibration Hessian residual",
    "directions_retained": len(kept),
    "before_max": max(before),
    "after_max": float(after_array.max()),
    "before_median": float(np.median(before)),
    "after_median": float(np.median(after_array)),
    "max_objective_increase": float(np.max(after_array - np.array(before))),
    "seconds": time.perf_counter() - start,
    "teacher_read_for_fit": False,
    "full_glu_recovered": False,
}
(p / "refine.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
