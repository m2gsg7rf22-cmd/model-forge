"""Read-only counterfactual cutoffs on saved public telemetry; no optimizer."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge.glu32_io import read_arrays

Array = NDArray[np.float64]
GRID = (
    1e-10,
    3e-10,
    1e-9,
    3e-9,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    1e-3,
)


def select(vectors: Array, losses: Array, threshold: float) -> list[int]:
    if (
        vectors.ndim != 2
        or losses.shape != (len(vectors),)
        or not np.isfinite(vectors).all()
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
        or not np.isfinite(threshold)
        or threshold < 0
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError(
            "Requires unit finite vectors, finite nonnegative losses/cutoff"
        )
    selected: list[int] = []
    for k in np.argsort(losses):
        if losses[k] <= threshold and all(
            abs(vectors[k] @ vectors[j]) < 0.9995 for j in selected
        ):
            selected.append(int(k))
    return selected


def diagnose(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if (
        config["thresholds"] != list(GRID)
        or config["duplicate_cosine"] != 0.9995
        or config["baseline_threshold"] != 1e-8
        or config["accept_candidates"] is not False
    ):
        raise ValueError("Fixed diagnostic grid only; no acceptance method")
    data = read_arrays(
        path,
        {
            "known": (52, 32),
            "v": (256, 32),
            "residual": (256,),
            "initial_loss": (256,),
            "b": (128, 528),
        },
        config["telemetry_sha256"],
    )
    v, losses, b = data["v"], data["residual"], data["b"]
    if not np.allclose(b @ b.T, np.eye(128), atol=1e-10, rtol=0):
        raise ValueError("Saved basis is not orthonormal")
    i, j = np.triu_indices(32)
    q = v[:, i] * v[:, j] * np.where(i == j, 1.0, np.sqrt(2.0))
    residual = q - (q @ b.T) @ b
    recomputed = np.sum(residual**2, axis=1)
    if not np.allclose(recomputed, losses, atol=1e-16, rtol=1e-8):
        raise ValueError("Stored refined residuals do not match basis")
    baseline = select(v, losses, 1e-8)
    if not np.array_equal(v[baseline], data["known"]):
        raise ValueError("Original52selection changed")
    rows = []
    for threshold in GRID:
        indices = select(v, losses, threshold)
        rows.append(
            {
                "threshold": threshold,
                "greedy_nonduplicate_count": len(indices),
                "selected_indices": indices,
                "max_selected_squared_loss": float(max(losses[indices]))
                if indices
                else None,
            }
        )
    return {
        "scope": "Public counterfactual threshold diagnostic ONLY; not an acceptance policy",
        "telemetry_sha256": config["telemetry_sha256"],
        "baseline_count": len(baseline),
        "rows": rows,
        "refined_loss_quantiles": np.quantile(losses, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "max_recomputed_loss_difference": float(np.max(np.abs(recomputed - losses))),
        "new_optimizer_calls": 0,
        "target_vector_calls": 0,
        "teacher_truth_used": False,
        "accepted_candidate": False,
        "model_built": False,
        "evaluation_calls": 0,
        "private_gate": "closed",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    request = json.load(sys.stdin)
    print(
        json.dumps(
            diagnose(Path(request["telemetry"]), request["config"]), allow_nan=False
        )
    )
