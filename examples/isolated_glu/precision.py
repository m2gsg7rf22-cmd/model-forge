"""Public-only curvature precision diagnostic; no learner or sealed evaluation."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CHANNELS = ("fp64", "fp64_return_fp32", "fp64_return_fp16", "fp32_compute")


def reference_weights(
    g: Array, u: Array, d: Array, channel: str
) -> tuple[Array, Array, Array]:
    """Smooth reference for rounded stored weights, not rounded arithmetic."""
    if channel not in CHANNELS:
        raise ValueError("Unsupported channel")
    return tuple(
        np.asarray(a, dtype=np.float32).astype(np.float64)
        if channel == "fp32_compute"
        else a
        for a in (g, u, d)
    )  # type: ignore[return-value]


def forward(g: Array, u: Array, d: Array, x: Array, channel: str) -> Array:
    if channel not in CHANNELS:
        raise ValueError("Unsupported channel")
    dtype = np.float32 if channel == "fp32_compute" else np.float64
    xx = np.asarray(x, dtype=dtype)
    gg, uu, dd = (np.asarray(a, dtype=dtype) for a in (g, u, d))
    z = xx @ gg.T
    y = (z / (1 + np.exp(-z)) * (xx @ uu.T)) @ dd.T
    if channel == "fp64_return_fp32":
        y = y.astype(np.float32)
    elif channel == "fp64_return_fp16":
        y = y.astype(np.float16)
    result = np.asarray(y, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite forward response")
    return result


def analytic_hessian(g: Array, u: Array, d: Array, x: Array) -> Array:
    """Diagnostic truth only: H=down*(s''(gx)*(ux)*gg'+s'(gx)*(gu'+ug'))."""
    z, value = x @ g.T, x @ u.T
    sig = 1 / (1 + np.exp(-z))
    first = sig + z * sig * (1 - sig)
    second = sig * (1 - sig) * (2 + z * (1 - 2 * sig))
    result = np.empty((len(x), len(d), g.shape[1], g.shape[1]))
    for k in range(len(x)):
        gg = np.einsum("ow,wi,wj->oij", d * (second[k] * value[k]), g, g, optimize=True)
        cross = np.einsum("ow,wi,wj->oij", d * first[k], g, u, optimize=True)
        result[k] = gg + cross + cross.swapaxes(-1, -2)
    return result


def curvature(
    query: Callable[[Array], Array], probes: Array, step: float
) -> tuple[Array, int]:
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Positive finite step required")
    dimension = probes.shape[1]
    i, j = np.triu_indices(dimension)
    ei, ej = np.eye(dimension)[i] * step, np.eye(dimension)[j] * step
    result: Array | None = None
    calls = 0
    for k, x in enumerate(probes):
        value = (
            query(x + ei + ej)
            - query(x + ei - ej)
            - query(x - ei + ej)
            + query(x - ei - ej)
        ) / (4 * step**2)
        calls += 4 * len(i)
        if result is None:
            result = np.empty((len(probes), value.shape[1], dimension, dimension))
        result[k, :, i, j] = value
        result[k, :, j, i] = value
    if result is None or not np.isfinite(result).all():
        raise ValueError("Empty/nonfinite curvature")
    return result, calls


def richardson(fine: Array, coarse: Array) -> Array:
    return np.asarray((4 * fine - coarse) / 3)


def matrix(h: Array) -> Array:
    i, j = np.triu_indices(h.shape[-1])
    return np.asarray(
        h[:, :, i, j].reshape(-1, len(i)) * np.where(i == j, 1.0, np.sqrt(2.0))
    )


def metrics(estimate: Array, exact: Array, rank: int) -> dict[str, float]:
    if (
        estimate.shape != exact.shape
        or not np.isfinite(estimate).all()
        or not np.isfinite(exact).all()
    ):
        raise ValueError("Mismatched/nonfinite Hessians")
    a, b = matrix(estimate), matrix(exact)
    _, s, v = np.linalg.svd(a, full_matrices=False)
    _, se, ve = np.linalg.svd(b, full_matrices=False)
    if rank < 1 or rank > len(s) or se[rank - 1] <= 1e-12 * se[0] or s[rank - 1] == 0:
        raise ValueError("Diagnostic reference/estimate rank inadequate")
    reference_next = float(se[rank]) if rank < len(se) else 0.0
    reference_gap = float(se[rank - 1] / max(reference_next, np.finfo(float).tiny))
    if reference_gap < 100:
        raise ValueError("Analytic reference rank not spectrally separated")
    vb, eb = v[:rank], ve[:rank]
    distance = np.linalg.norm(vb - (vb @ eb.T) @ eb) / np.sqrt(rank)
    return {
        "reference_singular_rank": float(se[rank - 1]),
        "reference_singular_next": reference_next,
        "reference_rank_gap": reference_gap,
        "hessian_relative_error": float(
            np.linalg.norm(estimate - exact) / np.linalg.norm(exact)
        ),
        "subspace_rms_sine": float(distance),
        "singular_rank": float(s[rank - 1]),
        "singular_next": float(s[rank]) if rank < len(s) else 0.0,
        "rank_gap": float(
            s[rank - 1] / max(s[rank] if rank < len(s) else 0.0, np.finfo(float).tiny)
        ),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Only the committed fixed public diagnostic; no adjustable teacher seeds."""
    if (
        config.get("teacher_seed") != 572041
        or config.get("input_dim") != 32
        or config.get("width") != 64
        or config.get("output_dim") != 32
        or config.get("channels") != list(CHANNELS)
        or config.get("steps") != [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        or config.get("probe_seed") != 20261024
        or config.get("probe_count") != 64
        or config.get("weight_std") != 0.2
        or config.get("probe_std") != 4
        or config.get("rank") != 128
        or config.get("reference_gap_min") != 100
    ):
        raise ValueError("Fixed public diagnostic configuration required")
    start = time.perf_counter()
    rng = np.random.default_rng(572041)
    g, u, d = (
        rng.normal(size=(64, 32)) * 0.2,
        rng.normal(size=(64, 32)) * 0.2,
        rng.normal(size=(32, 64)) * 0.2,
    )
    probes = np.random.default_rng(20261024).normal(size=(64, 32)) * 4
    records: list[dict[str, Any]] = []
    calls = 0
    for channel in CHANNELS:
        reference = analytic_hessian(*reference_weights(g, u, d, channel), probes)
        for step in config["steps"]:

            def query(x: Array) -> Array:
                return forward(g, u, d, x, channel)

            fine, fc = curvature(query, probes, step)
            coarse, cc = curvature(query, probes, 2 * step)
            calls += fc + cc
            for method, value in (
                ("central", fine),
                ("richardson", richardson(fine, coarse)),
            ):
                records.append(
                    {
                        "channel": channel,
                        "step": step,
                        "method": method,
                        **metrics(value, reference, 128),
                    }
                )
    if calls != 6488064 or len(records) != 48:
        raise ValueError("Fixed observation inventory mismatch")
    return {
        "scope": "public development diagnostic, no fitting or sealed success",
        "records": records,
        "vector_calls": calls,
        "analytic_reference_used_only_for_diagnostic_metrics": True,
        "heldout_calls": 0,
        "fits": 0,
        "fresh_secret_teachers": 0,
        "seconds": time.perf_counter() - start,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_final_model_gate": "closed",
        "full_map_recovery_verified": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(json.load(sys.stdin)), allow_nan=False))
