"""UMRRE: bounded, reproducible reconstruction experiments on project-owned models.

A solver receives an oracle, never its weights. Ground truth is used only by
separate evaluation code. These experiments make no claim about proprietary LLMs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import psutil

Array = npt.NDArray[np.float64]
SOURCES = [
    "https://proceedings.mlr.press/v235/carlini24a.html",
    "https://arxiv.org/abs/2003.04884",
    "https://proceedings.mlr.press/v235/ziyin24a.html",
    "https://arxiv.org/abs/2306.00978",
]


def recover_affine(
    oracle: Callable[[Array], Array], dimensions: int
) -> tuple[Array, Array, int]:
    """Recover y=xW+b using zero and basis queries; no target parameter access."""
    if type(dimensions) is not int or not 1 <= dimensions <= 64:
        raise ValueError("dimensions must be an integer in [1, 64]")
    queries = np.vstack([np.zeros((1, dimensions)), np.eye(dimensions)])
    outputs = np.asarray(oracle(queries), dtype=np.float64)
    if (
        outputs.ndim != 2
        or outputs.shape[0] != dimensions + 1
        or not np.isfinite(outputs).all()
    ):
        raise ValueError("Oracle must return a finite matrix with one row per query")
    bias = outputs[0].copy()
    return outputs[1:] - bias, bias, dimensions + 1


def _mse(actual: Array, expected: Array) -> float:
    return float(np.mean((actual - expected) ** 2))


def _quantize(value: Array) -> tuple[Array, int]:
    scale = max(float(np.max(np.abs(value))) / 127, np.finfo(np.float64).tiny)
    packed = np.rint(value / scale).clip(-127, 127).astype(np.int8)
    return packed.astype(np.float64) * scale, packed.nbytes + 8


def _linear(rng: np.random.Generator) -> dict[str, Any]:
    weight, bias = rng.normal(size=(8, 3)), rng.normal(size=3)

    def oracle(x: Array) -> Array:
        return np.asarray(x @ weight + bias, dtype=np.float64)

    recovered, recovered_bias, count = recover_affine(oracle, 8)
    heldout = rng.normal(size=(256, 8))
    return {
        "query_count": count,
        "evaluation_query_count": 256,
        "parameter_count": 27,
        "input_dimension": 8,
        "output_dimension": 3,
        "max_parameter_error": float(
            max(
                np.max(np.abs(weight - recovered)),
                np.max(np.abs(bias - recovered_bias)),
            )
        ),
        "heldout_mse": _mse(heldout @ recovered + recovered_bias, oracle(heldout)),
        "conditions": "Known affine family; noiseless full real-valued output; unrestricted basis inputs; float64 arithmetic.",
        "conclusion": "Recovery to measured floating-point tolerance, not bit-identical source bytes.",
    }


def _nonidentifiability(rng: np.random.Generator) -> dict[str, Any]:
    x = rng.normal(size=(4, 8))
    weight = rng.normal(size=(8, 3))
    _, _, vh = np.linalg.svd(x, full_matrices=True)
    direction = vh[-1]
    alternative = weight + np.outer(direction, np.array([1.0, -2.0, 3.0]))
    return {
        "observations": 4,
        "input_dimension": 8,
        "design_rank": int(np.linalg.matrix_rank(x)),
        "query_count": 4,
        "parameter_l2_distance": float(np.linalg.norm(weight - alternative)),
        "observed_max_output_difference": float(
            np.max(np.abs(x @ weight - x @ alternative))
        ),
        "witness_input": direction.tolist(),
        "witness_max_output_difference": float(
            np.max(np.abs(direction @ weight - direction @ alternative))
        ),
        "conclusion": "A nullspace perturbation produces indistinguishable observations and different unseen outputs. Infinite affine solutions exist.",
    }


def _symmetry(rng: np.random.Generator) -> dict[str, Any]:
    w1, b1, w2 = rng.normal(size=(8, 12)), rng.normal(size=12), rng.normal(size=(12, 3))
    permutation = np.roll(np.arange(12), 1)
    scale = np.linspace(0.5, 2.0, 12)
    transformed_w1 = w1[:, permutation] * scale
    transformed_b1 = b1[permutation] * scale
    transformed_w2 = w2[permutation] / scale[:, None]
    x = rng.normal(size=(256, 8))
    original = np.maximum(x @ w1 + b1, 0) @ w2
    transformed = np.maximum(x @ transformed_w1 + transformed_b1, 0) @ transformed_w2
    # ReLU(cx)=c ReLU(x), c>0, plus consistent hidden permutation: an all-input identity.
    return {
        "query_count": 0,
        "evaluation_query_count": 256,
        "weight_l2_distance": float(np.linalg.norm(w1 - transformed_w1)),
        "heldout_max_output_difference": float(np.max(np.abs(original - transformed))),
        "algebraic_identity": "ReLU((xW+b)PD) D^-1 P^T V = ReLU(xW+b)V for permutation P and positive diagonal D",
        "conclusion": "Function equivalence cannot identify a unique neural parameter vector even with all-input observations.",
    }


def _approximation(rng: np.random.Generator) -> dict[str, Any]:
    target_w = rng.normal(size=(8, 16)) / np.sqrt(8)
    target_v = rng.normal(size=(16, 3)) / 4

    def target(x: Array) -> Array:
        return np.asarray(np.tanh(x @ target_w) @ target_v, dtype=np.float64)

    # Only oracle outputs enter student fitting. Validation and test queries are separate.
    train, validation, test = (rng.normal(size=(n, 8)) for n in (256, 128, 256))
    train_y, validation_y, test_y = target(train), target(validation), target(test)
    candidates: list[dict[str, Any]] = []
    fitted: list[tuple[Array, Array]] = []
    for width in (8, 16, 32):
        projection = rng.normal(size=(8, width)) / np.sqrt(8)
        features = np.tanh(train @ projection)
        coefficient = np.linalg.solve(
            features.T @ features + 0.01 * np.eye(width), features.T @ train_y
        )
        candidates.append(
            {
                "width": width,
                "parameter_count": 11 * width,
                "validation_mse": _mse(
                    np.tanh(validation @ projection) @ coefficient, validation_y
                ),
                "multiply_adds_per_input": 11 * width,
            }
        )
        fitted.append((projection, coefficient))
    chosen = min(range(len(candidates)), key=lambda i: candidates[i]["validation_mse"])
    p, c = fitted[chosen]
    prediction = np.tanh(test @ p) @ c
    # Bootstrap independent held-out examples; this interval does not estimate universal behavior.
    errors = np.mean((prediction - test_y) ** 2, axis=1)
    means = np.mean(
        errors[rng.integers(0, len(errors), size=(200, len(errors)))], axis=1
    )
    return {
        "query_count": 384,
        "evaluation_query_count": 256,
        "candidates": candidates,
        "selected_width": candidates[chosen]["width"],
        "heldout_mse": _mse(prediction, test_y),
        "zero_predictor_heldout_mse": _mse(np.zeros_like(test_y), test_y),
        "heldout_mse_bootstrap_95_percent_interval": np.quantile(
            means, [0.025, 0.975]
        ).tolist(),
        "bootstrap_replicates": 200,
        "selection_uses_test_set": False,
        "conclusion": "Bounded random-feature architecture search and behavioral approximation on Gaussian inputs; neither architecture nor original weights recovered.",
    }


def _compression(rng: np.random.Generator) -> dict[str, Any]:
    w = rng.normal(size=(32, 24))
    x = rng.normal(size=(256, 32))
    baseline = x @ w
    u, s, vh = np.linalg.svd(w, full_matrices=False)
    lowrank = (u[:, :4] * s[:4]) @ vh[:4]
    pruned = w.copy()
    removed = np.argsort(np.linalg.norm(w, axis=0))[:12]
    pruned[:, removed] = 0
    quantized, packed_bytes = _quantize(w)
    rows = []
    for name, candidate, stored in [
        ("rank_4", lowrank, (32 * 4 + 4 * 24) * 8),
        ("structured_column_pruning_50_percent", pruned, (32 * 12) * 8 + 12 * 8),
        ("symmetric_int8", quantized, packed_bytes),
    ]:
        rows.append(
            {
                "method": name,
                "weight_mse": _mse(candidate, w),
                "output_mse": _mse(x @ candidate, baseline),
                "representation_bytes": stored,
                "ratio_to_dense_float64": stored / w.nbytes,
                "lossless": bool(np.array_equal(candidate, w)),
            }
        )
    # Actual small attention calculation, quantized storage then float64 dequantized compute.
    key, value, query = (
        rng.normal(size=(64, 16)),
        rng.normal(size=(64, 16)),
        rng.normal(size=(8, 16)),
    )

    def attention(k: Array, v: Array) -> Array:
        score = query @ k.T / 4
        prob = np.exp(score - score.max(axis=1, keepdims=True))
        return (prob / prob.sum(axis=1, keepdims=True)) @ v

    qkey, keybytes = _quantize(key)
    qvalue, valuebytes = _quantize(value)
    return {
        "query_count": 0,
        "evaluation_input_count": 256,
        "weight_shape": list(w.shape),
        "dense_bytes": w.nbytes,
        "methods": rows,
        "kv_cache": {
            "tokens": 64,
            "head_dimension": 16,
            "dense_bytes": key.nbytes + value.nbytes,
            "int8_representation_bytes": keybytes + valuebytes,
            "attention_output_mse": _mse(
                attention(qkey, qvalue), attention(key, value)
            ),
            "compute_dtype": "float64 after dequantization",
            "lossless": False,
        },
        "conclusion": "Representational size reductions have measured error; no speedup or production LLM quality claim.",
    }


def _modalities(rng: np.random.Generator) -> dict[str, Any]:
    rows = []
    for name, shape in [
        ("image", (4, 4, 3)),
        ("audio", (64,)),
        ("video", (2, 4, 4, 1)),
    ]:
        dimension = int(np.prod(shape))
        weight = rng.normal(size=(dimension, 2))
        bias = rng.normal(size=2)

        # Tensor adapter flattens an owned, known affine synthetic target.
        def oracle(batch: Array) -> Array:
            return np.asarray(batch @ weight + bias, dtype=np.float64)

        recovered, offset, count = recover_affine(oracle, dimension)
        test = rng.normal(size=(16, *shape)).reshape(16, dimension)
        rows.append(
            {
                "modality": name,
                "tensor_shape": list(shape),
                "query_count": count,
                "evaluation_query_count": 16,
                "heldout_mse": _mse(test @ recovered + offset, oracle(test)),
                "scope": "Synthetic affine tensor projection only; no pretrained generative model support",
            }
        )
    return {
        "experiments": rows,
        "query_count": sum(r["query_count"] for r in rows),
        "evaluation_query_count": 48,
        "unsupported": [
            "diffusion weight recovery",
            "closed video model reconstruction",
            "semantic multimodal equivalence",
        ],
    }


def run_research(
    destination: str | Path | None = None, *, seed: int = 7
) -> dict[str, Any]:
    """Execute six fixed-cost local experiments and optionally atomically save JSON.

    Destination must be a new directory. No downloads, API credentials, or remote
    queries. Dimensions and dataset sizes are fixed to bound CPU and allocations.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    target = (
        Path(destination).expanduser().absolute() if destination is not None else None
    )
    if target is not None:
        if target.exists() or target.is_symlink():
            raise ValueError("Research destination must be a new directory")
        if any(p.is_symlink() for p in target.parents):
            raise ValueError("Research destination ancestors must not be symlinks")
        target.mkdir(parents=True)
    start = time.perf_counter()
    process = psutil.Process()
    samples = [process.memory_info().rss]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.01):
            samples.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    experiments: dict[str, Any] = {}
    try:
        names = [
            "affine_recovery",
            "rank_deficient_nonidentifiability",
            "neural_symmetry",
            "functional_approximation",
            "compression",
            "multimodal_adapters",
        ]
        functions = [
            _linear,
            _nonidentifiability,
            _symmetry,
            _approximation,
            _compression,
            _modalities,
        ]
        for name, function, child_seed in zip(
            names, functions, np.random.SeedSequence(seed).spawn(len(names))
        ):
            before = time.perf_counter()
            result = function(np.random.default_rng(child_seed))
            result["elapsed_seconds"] = time.perf_counter() - before
            experiments[name] = result
    finally:
        stop.set()
        monitor.join(timeout=1)
    samples.append(process.memory_info().rss)
    # Content hash excludes timings/environment so numerical reproducibility is independently checkable.
    numerical = {
        key: {k: v for k, v in value.items() if k != "elapsed_seconds"}
        for key, value in experiments.items()
    }
    fingerprint = hashlib.sha256(
        json.dumps(numerical, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    report: dict[str, Any] = {
        "engine": "UMRRE",
        "protocol_version": 1,
        "seed": seed,
        "ground_truth_owner": "ModelForge synthetic models",
        "scope": "Controlled synthetic models; no universal or proprietary model reconstruction demonstrated",
        "experiments": experiments,
        "numerical_sha256": fingerprint,
        "cost": {
            "remote_requests": 0,
            "paid_api_cost_usd": 0,
            "oracle_query_count": sum(
                int(row.get("query_count", 0)) for row in experiments.values()
            ),
            "evaluation_query_count": sum(
                int(row.get("evaluation_query_count", 0))
                for row in experiments.values()
            ),
            "elapsed_seconds": time.perf_counter() - start,
            "sampled_peak_process_rss_bytes": max(samples),
            "rss_before_bytes": samples[0],
            "rss_after_bytes": samples[-1],
            "memory_measurement": "10ms process RSS sampling, includes interpreter and libraries; may miss transient peaks",
        },
        "environment": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "sources": SOURCES,
        "limitations": [
            "No original closed-model weights or architecture recovered",
            "Fixed synthetic families and input distributions",
            "Floating-point tolerance is not byte identity",
            "Sampled memory is not an OS-enforced cap",
            "Architecture search is three random-feature widths; not transformer discovery",
            "MoE routing optimization is not implemented",
        ],
    }
    if target is not None:
        report["report_path"] = str(target / "umrre-results.json")
        temporary = target / ".umrre-results.tmp"
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, target / "umrre-results.json")
    return report
