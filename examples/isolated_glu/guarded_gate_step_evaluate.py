"""Separate one-time PUBLIC development evaluation, not sealed equivalence."""

from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from modelforge import glu32
from modelforge.glu32_io import read_json, read_arrays, raw


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def public_reference() -> Any:
    spec = importlib.util.spec_from_file_location(
        "public_fp32_fixture", Path(__file__).with_name("fp32_fullmap.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(output: Path) -> dict[str, Any]:
    with (output / "evaluation-attempt.json").open("x") as f:
        json.dump(
            {
                "public_reference_calls_reserved": 2048,
                "sealed": False,
                "automatic_retry": False,
            },
            f,
        )
    report = read_json(output / "fit-report.json")
    if report.get("candidate_built") is not True:
        raise ValueError("No development candidate")
    if sha(output / "candidate.npz") != report["candidate_sha256"]:
        raise ValueError("Candidate commitment changed")
    if (
        report.get("method") != "isolated-glu-fp32-guarded-gate-step-v1"
        or sha(output / "public/manifest.json") != report["calibration_manifest_sha256"]
    ):
        raise ValueError("Method/calibration commitment changed")
    student = read_arrays(
        output / "candidate.npz",
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        report["candidate_sha256"],
    )
    baseline = read_arrays(
        output / "baseline.npz",
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        report["baseline_sha256"],
    )
    if sha(output / "baseline.npz") != report["baseline_sha256"]:
        raise ValueError("Baseline commitment changed")
    reference = public_reference()
    g, u, d = reference.public_weights()
    rng = np.random.default_rng(941032)
    inputs = []
    references = []
    predictions = []
    groups = []
    baseline_groups = []
    baseline_predictions = []
    calls = 0
    for scale in [0.01, 1.0, 10.0, 100.0]:
        x = rng.normal(size=(512, 32)) * scale
        with np.errstate(over="ignore"):
            truth = reference.precision.forward(g, u, d, x, "fp32_compute")
        calls += len(x)
        predicted = glu32.predict(student["gates"], student["couplings"], x)
        base_predicted = glu32.predict(baseline["gates"], baseline["couplings"], x)
        base_delta = base_predicted - truth
        baseline_groups.append(
            {
                "scale": scale,
                "relative_frobenius": float(
                    np.linalg.norm(base_delta) / np.linalg.norm(truth)
                ),
                "max_absolute": float(np.max(np.abs(base_delta))),
            }
        )
        baseline_predictions.append(base_predicted)
        delta = predicted - truth
        groups.append(
            {
                "scale": scale,
                "relative_frobenius": float(
                    np.linalg.norm(delta) / np.linalg.norm(truth)
                ),
                "max_absolute": float(np.max(np.abs(delta))),
            }
        )
        inputs.append(x)
        references.append(truth)
        predictions.append(predicted)
    if calls != 2048:
        raise ValueError("Public evaluation call inventory mismatch")
    np.savez(
        output / "public-validation.npz",
        inputs=np.vstack(inputs),
        reference=np.vstack(references),
        prediction=np.vstack(predictions),
        baseline_prediction=np.vstack(baseline_predictions),
    )
    result = {
        "stage": "public_development_evaluation",
        "vector_calls": calls,
        "groups": groups,
        "baseline_groups": baseline_groups,
        "paired_same_reference": True,
        "baseline_sha256": report["baseline_sha256"],
        "candidate_sha256": report["candidate_sha256"],
        "validation_sha256": sha(output / "public-validation.npz"),
        "development_max_group_relative": 1e-3,
        "all_groups_relative_at_most_1e_minus3": all(
            x["relative_frobenius"] <= 1e-3 for x in groups
        ),
        "sealed": False,
        "original_weights_recovered": False,
        "all_inputs_equivalence_proven": False,
        "universal_reconstruction": False,
        "private_gate": "closed",
    }
    (output / "evaluation-report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    req = json.load(sys.stdin)
    print(json.dumps(evaluate(Path(req["output"])), allow_nan=False))
