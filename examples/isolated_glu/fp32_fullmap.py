"""Public development only: consistent FP32 observations and frozen learner."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
from modelforge import glu32
from modelforge.antipodal import canonical_probe_points
from modelforge.glu32_io import raw, read_json, read_arrays, SHAPES

spec = importlib.util.spec_from_file_location(
    "public_precision", Path(__file__).with_name("precision.py")
)
assert spec and spec.loader
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)
Array = NDArray[np.float64]


def write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def public_weights() -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(572041)
    return (
        rng.normal(size=(64, 32)) * 0.2,
        rng.normal(size=(64, 32)) * 0.2,
        rng.normal(size=(32, 64)) * 0.2,
    )


def collect_multiscale(query: Callable[[Array], Array]) -> dict[str, Array]:
    rng = np.random.default_rng(931082)
    small, large = rng.normal(size=(4096, 32)), rng.normal(size=(512, 32))
    even = canonical_probe_points(32)
    return {
        "small_x": small,
        "small_odd": 0.5 * (query(0.001 * small) - query(-0.001 * small)),
        "large_x": large,
        "large_odd": 0.5 * (query(1000 * large) - query(-1000 * large)),
        "even_pos": query(even),
        "even_neg": query(-even),
    }


def prepare(root: Path) -> dict[str, Any]:
    public = root / "public"
    public.mkdir(exist_ok=False)
    g, u, d = public_weights()
    calls = 0

    def query(x: Array) -> Array:
        nonlocal calls
        calls += len(x)
        # Negative saturated inputs may overflow exp; resulting SiLU remains finite.
        with np.errstate(over="ignore"):
            return np.asarray(precision.forward(g, u, d, x, "fp32_compute"))

    probes = np.random.default_rng(20261024).normal(size=(64, 32)) * 4
    fine, fc = precision.curvature(query, probes, 0.3)
    coarse, cc = precision.curvature(query, probes, 0.6)
    np.savez(public / "curvature-pair.npz", fine=fine, coarse=coarse)
    np.savez(public / "hessians.npz", hessian=precision.richardson(fine, coarse))
    observations = collect_multiscale(query)
    np.savez(
        public / "multiscale.npz",
        small_x=observations["small_x"],
        small_odd=observations["small_odd"],
        large_x=observations["large_x"],
        large_odd=observations["large_odd"],
        even_pos=observations["even_pos"],
        even_neg=observations["even_neg"],
    )
    if fc + cc != 270336 or calls != 280608:
        raise ValueError("Calibration call inventory mismatch")
    files = {
        n: sha(public / n)
        for n in ["hessians.npz", "multiscale.npz", "curvature-pair.npz"]
    }
    write(public / "manifest.json", {"channel": "fp32_compute", "files": files})
    return {
        "stage": "prepare",
        "channel": "fp32_compute",
        "vector_calls": calls,
        "files": files,
        "manifest_sha256": sha(public / "manifest.json"),
    }


def fit(root: Path) -> dict[str, Any]:
    # Exclusive attempt marker survives timeout, process death and input errors.
    with (root / "fit-attempt.json").open("x") as handle:
        json.dump({"fits_reserved": 1, "automatic_retry": False}, handle)
    public = root / "public"
    meta = read_json(public / "manifest.json")
    if (
        set(meta) != {"channel", "files"}
        or meta["channel"] != "fp32_compute"
        or set(meta["files"])
        != {"hessians.npz", "multiscale.npz", "curvature-pair.npz"}
    ):
        raise ValueError("Invalid declared channel/observation inventory")
    for name, commitment in meta["files"].items():
        if sha(public / name) != commitment:
            raise ValueError("Calibration commitment mismatch")
    h = read_arrays(
        public / "hessians.npz",
        {"hessian": (64, 32, 32, 32)},
        meta["files"]["hessians.npz"],
    )["hessian"]
    observations = read_arrays(
        public / "multiscale.npz", SHAPES, meta["files"]["multiscale.npz"]
    )
    stage = "directions"
    report: dict[str, Any] = {
        "candidate_built": False,
        "stage": stage,
        "calibration_manifest_sha256": sha(public / "manifest.json"),
        "sealed": False,
    }
    if (root / "candidate.npz").exists() or (root / "fit-report.json").exists():
        raise FileExistsError("Development fit already attempted")
    try:
        directions, first = glu32.directions(h)
        report["directions"] = first
        stage = "multiscale"
        gates, couplings, second = glu32.multiscale(directions, observations)
        if (
            gates.shape != (64, 32)
            or couplings.shape != (64, 32, 32)
            or not np.isfinite(gates).all()
            or not np.isfinite(couplings).all()
        ):
            raise ValueError("Invalid candidate")
        np.savez(root / "candidate.npz", gates=gates, couplings=couplings)
        report.update(
            candidate_built=True,
            stage="candidate_built",
            multiscale=second,
            candidate_sha256=sha(root / "candidate.npz"),
        )
    except ValueError as exc:
        report.update(stage=stage, error=str(exc))
    write(root / "fit-report.json", report)
    return report


def evaluate(root: Path) -> dict[str, Any]:
    report = read_json(root / "fit-report.json")
    if report.get("candidate_built") is not True:
        raise ValueError("No accepted development candidate")
    if sha(root / "candidate.npz") != report["candidate_sha256"]:
        raise ValueError("Candidate commitment changed")
    if sha(root / "public/manifest.json") != report["calibration_manifest_sha256"]:
        raise ValueError("Calibration manifest changed")
    student = read_arrays(
        root / "candidate.npz",
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        report["candidate_sha256"],
    )
    g, u, d = public_weights()
    rng = np.random.default_rng(941027)
    groups = []
    for scale in [0.01, 1.0, 10.0, 100.0]:
        x = rng.normal(size=(512, 32)) * scale
        with np.errstate(over="ignore"):
            truth = np.asarray(precision.forward(g, u, d, x, "fp32_compute"))
        predicted = glu32.predict(student["gates"], student["couplings"], x)
        difference = predicted - truth
        groups.append(
            {
                "scale": scale,
                "relative_frobenius": float(
                    np.linalg.norm(difference) / np.linalg.norm(truth)
                ),
                "max_absolute": float(np.max(np.abs(difference))),
            }
        )
    return {
        "stage": "public_development_evaluation",
        "vector_calls": 2048,
        "groups": groups,
        "sealed": False,
        "original_weights_recovered": False,
        "universal_reconstruction": False,
    }


if __name__ == "__main__":
    request = json.load(sys.stdin)
    functions = {"prepare": prepare, "fit": fit, "evaluate": evaluate}
    print(
        json.dumps(functions[request["stage"]](Path(request["root"])), allow_nan=False)
    )
