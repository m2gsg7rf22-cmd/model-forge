"""Owned PUBLIC FP32 collector; separate from the fitting process."""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from modelforge.glu32_io import raw, read_json, read_arrays, SHAPES
import hashlib

Array = NDArray[np.float64]


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def observe(x: Array, query: Callable[[Array], Array]) -> dict[str, Array]:
    return {
        "small_x": x.copy(),
        "fine_pos": query(0.03 * x),
        "fine_neg": query(-0.03 * x),
        "coarse_pos": query(0.06 * x),
        "coarse_neg": query(-0.06 * x),
    }


def collect(calibration: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    with (output / "collection-attempt.json").open("x") as f:
        json.dump({"vector_calls_reserved": 16384, "automatic_retry": False}, f)
    digest = sha(calibration / "manifest.json")
    if digest != config["calibration_manifest_sha256"]:
        raise ValueError("Calibration manifest commitment changed")
    if read_json(calibration / "manifest.json") != {
        "channel": "fp32_compute",
        "files": config["expected_files"],
    }:
        raise ValueError("Invalid calibration provenance")
    for name, h in config["expected_files"].items():
        if sha(calibration / name) != h:
            raise ValueError("Calibration commitment changed")
    old = read_arrays(
        calibration / "multiscale.npz",
        SHAPES,
        config["expected_files"]["multiscale.npz"],
    )
    spec = importlib.util.spec_from_file_location(
        "owned_public_fixture", Path(__file__).with_name("fp32_fullmap.py")
    )
    assert spec and spec.loader
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    g, u, d = reference.public_weights()
    calls = 0

    def query(x: Array) -> Array:
        nonlocal calls
        calls += len(x)
        with np.errstate(over="ignore"):
            result: Array = reference.precision.forward(g, u, d, x, "fp32_compute")
        return result

    arrays = observe(old["small_x"], query)
    if calls != 16384:
        raise ValueError("Calibration call count mismatch")
    np.savez(
        output / "signed.npz",
        small_x=arrays["small_x"],
        fine_pos=arrays["fine_pos"],
        fine_neg=arrays["fine_neg"],
        coarse_pos=arrays["coarse_pos"],
        coarse_neg=arrays["coarse_neg"],
    )
    meta = {
        "channel": "fp32_compute",
        "rho": 0.03,
        "coarse_rho": 0.06,
        "vector_calls": calls,
        "calibration_manifest_sha256": digest,
        "raw_sha256": sha(output / "signed.npz"),
    }
    (output / "manifest.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n"
    )
    return dict(meta, manifest_sha256=sha(output / "manifest.json"))


if __name__ == "__main__":
    req = json.load(sys.stdin)
    print(
        json.dumps(
            collect(Path(req["calibration"]), Path(req["output"]), req["config"]),
            allow_nan=False,
        )
    )
