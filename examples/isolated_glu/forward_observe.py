"""New owned PUBLIC FP32 training/guard collector, separate from fitting."""

from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from modelforge.glu32_io import raw

METHOD = "isolated-glu-fp32-guarded-forward-polish-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def collect(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    with (output / "collection-attempt.json").open("x") as f:
        json.dump({"vector_calls_reserved": 2048, "automatic_retry": False}, f)
    if (
        config.get("id") != METHOD
        or config.get("baseline_sha256")
        != "b2d5fdd4cf8be40954562c2ce9393cd5b9a3c3cbd3cd8cbfa9bc18728d1de41e"
    ):
        raise ValueError("Invalid prospective collector identity")
    spec = importlib.util.spec_from_file_location(
        "owned_public_fixture", Path(__file__).with_name("fp32_fullmap.py")
    )
    assert spec and spec.loader
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    g, u, d = ref.public_weights()
    arrays = {}
    calls = 0
    for name, seed in [("train", 882030), ("guard", 882031)]:
        rng = np.random.default_rng(seed)
        x = np.vstack(
            [rng.normal(size=(256, 32)) * scale for scale in [0.01, 1.0, 10.0, 100.0]]
        )
        with np.errstate(over="ignore"):
            y = ref.precision.forward(g, u, d, x, "fp32_compute")
        calls += len(x)
        arrays[name + "_x"] = x
        arrays[name + "_y"] = y
    if calls != 2048:
        raise ValueError("Calibration call count mismatch")
    np.savez(
        output / "observations.npz",
        train_x=arrays["train_x"],
        train_y=arrays["train_y"],
        guard_x=arrays["guard_x"],
        guard_y=arrays["guard_y"],
    )
    meta = {
        "method": METHOD,
        "channel": "fp32_compute",
        "train_seed": 882030,
        "guard_seed": 882031,
        "scales": [0.01, 1.0, 10.0, 100.0],
        "group_rows": 256,
        "vector_calls": calls,
        "baseline_sha256": config["baseline_sha256"],
        "raw_sha256": sha(output / "observations.npz"),
    }
    (output / "manifest.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n"
    )
    return dict(meta, manifest_sha256=sha(output / "manifest.json"))


if __name__ == "__main__":
    req = json.load(sys.stdin)
    print(json.dumps(collect(Path(req["output"]), req["config"]), allow_nan=False))
