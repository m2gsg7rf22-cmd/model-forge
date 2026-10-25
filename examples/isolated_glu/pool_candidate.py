"""NEW public all-pool hypothesis generation; original FP32 failure unchanged."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge import glu32
from modelforge.glu32_io import raw, read_json, read_arrays, SHAPES

Array = NDArray[np.float64]
METHOD = "isolated-glu-fp32-all-pool-candidate-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def pool_indices(v: Array, losses: Array, width: int) -> list[int]:
    if (
        v.ndim != 2
        or losses.shape != (len(v),)
        or not np.isfinite(v).all()
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
        or not np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Invalid finite unit pool/residuals")
    selected: list[int] = []
    for k in np.argsort(losses):
        if all(abs(v[k] @ v[j]) < 0.9995 for j in selected):
            selected.append(int(k))
    if len(selected) != width:
        raise ValueError(
            "All-pool hypothesis count differs from known width; no truncation"
        )
    return selected


def build(
    calibration: Path, telemetry: Path, output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    with (output / "hypothesis-attempt.json").open("x") as f:
        json.dump(
            {
                "multiscale_fits_reserved": 1,
                "direction_optimizer_calls": 0,
                "new_calibration_calls": 0,
            },
            f,
        )
    if sha(calibration / "manifest.json") != config["calibration_manifest_sha256"]:
        raise ValueError("Calibration manifest commitment changed")
    meta = read_json(calibration / "manifest.json")
    if meta != {"channel": "fp32_compute", "files": config["expected_files"]}:
        raise ValueError("Invalid calibration channel/inventory")
    for name, h in config["expected_files"].items():
        if sha(calibration / name) != h:
            raise ValueError("Calibration commitment changed")
    data = read_arrays(
        telemetry,
        {
            "known": (52, 32),
            "v": (256, 32),
            "residual": (256,),
            "initial_loss": (256,),
            "b": (128, 528),
        },
        config["telemetry_sha256"],
    )
    observations = read_arrays(
        calibration / "multiscale.npz",
        SHAPES,
        config["expected_files"]["multiscale.npz"],
    )
    report: dict[str, Any] = {
        "method": METHOD,
        "candidate_built": False,
        "residual_acceptance_used": False,
        "plateau_confidence_used": False,
        "calibration_manifest_sha256": config["calibration_manifest_sha256"],
        "telemetry_sha256": config["telemetry_sha256"],
        "sealed": False,
        "direction_optimizer_calls": 0,
        "new_calibration_calls": 0,
        "private_gate": "closed",
    }
    stage = "all_pool_hypothesis"
    try:
        indices = pool_indices(data["v"], data["residual"], 64)
        report.update(
            selected_indices=indices,
            hypothesis_count=len(indices),
            max_selected_loss=float(np.max(data["residual"][indices])),
        )
        stage = "multiscale"
        gates, couplings, diagnostics = glu32.multiscale(
            data["v"][indices], observations
        )
        singular = np.linalg.svd(couplings, compute_uv=False)
        if (
            gates.shape != (64, 32)
            or couplings.shape != (64, 32, 32)
            or not np.isfinite(gates).all()
            or not np.isfinite(couplings).all()
            or np.any(np.linalg.norm(gates, axis=1) <= 0)
            or np.any(singular[:, 0] <= 0)
            or np.any(singular[:, 1] > 1e-10 * singular[:, 0])
        ):
            raise ValueError("Invalid finite nonzero rankone candidate")
        np.savez(output / "candidate.npz", gates=gates, couplings=couplings)
        (output / "public").mkdir()
        (output / "public/manifest.json").write_bytes(
            raw(calibration / "manifest.json", 2_000_000)
        )
        report.update(
            candidate_built=True,
            stage="candidate_built",
            multiscale=diagnostics,
            candidate_sha256=sha(output / "candidate.npz"),
        )
    except ValueError as exc:
        report.update(stage=stage, error=str(exc))
    (output / "fit-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    req = json.load(sys.stdin)
    print(
        json.dumps(
            build(
                Path(req["calibration"]),
                Path(req["telemetry"]),
                Path(req["output"]),
                req["config"],
            ),
            allow_nan=False,
        )
    )
