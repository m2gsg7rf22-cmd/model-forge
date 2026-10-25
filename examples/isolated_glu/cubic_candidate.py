"""Derived normalized-odd Richardson RHS; no teacher import in fitting process."""

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
METHOD = "isolated-glu-fp32-cubic-richardson-v1"


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


def required_digest(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Missing or invalid raw calibration SHA256")
    return value


def cubic_rhs(
    fine_pos: Array, fine_neg: Array, coarse_pos: Array, coarse_neg: Array, rho: float
) -> Array:
    if rho != 0.03 or any(
        x.shape != fine_pos.shape or not np.isfinite(x).all()
        for x in (fine_pos, fine_neg, coarse_pos, coarse_neg)
    ):
        raise ValueError("Invalid fixed-rho signed observations")
    fine = (fine_pos - fine_neg) * 0.5 / (0.25 * rho**3)
    coarse = (coarse_pos - coarse_neg) * 0.5 / (0.25 * (2 * rho) ** 3)
    result = (4 * fine - coarse) / 3
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite derived cubic RHS")
    return result


def bridge(rhs: Array) -> Array:
    return rhs * (0.25 * 0.001**3)


def build(
    calibration: Path,
    telemetry: Path,
    output: Path,
    config: dict[str, Any],
    additional: Path,
    additional_manifest_sha256: str,
) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    with (output / "hypothesis-attempt.json").open("x") as f:
        json.dump(
            {
                "multiscale_fits_reserved": 1,
                "direction_optimizer_calls": 0,
                "new_calibration_calls": 0,
                "additional_calibration_calls_used": 16384,
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
    if sha(additional / "manifest.json") != additional_manifest_sha256:
        raise ValueError("Additional manifest commitment changed")
    meta = read_json(additional / "manifest.json")
    if meta != {
        "channel": "fp32_compute",
        "rho": 0.03,
        "coarse_rho": 0.06,
        "vector_calls": 16384,
        "calibration_manifest_sha256": config["calibration_manifest_sha256"],
        "raw_sha256": meta.get("raw_sha256"),
    }:
        raise ValueError("Invalid additional calibration provenance")
    raw_digest = required_digest(meta.get("raw_sha256"))
    signed = read_arrays(
        additional / "signed.npz",
        {
            k: (4096, 32)
            for k in ("small_x", "fine_pos", "fine_neg", "coarse_pos", "coarse_neg")
        },
        raw_digest,
    )
    if not np.array_equal(signed["small_x"], observations["small_x"]):
        raise ValueError("Calibration inputs changed")
    rhs = cubic_rhs(
        signed["fine_pos"],
        signed["fine_neg"],
        signed["coarse_pos"],
        signed["coarse_neg"],
        0.03,
    )
    observations["small_odd"] = bridge(rhs)
    np.savez(
        output / "derived-cubic.npz",
        rhs=rhs,
        bridged_small_odd=observations["small_odd"],
    )
    report: dict[str, Any] = {
        "method": METHOD,
        "additional_manifest_sha256": additional_manifest_sha256,
        "derived_rhs_sha256": sha(output / "derived-cubic.npz"),
        "small_odd_semantics": "derived_rhs_bridge_not_physical_0.001_response",
        "candidate_built": False,
        "residual_acceptance_used": False,
        "plateau_confidence_used": False,
        "calibration_manifest_sha256": config["calibration_manifest_sha256"],
        "telemetry_sha256": config["telemetry_sha256"],
        "sealed": False,
        "direction_optimizer_calls": 0,
        "new_calibration_calls": 0,
        "additional_calibration_calls_used": 16384,
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
                Path(req["additional"]),
                req["additional_manifest_sha256"],
            ),
            allow_nan=False,
        )
    )
