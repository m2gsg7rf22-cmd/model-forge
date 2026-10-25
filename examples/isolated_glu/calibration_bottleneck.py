"""Read-only saved calibration diagnostics; no target, optimizer or fitting access."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge.antipodal import canonical_probe_points
from modelforge.glu32_io import raw, read_arrays, read_json, SHAPES

Array = NDArray[np.float64]
NAMES = {
    "multiscale",
    "old_manifest",
    "signed",
    "additional_manifest",
    "derived",
    "candidate",
    "fit_report",
}


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def metrics(prediction: Array, reference: Array) -> dict[str, Any]:
    if (
        prediction.shape != reference.shape
        or prediction.ndim != 2
        or not np.isfinite(prediction).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Invalid metric arrays")
    denominator = float(np.linalg.norm(reference))
    if denominator <= 0:
        raise ValueError("Zero reference norm")
    delta = prediction - reference
    result = {
        "rows": len(reference),
        "outputs": reference.shape[1],
        "reference_frobenius": denominator,
        "prediction_frobenius": float(np.linalg.norm(prediction)),
        "relative_frobenius": float(np.linalg.norm(delta) / denominator),
        "max_absolute": float(np.max(np.abs(delta))),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite metric")
    return result


def components(g: Array, c: Array, x: Array) -> dict[str, Array]:
    shape = (len(x), c.shape[1])
    cubic = np.zeros(shape)
    even = np.zeros(shape)
    odd = np.zeros(shape)
    saturation = np.zeros(shape)
    for gate, coupling in zip(g, c):
        t = x @ gate
        value = x @ coupling.T
        cubic += (t * t)[:, None] * value
        even += 0.5 * t[:, None] * value
        odd += 0.5 * (t * np.tanh(0.5 * t))[:, None] * value
        saturation += 0.5 * np.abs(t)[:, None] * value
    return {"cubic": cubic, "even": even, "odd": odd, "saturation": saturation}


def diagnose(
    paths: dict[str, Path], output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    with (output / "diagnostic-attempt.json").open("x") as f:
        json.dump(
            {
                "queries": 0,
                "fits": 0,
                "optimizers": 0,
                "evaluation_reads": 0,
                "automatic_retry": False,
            },
            f,
        )
    if set(paths) != NAMES or set(config.get("inputs", {})) != NAMES:
        raise ValueError("Exact calibration-only input inventory required")
    before = {k: sha(p) for k, p in paths.items()}
    if before != config["inputs"]:
        raise ValueError("Input commitments changed")
    old_meta = read_json(paths["old_manifest"])
    add_meta = read_json(paths["additional_manifest"])
    fit = read_json(paths["fit_report"])
    if (
        old_meta.get("channel") != "fp32_compute"
        or old_meta["files"]["multiscale.npz"] != before["multiscale"]
        or add_meta.get("channel") != "fp32_compute"
        or add_meta.get("rho") != 0.03
        or add_meta.get("coarse_rho") != 0.06
        or add_meta.get("vector_calls") != 16384
        or add_meta.get("calibration_manifest_sha256") != before["old_manifest"]
        or add_meta.get("raw_sha256") != before["signed"]
    ):
        raise ValueError("Calibration provenance mismatch")
    if (
        fit.get("method") != "isolated-glu-fp32-cubic-richardson-v1"
        or fit.get("candidate_built") is not True
        or fit.get("candidate_sha256") != before["candidate"]
        or fit.get("derived_rhs_sha256") != before["derived"]
        or fit.get("additional_manifest_sha256") != before["additional_manifest"]
        or fit.get("calibration_manifest_sha256") != before["old_manifest"]
        or fit.get("small_odd_semantics")
        != "derived_rhs_bridge_not_physical_0.001_response"
    ):
        raise ValueError("Candidate provenance mismatch")
    obs = read_arrays(paths["multiscale"], SHAPES, before["multiscale"])
    signed = read_arrays(
        paths["signed"],
        {
            k: (4096, 32)
            for k in ["small_x", "fine_pos", "fine_neg", "coarse_pos", "coarse_neg"]
        },
        before["signed"],
    )
    derived = read_arrays(
        paths["derived"],
        {"rhs": (4096, 32), "bridged_small_odd": (4096, 32)},
        before["derived"],
    )
    candidate = read_arrays(
        paths["candidate"],
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        before["candidate"],
    )
    fine = (signed["fine_pos"] - signed["fine_neg"]) * 0.5 / (0.25 * 0.03**3)
    coarse = (signed["coarse_pos"] - signed["coarse_neg"]) * 0.5 / (0.25 * 0.06**3)
    rhs = (4 * fine - coarse) / 3
    if (
        not np.array_equal(signed["small_x"], obs["small_x"])
        or not np.array_equal(rhs, derived["rhs"])
        or not np.array_equal(rhs * (0.25 * 0.001**3), derived["bridged_small_odd"])
    ):
        raise ValueError("Signed/RHS/bridge audit failed")
    g, c = candidate["gates"], candidate["couplings"]
    small = components(g, c, obs["small_x"])
    even = components(g, c, canonical_probe_points(32))["even"]
    large = components(g, c, obs["large_x"] * 1000)
    old_rhs = obs["small_odd"] / (0.25 * 0.001**3)
    even_observed = obs["even_pos"] * 0.5 + obs["even_neg"] * 0.5
    comparisons = {
        "old_rhs_vs_new_rhs": metrics(old_rhs, rhs),
        "candidate_cubic_vs_old_rhs": metrics(small["cubic"], old_rhs),
        "candidate_cubic_vs_new_rhs": metrics(small["cubic"], rhs),
        "candidate_even_vs_saved_even": metrics(even, even_observed),
        "candidate_exact_odd_vs_saved_large_odd": metrics(
            large["odd"], obs["large_odd"]
        ),
        "candidate_saturation_vs_saved_large_odd": metrics(
            large["saturation"], obs["large_odd"]
        ),
        "candidate_saturation_vs_candidate_exact_odd": metrics(
            large["saturation"], large["odd"]
        ),
    }
    np.savez(
        output / "calibration-comparisons.npz",
        old_rhs=old_rhs,
        new_rhs=rhs,
        cubic=small["cubic"],
        even=even,
        even_observed=even_observed,
        large_odd=large["odd"],
        large_saturation=large["saturation"],
        large_observed=obs["large_odd"],
    )
    after = {k: sha(p) for k, p in paths.items()}
    if before != after:
        raise ValueError("Read-only inputs changed")
    report = {
        "id": config["id"],
        "status": "calibration_diagnostic_completed",
        "comparisons": comparisons,
        "inputs_before": before,
        "inputs_after": after,
        "inputs_unchanged": True,
        "arrays_sha256": sha(output / "calibration-comparisons.npz"),
        "reported_unprojected_coupling_tail": fit["multiscale"][
            "coupling_rank1_max_relative_tail_before_projection"
        ],
        "preprojection_coefficients_saved": False,
        "projection_effect_recomputed": False,
        "causal_attribution_proven": False,
        "target_queries": 0,
        "fits": 0,
        "optimizers": 0,
        "evaluation_reads": 0,
        "candidate_quality_remains_failed": True,
        "private_gate": "closed",
        "sealed": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    req = json.load(sys.stdin)
    print(
        json.dumps(
            diagnose(
                {k: Path(v) for k, v in req["paths"].items()},
                Path(req["output"]),
                req["config"],
            ),
            allow_nan=False,
        )
    )
