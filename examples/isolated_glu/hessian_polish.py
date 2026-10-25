"""Bounded PUBLIC fixed-gate coupling polish on saved FP32 Hessian observations."""

from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge.glu32_io import raw, read_json, read_arrays

Array = NDArray[np.float64]
METHOD = "isolated-glu-fp32-hessian-coupling-polish-v1"
_spec = importlib.util.spec_from_file_location(
    "reviewed_hessian_factor", Path(__file__).with_name("hessian_factor.py")
)
assert _spec and _spec.loader
factor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(factor)


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def cost(residual: Array) -> float:
    value = float(np.sum(residual * residual))
    if not np.isfinite(value):
        raise ValueError("Nonfinite complete residual cost")
    return value


def margin(value: float) -> float:
    return max(1e-12 * value, 1e-24)


def polish(
    g: Array, c: Array, h: Array, x: Array, max_units: int
) -> tuple[Array, dict[str, Any], Array, Array, dict[str, Array]]:
    if (
        g.ndim != 2
        or not 1 <= len(g) <= 64
        or c.ndim != 3
        or c.shape[0] != len(g)
        or c.shape[2] != g.shape[1]
        or h.shape != (len(x), c.shape[1], g.shape[1], g.shape[1])
        or h.dtype != np.float64
        or not np.isfinite(h).all()
        or isinstance(max_units, bool)
        or not isinstance(max_units, int)
        or not 1 <= max_units <= 8
    ):
        raise ValueError("Invalid bounded polish dimensions or budget")
    current = c.copy()
    residual = h.copy()
    for gate, coupling in zip(g, current):
        residual -= factor.operator(gate, coupling, x)
    initial = residual.copy()
    base_cost = cost(initial)
    screening = []
    screen_proposals = np.empty_like(current)
    for index, (gate, coupling) in enumerate(zip(g, current)):
        partial = residual + factor.operator(gate, coupling, x)
        proposal, local = factor.rankone_update(gate, partial, x)
        screen_proposals[index] = proposal
        screening.append(
            {
                "unit": index,
                "initial_proposed_cost": local["objective"],
                "initial_decrease": base_cost - local["objective"],
                "gram_condition": local["gram_condition"],
            }
        )
    ordered = sorted(
        screening, key=lambda item: (-item["initial_decrease"], item["unit"])
    )
    selected = [
        int(item["unit"])
        for item in ordered
        if item["initial_decrease"] > margin(base_cost)
    ][:max_units]
    decisions = []
    accepted = []
    sequential_proposals = np.empty((len(selected), *c.shape[1:]), dtype=np.float64)
    for position, index in enumerate(selected):
        old_cost = cost(residual)
        partial = residual + factor.operator(g[index], current[index], x)
        proposal, local = factor.rankone_update(g[index], partial, x)
        sequential_proposals[position] = proposal
        next_residual = partial - factor.operator(g[index], proposal, x)
        next_cost = cost(next_residual)
        retain = next_cost < old_cost - margin(old_cost)
        decisions.append(
            {
                "unit": index,
                "old_cost": old_cost,
                "proposed_cost": next_cost,
                "required_decrease": margin(old_cost),
                "accepted": retain,
                "gram_condition": local["gram_condition"],
            }
        )
        if retain:
            current[index] = proposal
            residual = next_residual
            accepted.append(index)
    report = {
        "initial_cost": base_cost,
        "final_cost": cost(residual),
        "screening": screening,
        "selected_units": selected,
        "accepted_units": accepted,
        "decisions": decisions,
        "local_rankone_solves": len(g) + len(selected),
        "sweeps": 1,
        "direction_updates": 0,
    }
    return (
        current,
        report,
        initial,
        residual,
        {"screen": screen_proposals, "sequential": sequential_proposals},
    )


def build(
    paths: dict[str, Path], output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    with (output / "polish-attempt.json").open("x") as f:
        json.dump(
            {
                "automatic_retry": False,
                "max_local_solves": 72,
                "new_calibration_calls": 0,
            },
            f,
        )
    names = {"hessians", "calibration_manifest", "candidate", "fit_report"}
    if set(paths) != names or set(config.get("inputs", {})) != names:
        raise ValueError("Exact polish input inventory required")
    before = {k: sha(p) for k, p in paths.items()}
    if before != config["inputs"]:
        raise ValueError("Polish input commitment changed")
    meta = read_json(paths["calibration_manifest"])
    old = read_json(paths["fit_report"])
    if (
        meta.get("channel") != "fp32_compute"
        or meta["files"]["hessians.npz"] != before["hessians"]
        or old.get("candidate_built") is not True
        or old.get("method") != "isolated-glu-fp32-cubic-richardson-v1"
        or old.get("candidate_sha256") != before["candidate"]
        or old.get("calibration_manifest_sha256") != before["calibration_manifest"]
    ):
        raise ValueError("Polish provenance mismatch")
    h = read_arrays(
        paths["hessians"], {"hessian": (64, 32, 32, 32)}, before["hessians"]
    )["hessian"]
    initial = read_arrays(
        paths["candidate"],
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        before["candidate"],
    )
    # Exact deterministic base-probe convention of source-frozen fp32_fullmap.prepare.
    x = np.random.default_rng(20261024).normal(size=(64, 32)) * 4
    report: dict[str, Any] = {
        "method": METHOD,
        "candidate_built": False,
        "private_gate": "closed",
        "sealed": False,
        "new_calibration_calls": 0,
        "direction_updates": 0,
        "calibration_manifest_sha256": before["calibration_manifest"],
        "initial_candidate_sha256": before["candidate"],
        "original_candidate_quality_failed": True,
    }
    try:
        updated, details, initial_residual, final_residual, proposals = polish(
            initial["gates"], initial["couplings"], h, x, 8
        )
        report["polish"] = details
        np.savez(
            output / "coupling-proposals.npz",
            screen=proposals["screen"],
            sequential=proposals["sequential"],
        )
        report["proposals_sha256"] = sha(output / "coupling-proposals.npz")
        np.savez(
            output / "hessian-residuals.npz",
            initial=initial_residual,
            final=final_residual,
        )
        report["residuals_sha256"] = sha(output / "hessian-residuals.npz")
        if not details["accepted_units"]:
            raise ValueError("No retained coupling update; no changed candidate")
        singular = np.linalg.svd(updated, compute_uv=False)
        if (
            not np.isfinite(updated).all()
            or np.any(singular[:, 0] <= 0)
            or np.any(singular[:, 1] > 1e-10 * singular[:, 0])
        ):
            raise ValueError("Invalid nonzero rankone coupling candidate")
        np.savez(output / "candidate.npz", gates=initial["gates"], couplings=updated)
        (output / "public").mkdir()
        (output / "public/manifest.json").write_bytes(
            raw(paths["calibration_manifest"], 2_000_000)
        )
        report.update(
            candidate_built=True, candidate_sha256=sha(output / "candidate.npz")
        )
    except ValueError as exc:
        report["error"] = str(exc)
    after = {k: sha(p) for k, p in paths.items()}
    if after != before:
        raise ValueError("Original inputs changed")
    report.update(inputs_before=before, inputs_after=after, inputs_unchanged=True)
    (output / "fit-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    req = json.load(sys.stdin)
    print(
        json.dumps(
            build(
                {k: Path(v) for k, v in req["paths"].items()},
                Path(req["output"]),
                req["config"],
            ),
            allow_nan=False,
        )
    )
