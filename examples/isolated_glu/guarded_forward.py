"""Prospective PUBLIC scale-balanced forward coupling polish with guard filter."""

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
METHOD = "isolated-glu-fp32-guarded-forward-polish-v1"
spec = importlib.util.spec_from_file_location(
    "reviewed_forward_factor", Path(__file__).with_name("forward_factor.py")
)
assert spec and spec.loader
factor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factor)


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def cost(residual: Array) -> float:
    value = float(np.sum(residual * residual))
    if not np.isfinite(value):
        raise ValueError("Nonfinite complete training cost")
    return value


def margin(value: float) -> float:
    return max(1e-12 * value, 1e-24)


def group_relative(residual: Array, reference: Array, rows: int) -> list[float]:
    if (
        rows < 1
        or reference.ndim != 2
        or len(reference) != 4 * rows
        or residual.shape != reference.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(residual).all()
    ):
        raise ValueError("Invalid four-group arrays")
    values = []
    for group in range(4):
        block = slice(group * rows, (group + 1) * rows)
        denominator = float(np.linalg.norm(reference[block]))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("Invalid reference group norm")
        value = float(np.linalg.norm(residual[block]) / denominator)
        if not np.isfinite(value):
            raise ValueError("Invalid group relative error")
        values.append(value)
    return values


def prepare_weights(y: Array, rows: int) -> tuple[Array, Array, list[float]]:
    group_relative(y, y, rows)
    norms = [float(np.linalg.norm(y[i * rows : (i + 1) * rows])) for i in range(4)]
    weights: Array = np.repeat(1 / np.asarray(norms), rows)
    weighted: Array = y * weights[:, None]
    return weights, weighted, norms


def polish(
    g: Array,
    c: Array,
    tx: Array,
    ty: Array,
    gx: Array,
    gy: Array,
    rows: int,
    max_units: int,
) -> tuple[Array, dict[str, Any], dict[str, Array]]:
    if (
        g.ndim != 2
        or not 1 <= len(g) <= 64
        or c.ndim != 3
        or len(c) != len(g)
        or c.shape[2] != g.shape[1]
        or ty.shape != (len(tx), c.shape[1])
        or gy.shape != (len(gx), c.shape[1])
        or isinstance(max_units, bool)
        or not isinstance(max_units, int)
        or not 1 <= max_units <= 8
    ):
        raise ValueError("Invalid bounded forward profile")
    w, weighted, norms = prepare_weights(ty, rows)
    group_relative(gy, gy, rows)
    current = c.copy()
    residual = weighted.copy()
    guard_residual = gy.copy()
    ones = np.ones(len(gx))
    for gate, coupling in zip(g, current):
        residual -= factor.operator(gate, coupling, tx, w)
        guard_residual -= factor.operator(gate, coupling, gx, ones)
    initial = residual.copy()
    initial_guard = guard_residual.copy()
    base_cost = cost(residual)
    screening = []
    screen = np.empty_like(c)
    for index, (gate, coupling) in enumerate(zip(g, current)):
        partial = residual + factor.operator(gate, coupling, tx, w)
        proposal, local = factor.rankone_update(gate, partial, tx, w)
        screen[index] = proposal
        screening.append(
            {
                "unit": index,
                "initial_proposed_cost": local["weighted_objective"],
                "initial_decrease": base_cost - local["weighted_objective"],
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
    sequential = np.empty((len(selected), *c.shape[1:]), dtype=np.float64)
    decisions = []
    accepted = []
    for position, index in enumerate(selected):
        before = cost(residual)
        guard_before = group_relative(guard_residual, gy, rows)
        partial = residual + factor.operator(g[index], current[index], tx, w)
        proposal, local = factor.rankone_update(g[index], partial, tx, w)
        sequential[position] = proposal
        next_residual = partial - factor.operator(g[index], proposal, tx, w)
        next_guard = (
            guard_residual
            + factor.operator(g[index], current[index], gx, ones)
            - factor.operator(g[index], proposal, gx, ones)
        )
        after = cost(next_residual)
        guard_after = group_relative(next_guard, gy, rows)
        decreased = after < before - margin(before)
        guard_passed = all(
            new <= old + 1e-10 for new, old in zip(guard_after, guard_before)
        )
        retain = decreased and guard_passed
        decisions.append(
            {
                "unit": index,
                "old_cost": before,
                "proposed_cost": after,
                "required_decrease": margin(before),
                "guard_before": guard_before,
                "guard_after": guard_after,
                "training_decreased": decreased,
                "guard_passed": guard_passed,
                "accepted": retain,
                "gram_condition": local["gram_condition"],
            }
        )
        if retain:
            current[index] = proposal
            residual = next_residual
            guard_residual = next_guard
            accepted.append(index)
    report = {
        "initial_cost": base_cost,
        "final_cost": cost(residual),
        "train_group_norms": norms,
        "initial_guard_relative": group_relative(initial_guard, gy, rows),
        "final_guard_relative": group_relative(guard_residual, gy, rows),
        "screening": screening,
        "selected_units": selected,
        "accepted_units": accepted,
        "decisions": decisions,
        "local_rankone_solves": len(g) + len(selected),
        "sweeps": 1,
        "direction_updates": 0,
        "guard_is_calibration": True,
    }
    arrays = {
        "weights": w,
        "weighted_target": weighted,
        "screen": screen,
        "sequential": sequential,
        "initial_train_residual": initial,
        "final_train_residual": residual,
        "initial_guard_residual": initial_guard,
        "final_guard_residual": guard_residual,
    }
    return current, report, arrays


def build(
    paths: dict[str, Path],
    calibration: Path,
    output: Path,
    config: dict[str, Any],
    manifest_sha256: str,
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
    if set(paths) != {"baseline", "fit_report", "old_manifest"}:
        raise ValueError("Exact baseline input inventory required")
    expected = {
        "baseline": config["baseline_sha256"],
        "fit_report": config["fit_report_sha256"],
        "old_manifest": config["old_manifest_sha256"],
    }
    before = {k: sha(p) for k, p in paths.items()}
    if before != expected or sha(calibration / "manifest.json") != manifest_sha256:
        raise ValueError("Forward input commitment changed")
    meta = read_json(calibration / "manifest.json")
    digest = meta.get("raw_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
    ):
        raise ValueError("Invalid raw calibration SHA256")
    expected_meta = {
        "method": METHOD,
        "channel": "fp32_compute",
        "train_seed": 882030,
        "guard_seed": 882031,
        "scales": [0.01, 1.0, 10.0, 100.0],
        "group_rows": 256,
        "vector_calls": 2048,
        "baseline_sha256": before["baseline"],
        "raw_sha256": digest,
    }
    if meta != expected_meta:
        raise ValueError("Forward calibration provenance mismatch")
    old = read_json(paths["fit_report"])
    old_meta = read_json(paths["old_manifest"])
    if (
        old.get("method") != "isolated-glu-fp32-cubic-richardson-v1"
        or old.get("candidate_built") is not True
        or old.get("candidate_sha256") != before["baseline"]
        or old.get("calibration_manifest_sha256") != before["old_manifest"]
        or old_meta.get("channel") != "fp32_compute"
    ):
        raise ValueError("Baseline provenance mismatch")
    observations = read_arrays(
        calibration / "observations.npz",
        {k: (1024, 32) for k in ["train_x", "train_y", "guard_x", "guard_y"]},
        digest,
    )
    for name, seed in [("train", 882030), ("guard", 882031)]:
        rng = np.random.default_rng(seed)
        x = np.vstack(
            [rng.normal(size=(256, 32)) * scale for scale in [0.01, 1.0, 10.0, 100.0]]
        )
        if not np.array_equal(x, observations[name + "_x"]):
            raise ValueError("Calibration seed/input mismatch")
    initial = read_arrays(
        paths["baseline"],
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        before["baseline"],
    )
    report: dict[str, Any] = {
        "method": METHOD,
        "candidate_built": False,
        "baseline_sha256": before["baseline"],
        "calibration_manifest_sha256": before["old_manifest"],
        "forward_manifest_sha256": manifest_sha256,
        "forward_raw_sha256": digest,
        "new_calibration_calls": 0,
        "additional_calibration_calls_used": 2048,
        "private_gate": "closed",
        "sealed": False,
        "direction_updates": 0,
    }
    try:
        updated, details, arrays = polish(
            initial["gates"],
            initial["couplings"],
            observations["train_x"],
            observations["train_y"],
            observations["guard_x"],
            observations["guard_y"],
            256,
            8,
        )
        report["polish"] = details
        np.savez(
            output / "polish-arrays.npz",
            weights=arrays["weights"],
            weighted_target=arrays["weighted_target"],
            screen=arrays["screen"],
            sequential=arrays["sequential"],
            initial_train_residual=arrays["initial_train_residual"],
            final_train_residual=arrays["final_train_residual"],
            initial_guard_residual=arrays["initial_guard_residual"],
            final_guard_residual=arrays["final_guard_residual"],
        )
        report["arrays_sha256"] = sha(output / "polish-arrays.npz")
        if not details["accepted_units"]:
            raise ValueError("No guarded update retained; no changed candidate")
        singular = np.linalg.svd(updated, compute_uv=False)
        if (
            not np.isfinite(updated).all()
            or np.any(singular[:, 0] <= 0)
            or np.any(singular[:, 1] > 1e-10 * singular[:, 0])
        ):
            raise ValueError("Invalid nonzero finite rankone coupling candidate")
        np.savez(output / "candidate.npz", gates=initial["gates"], couplings=updated)
        (output / "baseline.npz").write_bytes(raw(paths["baseline"]))
        (output / "public").mkdir()
        (output / "public/manifest.json").write_bytes(
            raw(paths["old_manifest"], 2_000_000)
        )
        report.update(
            candidate_built=True, candidate_sha256=sha(output / "candidate.npz")
        )
    except ValueError as exc:
        report["error"] = str(exc)
    after = {k: sha(p) for k, p in paths.items()}
    if (
        before != after
        or sha(calibration / "manifest.json") != manifest_sha256
        or sha(calibration / "observations.npz") != digest
    ):
        raise ValueError("Source observations changed")
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
                Path(req["calibration"]),
                Path(req["output"]),
                req["config"],
                req["manifest_sha256"],
            ),
            allow_nan=False,
        )
    )
