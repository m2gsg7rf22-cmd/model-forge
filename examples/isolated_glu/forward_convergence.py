"""Distinct bounded fixed-gate convergence study on reused PUBLIC calibration."""

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
METHOD = "isolated-glu-fp32-guarded-forward-convergence-v1"
spec = importlib.util.spec_from_file_location(
    "frozen_guarded_core", Path(__file__).with_name("guarded_forward.py")
)
assert spec and spec.loader
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def guard_ok(new: list[float], current: list[float], initial: list[float]) -> bool:
    if (
        len(new) != 4
        or len(current) != 4
        or len(initial) != 4
        or not np.isfinite(new + current + initial).all()
    ):
        raise ValueError("Invalid four-group guard metrics")
    return all(
        n <= c + 1e-10 and n <= i + 1e-10 for n, c, i in zip(new, current, initial)
    )


def converge(
    g: Array,
    c: Array,
    tx: Array,
    ty: Array,
    gx: Array,
    gy: Array,
    rows: int,
    max_sweeps: int,
) -> tuple[Array, dict[str, Any], dict[str, Array]]:
    if (
        g.ndim != 2
        or not 1 <= len(g) <= 64
        or c.ndim != 3
        or len(c) != len(g)
        or c.shape[2] != g.shape[1]
        or ty.shape != (len(tx), c.shape[1])
        or gy.shape != (len(gx), c.shape[1])
        or isinstance(max_sweeps, bool)
        or not isinstance(max_sweeps, int)
        or not 1 <= max_sweeps <= 4
    ):
        raise ValueError("Invalid convergence dimensions or sweep budget")
    weights, weighted, norms = core.prepare_weights(ty, rows)
    core.group_relative(gy, gy, rows)
    current = c.copy()
    residual = weighted.copy()
    guard_residual = gy.copy()
    ones = np.ones(len(gx))
    for gate, coupling in zip(g, current):
        residual -= core.factor.operator(gate, coupling, tx, weights)
        guard_residual -= core.factor.operator(gate, coupling, gx, ones)
    initial = residual.copy()
    initial_guard = guard_residual.copy()
    initial_guard_relative = core.group_relative(initial_guard, gy, rows)
    arrays = {
        "weights": weights,
        "weighted_target": weighted,
        "initial_train_residual": initial,
        "initial_guard_residual": initial_guard,
    }
    sweeps = []
    total_solves = 0
    accepted_count = 0
    stop_reason = "max_sweeps"
    for sweep in range(max_sweeps):
        base_cost = core.cost(residual)
        screening = []
        screen = np.empty_like(c)
        for index, (gate, coupling) in enumerate(zip(g, current)):
            partial = residual + core.factor.operator(gate, coupling, tx, weights)
            proposal, local = core.factor.rankone_update(gate, partial, tx, weights)
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
            if item["initial_decrease"] > core.margin(base_cost)
        ]
        sequential = np.empty((len(selected), *c.shape[1:]), dtype=np.float64)
        decisions = []
        accepted = []
        for position, index in enumerate(selected):
            before = core.cost(residual)
            guard_before = core.group_relative(guard_residual, gy, rows)
            partial = residual + core.factor.operator(
                g[index], current[index], tx, weights
            )
            proposal, local = core.factor.rankone_update(g[index], partial, tx, weights)
            sequential[position] = proposal
            next_residual = partial - core.factor.operator(
                g[index], proposal, tx, weights
            )
            next_guard = (
                guard_residual
                + core.factor.operator(g[index], current[index], gx, ones)
                - core.factor.operator(g[index], proposal, gx, ones)
            )
            after = core.cost(next_residual)
            guard_after = core.group_relative(next_guard, gy, rows)
            decreased = after < before - core.margin(before)
            passed = guard_ok(guard_after, guard_before, initial_guard_relative)
            retain = decreased and passed
            decisions.append(
                {
                    "unit": index,
                    "old_cost": before,
                    "proposed_cost": after,
                    "required_decrease": core.margin(before),
                    "guard_before": guard_before,
                    "guard_after": guard_after,
                    "training_decreased": decreased,
                    "guard_passed": passed,
                    "accepted": retain,
                    "gram_condition": local["gram_condition"],
                }
            )
            if retain:
                current[index] = proposal
                residual = next_residual
                guard_residual = next_guard
                accepted.append(index)
        count = len(g) + len(selected)
        total_solves += count
        accepted_count += len(accepted)
        arrays[f"screen_{sweep}"] = screen
        arrays[f"sequential_{sweep}"] = sequential
        sweeps.append(
            {
                "sweep": sweep,
                "initial_cost": base_cost,
                "final_cost": core.cost(residual),
                "screening": screening,
                "selected_units": selected,
                "accepted_units": accepted,
                "decisions": decisions,
                "local_rankone_solves": count,
                "final_guard_relative": core.group_relative(guard_residual, gy, rows),
            }
        )
        if not accepted:
            stop_reason = "zero_accepted"
            break
    if total_solves > 512:
        raise ValueError("Local solve budget exceeded")
    arrays["final_train_residual"] = residual
    arrays["final_guard_residual"] = guard_residual
    report = {
        "initial_cost": core.cost(initial),
        "final_cost": core.cost(residual),
        "initial_guard_relative": initial_guard_relative,
        "final_guard_relative": core.group_relative(guard_residual, gy, rows),
        "train_group_norms": norms,
        "sweeps": sweeps,
        "local_rankone_solves": total_solves,
        "accepted_update_count": accepted_count,
        "stop_reason": stop_reason,
        "direction_updates": 0,
        "guard_is_reused_calibration": True,
    }
    return current, report, arrays


def build(
    paths: dict[str, Path], output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    with (output / "convergence-attempt.json").open("x") as f:
        json.dump(
            {
                "automatic_retry": False,
                "max_sweeps": 4,
                "max_local_solves": 512,
                "new_calibration_calls": 0,
            },
            f,
        )
    names = {
        "baseline",
        "fit_report",
        "old_manifest",
        "forward_manifest",
        "observations",
    }
    if set(paths) != names or set(config.get("inputs", {})) != names:
        raise ValueError("Exact convergence input inventory required")
    before = {k: sha(p) for k, p in paths.items()}
    if before != config["inputs"]:
        raise ValueError("Convergence input commitment changed")
    previous = read_json(paths["fit_report"])
    meta = read_json(paths["forward_manifest"])
    old_meta = read_json(paths["old_manifest"])
    if (
        previous.get("method") != "isolated-glu-fp32-guarded-forward-polish-v1"
        or previous.get("candidate_built") is not True
        or previous.get("candidate_sha256") != before["baseline"]
        or previous.get("calibration_manifest_sha256") != before["old_manifest"]
        or previous.get("forward_manifest_sha256") != before["forward_manifest"]
        or previous.get("forward_raw_sha256") != before["observations"]
        or old_meta.get("channel") != "fp32_compute"
    ):
        raise ValueError("Convergence baseline provenance mismatch")
    expected_meta = {
        "method": "isolated-glu-fp32-guarded-forward-polish-v1",
        "channel": "fp32_compute",
        "train_seed": 882030,
        "guard_seed": 882031,
        "scales": [0.01, 1.0, 10.0, 100.0],
        "group_rows": 256,
        "vector_calls": 2048,
        "baseline_sha256": previous["baseline_sha256"],
        "raw_sha256": before["observations"],
    }
    if meta != expected_meta:
        raise ValueError("Reused calibration provenance mismatch")
    data = read_arrays(
        paths["observations"],
        {k: (1024, 32) for k in ["train_x", "train_y", "guard_x", "guard_y"]},
        before["observations"],
    )
    for name, seed in [("train", 882030), ("guard", 882031)]:
        rng = np.random.default_rng(seed)
        expected_x = np.vstack(
            [rng.normal(size=(256, 32)) * scale for scale in [0.01, 1.0, 10.0, 100.0]]
        )
        if not np.array_equal(expected_x, data[name + "_x"]):
            raise ValueError("Reused calibration seed/input mismatch")
    baseline = read_arrays(
        paths["baseline"],
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        before["baseline"],
    )
    report: dict[str, Any] = {
        "method": METHOD,
        "candidate_built": False,
        "baseline_sha256": before["baseline"],
        "calibration_manifest_sha256": before["old_manifest"],
        "forward_manifest_sha256": before["forward_manifest"],
        "forward_raw_sha256": before["observations"],
        "new_calibration_calls": 0,
        "reused_calibration_calls": 2048,
        "private_gate": "closed",
        "sealed": False,
        "direction_updates": 0,
        "starting_experimental_candidate_quality_failed": True,
    }
    try:
        updated, details, arrays = converge(
            baseline["gates"],
            baseline["couplings"],
            data["train_x"],
            data["train_y"],
            data["guard_x"],
            data["guard_y"],
            256,
            4,
        )
        report["convergence"] = details
        np.savez(
            output / "residuals-weights.npz",
            weights=arrays["weights"],
            weighted_target=arrays["weighted_target"],
            initial_train_residual=arrays["initial_train_residual"],
            final_train_residual=arrays["final_train_residual"],
            initial_guard_residual=arrays["initial_guard_residual"],
            final_guard_residual=arrays["final_guard_residual"],
        )
        evidence = {"residuals-weights.npz": sha(output / "residuals-weights.npz")}
        for index in range(len(details["sweeps"])):
            name = f"sweep-{index}.npz"
            np.savez(
                output / name,
                screen=arrays[f"screen_{index}"],
                sequential=arrays[f"sequential_{index}"],
            )
            evidence[name] = sha(output / name)
        report["evidence_files"] = evidence
        if not details["accepted_update_count"]:
            raise ValueError("No convergence update retained; no changed candidate")
        singular = np.linalg.svd(updated, compute_uv=False)
        if (
            not np.isfinite(updated).all()
            or np.any(singular[:, 0] <= 0)
            or np.any(singular[:, 1] > 1e-10 * singular[:, 0])
        ):
            raise ValueError("Invalid nonzero finite rankone candidate")
        np.savez(output / "candidate.npz", gates=baseline["gates"], couplings=updated)
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
    if before != after:
        raise ValueError("Convergence inputs changed")
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
