"""One bounded fixed-C gate-vector step on reused PUBLIC calibration."""

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
METHOD = "isolated-glu-fp32-guarded-gate-step-v1"


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        "frozen_" + name, Path(__file__).with_name(name + ".py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load("guarded_forward")
gradient = load("gate_gradient")


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def direction(g: Array, d: Array, k: Array) -> tuple[Array, dict[str, Any]]:
    if (
        g.ndim != 1
        or d.shape != g.shape
        or k.shape != (len(g), len(g))
        or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in [g, d, k])
        or not np.array_equal(k, k.T)
        or np.linalg.norm(g) <= 0
    ):
        raise ValueError("Invalid finite gate/gradient/symmetric Gram")
    eig = np.linalg.eigvalsh(k)
    if not np.isfinite(eig).all() or eig[-1] <= 0 or eig[0] <= 1e-12 * eig[-1]:
        raise ValueError("Singular/ill-conditioned Gram; no ridge or skipped unit")
    unbounded = -0.5 * np.linalg.solve(k, d)
    length = float(np.linalg.norm(unbounded))
    delta = (
        unbounded.copy()
        if length == 0
        else unbounded * min(1.0, 0.05 * float(np.linalg.norm(g)) / length)
    )
    score = -float(d @ delta + delta @ k @ delta)
    if (
        not np.isfinite(unbounded).all()
        or not np.isfinite(delta).all()
        or not np.isfinite(score)
    ):
        raise ValueError("Nonfinite gate direction")
    return delta, {
        "unclipped": unbounded.tolist(),
        "eigenvalues": eig.tolist(),
        "score": score,
        "zero_direction": length == 0,
    }


def guard_ok(
    nr: list[float],
    na: list[float],
    cr: list[float],
    ca: list[float],
    ir: list[float],
    ia: list[float],
) -> bool:
    if (
        any(len(v) != 4 for v in [nr, na, cr, ca, ir, ia])
        or not np.isfinite(nr + na + cr + ca + ir + ia).all()
    ):
        raise ValueError("Invalid four-group guard metrics")
    return all(
        n <= c + 1e-10 and n <= i + 1e-10 for n, c, i in zip(nr, cr, ir)
    ) and all(
        n <= c + 1e-10 * max(1.0, i) and n <= i + 1e-10 * max(1.0, i)
        for n, c, i in zip(na, ca, ia)
    )


def maxabs(residual: Array, rows: int) -> list[float]:
    return [
        float(np.max(np.abs(residual[i * rows : (i + 1) * rows]))) for i in range(4)
    ]


def activation(t: Array) -> Array:
    return t * (0.5 * (1 + np.tanh(0.5 * t)))


def proposal(
    g: Array, c: Array, x: Array, w: Array, residual: Array
) -> tuple[Array, dict[str, Any]]:
    t = x @ g
    sigmoid = 0.5 * (1 + np.tanh(0.5 * t))
    prime = sigmoid + t * sigmoid * (1 - sigmoid)
    branch = x @ c.T
    d = -2 * ((w * prime * np.sum(residual * branch, axis=1)) @ x)
    k = gradient.gate_gram(g, c, x, w)
    delta, info = direction(g, d, k)
    return delta, dict(
        info,
        gradient=d.tolist(),
        gram=k.tolist(),
        gate=g.tolist(),
        direction=delta.tolist(),
    )


def step(
    g: Array, c: Array, tx: Array, ty: Array, gx: Array, gy: Array, rows: int
) -> tuple[Array, dict[str, Any], dict[str, Array]]:
    gradient.forward(g, c, tx)
    gradient.forward(g, c, gx)
    if ty.shape != (len(tx), c.shape[1]) or gy.shape != (len(gx), c.shape[1]):
        raise ValueError("Invalid physical response shapes")
    singular = np.linalg.svd(c, compute_uv=False)
    if (
        not np.isfinite(singular).all()
        or np.any(singular[:, 0] <= 0)
        or (singular.shape[1] > 1 and np.any(singular[:, 1] > 1e-10 * singular[:, 0]))
    ):
        raise ValueError("Fixed C must be finite nonzero rankone")
    if np.any(np.linalg.norm(g, axis=1) <= 0):
        raise ValueError("Nonzero finite gates required")
    weights, weighted, norms = core.prepare_weights(ty, rows)
    core.group_relative(gy, gy, rows)
    current = g.copy()
    cbytes = c.tobytes()
    pred = gradient.forward(g, c, tx)
    guardpred = gradient.forward(g, c, gx)
    residual = weighted - pred * weights[:, None]
    gres = gy - guardpred
    initial = residual.copy()
    initial_guard = gres.copy()
    ir = core.group_relative(gres, gy, rows)
    ia = maxabs(gres, rows)
    screen = []
    for index in range(len(g)):
        _, info = proposal(current[index], c[index], tx, weights, residual)
        screen.append(dict(info, unit=index))
    ordered = sorted(range(len(g)), key=lambda i: (-screen[i]["score"], i))
    selected = [
        i for i in ordered if screen[i]["score"] > core.margin(core.cost(residual))
    ][:8]
    sequential = []
    trials = []
    trial_gates = []
    accepted = []
    for index in selected:
        delta, info = proposal(current[index], c[index], tx, weights, residual)
        sequential.append(dict(info, unit=index))
        before = core.cost(residual)
        cr = core.group_relative(gres, gy, rows)
        ca = maxabs(gres, rows)
        old = current[index].copy()
        train_branch = tx @ c[index].T
        guard_branch = gx @ c[index].T
        for power in range(8):
            alpha = 2.0**-power
            trial_gate = old + alpha * delta
            trial_gates.append(trial_gate.copy())
            nextpred = (
                pred
                + (activation(tx @ trial_gate) - activation(tx @ old))[:, None]
                * train_branch
            )
            nextguardpred = (
                guardpred
                + (activation(gx @ trial_gate) - activation(gx @ old))[:, None]
                * guard_branch
            )
            nr = weighted - nextpred * weights[:, None]
            ngr = gy - nextguardpred
            after = core.cost(nr)
            ar = core.group_relative(ngr, gy, rows)
            aa = maxabs(ngr, rows)
            decreased = after < before - core.margin(before)
            passed = guard_ok(ar, aa, cr, ca, ir, ia)
            retain = decreased and passed
            trials.append(
                {
                    "unit": index,
                    "alpha": alpha,
                    "old_cost": before,
                    "proposed_cost": after,
                    "required_decrease": core.margin(before),
                    "guard_before_relative": cr,
                    "guard_after_relative": ar,
                    "guard_before_max_absolute": ca,
                    "guard_after_max_absolute": aa,
                    "training_decreased": decreased,
                    "guard_passed": passed,
                    "accepted": retain,
                }
            )
            if retain:
                current[index] = trial_gate
                pred = nextpred
                guardpred = nextguardpred
                residual = nr
                gres = ngr
                accepted.append(index)
                break
    full = gradient.forward(current, c, tx)
    fullguard = gradient.forward(current, c, gx)
    # Delta accumulation is used only for trials; independently recomputed final
    # predictions must agree at a roundoff bound before any candidate is emitted.
    for actual, expected in [(pred, full), (guardpred, fullguard)]:
        if not np.allclose(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12 * max(1.0, float(np.max(np.abs(expected)))),
        ):
            raise ValueError("Cached/full prediction invariant failed")
    if c.tobytes() != cbytes or len(trials) > 64 or len(screen) + len(sequential) > 72:
        raise ValueError("C/budget invariant failed")
    arrays = {
        "weights": weights,
        "weighted_target": weighted,
        "train_x": tx,
        "train_y": ty,
        "guard_x": gx,
        "guard_y": gy,
        "initial_train_residual": initial,
        "initial_guard_residual": initial_guard,
        "final_train_residual": residual,
        "final_guard_residual": gres,
        "initial_train_prediction": gradient.forward(g, c, tx),
        "final_train_prediction": full,
        "initial_guard_prediction": gradient.forward(g, c, gx),
        "final_guard_prediction": fullguard,
        "trial_gates": np.asarray(trial_gates, dtype=np.float64).reshape(
            (-1, g.shape[1])
        ),
    }
    for name, records in [("screen", screen), ("sequential", sequential)]:
        for field in [
            "gradient",
            "gram",
            "eigenvalues",
            "unclipped",
            "direction",
            "gate",
            "score",
        ]:
            shape = (
                (0, g.shape[1], g.shape[1])
                if field == "gram"
                else ((0,) if field == "score" else (0, g.shape[1]))
            )
            arrays[name + "_" + field] = (
                np.asarray([v[field] for v in records], dtype=np.float64)
                if records
                else np.empty(shape)
            )
    report = {
        "initial_cost": core.cost(initial),
        "final_cost": core.cost(residual),
        "initial_guard_relative": ir,
        "initial_guard_max_absolute": ia,
        "final_guard_relative": core.group_relative(gres, gy, rows),
        "final_guard_max_absolute": maxabs(gres, rows),
        "train_group_norms": norms,
        "screening": screen,
        "screen_order": ordered,
        "selected_units": selected,
        "sequential": sequential,
        "trials": trials,
        "accepted_units": accepted,
        "accepted_update_count": len(accepted),
        "local_linear_solves": len(screen) + len(sequential),
        "trial_count": len(trials),
        "couplings_byte_identical": True,
        "final_gate_norms": np.linalg.norm(current, axis=1).tolist(),
        "guard_is_reused_calibration": True,
        "direction_updates": len(accepted),
    }
    return current, report, arrays


def build(
    paths: dict[str, Path], output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    with (output / "gate-step-attempt.json").open("x") as f:
        json.dump(
            {
                "automatic_retry": False,
                "max_trials": 64,
                "max_local_solves": 72,
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
        raise ValueError("Exact gate_step input inventory required")
    before = {k: sha(p) for k, p in paths.items()}
    if before != config["inputs"]:
        raise ValueError("Gate-step input commitment changed")
    previous = read_json(paths["fit_report"])
    meta = read_json(paths["forward_manifest"])
    old_meta = read_json(paths["old_manifest"])
    if (
        previous.get("method") != "isolated-glu-fp32-guarded-forward-convergence-v1"
        or previous.get("candidate_built") is not True
        or previous.get("candidate_sha256") != before["baseline"]
        or previous.get("calibration_manifest_sha256") != before["old_manifest"]
        or previous.get("forward_manifest_sha256") != before["forward_manifest"]
        or previous.get("forward_raw_sha256") != before["observations"]
        or old_meta.get("channel") != "fp32_compute"
    ):
        raise ValueError("Gate-step baseline provenance mismatch")
    expected_meta = {
        "method": "isolated-glu-fp32-guarded-forward-polish-v1",
        "channel": "fp32_compute",
        "train_seed": 882030,
        "guard_seed": 882031,
        "scales": [0.01, 1.0, 10.0, 100.0],
        "group_rows": 256,
        "vector_calls": 2048,
        "baseline_sha256": "b2d5fdd4cf8be40954562c2ce9393cd5b9a3c3cbd3cd8cbfa9bc18728d1de41e",
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
        "couplings_updated": False,
        "starting_experimental_candidate_quality_failed": True,
    }
    try:
        updated, details, arrays = step(
            baseline["gates"],
            baseline["couplings"],
            data["train_x"],
            data["train_y"],
            data["guard_x"],
            data["guard_y"],
            256,
        )
        report["gate_step"] = details
        evidence_arrays: dict[str, Any] = dict(arrays)
        np.savez(output / "gate-step-evidence.npz", **evidence_arrays)
        report["evidence_files"] = {
            "gate-step-evidence.npz": sha(output / "gate-step-evidence.npz")
        }
        if not details["accepted_update_count"]:
            raise ValueError("No gate_step update retained; no changed candidate")
        if not np.isfinite(updated).all() or np.array_equal(updated, baseline["gates"]):
            raise ValueError("No finite changed gate candidate")
        np.savez(
            output / "candidate.npz", gates=updated, couplings=baseline["couplings"]
        )
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
        raise ValueError("Gate-step inputs changed")
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
