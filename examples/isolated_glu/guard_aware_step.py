"""Bounded fixed-C guard-aware gate step; caller supplies physical observations.

No target API, evaluation data, file-input loader or automatic retry lives here.
A successful synthetic run does not authorize the actual research protocol.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge.glu32_io import raw

Array = NDArray[np.float64]
METHOD = "isolated-glu-fp32-guard-aware-gate-step-v1"


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        "step_" + name, Path(__file__).with_name(name + ".py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gradient = load("gate_gradient")
metric = load("guard_metric_normals")
previous = load("guarded_gate_step")
core = previous.core
evidence = load("guard_step_evidence")


class StepError(ValueError):
    """A terminal failed attempt; partial artifacts remain evidence of failure."""


def write_json(path: Path, value: dict[str, Any]) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def cached_trial(
    prediction: Array, x: Array, old: Array, new: Array, c: Array
) -> Array:
    result: Array = prediction + (
        previous.activation(x @ new) - previous.activation(x @ old)
    )[:, None] * (x @ c.T)
    return result


def compare_cache(cached: Array, full: Array) -> None:
    if not np.isfinite(cached).all() or not np.allclose(
        cached, full, rtol=1e-12, atol=1e-12 * max(1.0, float(np.max(np.abs(full))))
    ):
        raise ValueError("Cached/full prediction invariant failed")


class Study:
    def __init__(self, root: Path):
        if any(p.is_symlink() for p in [root, *root.parents]):
            raise ValueError("No symlink output ancestors")
        root.mkdir(exist_ok=False)
        self.root = root
        (root / "proposals").mkdir()
        (root / "trials").mkdir()
        self.report: dict[str, Any] = {
            "method": METHOD,
            "status": "started",
            "stage": "initialization",
            "screen": [],
            "sequential": [],
            "trials": [],
            "accepted": [],
            "ordered": [],
            "selected": [],
            "changed": False,
            "actual_target_queries": 0,
            "new_evaluations": 0,
            "automatic_retry": False,
            "private_gate": "closed",
        }
        self.ordinal = 0
        write_json(root / "report.json", self.report)
        self.store = evidence.TraceStore(root / "cones")

    def save(self) -> None:
        write_json(self.root / "report.json", self.report)

    def event(self, kind: str, **payload: Any) -> None:
        event = dict(
            kind=kind, ordinal=self.ordinal, stage=self.report["stage"], **payload
        )
        with (self.root / "events.jsonl").open("a") as sink:
            sink.write(json.dumps(event, allow_nan=False) + "\n")
            sink.flush()
        self.ordinal += 1
        self.save()

    def inventory(self) -> None:
        files = {}
        paths = sorted(
            p
            for p in self.root.rglob("*")
            if p.is_file() and p.name != "artifact-manifest.json"
        )
        if len(paths) > 1024:
            raise ValueError("Bounded artifact inventory exceeded")
        for path in paths:
            files[str(path.relative_to(self.root))] = hashlib.sha256(
                raw(path)
            ).hexdigest()
        write_json(
            self.root / "artifact-manifest.json",
            {"status": self.report["status"], "files": files, "private_gate": "closed"},
        )

    def normals(
        self,
        pid: str,
        unit: int,
        g: Array,
        c: Array,
        gx: Array,
        gy: Array,
        error: Array,
        rows: int,
        record: dict[str, Any],
    ) -> Array:
        report: dict[str, Any] = {
            "unit": unit,
            "groups": [],
            "records": [],
            "logical_plane_count": 0,
            "matrix_plane_count": 0,
            "error_convention": "physical_candidate_minus_physical_reference",
        }
        record["normals"] = report
        matrices = []
        for group in range(4):
            record["active_guard_group"] = group
            write_json(self.root / "proposals" / (pid + ".json"), record)
            block = slice(group * rows, (group + 1) * rows)
            a, info = metric.metric_group(
                g[unit],
                c[unit],
                gx[block],
                gy[block],
                error[block],
                group,
                group * rows,
            )
            offset = report["matrix_plane_count"]
            for item in info["records"]:
                item = dict(item)
                item["logical_index"] = len(report["records"])
                if item["matrix_row"] is not None:
                    item["matrix_row"] += offset
                report["records"].append(item)
            report["groups"].append({k: v for k, v in info.items() if k != "records"})
            report["logical_plane_count"] = len(report["records"])
            report["matrix_plane_count"] += len(a)
            matrices.append(a)
            write_json(self.root / "proposals" / (pid + ".json"), record)
            if report["logical_plane_count"] > metric.MAX_LOGICAL_PLANES:
                raise ValueError("Full logical plane cap exceeded; no truncation")
        return np.vstack(matrices)

    def proposal(
        self,
        pid: str,
        unit: int,
        g: Array,
        c: Array,
        tx: Array,
        ty: Array,
        gx: Array,
        gy: Array,
        w: Array,
        weighted: Array,
        rows: int,
    ) -> tuple[Array, float]:
        record: dict[str, Any] = {"proposal_id": pid, "unit": unit, "status": "started"}
        self.report["stage"] = "proposal_forward"
        self.event("proposal_started", proposal_id=pid, unit=unit)
        path = self.root / "proposals" / (pid + ".json")
        write_json(path, record)
        # Current FULL predictions, not cached delta accumulation, define d and A.
        train = gradient.forward(g, c, tx)
        guard = gradient.forward(g, c, gx)
        residual = weighted - train * w[:, None]
        error = guard - gy
        np.savez(
            self.root / "proposals" / (pid + ".npz"),
            current_G=g,
            train_prediction=train,
            guard_prediction=guard,
            train_residual=residual,
            guard_error=error,
        )
        self.report["stage"] = "training_derivatives"
        self.save()
        t = tx @ g[unit]
        sigmoid = 0.5 * (1 + np.tanh(0.5 * t))
        prime = sigmoid + t * sigmoid * (1 - sigmoid)
        branch = tx @ c[unit].T
        d = -2 * ((w * prime * np.sum(residual * branch, axis=1)) @ tx)
        np.savez(self.root / "proposals" / (pid + "-gradient.npz"), gradient=d)
        k = gradient.gate_gram(g[unit], c[unit], tx, w)
        if not np.isfinite(d).all():
            raise ValueError("Nonfinite training gradient")
        record.update(gradient=d.tolist(), gram=k.tolist())
        write_json(path, record)
        self.report["stage"] = "guard_normals"
        self.save()
        a = self.normals(pid, unit, g, c, gx, gy, error, rows, record)
        self.report["stage"] = "cone"
        self.save()
        delta, trace = self.store.run_proposal(pid, g[unit], d, k, a)
        score = trace["result"]["score"]
        record.update(
            status="completed",
            direction=delta.tolist(),
            score=score,
            cone_trace=str(Path("../cones/proposals") / pid / "trace.json"),
        )
        write_json(path, record)
        self.event("proposal_completed", proposal_id=pid, score=score)
        return delta, float(score)


def validate(
    g: Array, c: Array, tx: Array, ty: Array, gx: Array, gy: Array, rows: int
) -> None:
    if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= 1024:
        raise ValueError("Bounded four-group row count required")
    gradient.forward(g, c, tx)
    gradient.forward(g, c, gx)
    if (
        len(tx) != 4 * rows
        or len(gx) != 4 * rows
        or ty.shape != (len(tx), c.shape[1])
        or gy.shape != (len(gx), c.shape[1])
        or any(v.dtype != np.float64 or not np.isfinite(v).all() for v in [ty, gy])
    ):
        raise ValueError("Finite complete physical train/guard arrays required")
    singular = np.linalg.svd(c, compute_uv=False)
    if (
        not np.isfinite(singular).all()
        or np.any(singular[:, 0] <= 0)
        or (singular.shape[1] > 1 and np.any(singular[:, 1] > 1e-10 * singular[:, 0]))
        or np.any(np.linalg.norm(g, axis=1) <= 0)
    ):
        raise ValueError("Nonzero rankone C and nonzero gates required")


def run_step(
    root: Path,
    g: Array,
    c: Array,
    tx: Array,
    ty: Array,
    gx: Array,
    gy: Array,
    *,
    rows: int,
) -> tuple[dict[str, Array] | None, dict[str, Any]]:
    study = Study(root)
    s = study.report
    try:
        s["stage"] = "input_validation"
        study.save()
        values = [g, c, tx, ty, gx, gy]
        if any(
            not isinstance(v, np.ndarray)
            or v.dtype != np.float64
            or v.nbytes > 8 * 1024**2
            for v in values
        ):
            raise ValueError("Bounded FP64 raw inputs required")
        np.savez(
            root / "initial-problem.npz",
            G=g,
            C=c,
            train_x=tx,
            train_y=ty,
            guard_x=gx,
            guard_y=gy,
        )
        validate(g, c, tx, ty, gx, gy, rows)
        current = g.copy()
        cbytes = c.tobytes()
        weights, weighted, _ = core.prepare_weights(ty, rows)
        pred = gradient.forward(current, c, tx)
        gp = gradient.forward(current, c, gx)
        residual = weighted - pred * weights[:, None]
        error = gp - gy
        initial_residual = residual.copy()
        initial_error = error.copy()
        np.savez(
            root / "derived-initial.npz",
            weights=weights,
            weighted_target=weighted,
            train_prediction=pred,
            guard_prediction=gp,
            train_residual=residual,
            guard_error=error,
        )
        ir = core.group_relative(error, gy, rows)
        ia = previous.maxabs(error, rows)
        initial_cost = core.cost(residual)
        s.update(
            initial_cost=initial_cost,
            initial_guard_relative=ir,
            initial_guard_max_absolute=ia,
        )
        scores = []
        for index in range(len(g)):
            pid = "screen-" + str(index)
            s["screen"].append(pid)
            _, score = study.proposal(
                pid, index, current, c, tx, ty, gx, gy, weights, weighted, rows
            )
            scores.append(score)
        ordered = sorted(range(len(g)), key=lambda i: (-scores[i], i))
        selected = [i for i in ordered if scores[i] > core.margin(initial_cost)][:8]
        s.update(ordered=ordered, selected=selected, screening_scores=scores)
        study.event("screening_completed")
        trial_gates = []
        for index in selected:
            pid = "sequential-" + str(index)
            s["sequential"].append(pid)
            delta, _ = study.proposal(
                pid, index, current, c, tx, ty, gx, gy, weights, weighted, rows
            )
            # Recompute complete current state for the strict nonlinear checks.
            pred = gradient.forward(current, c, tx)
            gp = gradient.forward(current, c, gx)
            residual = weighted - pred * weights[:, None]
            error = gp - gy
            before = core.cost(residual)
            cr = core.group_relative(error, gy, rows)
            ca = previous.maxabs(error, rows)
            old = current[index].copy()
            for power in range(8):
                s["stage"] = "trial_forward"
                alpha = 2.0**-power
                gate = old + alpha * delta
                trial_gates.append(gate.copy())
                number = len(s["trials"])
                trial: dict[str, Any] = {
                    "index": number,
                    "unit": index,
                    "alpha": alpha,
                    "gate": gate.tolist(),
                    "status": "started",
                }
                s["trials"].append(trial)
                study.event("trial_started", trial_index=number)
                candidate = current.copy()
                candidate[index] = gate
                full = gradient.forward(candidate, c, tx)
                fullguard = gradient.forward(candidate, c, gx)
                cached_train = cached_trial(pred, tx, old, gate, c[index])
                cached_guard = cached_trial(gp, gx, old, gate, c[index])
                np.savez(
                    root / "trials" / (str(number) + "-predictions.npz"),
                    full_train_prediction=full,
                    full_guard_prediction=fullguard,
                    cached_train_prediction=cached_train,
                    cached_guard_prediction=cached_guard,
                )
                compare_cache(cached_train, full)
                compare_cache(cached_guard, fullguard)
                nr = weighted - full * weights[:, None]
                ne = fullguard - gy
                after = core.cost(nr)
                ar = core.group_relative(ne, gy, rows)
                aa = previous.maxabs(ne, rows)
                decreased = after < before - core.margin(before)
                passed = previous.guard_ok(ar, aa, cr, ca, ir, ia)
                retain = decreased and passed
                trial.update(
                    status="completed",
                    old_cost=before,
                    proposed_cost=after,
                    required_decrease=core.margin(before),
                    guard_before_relative=cr,
                    guard_before_max_absolute=ca,
                    guard_after_relative=ar,
                    guard_after_max_absolute=aa,
                    training_decreased=decreased,
                    guard_passed=passed,
                    accepted=retain,
                    full_forward_checked=True,
                )
                np.savez(
                    root / "trials" / (str(number) + ".npz"),
                    G=candidate,
                    train_prediction=full,
                    guard_prediction=fullguard,
                    train_residual=nr,
                    guard_error=ne,
                )
                study.event("trial_completed", trial_index=number, accepted=retain)
                if retain:
                    current = candidate
                    pred, gp, residual, error = full, fullguard, nr, ne
                    s["accepted"].append(index)
                    break
        s["stage"] = "final_forward"
        study.save()
        full = gradient.forward(current, c, tx)
        fullguard = gradient.forward(current, c, gx)
        compare_cache(pred, full)
        compare_cache(gp, fullguard)
        final_residual = weighted - full * weights[:, None]
        final_error = fullguard - gy
        final_cost = core.cost(final_residual)
        fr = core.group_relative(final_error, gy, rows)
        fa = previous.maxabs(final_error, rows)
        changed = not np.array_equal(current, g)
        if (
            c.tobytes() != cbytes
            or len(s["trials"]) > 64
            or len(s["screen"]) + len(s["sequential"]) > 72
            or changed != bool(s["accepted"])
            or not previous.guard_ok(fr, fa, ir, ia, ir, ia)
            or (changed and final_cost >= initial_cost - core.margin(initial_cost))
        ):
            raise ValueError("Final C/budget/change/guard/cost invariant failed")
        arrays = {
            "initial_G": g,
            "final_G": current,
            "fixed_C": c,
            "physical_train_x": tx,
            "physical_train_y": ty,
            "physical_guard_x": gx,
            "physical_guard_y": gy,
            "weights": weights,
            "weighted_target": weighted,
            "initial_train_residual": initial_residual,
            "final_train_residual": final_residual,
            "initial_guard_error": initial_error,
            "final_guard_error": final_error,
            "trial_gates": np.asarray(trial_gates, dtype=np.float64).reshape(
                (-1, g.shape[1])
            ),
        }
        # Cones and whole study use distinct manifests; root status is authoritative.
        study.store.finalize(
            {"trial_gates": arrays["trial_gates"]},
            {"root_report": "../report.json", "scope": "cone_helper_only"},
        )
        evidence.transport.save_evidence(
            root / "evidence",
            arrays,
            {
                "method": METHOD,
                "root_report": "../report.json",
                "scope": "supplied_data_only_not_heldout",
            },
        )
        if changed:
            np.savez(root / "candidate.npz", gates=current, couplings=c)
        s.update(
            status="completed",
            stage="completed",
            final_cost=final_cost,
            final_guard_relative=fr,
            final_guard_max_absolute=fa,
            changed=changed,
        )
        study.event("study_completed")
        study.inventory()
        return ({"G": current, "C": c.copy()} if changed else None), s
    except Exception as exc:
        s.update(
            status="failed",
            changed=False,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        # A failed terminal root report dominates any previously written candidate.
        try:
            study.save()
            study.inventory()
        except Exception as write:
            raise StepError("Failure report unavailable: " + str(write)) from exc
        raise StepError(str(exc)) from exc
