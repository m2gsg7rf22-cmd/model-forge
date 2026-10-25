"""One-time paired PUBLIC evaluation; no fitting and no promotion by example."""

from __future__ import annotations
from collections.abc import Callable
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge import glu32
from modelforge.glu32_io import raw, read_arrays, read_json

Array = NDArray[np.float64]
Predict = Callable[[Array], Array]
METHOD = "isolated-glu-fp32-guard-aware-paired-evaluation-v1"
SEED = 941033
SCALES = [0.01, 1.0, 10.0, 100.0]
INPUT_NAMES = {
    "candidate",
    "baseline",
    "fit_controller",
    "fit_worker",
    "fit_step",
    "fit_artifact_manifest",
    "outcome_audit",
}


class EvaluationError(ValueError):
    """Terminal failed evaluation; reserved calls are never retried."""


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def save(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def directory(path: Path) -> None:
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("No symlink output ancestors")
    path.mkdir(exist_ok=False)


def metrics(prediction: Array, reference: Array) -> tuple[float, float]:
    if prediction.shape != reference.shape or any(
        v.dtype != np.float64 or not np.isfinite(v).all()
        for v in [prediction, reference]
    ):
        raise ValueError("Finite paired FP64-storage responses required")
    norm = float(np.linalg.norm(reference))
    delta = prediction - reference
    delta_norm = float(np.linalg.norm(delta))
    if np.any(delta != 0) and delta_norm == 0:
        raise ValueError("Nonzero error norm underflow")
    relative = delta_norm / norm if norm > 0 else float("nan")
    if delta_norm > 0 and relative == 0:
        raise ValueError("Nonzero relative metric underflow")
    maximum = float(np.max(np.abs(delta)))
    if norm <= 0 or not np.isfinite([norm, relative, maximum]).all():
        raise ValueError("Invalid paired metric/reference norm")
    return relative, maximum


def evaluate_batches(
    output: Path,
    query: Predict,
    student: Predict,
    baseline: Predict,
    *,
    seed: int,
    rows: int,
    dimension: int,
) -> dict[str, Any]:
    directory(output)
    report: dict[str, Any] = {
        "status": "started",
        "seed": seed,
        "calls_reserved": 4 * rows,
        "calls_attempted": 0,
        "calls_returned": 0,
        "calls_persisted": 0,
        "groups": [],
        "all_groups_relative_at_most_1e_minus3": None,
        "private_gate": "closed",
        "sealed": False,
        "attempted_means_prequery_intent": True,
    }
    progress = output / "progress.json"
    save(progress, report)
    try:
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or not 1 <= rows <= 512
            or isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or not 1 <= dimension <= 32
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError("Bounded rows/dimension/integer seed required")
        rng = np.random.default_rng(seed)
        xs, ys, ps, bs = [], [], [], []
        for group, scale in enumerate(SCALES):
            x = rng.normal(size=(rows, dimension)) * scale
            snapshot = x.copy()
            x.setflags(write=False)
            prefix = output / ("group-" + str(group))
            np.savez(str(prefix) + "-inputs.npz", inputs=x)
            report.update(stage="query", active_group=group)
            report["calls_attempted"] += rows
            save(progress, report)
            truth = query(x)
            report["calls_returned"] += rows
            # Persist raw response BEFORE predictor/metric code can fail.
            if not isinstance(truth, np.ndarray) or truth.dtype.kind not in "fiu":
                raise ValueError("Numeric array reference required")
            truth = truth.copy()
            np.savez(str(prefix) + "-reference.npz", reference=truth)
            report["calls_persisted"] += rows
            if x.tobytes() != snapshot.tobytes():
                raise ValueError("Query input mutation")
            report["stage"] = "prediction"
            save(progress, report)
            if (
                truth.shape != x.shape
                or truth.dtype != np.float64
                or not np.isfinite(truth).all()
            ):
                raise ValueError("Finite full-vector reference profile required")
            base = baseline(x).copy()
            np.savez(str(prefix) + "-baseline.npz", prediction=base)
            if x.tobytes() != snapshot.tobytes():
                raise ValueError("Baseline input mutation")
            pred = student(x).copy()
            np.savez(str(prefix) + "-student.npz", prediction=pred)
            if x.tobytes() != snapshot.tobytes():
                raise ValueError("Student input mutation")
            rel, absolute = metrics(pred, truth)
            br, ba = metrics(base, truth)
            report["groups"].append(
                {
                    "scale": scale,
                    "relative_frobenius": rel,
                    "max_absolute": absolute,
                    "baseline_relative_frobenius": br,
                    "baseline_max_absolute": ba,
                    "relative_difference": rel - br,
                    "max_absolute_difference": absolute - ba,
                    "relative_regressed": rel > br,
                    "max_absolute_regressed": absolute > ba,
                }
            )
            report["stage"] = "group_completed"
            save(progress, report)
            xs.append(snapshot)
            ys.append(truth)
            ps.append(pred)
            bs.append(base)
        if any(
            report[name] != 4 * rows
            for name in ["calls_attempted", "calls_returned", "calls_persisted"]
        ):
            raise ValueError("Query inventory mismatch")
        np.savez(
            output / "paired-validation.npz",
            inputs=np.vstack(xs),
            reference=np.vstack(ys),
            prediction=np.vstack(ps),
            baseline_prediction=np.vstack(bs),
        )
        report.update(
            status="completed",
            stage="completed",
            validation_sha256=sha(output / "paired-validation.npz"),
            development_max_group_relative=1e-3,
            all_groups_relative_at_most_1e_minus3=all(
                v["relative_frobenius"] <= 1e-3 for v in report["groups"]
            ),
            original_weights_recovered=False,
            all_inputs_equivalence_proven=False,
            universal_reconstruction=False,
        )
        save(progress, report)
        return report
    except Exception as exc:
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            all_groups_relative_at_most_1e_minus3=None,
        )
        save(progress, report)
        raise EvaluationError(str(exc)) from exc


def reserve_once(fit_root: Path, proof: dict[str, Any], *, seed: int) -> Path:
    if any(p.is_symlink() for p in [fit_root, *fit_root.parents]):
        raise ValueError("No symlink fit ancestors")
    marker = fit_root / ("public" + str(seed) + "-evaluation-attempt.json")
    with marker.open("x") as sink:
        sink.write(
            json.dumps(
                dict(proof, evaluation_seed=seed, automatic_retry=False),
                allow_nan=False,
            )
            + "\n"
        )
        sink.flush()
    return marker


def public_query() -> Predict:
    spec = importlib.util.spec_from_file_location(
        "reviewed_public_reference", Path(__file__).with_name("fp32_fullmap.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    g, u, d = module.public_weights()

    def query(x: Array) -> Array:
        with np.errstate(over="ignore"):
            value: Array = module.precision.forward(g, u, d, x, "fp32_compute")
        return value

    return query


def validate_fit(
    paths: dict[str, Path], config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Array], dict[str, Array]]:
    if set(paths) != INPUT_NAMES or set(config.get("inputs", {})) != INPUT_NAMES:
        raise ValueError("Exact paired-evaluation input inventory required")
    before = {name: sha(path) for name, path in paths.items()}
    if before != config["inputs"]:
        raise ValueError("Frozen evaluation input commitments changed")
    controller = read_json(paths["fit_controller"])
    worker = read_json(paths["fit_worker"])
    step = read_json(paths["fit_step"])
    manifest = read_json(paths["fit_artifact_manifest"])
    audit = read_json(paths["outcome_audit"])
    root = paths["fit_controller"].parent
    if (
        paths["candidate"] != root / "worker/step/candidate.npz"
        or paths["fit_step"] != root / "worker/step/report.json"
        or paths["fit_worker"] != root / "worker/worker-report.json"
        or paths["fit_artifact_manifest"] != root / "worker/step/artifact-manifest.json"
    ):
        raise ValueError("Exact immutable fit-root layout required")
    if (
        controller.get("status") != "completed"
        or controller.get("evaluation_eligible") is not True
        or controller.get("source_commit") != config["fit_source_commit"]
        or controller.get("preflight", {}).get("process_monitor_scope")
        != "process_and_children"
        or controller.get("worker_receipt", {}).get("process_monitor_scope")
        != "process_and_children"
        or controller.get("worker_receipt", {}).get("worker_report_sha256")
        != before["fit_worker"]
    ):
        raise ValueError("Successful audited resource/source fit required")
    if (
        worker.get("status") != "completed"
        or worker.get("candidate_built") is not True
        or worker.get("inputs_unchanged") is not True
        or worker.get("candidate_sha256") != before["candidate"]
        or worker.get("step_report_sha256") != before["fit_step"]
        or worker.get("artifact_manifest_sha256") != before["fit_artifact_manifest"]
        or step.get("status") != "completed"
        or step.get("changed") is not True
        or not step.get("accepted")
        or manifest.get("status") != "completed"
    ):
        raise ValueError("Changed terminal successful candidate required")
    if (
        audit.get("approved") is not True
        or audit.get("candidate_sha256") != before["candidate"]
        or audit.get("report_sha256") != before["fit_controller"]
        or audit.get("source_commit") != config["fit_source_commit"]
    ):
        raise ValueError("Frozen independent actual-fit audit required")
    step_root = root / "worker/step"
    files = manifest["files"]
    actual = {
        str(p.relative_to(step_root))
        for p in step_root.rglob("*")
        if p.is_file() and p.name != "artifact-manifest.json"
    }
    # Reservation is at fit_root, outside this immutable step inventory.
    if set(files) != actual or any(
        sha(step_root / name) != digest for name, digest in files.items()
    ):
        raise ValueError("Immutable fit artifact inventory changed")
    shapes = {"gates": (64, 32), "couplings": (64, 32, 32)}
    student = read_arrays(paths["candidate"], shapes, before["candidate"])
    baseline = read_arrays(paths["baseline"], shapes, before["baseline"])
    if student["couplings"].tobytes() != baseline[
        "couplings"
    ].tobytes() or np.array_equal(student["gates"], baseline["gates"]):
        raise ValueError("Fixed-C changed-gate candidate required")
    return before, student, baseline


def build(
    paths: dict[str, Path],
    output: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    if (
        preflight.get("process_monitor_scope") != "process_and_children"
        or preflight.get("memory_samples", 0) < 1
    ):
        raise ValueError("Pre-data tree-scope preflight required")
    directory(output)
    report: dict[str, Any] = {
        "method": METHOD,
        "status": "started",
        "private_gate": "closed",
        "calls_reserved": 0,
        "calls_attempted": 0,
        "calls_returned": 0,
        "calls_persisted": 0,
    }
    path = output / "worker-report.json"
    save(path, report)
    failure: Exception | None = None
    try:
        if set(paths) != INPUT_NAMES or set(config.get("inputs", {})) != INPUT_NAMES:
            raise ValueError("Exact paired input inventory required")
        report["inputs_before"] = {name: sha(p) for name, p in paths.items()}
        save(path, report)
        before, student, baseline = validate_fit(paths, config)
        if before != report["inputs_before"]:
            raise ValueError("Inputs changed during validation")
        report.update(reservation_status="creation_started", calls_reserved=None)
        save(path, report)
        marker = reserve_once(
            paths["fit_controller"].parent,
            {
                "calls_reserved": 2048,
                "candidate_sha256": before["candidate"],
                "baseline_sha256": before["baseline"],
                "config_content_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()
                ).hexdigest(),
            },
            seed=SEED,
        )
        report.update(
            calls_reserved=2048,
            reservation_status="metadata_persisted",
            persistent_marker=str(marker),
            persistent_marker_sha256=sha(marker),
        )
        save(path, report)
        # Factory/target weights and new points are accessed ONLY after reservation.
        query = public_query()
        result = evaluate_batches(
            output / "batches",
            query,
            lambda x: glu32.predict(student["gates"], student["couplings"], x),
            lambda x: glu32.predict(baseline["gates"], baseline["couplings"], x),
            seed=SEED,
            rows=512,
            dimension=32,
        )
        report.update(
            status="completed",
            evaluation=result,
            paired_report_sha256=sha(output / "batches/progress.json"),
            candidate_sha256=before["candidate"],
            baseline_sha256=before["baseline"],
        )
    except Exception as exc:
        failure = exc
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        if (output / "batches/progress.json").exists():
            try:
                partial = read_json(output / "batches/progress.json")
                for name in [
                    "calls_reserved",
                    "calls_attempted",
                    "calls_returned",
                    "calls_persisted",
                ]:
                    report[name] = partial.get(name)
            except Exception as check:
                report.update(
                    status="failed",
                    progress_check_error=str(check),
                    calls_attempted=None,
                    calls_returned=None,
                    calls_persisted=None,
                )
                if failure is None:
                    failure = check
                    report.update(error_type=type(check).__name__, error=str(check))
        if "inputs_before" in report:
            report["inputs_unchanged"] = None
            try:
                after = {name: sha(p) for name, p in paths.items()}
                report.update(
                    inputs_after=after,
                    inputs_unchanged=after == report["inputs_before"],
                )
                if not report["inputs_unchanged"]:
                    raise ValueError("Evaluation input commitments changed")
            except Exception as check:
                report.update(status="failed", input_check_error=str(check))
                if failure is None:
                    failure = check
                    report.update(error_type=type(check).__name__, error=str(check))
        save(path, report)
    if failure is not None:
        raise failure
    return {
        "method": METHOD,
        "status": report["status"],
        "worker_report_sha256": sha(path),
        "calls_reserved": report["calls_reserved"],
        "calls_attempted": report["calls_attempted"],
        "calls_returned": report["calls_returned"],
        "calls_persisted": report["calls_persisted"],
        "criterion_met": report["evaluation"]["all_groups_relative_at_most_1e_minus3"],
        "private_gate": "closed",
    }


if __name__ == "__main__":
    req = json.load(sys.stdin)
    try:
        result = build(
            {k: Path(v) for k, v in req["paths"].items()},
            Path(req["output"]),
            req["config"],
            req["preflight"],
        )
    except Exception as exc:
        result = {"ok": False, "error": str(exc)[:500], "private_gate": "closed"}
    print(json.dumps(result, allow_nan=False))
