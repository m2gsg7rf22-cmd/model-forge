"""Frozen isolated-GLU replication roles; logical separation, not an OS sandbox."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, NoReturn
import zipfile

import numpy as np
from numpy.typing import NDArray
from modelforge.antipodal import canonical_probe_points
from modelforge.execution import run_json

Array = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "experiments/preregistrations/isolated-glu-fullmap-v1.json"
REG_SHA = "cf9921199370a5c204e2f27fb04ee2ee9f1f98ef331c3609c538e4633ee5e3a4"
READY = ROOT / "experiments/preregistrations/isolated-glu-runner-readiness.json"
AUDIT_FILES = [
    "solve.json",
    "refine.json",
    "multiscale-solve.json",
    "refined-solution.npz",
    "solution.npz",
]


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p: Path, value: Any) -> None:
    data = json.dumps(value, allow_nan=False, indent=2)
    with p.open("x", encoding="utf-8") as f:
        f.write(data + "\n")


def _bad_constant(value: str) -> NoReturn:
    raise ValueError("Nonfinite JSON constant: " + value)


def read(p: Path) -> dict[str, Any]:
    if p.is_symlink() or p.stat().st_size > 2_000_000:
        raise ValueError("Invalid JSON artifact")
    result: Any = json.loads(p.read_bytes(), parse_constant=_bad_constant)
    # Also rejects finite-looking literals such as 1e999, including nested fields.
    json.dumps(result, allow_nan=False)
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object")
    return result


def registration() -> dict[str, Any]:
    if digest(REG) != REG_SHA:
        raise ValueError("Preregistration changed")
    return read(REG)


def sources() -> dict[str, str]:
    paths = list(registration()["solver_source_sha256"]) + [
        "examples/isolated_glu/sealed.py",
        "src/modelforge/execution.py",
        "src/modelforge/antipodal.py",
    ]
    result = {p: digest(ROOT / p) for p in paths}
    for p, expected in registration()["solver_source_sha256"].items():
        if result[p] != expected:
            raise ValueError("Frozen solver source changed")
    return result


def require_readiness(path: Path) -> None:
    if not path.is_file():
        raise ValueError("Runner readiness certificate is missing")
    ready = read(path)
    if (
        ready.get("ready") is not True
        or ready.get("sources") != sources()
        or ready.get("registration_sha256") != REG_SHA
    ):
        raise ValueError("Runner readiness does not match frozen sources")
    review = ROOT / ready["review_path"]
    if (
        not review.resolve().is_relative_to(ROOT)
        or digest(review) != ready["review_sha256"]
    ):
        raise ValueError("Runner readiness review commitment mismatch")
    if read(review).get("status") != "approved_for_fresh_teacher_generation":
        raise ValueError("Runner readiness review not approved")
    for p in [
        *sources(),
        path.relative_to(ROOT).as_posix(),
        review.relative_to(ROOT).as_posix(),
    ]:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:" + p], cwd=ROOT, timeout=10
        )
        if hashlib.sha256(committed).hexdigest() != digest(ROOT / p):
            raise ValueError(
                "Runner readiness requires committed unchanged source/review"
            )


def arrays(path: Path, shapes: dict[str, tuple[int, ...]]) -> dict[str, Array]:
    if path.is_symlink() or path.stat().st_size > 128 * 1024**2:
        raise ValueError("Invalid array artifact")
    with zipfile.ZipFile(path) as z:
        if sum(x.file_size for x in z.infolist()) > 128 * 1024**2:
            raise ValueError("Array allocation budget exceeded")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(shapes):
            raise ValueError("Unexpected array fields")
        result = {}
        for key, shape in shapes.items():
            a = z[key]
            if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
                raise ValueError("Invalid finite float64 array: " + key)
            result[key] = a
    return result


def forward(t: dict[str, Array], x: Array) -> Array:
    z = x @ t["gate"].T
    return np.asarray(
        (z * 0.5 * (1 + np.tanh(z / 2)) * (x @ t["up"].T)) @ t["down"].T,
        dtype=np.float64,
    )


def student(g: Array, c: Array, x: Array) -> Array:
    result = np.zeros((len(x), 32), dtype=np.float64)
    for unit in range(64):
        z = x @ g[unit]
        result += (z * 0.5 * (1 + np.tanh(z / 2)))[:, None] * (x @ c[unit].T)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite student output")
    return result


def relative(error: Array, target: Array) -> float:
    if (
        error.shape != target.shape
        or not np.isfinite(error).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Invalid metric arrays")
    denom = float(np.linalg.norm(target))
    if not math.isfinite(denom) or denom <= 0:
        raise ValueError("Undefined relative metric denominator")
    result = float(np.linalg.norm(error) / denom)
    if not math.isfinite(result):
        raise ValueError("Nonfinite relative metric")
    return result


def materialize(
    public: Path, private: Path, index: int, teacher: dict[str, Array], heldout: Array
) -> dict[str, Any]:
    """Prepare supplied arrays; tests use only the previously public development fixture."""
    reg = registration()
    if type(index) is not int or index not in range(3):
        raise ValueError("Unknown teacher index")
    if (
        set(teacher) != {"gate", "up", "down"}
        or any(a.shape != (64, 32) for k, a in teacher.items() if k != "down")
        or teacher["down"].shape != (32, 64)
    ):
        raise ValueError("Invalid teacher shape")
    if (
        heldout.shape != (4, 512, 32)
        or not np.isfinite(heldout).all()
        or any(not np.isfinite(a).all() for a in teacher.values())
    ):
        raise ValueError("Invalid teacher or heldout")
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False, mode=0o700)
    np.savez(
        private / "teacher.npz",
        gate=teacher["gate"],
        up=teacher["up"],
        down=teacher["down"],
    )
    np.savez(private / "heldout.npz", inputs=heldout)
    queries = 0

    def query(x: Array) -> Array:
        nonlocal queries
        queries += len(x)
        return forward(teacher, x)

    rng = np.random.default_rng(20261024)
    probes = rng.normal(size=(64, 32)) * 4
    i, j = np.triu_indices(32)
    ei = np.eye(32)[i] * 0.003
    ej = np.eye(32)[j] * 0.003
    h = np.zeros((64, 32, 32, 32))
    for k, x in enumerate(probes):
        value = (
            query(x + ei + ej)
            - query(x + ei - ej)
            - query(x - ei + ej)
            + query(x - ei - ej)
        ) / (4 * 0.003**2)
        h[k, :, i, j] = value
        h[k, :, j, i] = value
    np.savez(public / "hessians.npz", hessian=h)
    rng = np.random.default_rng(931082)
    small = rng.normal(size=(4096, 32))
    large = rng.normal(size=(512, 32))
    even = canonical_probe_points(32)
    np.savez(
        public / "multiscale-observations.npz",
        small_x=small,
        small_odd=0.5 * (query(small * 0.001) - query(-small * 0.001)),
        large_x=large,
        large_odd=0.5 * (query(large * 1000) - query(-large * 1000)),
        even_pos=query(even),
        even_neg=query(-even),
    )
    if queries != 145440:
        raise ValueError("Calibration query count mismatch")
    write(public / "preregistration.json", reg)
    manifest = {
        "index": index,
        "weight_scale": reg["teacher_weight_scales"][index],
        "source_sha256": sources(),
        "registration_sha256": REG_SHA,
        "calibration_vector_queries": queries,
        "public_sha256": {
            name: digest(public / name)
            for name in ["hessians.npz", "multiscale-observations.npz"]
        },
        "private_sha256": {
            name: digest(private / name) for name in ["teacher.npz", "heldout.npz"]
        },
    }
    write(public / "manifest.json", manifest)
    write(
        private / "prepared.json", {"manifest_sha256": digest(public / "manifest.json")}
    )
    return {"prepared": True, "index": index, "calibration_vector_queries": queries}


def prepare(
    public: Path, private: Path, index: int, *, readiness: Path = READY
) -> dict[str, Any]:
    require_readiness(readiness)  # Before entropy, mkdir or teacher creation.
    reg = registration()
    if type(index) is not int or index not in range(3):
        raise ValueError("Unknown teacher index")
    rng = np.random.default_rng(secrets.randbits(128))
    scale = reg["teacher_weight_scales"][index]
    teacher = {
        "gate": rng.normal(size=(64, 32)) * scale,
        "up": rng.normal(size=(64, 32)) * scale,
        "down": rng.normal(size=(32, 64)) * scale,
    }
    held_rng = np.random.default_rng(secrets.randbits(128))
    held = np.stack(
        [
            held_rng.normal(size=(512, 32)) * s
            for s in reg["sealed_evaluation"]["scales"]
        ]
    )
    return materialize(public, private, index, teacher, held)


def environment() -> dict[str, str]:
    env = {
        k: os.environ[k]
        for k in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG")
        if k in os.environ
    }
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        PYTHONUTF8="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    return env


def public_inputs(public: Path) -> dict[str, Any]:
    manifest = read(public / "manifest.json")
    if (
        manifest["source_sha256"] != sources()
        or manifest["registration_sha256"] != REG_SHA
        or manifest["calibration_vector_queries"] != 145440
        or read(public / "preregistration.json") != registration()
    ):
        raise ValueError("Public preparation commitment changed")
    if set(manifest["public_sha256"]) != {
        "hessians.npz",
        "multiscale-observations.npz",
    }:
        raise ValueError("Unexpected observation commitment")
    if any(
        digest(public / name) != expected
        for name, expected in manifest["public_sha256"].items()
    ):
        raise ValueError("Public observations changed")
    arrays(public / "hessians.npz", {"hessian": (64, 32, 32, 32)})
    arrays(
        public / "multiscale-observations.npz",
        {
            "small_x": (4096, 32),
            "small_odd": (4096, 32),
            "large_x": (512, 32),
            "large_odd": (512, 32),
            "even_pos": (528, 32),
            "even_neg": (528, 32),
        },
    )
    return manifest


def fit_audit(public: Path) -> None:
    first = read(public / "solve.json")
    refine = read(public / "refine.json")
    multi = read(public / "multiscale-solve.json")
    if (
        first["subspace_rank"] != 128
        or first["singular_128"] <= 0
        or first["singular_129"] < 0
        or first["singular_128"] < 100 * first["singular_129"]
    ):
        raise ValueError("Insufficient Hessian rank/gap")
    directions = arrays(
        public / "refined-solution.npz",
        {"directions": (64, 32), "all_directions": (256, 32), "residual": (256,)},
    )["directions"]
    h = arrays(public / "hessians.npz", {"hessian": (64, 32, 32, 32)})["hessian"]
    ii, jj = np.triu_indices(32)
    weights = np.where(ii == jj, 1.0, np.sqrt(2.0))
    _, singular, vt = np.linalg.svd(
        h[:, :, ii, jj].reshape(-1, len(ii)) * weights, full_matrices=False
    )
    if singular[127] <= 1e-10 * singular[0] or singular[127] < 100 * singular[128]:
        raise ValueError("Actual Hessian rank/gap is inadequate")
    q = directions[:, ii] * directions[:, jj] * weights
    residual = q - (q @ vt[:128].T) @ vt[:128]
    if np.max(np.sum(residual**2, axis=1)) > 1e-8:
        raise ValueError("Retained candidate residual exceeds threshold")
    if refine["directions_retained"] != 64 or not np.allclose(
        np.linalg.norm(directions, axis=1), 1, atol=1e-10, rtol=0
    ):
        raise ValueError("Invalid candidate count/residual")
    cos = np.abs(directions @ directions.T)
    np.fill_diagonal(cos, 0)
    if cos.max() >= 0.9995:
        raise ValueError("Duplicate candidate directions")
    if (
        multi["small_design_rank"] != 2048
        or multi["small_design_columns"] != 2048
        or multi["magnitude_design_rank"] != 64
        or multi["orientation_rank"] != 64
        or multi["magnitude_min"] <= 0
    ):
        raise ValueError("Invalid multiscale rank or magnitude")
    arrays(
        public / "multiscale-solution.npz",
        {"gates": (64, 32), "couplings": (64, 32, 32)},
    )


def solve(public: Path) -> dict[str, Any]:
    write(public / "learner-attempt.json", {"started": True})
    started = time.monotonic()
    public_inputs(public)
    stages = []
    for name, args in [
        ("hessian_pilot.py", ["solve", str(public)]),
        ("refine.py", [str(public)]),
        ("multiscale.py", ["solve", str(public)]),
    ]:
        remaining = 180 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Learner budget exhausted")
        result = run_json(
            [sys.executable, str(ROOT / "examples/isolated_glu" / name), *args],
            {},
            cwd=public / "monitor",
            env=environment(),
            memory_limit_mb=512,
            timeout_seconds=remaining,
        )
        if result["process_monitor_scope"] != "process_and_children":
            raise ValueError("Incomplete process-tree memory monitoring")
        stages.append(
            {
                "script": name,
                **{
                    k: result[k]
                    for k in [
                        "process_monitor_scope",
                        "peak_sampled_rss_bytes",
                        "memory_samples",
                        "memory_limit_mb",
                    ]
                },
            }
        )
    fit_audit(public)
    report = {
        "ok": True,
        "seconds": time.monotonic() - started,
        "stages": stages,
        "source_sha256": sources(),
        "manifest_sha256": digest(public / "manifest.json"),
        "solution_sha256": digest(public / "multiscale-solution.npz"),
        "rss_limit_mib": 512,
        "peak_rss_recorded": True,
        "heldout_read": False,
        "audit_sha256": {n: digest(public / n) for n in AUDIT_FILES},
    }
    write(public / "learner-run.json", report)
    return report


def seal(public: Path, private: Path) -> None:
    fit_audit(public)
    report = read(public / "learner-run.json")
    if (
        report.get("ok") is not True
        or report["solution_sha256"] != digest(public / "multiscale-solution.npz")
        or report["manifest_sha256"] != digest(public / "manifest.json")
    ):
        raise ValueError("Invalid learner completion")
    if report.get("source_sha256") != sources() or report.get("audit_sha256") != {
        n: digest(public / n) for n in AUDIT_FILES
    }:
        raise ValueError("Learner audit/source commitment changed")
    stages = report.get("stages", [])
    if (
        [x.get("script") for x in stages]
        != ["hessian_pilot.py", "refine.py", "multiscale.py"]
        or not all(
            x.get("process_monitor_scope") == "process_and_children"
            and x.get("memory_samples", 0) > 0
            and 0 < x.get("peak_sampled_rss_bytes", 0) <= 512 * 1024**2
            for x in stages
        )
        or not 0 <= report.get("seconds", float("inf")) <= 180
    ):
        raise ValueError("Incomplete bounded learner stages")
    write(
        private / "seal.json",
        {
            "solution_sha256": digest(public / "multiscale-solution.npz"),
            "learner_sha256": digest(public / "learner-run.json"),
            "manifest_sha256": digest(public / "manifest.json"),
            "audit_sha256": {n: digest(public / n) for n in AUDIT_FILES},
        },
    )


def verify(public: Path, private: Path) -> dict[str, Any]:
    write(
        private / "evaluation-consumed.json", {"started": True}
    )  # First operation, before any read.
    manifest = public_inputs(public)
    prepared = read(private / "prepared.json")
    fixed = read(private / "seal.json")
    if (
        prepared["manifest_sha256"] != digest(public / "manifest.json")
        or fixed["manifest_sha256"] != digest(public / "manifest.json")
        or fixed["solution_sha256"] != digest(public / "multiscale-solution.npz")
        or fixed["learner_sha256"] != digest(public / "learner-run.json")
    ):
        raise ValueError("Preparation or solution seal changed")
    if set(manifest["private_sha256"]) != {"teacher.npz", "heldout.npz"} or any(
        digest(private / name) != expected
        for name, expected in manifest["private_sha256"].items()
    ):
        raise ValueError("Private commitment changed")
    if set(fixed["audit_sha256"]) != set(AUDIT_FILES) or any(
        digest(public / n) != v for n, v in fixed["audit_sha256"].items()
    ):
        raise ValueError("Audit commitment changed")
    fit_audit(public)
    teacher = arrays(
        private / "teacher.npz", {"gate": (64, 32), "up": (64, 32), "down": (32, 64)}
    )
    held = arrays(private / "heldout.npz", {"inputs": (4, 512, 32)})["inputs"]
    sol = arrays(
        public / "multiscale-solution.npz",
        {"gates": (64, 32), "couplings": (64, 32, 32)},
    )
    g, c = sol["gates"], sol["couplings"]
    true_g = teacher["gate"]
    true_c = np.einsum("ow,wd->wod", teacher["down"], teacher["up"])
    norms = np.linalg.norm(g, axis=1)
    tnorms = np.linalg.norm(true_g, axis=1)
    if np.any(norms <= 0) or np.any(tnorms <= 0):
        raise ValueError("Zero gate")
    cos = (true_g / tnorms[:, None]) @ (g / norms[:, None]).T
    work = np.abs(cos).copy()
    gate_errors = []
    coupling_errors = []
    matched_cos = []
    orientations = 0
    for _ in range(64):
        i, j = np.unravel_index(np.argmax(work), work.shape)
        matched_cos.append(float(work[i, j]))
        orientations += int(cos[i, j] > 0)
        gate_errors.append(relative(g[j] - true_g[i], true_g[i]))
        coupling_errors.append(relative(c[j] - true_c[i], true_c[i]))
        work[i, :] = -1
        work[:, j] = -1
    singular = np.linalg.svd(c, compute_uv=False)
    if np.any(singular[:, 0] <= 0) or np.any(singular[:, 1] > 1e-10 * singular[:, 0]):
        raise ValueError("Student coupling not rank one")
    validation = []
    for scale, x in zip(
        registration()["sealed_evaluation"]["scales"], held, strict=True
    ):
        target = forward(teacher, x)
        error = student(g, c, x) - target
        row_denom = np.maximum(np.linalg.norm(target, axis=1), 1e-30)
        row_rel = np.linalg.norm(error, axis=1) / row_denom
        validation.append(
            {
                "scale": scale,
                "rows": 512,
                "relative_frobenius_error": relative(error, target),
                "max_absolute_error": float(np.max(np.abs(error))),
                "row_relative_percentiles": np.percentile(
                    row_rel, [50, 95, 99, 100]
                ).tolist(),
                "row_denominator_floor": 1e-30,
            }
        )
    passed = (
        all(v["relative_frobenius_error"] <= 0.001 for v in validation)
        and min(matched_cos) >= 0.9995
        and max(gate_errors) <= 0.01
        and max(coupling_errors) <= 0.01
        and orientations == 64
    )
    result = {
        "index": manifest["index"],
        "weight_scale": manifest["weight_scale"],
        "criteria_met": passed,
        "validation": validation,
        "one_to_one_min_absolute_cosine": min(matched_cos),
        "max_relative_oriented_gate_error": max(gate_errors),
        "max_relative_coupling_error": max(coupling_errors),
        "orientations_correct": orientations,
        "calibration_vector_queries": 145440,
        "sealed_evaluation_vector_queries": 2048,
        "manifest_sha256": digest(public / "manifest.json"),
        "solution_sha256": fixed["solution_sha256"],
        "source_sha256": sources(),
        "private_model_gate": "closed",
        "whole_transformer_reconstructed": False,
        "original_weights_recovered": False,
        "all_inputs_equivalence_proven": False,
    }
    write(public / "verification.json", result)
    return result


def criteria(row: dict[str, Any]) -> bool:
    def number(value: Any) -> float:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid scientific metric")
        return float(value)

    try:
        scales = registration()["sealed_evaluation"]["scales"]
        v = row["validation"]
        if (
            len(v) != 4
            or [x["scale"] for x in v] != scales
            or any(type(x["rows"]) is not int or x["rows"] != 512 for x in v)
        ):
            return False
        return (
            all(number(x["relative_frobenius_error"]) <= 0.001 for x in v)
            and 0.9995 <= number(row["one_to_one_min_absolute_cosine"]) <= 1 + 1e-12
            and number(row["max_relative_oriented_gate_error"]) <= 0.01
            and number(row["max_relative_coupling_error"]) <= 0.01
            and type(row["orientations_correct"]) is int
            and row["orientations_correct"] == 64
            and row["calibration_vector_queries"] == 145440
            and row["sealed_evaluation_vector_queries"] == 2048
        )
    except (KeyError, TypeError, ValueError):
        return False


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(rows) != 3
        or any(type(r.get("index")) is not int for r in rows)
        or sorted(r["index"] for r in rows) != [0, 1, 2]
    ):
        raise ValueError("Need exactly three distinct teacher results")
    accepted = all(
        r.get("criteria_met") is True
        and r.get("source_sha256") == sources()
        and r.get("weight_scale") == registration()["teacher_weight_scales"][r["index"]]
        and all(
            isinstance(r.get(k), str)
            and len(r[k]) == 64
            and all(c in "0123456789abcdef" for c in r[k])
            for k in ["manifest_sha256", "solution_sha256"]
        )
        and r.get("private_model_gate") == "closed"
        and criteria(r)
        and r.get("process_monitor_scope") == "process_and_children"
        and r.get("memory_samples", 0) > 0
        and 0 < r.get("peak_sampled_rss_bytes", 0) <= 512 * 1024**2
        and 0 <= r.get("elapsed_seconds", float("inf")) <= 180
        for r in rows
    )
    return {
        "protocol": registration()["protocol"],
        "results": rows,
        "criteria_met": accepted,
        "source_sha256": sources(),
        "registration_sha256": REG_SHA,
        "external_calls": 0,
        "private_model_gate": "closed",
        "whole_transformer_reconstructed": False,
        "original_weights_recovered": False,
    }


def run_teacher(public: Path, private: Path, index: int) -> dict[str, Any]:
    started = time.monotonic()
    role_evidence = []

    def call(role: str, payload: dict[str, Any]) -> dict[str, Any]:
        remaining = 180 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Teacher budget exhausted")
        result = run_json(
            [sys.executable, str(Path(__file__).resolve()), role],
            payload,
            cwd=public.parent / "monitor",
            env=environment(),
            memory_limit_mb=512,
            timeout_seconds=remaining,
        )
        if result["process_monitor_scope"] != "process_and_children":
            raise ValueError("Incomplete process-tree monitoring")
        role_evidence.append(
            {
                "role": role,
                **{
                    k: result[k]
                    for k in [
                        "process_monitor_scope",
                        "peak_sampled_rss_bytes",
                        "memory_samples",
                        "memory_limit_mb",
                    ]
                },
            }
        )
        return result

    try:
        call(
            "prepare", {"public": str(public), "private": str(private), "index": index}
        )
        call("solve", {"public": str(public)})
        call("seal", {"public": str(public), "private": str(private)})
        row = call("verify", {"public": str(public), "private": str(private)})
        row.update(
            elapsed_seconds=time.monotonic() - started,
            rss_limit_mib=512,
            peak_rss_recorded=True,
            roles=role_evidence,
        )
    except (ValueError, RuntimeError, TimeoutError, MemoryError, OSError) as exc:
        row = {
            "index": index,
            "criteria_met": False,
            "failure_type": type(exc).__name__,
            "failure": str(exc)[:1000],
            "elapsed_seconds": time.monotonic() - started,
            "roles_completed": role_evidence,
        }
    return row


def run(output: Path) -> dict[str, Any]:
    require_readiness(READY)
    output.mkdir(parents=True, exist_ok=False)
    (output / "public").mkdir()
    rows = []
    for index in range(3):
        started = time.monotonic()
        try:
            row = run_json(
                [sys.executable, str(Path(__file__).resolve()), "teacher"],
                {
                    "public": str(output / "public" / str(index)),
                    "private": str(output / "private" / str(index)),
                    "index": index,
                },
                cwd=output / "monitor",
                env=environment(),
                memory_limit_mb=512,
                timeout_seconds=180,
            )
            row["elapsed_seconds"] = time.monotonic() - started
            row["memory_scope"] = (
                "teacher worker and all descendants, including preparation, learning, sealing and verification; top-level launcher excluded"
            )
        except (ValueError, RuntimeError, TimeoutError, MemoryError, OSError) as exc:
            row = {
                "index": index,
                "criteria_met": False,
                "failure_type": type(exc).__name__,
                "failure": str(exc)[:1000],
                "elapsed_seconds": time.monotonic() - started,
            }
        rows.append(row)
        write(output / f"teacher-{index}-outcome.json", row)
    report = aggregate(rows)
    write(output / "public/report.json", report)
    return report


def main() -> None:
    if sys.argv[1] in {"prepare", "solve", "verify", "seal", "teacher"}:
        payload = json.load(sys.stdin)
        public = Path(payload["public"])
        if sys.argv[1] == "solve":
            if set(payload) != {"public"}:
                raise ValueError("Learner payload must contain public only")
            result = solve(public)
        elif sys.argv[1] == "prepare":
            result = prepare(public, Path(payload["private"]), payload["index"])
        elif sys.argv[1] == "seal":
            seal(public, Path(payload["private"]))
            result = {"sealed": True}
        elif sys.argv[1] == "teacher":
            result = run_teacher(public, Path(payload["private"]), payload["index"])
        else:
            result = verify(public, Path(payload["private"]))
    elif sys.argv[1] == "run":
        result = run(Path(sys.argv[2]).resolve())
    else:
        raise ValueError("Unknown research role")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
