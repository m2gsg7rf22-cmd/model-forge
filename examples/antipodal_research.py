"""Owned sealed antipodal experiment. Learner receives only public observations.

Three process roles; not a security sandbox against the same OS account.
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any
import numpy as np
from modelforge.antipodal import canonical_probe_points
from modelforge.execution import run_json

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "experiments/preregistrations/antipodal-even-v1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf-8")


def sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / "src/modelforge/antipodal.py",
        ROOT / "src/modelforge/antipodal_io.py",
    ]
    return {p.name: digest(p) for p in paths}


def evaluate(teacher: Any, x: Any) -> Any:
    z = x @ teacher["gate"].T + teacher["bias"]
    return ((z / (1 + np.exp(-z))) * (x @ teacher["up"].T)) @ teacher["down"].T


def prepare(public: Path, private: Path, condition: str) -> dict[str, Any]:
    registration = json.loads(REGISTRATION.read_text())
    if condition not in registration["conditions"]:
        raise ValueError("Unknown preregistered condition")
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False, mode=0o700)
    rng = np.random.default_rng(secrets.randbits(128))
    d, w, o = (
        registration["input_dimension"],
        registration["hidden_width"],
        registration["output_dimension"],
    )
    teacher = {
        "gate": rng.normal(0, 0.2, (w, d)),
        "up": rng.normal(0, 0.2, (w, d)),
        "down": rng.normal(0, 0.2, (o, w)),
        "bias": 0.0 if condition == "bias_free" else 0.7,
    }
    np.savez(private / "teacher.npz", **teacher)
    np.save(
        private / "heldout.npy",
        rng.normal(size=(registration["heldout_rows"], d)),
        allow_pickle=False,
    )
    x = canonical_probe_points(d)
    write(
        public / "observations.json",
        {
            "schema_version": 1,
            "observation_kind": "isolated_glu_antipodal",
            "family": "bias_free_swiglu",
            "authorized": True,
            "dimension": d,
            "positive_outputs": evaluate(teacher, x).tolist(),
            "negative_outputs": evaluate(teacher, -x).tolist(),
        },
    )
    write(public / "preregistration.json", registration)
    commitment = {
        "condition": condition,
        "sources": sources(),
        "observations_sha256": digest(public / "observations.json"),
        "registration_sha256": digest(public / "preregistration.json"),
        "teacher_sha256": digest(private / "teacher.npz"),
        "heldout_sha256": digest(private / "heldout.npy"),
    }
    write(public / "manifest.json", commitment)
    write(
        private / "prepared.json", {"manifest_sha256": digest(public / "manifest.json")}
    )
    return {"prepared": True, "pid": os.getpid()}


def verify(public: Path, private: Path) -> dict[str, Any]:
    with (private / "evaluation-consumed.json").open("x") as f:
        json.dump({"started": True}, f)
    manifest = json.loads((public / "manifest.json").read_text())
    prepared = json.loads((private / "prepared.json").read_text())
    seal = json.loads((private / "seal.json").read_text())
    if (
        prepared["manifest_sha256"] != digest(public / "manifest.json")
        or manifest["sources"] != sources()
    ):
        raise ValueError("Preparation or source commitment changed")
    checks = [
        (public / "observations.json", manifest["observations_sha256"]),
        (public / "preregistration.json", manifest["registration_sha256"]),
        (private / "teacher.npz", manifest["teacher_sha256"]),
        (private / "heldout.npy", manifest["heldout_sha256"]),
        (public / "solution/quadratic-even.json", seal["solution_sha256"]),
    ]
    if any(digest(p) != expected for p, expected in checks):
        raise ValueError("Sealed artifact commitment mismatch")
    registration = json.loads((public / "preregistration.json").read_text())
    if registration != json.loads(REGISTRATION.read_text()):
        raise ValueError("Preregistration changed")
    q = np.asarray(
        json.loads((public / "solution/quadratic-even.json").read_text())[
            "quadratic_even"
        ]
    )
    if q.shape != (32, 32, 32) or not np.isfinite(q).all():
        raise ValueError("Invalid fixed-dimension student")
    # Audit the unique fixed interpolation independently, without learner code.
    obs = json.loads((public / "observations.json").read_text())
    even = (
        np.asarray(obs["positive_outputs"]) + np.asarray(obs["negative_outputs"])
    ) / 2
    for k in range(32):
        if not np.allclose(np.diag(q[k]), even[:32, k], rtol=0, atol=1e-12):
            raise ValueError("Non-whitelisted student diagonal")
    row = 32
    for i in range(32):
        for j in range(i + 1, 32):
            expected = (even[row] - even[i] - even[j]) / 2
            if not np.allclose(
                q[:, i, j], expected, rtol=0, atol=1e-12
            ) or not np.array_equal(q[:, i, j], q[:, j, i]):
                raise ValueError("Non-whitelisted student cross term")
            row += 1
    with np.load(private / "teacher.npz", allow_pickle=False) as teacher:
        x = np.load(private / "heldout.npy", allow_pickle=False)
        full = evaluate(teacher, x)
        target = (full + evaluate(teacher, -x)) / 2
        prediction = np.stack(
            [np.sum((x @ matrix) * x, axis=1) for matrix in q], axis=1
        )
        error = float(np.max(np.abs(prediction - target)))
        analytic = np.einsum(
            "ol,li,lj->oij", teacher["down"], teacher["gate"], teacher["up"]
        )
        analytic = (analytic + analytic.swapaxes(1, 2)) / 4
        coefficient_error = float(np.max(np.abs(q - analytic)))
    accepted = error <= registration["max_abs_even_heldout_error"]
    if manifest["condition"] == "bias_free":
        accepted = (
            accepted
            and coefficient_error
            <= registration["max_abs_quadratic_coefficient_error_bias_free"]
        )
    result = {
        "condition": manifest["condition"],
        "max_abs_even_heldout_error": error,
        "coefficient_error_vs_bias_free_formula": coefficient_error,
        "even_component_criteria_met": accepted,
        "max_abs_error_if_student_mislabeled_full_glu": float(
            np.max(np.abs(prediction - full))
        ),
        "full_glu_recovered": False,
        "original_weights_recovered": False,
        "heldout_rows": 64,
        "calibration_evaluations": 1056,
        "sealed_test_evaluations": 128,
        "manifest_sha256": digest(public / "manifest.json"),
        "solution_sha256": seal["solution_sha256"],
        "source_sha256": sources(),
        "private_model_gate": "closed",
    }
    write(public / "verification.json", result)
    return result


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"prepare", "verify"}:
        payload = json.load(sys.stdin)
        p, t = Path(payload["public"]), Path(payload["private"])
        result = (
            prepare(p, t, payload["condition"])
            if sys.argv[1] == "prepare"
            else verify(p, t)
        )
        print(json.dumps(result))
        return
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result_rows = []
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    for condition in json.loads(REGISTRATION.read_text())["conditions"]:
        public, private = output / "public" / condition, output / "private" / condition
        payload = {
            "public": str(public),
            "private": str(private),
            "condition": condition,
        }

        def budget() -> float:
            remaining = 120 - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Preregistered total research budget exhausted")
            return remaining

        run_json(
            [sys.executable, str(Path(__file__).resolve()), "prepare"],
            payload,
            cwd=ROOT,
            env=env,
            memory_limit_mb=512,
            timeout_seconds=budget(),
        )
        solve = run_json(
            [sys.executable, "-m", "modelforge.runtime_worker"],
            {
                "operation": "antipodal_reconstruct",
                "arguments": {
                    "observations_path": str(public / "observations.json"),
                    "destination": str(public / "solution"),
                },
            },
            cwd=ROOT,
            env=env,
            memory_limit_mb=512,
            timeout_seconds=budget(),
        )
        write(
            private / "seal.json",
            {"solution_sha256": digest(public / "solution/quadratic-even.json")},
        )
        row = run_json(
            [sys.executable, str(Path(__file__).resolve()), "verify"],
            payload,
            cwd=ROOT,
            env=env,
            memory_limit_mb=512,
            timeout_seconds=budget(),
        )
        row["learner_process_monitor_scope"] = solve["process_monitor_scope"]
        result_rows.append(row)
    write(
        output / "public/report.json",
        {
            "protocol": "owned-antipodal-even-v1",
            "preregistration_sha256": digest(REGISTRATION),
            "results": result_rows,
            "elapsed_seconds": time.monotonic() - started,
            "external_calls": 0,
            "scope": "Even component only under isolated paired-query access; misspecification control deliberately violates the bias-free assumption",
        },
    )
    print(
        json.dumps(
            {
                "saved_public": str(output / "public"),
                "seconds": time.monotonic() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
