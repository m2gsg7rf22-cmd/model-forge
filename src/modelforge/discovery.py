"""Process-separated, preregistered reconstruction of an owned quadratic oracle.

This is an experimental file protocol, not a security sandbox against a hostile
process running as the same operating-system user.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import psutil

Array = npt.NDArray[np.float64]
METHODS = ("linear_lstsq", "quadratic_lstsq", "quadratic_ridge_0.001")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def features(x: Array, degree: int = 2) -> Array:
    """Canonical monomials, avoiding the nonunique antisymmetric matrix part."""
    columns = [np.ones(x.shape[0]), *[x[:, i] for i in range(x.shape[1])]]
    if degree == 2:
        columns.extend(
            x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1])
        )
    return np.asarray(np.column_stack(columns), dtype=np.float64)


def preregistration(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "public_sampling_seed": seed,
        "family": "degree <=2 polynomial R^4 -> R^2 in canonical monomial basis",
        "dimension": 4,
        "outputs": 2,
        "coefficient_count": 30,
        "train_count": 64,
        "validation_count": 32,
        "sealed_test_count_per_condition": 128,
        "input_distribution": "independent uniform[-1,1] coordinates",
        "conditions": [
            {"name": "real_noiseless", "observation": "real", "noise_std": 0.0},
            {"name": "real_noise_0.0001", "observation": "real", "noise_std": 0.0001},
            {"name": "real_noise_0.1", "observation": "real", "noise_std": 0.1},
            {"name": "sign_only", "observation": "sign", "noise_std": 0.0},
        ],
        "methods": list(METHODS),
        "selection": "minimum validation MSE; fixed method order breaks ties; no test access",
        "validation_exact_retention_threshold_mse": 1e-12,
        "exact_recovery_criteria": {
            "heldout_mse_at_most": 1e-16,
            "max_coefficient_error_at_most": 1e-9,
        },
        "hypotheses": [
            "Noiseless real outputs and full-column-rank monomial design recover canonical polynomial coefficients to tolerance.",
            "Sign-only outputs do not identify coefficient amplitude: positive global scaling preserves all signs.",
            "Gaussian observation noise prevents exact recovery at preregistered tolerance; residual error is measured, not silently rounded away.",
        ],
        "max_total_seconds": 60,
        "max_worker_rss_bytes": 512 * 1024**2,
        "remote_requests": 0,
        "paid_api_cost_usd": 0,
    }


def solve(public: Path) -> dict[str, Any]:
    """Fit only public train/validation observations; no teacher/test arguments."""
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["preregistration"]
    if config != preregistration(config["public_sampling_seed"]):
        raise ValueError("Unrecognized preregistration")
    observation_file = public / "observations.json"
    if _digest(observation_file) != manifest["observations_sha256"]:
        raise ValueError("Observation commitment mismatch")
    observations = json.loads(observation_file.read_text(encoding="utf-8"))
    solutions: dict[str, Any] = {}
    for condition in config["conditions"]:
        name = condition["name"]
        row = observations[name]
        x, y, vx, vy = (
            np.asarray(row[key], dtype=np.float64)
            for key in ("train_x", "train_y", "validation_x", "validation_y")
        )
        candidates = []
        coefficients = []
        for method in METHODS:
            degree = 1 if method == "linear_lstsq" else 2
            a, va = features(x, degree), features(vx, degree)
            if method == "quadratic_ridge_0.001":
                coefficient = np.linalg.solve(
                    a.T @ a + 0.001 * np.eye(a.shape[1]), a.T @ y
                )
            else:
                coefficient = np.linalg.lstsq(a, y, rcond=None)[0]
            coefficients.append(coefficient)
            candidates.append(
                {
                    "method": method,
                    "validation_mse": float(np.mean((va @ coefficient - vy) ** 2)),
                    "design_rank": int(np.linalg.matrix_rank(a)),
                    "design_columns": a.shape[1],
                    "design_condition_number": float(np.linalg.cond(a)),
                }
            )
        selected = min(
            range(len(candidates)), key=lambda i: candidates[i]["validation_mse"]
        )
        chosen = candidates[selected]
        solutions[name] = {
            "selected_method": chosen["method"],
            "candidates": candidates,
            "coefficients": coefficients[selected].tolist(),
            "validation_retained_as_exact_candidate": chosen["validation_mse"]
            <= config["validation_exact_retention_threshold_mse"],
        }
    solution = {
        "protocol_version": 1,
        "observations_sha256": manifest["observations_sha256"],
        "manifest_sha256": _digest(public / "manifest.json"),
        "conditions": solutions,
        "solver_pid": os.getpid(),
        "test_labels_read": False,
        "teacher_parameters_read": False,
    }
    _write(public / "solution.json", solution)
    return {"pid": os.getpid(), "condition_count": len(solutions)}


def _worker(
    stage: str, arguments: list[str], deadline: float, memory_limit: int
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    before = time.perf_counter()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            [sys.executable, "-m", "modelforge.discovery_oracle", stage, *arguments],
            stdout=stdout,
            stderr=stderr,
            env=environment,
        )
        peak = 0
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Discovery total 60 second deadline exceeded")
                try:
                    rss = psutil.Process(process.pid).memory_info().rss
                    peak = max(peak, rss)
                    if rss > memory_limit:
                        raise MemoryError(
                            "Discovery worker exceeded 512 MiB RSS budget"
                        )
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.01)
        except BaseException:
            process.kill()
            process.wait(timeout=5)
            raise
        stderr.seek(0)
        error = stderr.read(65536).decode("utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(f"Discovery {stage} worker failed: {error}")
    return {
        "stage": stage,
        "pid": process.pid,
        "elapsed_seconds": time.perf_counter() - before,
        "sampled_peak_rss_bytes": peak,
        "returncode": process.returncode,
    }


def run_discovery(
    destination: str | Path,
    *,
    seed: int = 17,
    private_storage: str | Path | None = None,
) -> dict[str, Any]:
    """Run bounded CPU prepare → solve → seal → verify, retaining private truth in managed persistent storage.

    Destination must be new. Output contains observations, student, commitments and
    report only; the separate private directory must never enter public exports.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    public = Path(destination).expanduser().absolute()
    if (
        public.exists()
        or public.is_symlink()
        or any(p.is_symlink() for p in public.parents)
    ):
        raise ValueError("Destination must be new with no symlink ancestors")
    public.mkdir(parents=True)
    storage = (
        Path(private_storage).expanduser().absolute()
        if private_storage is not None
        else public.parent / ".discovery-private"
    )
    if storage.is_symlink() or any(p.is_symlink() for p in storage.parents):
        raise ValueError("Private storage must not have symlink ancestors")
    if storage.resolve().is_relative_to(
        public.resolve()
    ) or public.resolve().is_relative_to(storage.resolve()):
        raise ValueError(
            "Private truth must remain outside public experiment artifacts"
        )
    storage.mkdir(parents=True, exist_ok=True, mode=0o700)
    private = Path(tempfile.mkdtemp(prefix="teacher-", dir=storage))
    private.chmod(0o700)
    config = preregistration(seed)
    _write(private / "config.json", config)
    start = time.perf_counter()
    deadline = time.monotonic() + config["max_total_seconds"]
    stages = []
    stages.append(
        _worker(
            "prepare",
            [str(private), str(public)],
            deadline,
            config["max_worker_rss_bytes"],
        )
    )
    stages.append(
        _worker("solve", [str(public)], deadline, config["max_worker_rss_bytes"])
    )
    _write(
        private / "seal.json",
        {
            "solution_sha256": _digest(public / "solution.json"),
            "manifest_sha256": _digest(public / "manifest.json"),
        },
    )
    stages.append(
        _worker(
            "verify",
            [str(private), str(public)],
            deadline,
            config["max_worker_rss_bytes"],
        )
    )
    report: dict[str, Any] = json.loads(
        (public / "verification-core.json").read_text(encoding="utf-8")
    )
    report["verification_core_sha256"] = _digest(public / "verification-core.json")
    report["cost"].update(
        {
            "elapsed_seconds": time.perf_counter() - start,
            "stages": stages,
            "sampled_peak_worker_rss_bytes": max(
                s["sampled_peak_rss_bytes"] for s in stages
            ),
            "measurement": "10ms single-worker RSS sampling; transient peaks may be missed; worker killed on sampled budget excess",
        }
    )
    report["private_truth_retention"] = (
        "Managed persistent private storage retained for authorized audit/replay; deliberately excluded from this report/export. Same OS user can find it."
    )
    _write(public / "report.json", report)
    return report
