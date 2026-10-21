"""Owned single-head attention identifiability, with observation-only learners.

This known-family experiment recovers the composed score matrix, not the
original Q/K factorization, and makes no closed-LLM reconstruction claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import psutil

Array = npt.NDArray[np.float64]
SEEDS = (11, 23, 47)


def _softmax(score: Array) -> Array:
    shifted = score - score.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True), dtype=np.float64)


def recover_score_matrix(
    inputs: Array, probabilities: Array, *, head_dimension: int
) -> dict[str, Any]:
    """Recover A from log(P_ij/P_i0)=x_i^T A(x_j-x_0)/sqrt(h).

    Inputs/probabilities are legitimately exposed owned-oracle observations.
    No teacher seed, factors, composed matrix or heldout labels are accepted.
    Zero/clipped/underflow probabilities are rejected rather than silently fixed.
    """
    x, p = (
        np.asarray(inputs, dtype=np.float64),
        np.asarray(probabilities, dtype=np.float64),
    )
    if (
        x.ndim != 2
        or not 2 <= len(x) <= 32
        or not 1 <= x.shape[1] <= 8
        or not np.isfinite(x).all()
    ):
        raise ValueError("inputs must be a finite [2..32,1..8] matrix")
    if type(head_dimension) is not int or not 1 <= head_dimension <= 16:
        raise ValueError("head_dimension must be an integer in [1,16]")
    if (
        p.shape != (len(x), len(x))
        or not np.isfinite(p).all()
        or np.any(p <= 1e-12)
        or np.any(p > 1)
        or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError(
            "probabilities must be positive, finite, normalized and above precision floor 1e-12"
        )
    design = np.asarray(
        [np.outer(q, k - x[0]).ravel() for q in x for k in x[1:]], dtype=np.float64
    )
    observed = (np.log(p[:, 1:]) - np.log(p[:, :1])).ravel() * math.sqrt(head_dimension)
    coefficient, _, rank, singular = np.linalg.lstsq(design, observed, rcond=None)
    if rank != x.shape[1] ** 2:
        raise ValueError(
            "Observation design is rank deficient; score matrix is not identifiable"
        )
    return {
        "matrix": coefficient.reshape(x.shape[1], x.shape[1]).tolist(),
        "design_rank": int(rank),
        "parameter_count": x.shape[1] ** 2,
        "linear_equations": len(design),
        "design_condition_number": float(singular[0] / singular[-1]),
        "log_ratio_residual_mse": float(
            np.mean((design @ coefficient - observed) ** 2)
        ),
        "learner_pid": os.getpid(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(_canonical(value))
    os.replace(temporary, path)


def _recover_process(
    x: Array, p: Array, head: int, deadline: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "inputs": x.tolist(),
        "probabilities": p.tolist(),
        "head_dimension": head,
    }
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    with (
        tempfile.TemporaryFile() as stdin,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        stdin.write(_canonical(payload))
        stdin.seek(0)
        process = subprocess.Popen(
            [sys.executable, "-m", "modelforge.attention_research", "--recover"],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=environment,
        )
        peak = 0
        start = time.perf_counter()
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "Attention experiment exceeded total 60 second budget"
                    )
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                    if peak > 512 * 1024**2:
                        raise MemoryError(
                            "Attention learner exceeded sampled 512 MiB RSS budget"
                        )
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.01)
        except BaseException:
            process.kill()
            process.wait(timeout=5)
            raise
        if process.returncode:
            stderr.seek(0)
            raise RuntimeError(stderr.read(65536).decode("utf-8", errors="replace"))
        stdout.seek(0)
        result: dict[str, Any] = json.loads(stdout.read(262144))
        if (
            result["source_sha256"]
            != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise ValueError("Learner/verifier source commitment mismatch")
    return result, {
        "elapsed_seconds": time.perf_counter() - start,
        "sampled_peak_rss_bytes": peak,
        "payload_sha256": _sha(payload),
        "sealed_solution_sha256": _sha(result),
        "learner_payload_fields": sorted(payload),
    }


def _one(
    seed: int, teacher_rng: np.random.Generator, deadline: float, private_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Reserve before generating/writing a teacher so a repeated seed cannot
    # silently replace the private truth behind an already consumed evaluation.
    with (private_root / f"experiment-{seed}.claim").open(
        "x", encoding="utf-8"
    ) as claim:
        json.dump({"sampling_seed": seed, "verifier_pid": os.getpid()}, claim)
    d, head = 3, 3
    rng = np.random.default_rng(seed)
    x = np.vstack([np.zeros((1, d)), np.eye(d), rng.uniform(-0.5, 0.5, (2, d))])
    wq, wk, wv = (teacher_rng.normal(scale=0.3, size=(d, d)) for _ in range(3))
    a = wq @ wk.T
    _atomic(
        private_root / f"teacher-{seed}.json",
        {"wq": wq.tolist(), "wk": wk.tolist(), "wv": wv.tolist()},
    )
    # Teacher-side forward uses projected Q/K, independent of learner log-ratio design.
    score = (x @ wq) @ (x @ wk).T / math.sqrt(head)
    p = _softmax(score)
    solution, accounting = _recover_process(x, p, head, deadline)
    recovered = np.asarray(solution["matrix"], dtype=np.float64)
    if recovered.shape != (d, d) or not np.isfinite(recovered).all():
        raise ValueError("Invalid sealed learner matrix")
    # Claim a one-time evaluation before creating private heldout inputs.
    with (private_root / f"evaluation-{seed}.json").open(
        "x", encoding="utf-8"
    ) as marker:
        json.dump(
            {
                "sealed_solution_sha256": accounting["sealed_solution_sha256"],
                "status": "consumed",
            },
            marker,
        )
    heldout = teacher_rng.uniform(-1, 1, size=(12, d))
    _atomic(
        private_root / f"evaluation-inputs-{seed}.json",
        {"heldout_inputs": heldout.tolist()},
    )
    expected = _softmax((heldout @ wq) @ (heldout @ wk).T / math.sqrt(head))
    actual = _softmax(heldout @ recovered @ heldout.T / math.sqrt(head))
    truth_output, student_output = expected @ (heldout @ wv), actual @ (heldout @ wv)
    coefficient_error = float(np.max(np.abs(a - recovered)))
    probability_error = float(np.max(np.abs(expected - actual)))
    # General invertible gauge, not just hidden-unit permutation.
    r = np.asarray([[2.0, 0.2, 0.0], [0.0, 0.5, 0.1], [0.0, 0.0, 1.5]])
    q2, k2 = wq @ r, wk @ np.linalg.inv(r).T
    p2 = _softmax((heldout @ q2) @ (heldout @ k2).T / math.sqrt(head))
    row_shift = teacher_rng.normal(size=(len(x), 1))
    shifted_p = _softmax(score + row_shift)
    # On a rank-deficient observed subspace, e3 e3^T is invisible.
    deficient = x.copy()
    deficient[:, -1] = 0
    alternative = a + np.outer(np.eye(d)[-1], np.eye(d)[-1])
    deficient_p = _softmax(deficient @ a @ deficient.T / math.sqrt(head))
    alternative_p = _softmax(deficient @ alternative @ deficient.T / math.sqrt(head))
    unseen = np.vstack([np.zeros((1, d)), np.eye(d)])
    unseen_p = _softmax(unseen @ a @ unseen.T / math.sqrt(head))
    unseen_alt = _softmax(unseen @ alternative @ unseen.T / math.sqrt(head))
    # Output-only counterexample: valid but degenerate rank-0 value projection.
    zero_values = np.zeros((len(unseen), d))
    output_only_difference = float(
        np.max(np.abs(unseen_p @ zero_values - unseen_alt @ zero_values))
    )
    noise_rng = np.random.default_rng(seed + 100)
    noisy = np.maximum(p + noise_rng.normal(scale=1e-6, size=p.shape), 1e-10)
    noisy /= noisy.sum(axis=1, keepdims=True)
    noisy_solution, noise_cost = _recover_process(x, noisy, head, deadline)
    noisy_matrix = np.asarray(noisy_solution["matrix"])
    row = {
        "public_sampling_seed": seed,
        "authorized_observations": {
            "inputs": x.tolist(),
            "probabilities": p.tolist(),
            "head_dimension": head,
            "noisy_control_probabilities": noisy.tolist(),
        },
        "student_solution": solution,
        "noisy_student_solution": noisy_solution,
        "full_probability_recovery": {
            "max_composed_matrix_error": coefficient_error,
            "heldout_max_probability_difference": probability_error,
            "heldout_output_mse": float(np.mean((truth_output - student_output) ** 2)),
            "criteria_met": coefficient_error <= 1e-10 and probability_error <= 1e-10,
            "design_rank": solution["design_rank"],
            "design_condition_number": solution["design_condition_number"],
            "linear_equations": solution["linear_equations"],
            "smallest_training_probability": float(p.min()),
        },
        "gauge_symmetry": {
            "transformation": "Wq'=Wq R, Wk'=Wk R^{-T}; R invertible",
            "q_weight_l2_distance": float(np.linalg.norm(wq - q2)),
            "k_weight_l2_distance": float(np.linalg.norm(wk - k2)),
            "max_composed_matrix_difference": float(np.max(np.abs(a - q2 @ k2.T))),
            "max_output_difference": float(
                np.max(np.abs(p2 @ (heldout @ wv) - truth_output))
            ),
            "conclusion": "Even perfect composed-score recovery does not identify original Q/K factors",
        },
        "row_constant_invariance": {
            "max_probability_difference": float(np.max(np.abs(p - shifted_p))),
            "identity": "softmax(S+c*1^T)=softmax(S), independently per row; log ratios eliminate this offset",
        },
        "rank_deficient_control": {
            "input_rank": int(np.linalg.matrix_rank(deficient)),
            "dimension": d,
            "invisible_matrix_perturbation_l2": float(np.linalg.norm(alternative - a)),
            "training_max_probability_difference": float(
                np.max(np.abs(deficient_p - alternative_p))
            ),
            "unseen_max_probability_difference": float(
                np.max(np.abs(unseen_p - unseen_alt))
            ),
            "conclusion": "An unobserved direction prevents identifying A outside the observed subspace",
        },
        "outputs_only_control": {
            "value_projection_rank": 0,
            "same_output_max_difference": output_only_difference,
            "different_probability_max_difference": float(
                np.max(np.abs(unseen_p - unseen_alt))
            ),
            "scope": "Owned rank-0 value-projection counterexample only; not a generic impossibility result for full-rank or known invertible values",
        },
        "probability_noise_control": {
            "additive_std_before_normalization": 1e-6,
            "max_composed_matrix_error": float(np.max(np.abs(noisy_matrix - a))),
            "exact_matrix_tolerance_met": bool(
                np.max(np.abs(noisy_matrix - a)) <= 1e-10
            ),
            "scope": "Perturbed probability observations; not provider logit precision or realistic API access",
        },
        "verification": {
            "verifier_pid": os.getpid(),
            "learner_pid": solution["learner_pid"],
            "sealed_solution_sha256": accounting["sealed_solution_sha256"],
            "heldout_generated_after_solution_sealed": True,
            "learner_observes_teacher_weights": False,
            "learner_observes_heldout": False,
            "independent_external_scientist": False,
        },
        "cost": {
            "clean_learner": accounting,
            "noisy_learner": noise_cost,
            "observed_attention_matrices": 1,
            "noise_control_reuses_owned_probabilities": True,
            "observed_scalar_probabilities": p.size,
            "sealed_heldout_matrix_rows": len(heldout),
        },
    }
    private = {
        "sampling_seed": seed,
        "wq": wq.tolist(),
        "wk": wk.tolist(),
        "wv": wv.tolist(),
        "heldout_inputs": heldout.tolist(),
        "solution": solution,
    }
    return row, private


def run_attention_research(
    destination: str | Path | None = None, *, private_storage: str | Path | None = None
) -> dict[str, Any]:
    """Three preregistered small CPU experiments; optional new public report directory."""
    target = (
        Path(destination).expanduser().absolute() if destination is not None else None
    )
    if target is not None:
        candidate_private = (
            Path(private_storage).expanduser().absolute()
            if private_storage is not None
            else target.parent / ".attention-private" / target.name
        )
        resolved_public, resolved_private = (
            target.resolve(),
            candidate_private.resolve(),
        )
        if (
            resolved_public == resolved_private
            or resolved_public in resolved_private.parents
            or resolved_private in resolved_public.parents
        ):
            raise ValueError("Public and private artifact directories must be disjoint")
        if (
            target.exists()
            or target.is_symlink()
            or any(p.is_symlink() for p in target.parents)
        ):
            raise ValueError("destination must be new without symlink ancestors")
        target.mkdir(parents=True)
    registration = {
        "seeds": list(SEEDS),
        "input_dimension": 3,
        "head_dimension": 3,
        "training_sequence_rows": 6,
        "heldout_sequence_rows": 12,
        "recovery_method": "full-column-rank log-probability-ratio linear least squares",
        "max_matrix_error": 1e-10,
        "max_probability_error": 1e-10,
        "probability_floor_reject": 1e-12,
        "noise_std": 1e-6,
        "method_selection": "none; one fixed solver, no heldout tuning",
        "max_seconds": 60,
        "max_process_rss_bytes": 512 * 1024**2,
    }
    if target is not None:
        _atomic(target / "preregistration.json", registration)
    if private_storage is not None:
        private_root = Path(private_storage).expanduser().absolute()
    elif target is not None:
        private_root = target.parent / ".attention-private" / target.name
    else:
        private_root = Path(
            tempfile.mkdtemp(prefix="modelforge-attention-private-")
        ).resolve()
    if private_storage is not None or target is not None:
        if (
            private_root.exists()
            or private_root.is_symlink()
            or any(p.is_symlink() for p in private_root.parents)
        ):
            raise ValueError(
                "private_storage must be a new directory without symlink ancestors"
            )
        private_root.mkdir(parents=True, mode=0o700)
    private_root.chmod(0o700)
    _atomic(private_root / "preregistration.json", registration)
    start = time.perf_counter()
    deadline = time.monotonic() + 60
    secret_seed = secrets.randbits(128)
    teacher_rng = np.random.default_rng(secret_seed)
    experiment_rows = []
    private_rows = []
    rss = [psutil.Process().memory_info().rss]
    for seed in SEEDS:
        if (
            time.monotonic() > deadline
            or psutil.Process().memory_info().rss > 512 * 1024**2
        ):
            raise RuntimeError(
                "Attention verifier exceeded total time or sampled memory budget"
            )
        row, private = _one(seed, teacher_rng, deadline, private_root)
        experiment_rows.append(row)
        private_rows.append(private)
        rss.append(psutil.Process().memory_info().rss)
    private_fixture = {"secret_teacher_seed": secret_seed, "experiments": private_rows}
    _atomic(private_root / "truth.json", private_fixture)
    numerical = [
        {
            key: value
            for key, value in r.items()
            if key
            not in (
                "cost",
                "verification",
                "student_solution",
                "noisy_student_solution",
            )
        }
        for r in experiment_rows
    ]
    report = {
        "engine": "ModelForge owned attention identifiability",
        "protocol_version": 1,
        "preregistration": registration,
        "preregistration_sha256": _sha(registration),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_truth_sha256": _sha(private_fixture),
        "experiments": experiment_rows,
        "numerical_sha256": _sha(numerical),
        "cost": {
            "elapsed_seconds": time.perf_counter() - start,
            "sampled_peak_learner_rss_bytes": max(
                c["sampled_peak_rss_bytes"]
                for r in experiment_rows
                for c in (r["cost"]["clean_learner"], r["cost"]["noisy_learner"])
            ),
            "verifier_rss_before_after_seed_max_bytes": max(rss),
            "learner_processes": 6,
            "owned_training_probability_matrices": 3,
            "heldout_teacher_probability_matrices": 3,
            "verification_control_probability_matrices": 18,
            "student_heldout_probability_matrices": 3,
            "total_probability_forward_matrices": 27,
            "remote_requests": 0,
            "paid_api_cost_usd": 0,
            "memory_scope": "Learner RSS sampled every10ms with kill threshold; verifier RSS sampled between seeds. Transient peaks may be missed.",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "limitations": [
            "Known small single-head unmasked softmax family; no architecture discovery or decoder/LLM reconstruction",
            "Attention probabilities must be legitimately exposed; ordinary text APIs do not imply this access",
            "Full design rank, nonsaturated positive probabilities and known head dimension are required",
            "Q/K factors remain nonidentifiable under invertible gauge; row score offsets disappear under softmax",
            "Separate learner processes and file/stdin observations are scientific separation, not a security sandbox against the same OS user",
            "Verification software is project-held; separate review is still required for independent acceptance",
            "Private fixtures are retained in separate private_storage (or a hidden sibling directory when destination exists); no public truth paths. No destination defaults to temporary private retention. Public sampling seeds cannot reproduce hidden teachers",
            "Output-only negative control uses degenerate rank-0 value projection and is not a generic output-only impossibility theorem",
            "Numerical float64 tolerances do not establish bitwise identity or rigorous interval-certified equivalence",
        ],
        "references": [
            "https://arxiv.org/abs/1706.03762",
            "https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html",
        ],
    }
    if target is not None:
        _atomic(target / "attention-results.json", report)
    return report


def main() -> None:
    if sys.argv[1:] != ["--recover"]:
        raise SystemExit("Internal attention learner accepts --recover only")
    payload = json.loads(sys.stdin.buffer.read(262144))
    if set(payload) != {"inputs", "probabilities", "head_dimension"}:
        raise ValueError("Only authorized input/probability observations accepted")
    result = recover_score_matrix(
        payload["inputs"],
        payload["probabilities"],
        head_dimension=payload["head_dimension"],
    )
    sys.stdout.buffer.write(_canonical(result))


if __name__ == "__main__":
    main()
