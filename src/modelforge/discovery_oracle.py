"""Internal whitelisted subprocess stages for the Discovery Lab file protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
from typing import Any

import numpy as np

from .discovery import _digest, _write, preregistration, solve


def oracle_polynomial(
    x: np.ndarray[Any, Any], coefficient: np.ndarray[Any, Any], degree: int = 2
) -> np.ndarray[Any, Any]:
    """Independent direct evaluator, deliberately not the learner feature builder."""
    y = np.tile(coefficient[0], (len(x), 1))
    cursor = 1
    for i in range(x.shape[1]):
        y += x[:, i, None] * coefficient[cursor]
        cursor += 1
    if degree == 2:
        for i in range(x.shape[1]):
            for j in range(i, x.shape[1]):
                y += (x[:, i] * x[:, j])[:, None] * coefficient[cursor]
                cursor += 1
    return y


def prepare(private: Path, public: Path) -> None:
    config = json.loads((private / "config.json").read_text(encoding="utf-8"))
    if config != preregistration(config["public_sampling_seed"]):
        raise ValueError("Unrecognized configuration")
    secret_seed = secrets.randbits(128)
    teacher_rng = np.random.default_rng(secret_seed)
    teachers: dict[str, Any] = {}
    sampler = np.random.default_rng(config["public_sampling_seed"])
    observations: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    for condition in config["conditions"]:
        coefficient = teacher_rng.normal(size=(15, 2))
        teachers[condition["name"]] = coefficient.tolist()
        train = sampler.uniform(-1, 1, size=(config["train_count"], 4))
        validation = sampler.uniform(-1, 1, size=(config["validation_count"], 4))
        # Sealed inputs and labels use private randomness, not the public seed.
        heldout = teacher_rng.uniform(
            -1, 1, size=(config["sealed_test_count_per_condition"], 4)
        )
        outputs = []
        for x in (train, validation):
            y = oracle_polynomial(x, coefficient)
            if condition["observation"] == "sign":
                y = np.where(y >= 0, 1.0, -1.0)
            elif condition["noise_std"]:
                y = y + teacher_rng.normal(scale=condition["noise_std"], size=y.shape)
            outputs.append(y.tolist())
        observations[condition["name"]] = {
            "train_x": train.tolist(),
            "train_y": outputs[0],
            "validation_x": validation.tolist(),
            "validation_y": outputs[1],
        }
        tests[condition["name"]] = {
            "x": heldout.tolist(),
            "y": oracle_polynomial(heldout, coefficient).tolist(),
        }
    _write(
        private / "truth.json",
        {
            "private_teacher_seed": secret_seed,
            "teachers": teachers,
            "tests": tests,
        },
    )
    _write(public / "observations.json", observations)
    sources = {
        p.name: _digest(p)
        for p in (Path(__file__), Path(__file__).with_name("discovery.py"))
    }
    _write(
        public / "manifest.json",
        {
            "preregistration": config,
            "observations_sha256": _digest(public / "observations.json"),
            "private_truth_sha256": _digest(private / "truth.json"),
            "source_sha256": sources,
            "producer_pid": os.getpid(),
        },
    )


def verify(private: Path, public: Path) -> dict[str, Any]:
    # Claim the evaluation before reading labels: failed attempts cannot be retuned.
    with (private / "evaluation-consumed.json").open("x", encoding="utf-8") as marker:
        json.dump({"verifier_pid": os.getpid(), "status": "started"}, marker)
    manifest_path = public / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads((private / "seal.json").read_text(encoding="utf-8"))
    if (
        _digest(public / "solution.json") != seal["solution_sha256"]
        or _digest(manifest_path) != seal["manifest_sha256"]
    ):
        raise ValueError("Sealed artifact or manifest tampering")
    if _digest(private / "truth.json") != manifest["private_truth_sha256"]:
        raise ValueError("Private truth commitment mismatch")
    if _digest(public / "observations.json") != manifest["observations_sha256"]:
        raise ValueError("Observation commitment mismatch")
    current_sources = {
        p.name: _digest(p)
        for p in (Path(__file__), Path(__file__).with_name("discovery.py"))
    }
    if current_sources != manifest["source_sha256"]:
        raise ValueError("Source changed after preregistration")
    solution = json.loads((public / "solution.json").read_text(encoding="utf-8"))
    if (
        solution["manifest_sha256"] != _digest(manifest_path)
        or solution["observations_sha256"] != manifest["observations_sha256"]
    ):
        raise ValueError("Solution provenance mismatch")
    truth = json.loads((private / "truth.json").read_text(encoding="utf-8"))
    config = manifest["preregistration"]
    if config != preregistration(config["public_sampling_seed"]):
        raise ValueError("Preregistration changed")
    observations = json.loads(
        (public / "observations.json").read_text(encoding="utf-8")
    )
    results = []
    for condition in config["conditions"]:
        name = condition["name"]
        teacher = np.asarray(truth["teachers"][name], dtype=np.float64)
        fitted = solution["conditions"][name]
        if fitted["selected_method"] not in config["methods"]:
            raise ValueError("Method is not whitelisted")
        # Rebuild candidate fits from public data: validation scores/selection
        # are independently checked, not trusted solver self-reporting.
        observed = observations[name]
        train_x = np.asarray(observed["train_x"], dtype=np.float64)
        train_y = np.asarray(observed["train_y"], dtype=np.float64)
        validation_x = np.asarray(observed["validation_x"], dtype=np.float64)
        validation_y = np.asarray(observed["validation_y"], dtype=np.float64)
        audited_candidates: list[dict[str, Any]] = []
        audited_coefficients = []
        for method in config["methods"]:
            candidate_degree = 1 if method == "linear_lstsq" else 2
            columns = 5 if candidate_degree == 1 else 15
            # Independent direct evaluator applied to identity coefficients
            # builds a design without calling the learner feature function.
            design = oracle_polynomial(train_x, np.eye(columns), candidate_degree)
            if method == "quadratic_ridge_0.001":
                candidate = np.linalg.solve(
                    design.T @ design + 0.001 * np.eye(columns), design.T @ train_y
                )
            else:
                candidate = np.linalg.lstsq(design, train_y, rcond=None)[0]
            audited_coefficients.append(candidate)
            score = float(
                np.mean(
                    (
                        oracle_polynomial(validation_x, candidate, candidate_degree)
                        - validation_y
                    )
                    ** 2
                )
            )
            audited_candidates.append(
                {
                    "method": method,
                    "validation_mse": score,
                    "design_rank": int(np.linalg.matrix_rank(design)),
                    "design_columns": columns,
                    "design_condition_number": float(np.linalg.cond(design)),
                }
            )
        chosen = min(
            range(len(audited_candidates)),
            key=lambda i: audited_candidates[i]["validation_mse"],
        )
        claimed_candidates = fitted["candidates"]
        if len(claimed_candidates) != len(audited_candidates):
            raise ValueError("Invalid validation candidate count")
        for claimed, audited in zip(claimed_candidates, audited_candidates):
            if any(
                claimed[key] != audited[key]
                for key in ("method", "design_rank", "design_columns")
            ):
                raise ValueError("Validation candidate metadata mismatch")
            for key in ("validation_mse", "design_condition_number"):
                if not np.isclose(claimed[key], audited[key], rtol=1e-8, atol=1e-20):
                    raise ValueError("Validation candidate score mismatch")
        retained = (
            audited_candidates[chosen]["validation_mse"]
            <= config["validation_exact_retention_threshold_mse"]
        )
        if (
            fitted["selected_method"] != audited_candidates[chosen]["method"]
            or fitted["validation_retained_as_exact_candidate"] != retained
        ):
            raise ValueError("Preregistered validation selection mismatch")
        coefficients = np.asarray(fitted["coefficients"], dtype=np.float64)
        degree = 1 if fitted["selected_method"] == "linear_lstsq" else 2
        expected_shape = (5 if degree == 1 else 15, 2)
        if coefficients.shape != expected_shape or not np.isfinite(coefficients).all():
            raise ValueError("Invalid student coefficient shape/value")
        if not np.allclose(
            coefficients, audited_coefficients[chosen], rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Selected coefficients differ from whitelisted fit")
        heldout = truth["tests"][name]
        x, expected = np.asarray(heldout["x"]), np.asarray(heldout["y"])
        prediction = oracle_polynomial(x, coefficients, degree)
        errors = np.mean((prediction - expected) ** 2, axis=1)
        extended = np.zeros_like(teacher)
        extended[: len(coefficients)] = coefficients
        parameter_error = float(np.max(np.abs(extended - teacher)))
        mse = float(np.mean(errors))
        rng = np.random.default_rng(910)
        boot = np.mean(
            errors[rng.integers(0, len(errors), size=(300, len(errors)))], axis=1
        )
        exact = (
            mse <= config["exact_recovery_criteria"]["heldout_mse_at_most"]
            and parameter_error
            <= config["exact_recovery_criteria"]["max_coefficient_error_at_most"]
        )
        row = {
            "condition": name,
            "observation": condition["observation"],
            "noise_std": condition["noise_std"],
            "selected_method": fitted["selected_method"],
            "candidates": audited_candidates,
            "validation_retained_as_exact_candidate": retained,
            "heldout_mse": mse,
            "heldout_max_abs_error": float(np.max(np.abs(prediction - expected))),
            "max_canonical_coefficient_error": parameter_error,
            "exact_recovery_criteria_met": exact,
            "heldout_mse_bootstrap_95_percent_interval": np.quantile(
                boot, [0.025, 0.975]
            ).tolist(),
            "bootstrap_replicates": 300,
        }
        if condition["observation"] == "sign":
            train = np.asarray(observations[name]["train_x"])
            original = oracle_polynomial(train, teacher)
            alternative = oracle_polynomial(train, teacher * 2)
            row["nonidentifiability_witness"] = {
                "transformation": "multiply every canonical coefficient by positive scalar 2",
                "observed_sign_disagreements": int(
                    np.count_nonzero(np.sign(original) != np.sign(alternative))
                ),
                "all_input_sign_equivalence": "sign(2*f(x))=sign(f(x)) for every real x, including zero",
                "coefficient_l2_distance": float(np.linalg.norm(teacher)),
                "heldout_real_output_mse_between_witnesses": float(
                    np.mean(expected**2)
                ),
            }
        results.append(row)
    numerical = [{k: v for k, v in row.items()} for row in results]
    report = {
        "engine": "ModelForge Discovery Lab",
        "protocol_version": 1,
        "preregistration": config,
        "source_sha256": current_sources,
        "commitments": {
            "observations_sha256": manifest["observations_sha256"],
            "private_truth_sha256": manifest["private_truth_sha256"],
            "sealed_solution_sha256": seal["solution_sha256"],
            "manifest_sha256": seal["manifest_sha256"],
        },
        "verification": {
            "status": "completed",
            "verifier_pid": os.getpid(),
            "producer_pid": manifest["producer_pid"],
            "solver_pid": solution["solver_pid"],
            "process_separated": len(
                {os.getpid(), manifest["producer_pid"], solution["solver_pid"]}
            )
            == 3,
            "one_time_evaluation_claimed_before_truth_read": True,
            "teachers_independent_across_conditions": True,
            "independent_polynomial_evaluator": True,
            "validation_selection_independently_recomputed": True,
            "solver_inputs": [
                "manifest.json",
                "observations.json (train and validation only)",
            ],
            "sealed_test_used_for_selection": False,
            "independent_external_scientist": False,
            "scope": "Separate software verifier; same project authors and OS user. Not a malicious-process security sandbox.",
        },
        "results": results,
        "numerical_sha256": hashlib.sha256(
            json.dumps(numerical, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "cost": {
            "remote_requests": 0,
            "paid_api_cost_usd": 0,
            "train_oracle_examples": 256,
            "validation_oracle_examples": 128,
            "sealed_verification_examples": 512,
            "learner_observed_scalar_outputs": 768,
            "candidate_count_per_condition": 3,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "conclusion": "Tests bounded identifiable canonical quadratic function recovery and negative controls; inspect each measured criterion. No neural architecture, LLM or proprietary model reconstruction claim.",
        "uncertainty": [
            "Float64 tolerance is not byte-identical source parameter recovery.",
            "Bootstrap interval conditions on one owned teacher and sampled uniform inputs, not a population of architectures.",
            "Canonical monomial coefficients remove antisymmetric quadratic-matrix ambiguity; raw original parameterization need not be unique.",
            "Noise controls measure failure of exact recovery, not impossibility of useful approximation.",
            "Conditions have independent teachers, so noise rows are not a paired causal estimate of noise effects.",
            "Private truth is locally retained outside public artifact; same-user filesystem access is not isolated by OS sandbox.",
        ],
        "references": [
            {
                "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html",
                "role": "Primary implementation documentation: least squares and numerical rank",
            },
            {
                "url": "https://proceedings.mlr.press/v235/carlini24a.html",
                "role": "Related extraction research; does not establish this polynomial result or universal reconstruction",
            },
        ],
    }
    _write(public / "verification-core.json", report)
    _write(
        private / "evaluation-consumed.json",
        {
            "verifier_pid": os.getpid(),
            "status": "completed",
            "verified_core_sha256": _digest(public / "verification-core.json"),
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "solve", "verify"))
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    paths = [Path(p) for p in args.paths]
    if args.stage == "solve" and len(paths) == 1:
        solve(paths[0])
    elif args.stage == "prepare" and len(paths) == 2:
        prepare(*paths)
    elif args.stage == "verify" and len(paths) == 2:
        verify(*paths)
    else:
        parser.error(
            "solve requires public directory; prepare/verify require private and public directories"
        )


if __name__ == "__main__":
    main()
