"""Observation-only final-head recovery on a real, project-owned Llama decoder.

This protocol assumes authorized final hidden states and full float logits. It
recovers one known linear layer, not a decoder or closed model from text outputs.
"""

from __future__ import annotations

import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import secrets
import sys
import threading
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import psutil

from .discovery import _digest, _write
from .execution import run_json

Array = npt.NDArray[np.float64]


def _registration() -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "family": "project-owned randomly initialized Hugging Face Llama; final untied bias-free linear lm_head only",
        "hidden_size": 32,
        "vocab_size": 16,
        "sequence_length": 4,
        "calibration_examples": 64,
        "deficient_examples": 8,
        "heldout_examples": 32,
        "public_token_sampling_seed": 91,
        "method": "float64 least squares on exposed float32 final hidden states and full float32 logits",
        "max_weight_error": 1e-5,
        "max_heldout_logit_error": 1e-5,
        "max_condition_number_for_retention": 1e6,
        "fixed_method_verification_max_abs_difference": 1e-10,
        "selection": "one fixed method; no test-driven selection or retries",
        "max_seconds": 120,
        "max_worker_rss_bytes": 2 * 1024**3,
    }


def _sources() -> dict[str, str]:
    return {
        name: _digest(Path(__file__).with_name(name))
        for name in (
            "representation_research.py",
            "runtime_hf.py",
            "execution.py",
            "discovery.py",
        )
    }


def _teacher_files(private: Path) -> dict[str, str]:
    root = private / "teacher"
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Private teacher symlinks are not allowed")
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in paths
        if path.is_file()
    }


def recover_lm_head(hidden_states: Array, logits: Array) -> dict[str, Any]:
    """Solve H W=Y using observations only; W has [hidden,vocabulary] shape."""
    h, y = (
        np.asarray(hidden_states, dtype=np.float64),
        np.asarray(logits, dtype=np.float64),
    )
    if (
        h.ndim != 2
        or y.ndim != 2
        or len(h) != len(y)
        or not 1 <= len(h) <= 256
        or not 1 <= h.shape[1] <= 64
        or not 1 <= y.shape[1] <= 64
        or not np.isfinite(h).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError(
            "Expected bounded finite hidden-state/logit matrices sharing rows"
        )
    matrix, _, rank, singular = np.linalg.lstsq(h, y, rcond=None)
    if rank != h.shape[1]:
        raise ValueError(
            "Hidden-state observations are rank deficient; original head is not identified"
        )
    return {
        "matrix": matrix.tolist(),
        "rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]),
        "calibration_mse": float(np.mean((h @ matrix - y) ** 2)),
    }


def _load(private: Path) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(1)
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(private / "teacher"), local_files_only=True, trust_remote_code=False
        )
        .cpu()
        .eval()
    )
    if (
        model.config.hidden_size != 32
        or model.config.vocab_size != 16
        or model.config.tie_word_embeddings
        or model.get_output_embeddings().weight.data_ptr()
        == model.get_input_embeddings().weight.data_ptr()
    ):
        raise ValueError("Expected owned untied Llama head with hidden32/vocabulary16")
    return model


def _forward(model: Any, tokens: Any) -> tuple[Array, Array]:
    import torch

    with torch.inference_mode():
        outputs = model(
            input_ids=torch.tensor(tokens, dtype=torch.long, device="cpu"),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    return np.asarray(
        outputs.hidden_states[-1][:, -1, :].cpu().numpy(), dtype=np.float64
    ), np.asarray(outputs.logits[:, -1, :].cpu().numpy(), dtype=np.float64)


def _prepare(private: Path, public: Path) -> dict[str, Any]:
    from .runtime_hf import create_owned_fixture

    registration = json.loads(
        (public / "preregistration.json").read_text(encoding="utf-8")
    )
    if registration != _registration():
        raise ValueError("Unrecognized preregistration")
    with (private / "prepare.claim").open("x", encoding="utf-8") as claim:
        json.dump({"producer_pid": os.getpid()}, claim)
    teacher_seed, heldout_seed = secrets.randbelow(2**31), secrets.randbits(128)
    _write(
        private / "entropy.json",
        {"teacher_seed": teacher_seed, "heldout_seed": heldout_seed},
    )
    create_owned_fixture(private / "teacher", model_type="llama", seed=teacher_seed)
    model = _load(private)
    tokens = np.random.default_rng(registration["public_token_sampling_seed"]).integers(
        0, 16, size=(64, 4)
    )
    h, y = _forward(model, tokens.tolist())
    calibration_ids = {tuple(row) for row in tokens.tolist()}
    heldout_ids: set[tuple[int, ...]] = set()
    heldout_rng = np.random.default_rng(heldout_seed)
    for _ in range(1024):
        candidate = tuple(int(token) for token in heldout_rng.integers(0, 16, size=4))
        if candidate not in calibration_ids:
            heldout_ids.add(candidate)
        if len(heldout_ids) == 32:
            break
    if len(heldout_ids) != 32:
        raise RuntimeError(
            "Bounded heldout sampling failed to obtain 32 disjoint sequences"
        )
    heldout_tokens = [list(row) for row in sorted(heldout_ids)]
    _write(private / "heldout-inputs.json", {"tokens": heldout_tokens})
    _write(
        public / "observations.json",
        {
            "hidden_states": h.tolist(),
            "logits": y.tolist(),
            "calibration_token_ids": tokens.tolist(),
            "provenance": "Final-position hidden_states[-1] and full logits from real owned CPU Llama forward; no teacher weights",
        },
    )
    _write(
        public / "manifest.json",
        {
            "preregistration_sha256": _digest(public / "preregistration.json"),
            "observations_sha256": _digest(public / "observations.json"),
            "private_weight_sha256": _digest(private / "teacher" / "model.safetensors"),
            "private_teacher_files_sha256": _teacher_files(private),
            "private_heldout_inputs_sha256": _digest(private / "heldout-inputs.json"),
            "source_sha256": _sources(),
            "producer_pid": os.getpid(),
        },
    )
    return {"pid": os.getpid(), "calibration_examples": len(h)}


def _solve(public: Path) -> dict[str, Any]:
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != _sources() or manifest[
        "observations_sha256"
    ] != _digest(public / "observations.json"):
        raise ValueError("Source/observations commitment mismatch")
    observations = json.loads(
        (public / "observations.json").read_text(encoding="utf-8")
    )
    solution = recover_lm_head(observations["hidden_states"], observations["logits"])
    solution.update(
        {
            "solver_pid": os.getpid(),
            "observations_sha256": manifest["observations_sha256"],
            "manifest_sha256": _digest(public / "manifest.json"),
        }
    )
    _write(public / "solution.json", solution)
    return {"pid": os.getpid(), "rank": solution["rank"]}


def _verify(private: Path, public: Path) -> dict[str, Any]:
    # Exclusive claim is before any teacher or heldout read; failure consumes it.
    with (private / "evaluation-consumed.json").open("x", encoding="utf-8") as marker:
        json.dump({"status": "started", "pid": os.getpid()}, marker)
    seal = json.loads((private / "seal.json").read_text(encoding="utf-8"))
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    registration = json.loads(
        (public / "preregistration.json").read_text(encoding="utf-8")
    )
    checks = [
        (public / "solution.json", seal["solution_sha256"]),
        (public / "manifest.json", seal["manifest_sha256"]),
        (public / "observations.json", manifest["observations_sha256"]),
        (public / "preregistration.json", manifest["preregistration_sha256"]),
        (private / "teacher" / "model.safetensors", manifest["private_weight_sha256"]),
        (private / "heldout-inputs.json", manifest["private_heldout_inputs_sha256"]),
    ]
    if (
        registration != _registration()
        or manifest["source_sha256"] != _sources()
        or manifest["private_teacher_files_sha256"] != _teacher_files(private)
        or any(_digest(path) != expected for path, expected in checks)
    ):
        raise ValueError(
            "Sealed artifact, source or preregistration commitment mismatch"
        )
    observations = json.loads(
        (public / "observations.json").read_text(encoding="utf-8")
    )
    solution = json.loads((public / "solution.json").read_text(encoding="utf-8"))
    if solution["observations_sha256"] != manifest["observations_sha256"] or solution[
        "manifest_sha256"
    ] != _digest(public / "manifest.json"):
        raise ValueError("Student provenance mismatch")
    h, y, fitted = (
        np.asarray(value, dtype=np.float64)
        for value in (
            observations["hidden_states"],
            observations["logits"],
            solution["matrix"],
        )
    )
    if (
        h.shape != (64, 32)
        or y.shape != (64, 16)
        or fitted.shape != (32, 16)
        or not all(np.isfinite(x).all() for x in (h, y, fitted))
    ):
        raise ValueError("Unexpected observation/student dimensions")
    rank, condition = int(np.linalg.matrix_rank(h)), float(np.linalg.cond(h))
    if (
        solution["rank"] != rank
        or rank != 32
        or not np.isclose(solution["condition_number"], condition)
    ):
        raise ValueError("Student rank/conditioning mismatch")
    calibration_mse = float(np.mean((h @ fitted - y) ** 2))
    if not np.isclose(
        solution["calibration_mse"], calibration_mse, rtol=1e-8, atol=1e-24
    ):
        raise ValueError("Student calibration residual mismatch")
    # Independently reconstruct the fixed least-squares fit using explicit SVD,
    # not the learner's lstsq call, before any private model/heldout evaluation.
    u, singular, vh = np.linalg.svd(h, full_matrices=False)
    audited_fit = (vh.T / singular) @ (u.T @ y)
    fixed_method_difference = float(np.max(np.abs(audited_fit - fitted)))
    if (
        fixed_method_difference
        > registration["fixed_method_verification_max_abs_difference"]
    ):
        raise ValueError(
            "Sealed student differs from the whitelisted least-squares fit"
        )
    model = _load(private)
    truth = np.asarray(
        model.get_output_embeddings().weight.detach().cpu().numpy().T, dtype=np.float64
    )
    heldout_tokens = json.loads(
        (private / "heldout-inputs.json").read_text(encoding="utf-8")
    )["tokens"]
    heldout_h, heldout_y = _forward(model, heldout_tokens)
    coefficient_error = float(np.max(np.abs(fitted - truth)))
    logit_error = float(np.max(np.abs(heldout_h @ fitted - heldout_y)))
    # SVD-derived nullspace witness; its direction depends on calibration only.
    deficient = h[:8]
    _, _, vh = np.linalg.svd(deficient, full_matrices=True)
    direction = vh[-1]
    delta = np.outer(direction, np.linspace(0.5, 1.5, 16))
    alternative = truth + delta
    deficient_gap = float(np.max(np.abs(deficient @ delta)))
    unseen_gap = float(np.max(np.abs(heldout_h @ delta)))
    scaled = truth * 2
    argmax_disagreements = int(
        np.count_nonzero(np.argmax(h @ truth, axis=1) != np.argmax(h @ scaled, axis=1))
    )
    criteria = (
        rank == 32
        and condition <= registration["max_condition_number_for_retention"]
        and coefficient_error <= registration["max_weight_error"]
        and logit_error <= registration["max_heldout_logit_error"]
    )
    core = {
        "engine": "ModelForge owned decoder final-head reconstruction",
        "protocol_version": 1,
        "preregistration": registration,
        "source_sha256": _sources(),
        "commitments": {
            **seal,
            "observations_sha256": manifest["observations_sha256"],
            "preregistration_sha256": manifest["preregistration_sha256"],
            "private_weight_sha256": manifest["private_weight_sha256"],
            "private_heldout_inputs_sha256": manifest["private_heldout_inputs_sha256"],
        },
        "full_rank_recovery": {
            "rank": rank,
            "condition_number": condition,
            "coefficient_count": 512,
            "fixed_method_svd_verification_max_abs_difference": fixed_method_difference,
            "max_weight_error": coefficient_error,
            "calibration_mse": calibration_mse,
            "heldout_max_logit_error": logit_error,
            "heldout_logit_mse": float(np.mean((heldout_h @ fitted - heldout_y) ** 2)),
            "criteria_met": criteria,
        },
        "deficient_observation_control": {
            "observed_rows": 8,
            "rank": int(np.linalg.matrix_rank(deficient)),
            "hidden_dimension": 32,
            "nullspace_direction_from_calibration_only": direction.tolist(),
            "weight_l2_distance": float(np.linalg.norm(alternative - truth)),
            "calibration_max_logit_difference": deficient_gap,
            "heldout_max_logit_difference": unseen_gap,
            "conclusion": "Distinct final heads agree on rank-deficient exposed representations but differ on private heldout representations",
        },
        "argmax_only_control": {
            "positive_scaling": 2,
            "calibration_argmax_disagreements": argmax_disagreements,
            "weight_l2_distance": float(np.linalg.norm(scaled - truth)),
            "all_hidden_state_argmax_identity": "argmax(2*H W)=argmax(H W) including exactly tied logits; float64 computations measured separately",
            "conclusion": "Argmax labels alone do not identify logit scale or original linear-head weights",
        },
        "verification": {
            "producer_pid": manifest["producer_pid"],
            "solver_pid": solution["solver_pid"],
            "verifier_pid": os.getpid(),
            "separate_producer_solver_verifier_processes": len(
                {manifest["producer_pid"], solution["solver_pid"], os.getpid()}
            )
            == 3,
            "one_time_evaluation_claimed_before_teacher_read": True,
            "solver_inputs": [
                "public manifest",
                "public calibration hidden_states and full float logits",
            ],
            "teacher_weights_given_to_solver": False,
            "heldout_used_for_fit_or_selection": False,
            "independent_external_scientist": False,
            "scope": "Separate software processes and retained private teacher; same project and OS user, not a malicious same-user sandbox",
        },
        "cost": {
            "calibration_forward_sequences": 64,
            "heldout_forward_sequences": 32,
            "sequence_length": 4,
            "observed_hidden_scalars": 2048,
            "observed_logit_scalars": 1024,
            "remote_requests": 0,
            "paid_api_cost_usd": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": version("torch"),
            "transformers": version("transformers"),
            "teacher_forward_device": "cpu",
            "teacher_forward_dtype": "float32",
            "platform": platform.platform(),
        },
        "limitations": [
            "Randomly initialized owned decoder fixture, not pretrained intelligence",
            "Final known linear head only; backbone/tokenizer/architecture were not reconstructed",
            "Authorized exposed final activations and complete float logits are stronger access than ordinary text APIs",
            "Float32 forward rounding and finite conditioning prevent byte-identical recovery claims",
            "No output-only closed-model attack, no provider credentials, no target outputs or downloads",
            "Deficient/argmax counterexamples concern head identifiability; no universal model reconstruction conclusion",
        ],
        "references": [
            "https://huggingface.co/docs/transformers/main_classes/output",
            "https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html",
            "https://proceedings.mlr.press/v235/carlini24a.html",
        ],
    }
    _write(public / "verification-core.json", core)
    _write(
        private / "evaluation-consumed.json",
        {
            "status": "completed",
            "pid": os.getpid(),
            "verified_core_sha256": _digest(public / "verification-core.json"),
        },
    )
    return {
        "pid": os.getpid(),
        "verified_core_sha256": _digest(public / "verification-core.json"),
    }


def run_representation_research(
    destination: str | Path, *, private_storage: str | Path | None = None
) -> dict[str, Any]:
    """Fresh public/private protocol directories, bounded 120s CPU-only workers."""
    public = Path(destination).expanduser().absolute()
    private = (
        Path(private_storage).expanduser().absolute()
        if private_storage is not None
        else public.parent / ".representation-private" / public.name
    )
    resolved_public, resolved_private = public.resolve(), private.resolve()
    if (
        resolved_public == resolved_private
        or resolved_public in resolved_private.parents
        or resolved_private in resolved_public.parents
    ):
        raise ValueError("Public and private directories must be disjoint")
    if any(
        p.exists() or p.is_symlink() or any(a.is_symlink() for a in p.parents)
        for p in (public, private)
    ):
        raise ValueError(
            "Public/private destinations must be new without symlink ancestors"
        )
    public.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    private.chmod(0o700)
    _write(public / "preregistration.json", _registration())
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        str(Path(p).absolute())
        for p in environment.get("PYTHONPATH", "").split(os.pathsep)
        if p
    )
    start = time.perf_counter()
    deadline = time.monotonic() + 120
    stages = []
    for stage in ("prepare", "solve", "verify"):
        if stage == "verify":
            _write(
                private / "seal.json",
                {
                    "solution_sha256": _digest(public / "solution.json"),
                    "manifest_sha256": _digest(public / "manifest.json"),
                },
            )
        payload = {"public": str(public)}
        if stage != "solve":
            payload["private"] = str(private)
        result = run_json(
            [sys.executable, "-m", "modelforge.representation_research", stage],
            payload,
            cwd=public,
            memory_limit_mb=2048,
            timeout_seconds=max(0.001, deadline - time.monotonic()),
            env=environment,
        )
        stages.append({"stage": stage, **result})
    report: dict[str, Any] = json.loads(
        (public / "verification-core.json").read_text(encoding="utf-8")
    )
    report["verification_core_sha256"] = _digest(public / "verification-core.json")
    report["cost"].update(
        {
            "elapsed_seconds": time.perf_counter() - start,
            "stages": stages,
            "sampled_peak_worker_rss_bytes": max(
                s["sampled_peak_self_rss_bytes"] for s in stages
            ),
            "memory_measurement": "Worker self RSS every10ms; generic execution monitor enforces 2048MiB every50ms, transient peaks may be missed",
        }
    )
    _write(public / "report.json", report)
    return report


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "solve", "verify"}:
        raise SystemExit(
            "Internal representation protocol: prepare/solve/verify JSON stdin only"
        )
    stage = sys.argv[1]
    payload = json.loads(sys.stdin.buffer.read(262144))
    expected = {"public"} if stage == "solve" else {"public", "private"}
    if set(payload) != expected:
        raise ValueError(
            "Unexpected protocol inputs; solver must not receive private paths"
        )
    process = psutil.Process()
    samples = [process.memory_info().rss]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.01):
            samples.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    before = time.perf_counter()
    try:
        public = Path(payload["public"])
        if stage == "solve":
            result = _solve(public)
        else:
            result = (_prepare if stage == "prepare" else _verify)(
                Path(payload["private"]), public
            )
    finally:
        stop.set()
        monitor.join(timeout=1)
    result.update(
        {
            "sampled_peak_self_rss_bytes": max(samples),
            "elapsed_seconds": time.perf_counter() - before,
        }
    )
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
