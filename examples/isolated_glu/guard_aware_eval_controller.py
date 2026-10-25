"""One exclusive paired-evaluation attempt; pre-data tree monitoring required.

No fitting or retry. Failed controller status dominates completed metric files.
"""

from __future__ import annotations
from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from modelforge.execution import run_json
from modelforge.glu32_io import raw, read_json

DEPENDENCIES = {
    "examples/isolated_glu/guard_aware_eval_controller.py",
    "examples/isolated_glu/guard_aware_evaluate.py",
    "examples/isolated_glu/fp32_fullmap.py",
    "examples/isolated_glu/precision.py",
    "src/modelforge/glu32.py",
    "src/modelforge/antipodal.py",
    "src/modelforge/execution.py",
    "src/modelforge/glu32_io.py",
}


class ProtocolError(ValueError):
    """Terminal controller failure, including unavailable monitoring."""


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def save(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def verify_source(
    repo: Path, config_path: Path, review_path: Path
) -> tuple[dict[str, Any], str]:
    config = read_json(config_path)
    review = read_json(review_path)
    if review.get("actual_evaluation_protocol_approved") is not True:
        raise ValueError("Complete protocol independent source approval required")
    pins = config.get("frozen_dependencies", {})
    if set(pins) != DEPENDENCIES or any(
        sha(repo / name) != digest for name, digest in pins.items()
    ):
        raise ValueError("Frozen dependency inventory/commitments mismatch")
    reviewed = review.get("source_sha256", {})
    if any(
        reviewed.get(name) != digest for name, digest in pins.items()
    ) or reviewed.get(str(config_path.relative_to(repo))) != sha(config_path):
        raise ValueError("Review does not bind exact sources/preregistration")
    if (
        config.get("id") != "isolated-glu-fp32-guard-aware-paired-evaluation-v1"
        or config.get("timeout_seconds") != 180
        or config.get("memory_limit_mb") != 512
        or config.get("worker_output_limit_bytes") != 2048
        or config.get("blas_threads") != 1
    ):
        raise ValueError("Frozen protocol limits changed")

    if (
        config.get("evaluation_seed") != 941033
        or config.get("rows_per_group") != 512
        or config.get("scales") != [0.01, 1.0, 10.0, 100.0]
        or config.get("new_reference_vector_budget") != 2048
        or config.get("quality_criterion") != 0.001
    ):
        raise ValueError("Frozen evaluation policy changed")

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, timeout=10
        ).strip()

    if (
        git("status", "--porcelain")
        or git("branch", "--show-current") != "discovery-lab"
    ):
        raise ValueError("Clean discovery-lab commit required before actual evaluation")
    return config, git("rev-parse", "HEAD")


def validate_artifacts(
    worker: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    if sha(worker / "worker-report.json") != summary.get("worker_report_sha256"):
        raise ValueError("Worker report commitment mismatch")
    report: dict[str, Any] = read_json(worker / "worker-report.json")
    if (
        report.get("status") != "completed"
        or report.get("inputs_unchanged") is not True
        or report.get("inputs_before") != config["inputs"]
        or report.get("inputs_after") != config["inputs"]
    ):
        raise ValueError("Worker failed or immutable input commitments missing")
    keys = ["calls_reserved", "calls_attempted", "calls_returned", "calls_persisted"]
    if any(report.get(k) != 2048 or summary.get(k) != 2048 for k in keys):
        raise ValueError("Complete2048 vector-call accounting required")
    pair_path = worker / "batches/progress.json"
    if sha(pair_path) != report.get("paired_report_sha256"):
        raise ValueError("Paired report commitment mismatch")
    pair = read_json(pair_path)
    if (
        pair.get("status") != "completed"
        or pair.get("seed") != 941033
        or [g["scale"] for g in pair.get("groups", [])] != [0.01, 1.0, 10.0, 100.0]
        or pair.get("development_max_group_relative") != 0.001
        or any(pair.get(k) != 2048 for k in keys)
    ):
        raise ValueError("Incomplete/wrong paired evaluation profile")
    if pair != report.get("evaluation") or summary.get("criterion_met") != pair.get(
        "all_groups_relative_at_most_1e_minus3"
    ):
        raise ValueError("Evaluation quality/report disagreement")
    if sha(worker / "batches/paired-validation.npz") != pair.get("validation_sha256"):
        raise ValueError("Numerical validation commitment mismatch")
    if (
        report.get("candidate_sha256") != config["inputs"]["candidate"]
        or report.get("baseline_sha256") != config["inputs"]["baseline"]
    ):
        raise ValueError("Frozen candidate/incumbent mismatch")
    marker = paths["fit_controller"].parent / "public941033-evaluation-attempt.json"
    if report.get("persistent_marker") != str(marker) or sha(marker) != report.get(
        "persistent_marker_sha256"
    ):
        raise ValueError("Persistent one-time marker missing/changed")
    value = read_json(marker)
    canonical = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (
        value.get("evaluation_seed") != 941033
        or value.get("calls_reserved") != 2048
        or value.get("candidate_sha256") != config["inputs"]["candidate"]
        or value.get("baseline_sha256") != config["inputs"]["baseline"]
        or value.get("config_content_sha256") != canonical
    ):
        raise ValueError("Marker does not bind frozen evaluation")
    return report


def run_protocol(
    repo: Path,
    output: Path,
    paths: dict[str, Path],
    config_path: Path,
    review_path: Path,
    *,
    engine: Callable[..., dict[str, Any]] = run_json,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    started = clock()
    if any(p.is_symlink() for p in [output, *output.parents]):
        raise ValueError("No symlink attempt ancestors")
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = {
        "status": "started",
        "worker_started": False,
        "evaluation_completed": False,
        "quality_criterion_met": None,
        "automatic_retry": False,
        "private_gate": "closed",
    }
    report_path = output / "controller-report.json"
    save(report_path, report)

    def remaining() -> float:
        value = 180 - (clock() - started)
        if value <= 0:
            raise TimeoutError("Shared protocol deadline expired")
        return value

    try:
        config, head = verify_source(repo, config_path, review_path)
        report.update(
            source_commit=head,
            config_sha256=sha(config_path),
            review_sha256=sha(review_path),
        )
        save(report_path, report)
        env = dict(os.environ)
        env.update(
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            PYTHONPATH=str(repo / "src"),
        )
        # Nothing below reads or hashes a scientific input until preflight succeeds.
        preflight = engine(
            [
                sys.executable,
                "-c",
                "import json,time;time.sleep(.15);print(json.dumps({'synthetic_scope_preflight':True}))",
            ],
            {},
            cwd=output / "preflight",
            memory_limit_mb=512,
            timeout_seconds=remaining(),
            output_limit_bytes=2048,
            env=env,
        )
        report["preflight"] = preflight
        save(report_path, report)
        if (
            preflight.get("process_monitor_scope") != "process_and_children"
            or preflight.get("memory_samples", 0) < 1
        ):
            raise ValueError(
                "Pre-data tree monitor scope unavailable; scientific inputs not opened"
            )
        budget = remaining()
        report["worker_started"] = True
        save(report_path, report)
        summary = engine(
            [
                sys.executable,
                str(repo / "examples/isolated_glu/guard_aware_evaluate.py"),
            ],
            {
                "paths": {k: str(p) for k, p in paths.items()},
                "output": str(output / "worker"),
                "config": config,
                "preflight": preflight,
            },
            cwd=output / "engine",
            memory_limit_mb=512,
            timeout_seconds=budget,
            output_limit_bytes=2048,
            env=env,
        )
        report["worker_receipt"] = summary
        save(report_path, report)
        if (
            summary.get("process_monitor_scope") != "process_and_children"
            or summary.get("memory_samples", 0) < 1
        ):
            raise ValueError(
                "Worker tree monitoring unavailable; artifacts are not resource-protocol success"
            )
        result = validate_artifacts(output / "worker", summary, config, paths)
        _, after_head = verify_source(repo, config_path, review_path)
        if after_head != head:
            raise ValueError("Source commit changed")
        remaining()
        report.update(
            status="completed",
            evaluation_completed=True,
            quality_criterion_met=result["evaluation"][
                "all_groups_relative_at_most_1e_minus3"
            ],
            elapsed_seconds_before_final_report=clock() - started,
            worker_files_sha256={
                str(p.relative_to(output / "worker")): sha(p)
                for p in (output / "worker").rglob("*")
                if p.is_file()
            },
            worker_report=result,
        )
        save(report_path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed",
            evaluation_completed=False,
            quality_criterion_met=None,
            query_counts_unknown_if_worker_killed=bool(report["worker_started"]),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        save(report_path, report)
        raise ProtocolError(str(exc)) from exc


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        result = run_protocol(
            Path(request["repo"]),
            Path(request["output"]),
            {k: Path(v) for k, v in request["paths"].items()},
            Path(request["config"]),
            Path(request["review"]),
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "quality_criterion_met": result["quality_criterion_met"],
                    "private_gate": "closed",
                }
            )
        )
    except Exception as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)[:500], "private_gate": "closed"})
        )
