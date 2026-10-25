"""One exclusive reviewed protocol attempt; resource gate before scientific input.

No evaluation or retry. Controller report dominates any partial worker artifacts.
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
    "examples/isolated_glu/" + name + ".py"
    for name in [
        "guard_aware_controller",
        "guard_aware_worker",
        "guard_aware_step",
        "guard_aware_trace",
        "guard_aware_direction",
        "guard_step_evidence",
        "guard_metric_normals",
        "gate_gradient",
        "guarded_gate_step",
        "guarded_forward",
        "forward_factor",
        "evidence_transport",
    ]
} | {"src/modelforge/execution.py", "src/modelforge/glu32_io.py"}


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
    if review.get("actual_fitting_protocol_approved") is not True:
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
        config.get("id") != "isolated-glu-fp32-guard-aware-gate-step-v1"
        or config.get("timeout_seconds") != 180
        or config.get("memory_limit_mb") != 512
        or config.get("worker_output_limit_bytes") != 2048
        or config.get("blas_threads") != 1
    ):
        raise ValueError("Frozen protocol limits changed")

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, timeout=10
        ).strip()

    if (
        git("status", "--porcelain")
        or git("branch", "--show-current") != "discovery-lab"
    ):
        raise ValueError("Clean discovery-lab commit required before actual fitting")
    return config, git("rev-parse", "HEAD")


def validate_artifacts(worker: Path, summary: dict[str, Any]) -> dict[str, Any]:
    if sha(worker / "worker-report.json") != summary.get("worker_report_sha256"):
        raise ValueError("Worker report commitment mismatch")
    report: dict[str, Any] = read_json(worker / "worker-report.json")
    root = worker / "step"
    if (
        report.get("status") != "completed"
        or report.get("inputs_unchanged") is not True
    ):
        raise ValueError("Worker failed or immutable inputs not verified")
    for name, field in [
        ("report.json", "step_report_sha256"),
        ("artifact-manifest.json", "artifact_manifest_sha256"),
    ]:
        if sha(root / name) != report.get(field):
            raise ValueError("Step report/manifest commitment mismatch")
    details = read_json(root / "report.json")
    manifest = read_json(root / "artifact-manifest.json")
    if details.get("status") != "completed" or manifest.get("status") != "completed":
        raise ValueError("Terminal failed status dominates all candidates")
    files = manifest.get("files", {})
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "artifact-manifest.json"
    }
    if set(files) != actual or any(
        sha(root / name) != digest for name, digest in files.items()
    ):
        raise ValueError("Complete artifact inventory/hash mismatch")
    changed = bool(details.get("changed"))
    if changed != bool(report.get("candidate_built")) or changed != bool(
        summary.get("changed")
    ):
        raise ValueError("Changed candidate status mismatch")
    if changed and (
        not details.get("accepted")
        or sha(root / "candidate.npz") != report.get("candidate_sha256")
    ):
        raise ValueError("Changed candidate commitment/updates missing")
    if not changed and (root / "candidate.npz").exists():
        raise ValueError("Unexpected candidate on nochange run")
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
        "evaluation_eligible": False,
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
            [sys.executable, str(repo / "examples/isolated_glu/guard_aware_worker.py")],
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
        result = validate_artifacts(output / "worker", summary)
        _, after_head = verify_source(repo, config_path, review_path)
        if after_head != head:
            raise ValueError("Source commit changed")
        remaining()
        report.update(
            status="completed",
            evaluation_eligible=bool(result["candidate_built"]),
            worker_report=result,
        )
        save(report_path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed",
            evaluation_eligible=False,
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
                    "evaluation_eligible": result["evaluation_eligible"],
                    "private_gate": "closed",
                }
            )
        )
    except Exception as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)[:500], "private_gate": "closed"})
        )
