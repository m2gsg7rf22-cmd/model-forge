"""One prospective source-reviewed dimension diagnostic under bounded monitoring."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from modelforge.execution import run_json
from modelforge.glu32_io import read_json
from scaling_domain_run import POLICY, inventory_plan, save_json
from scaling_domain_store import _ancestry, sha, verify_manifest

DEPENDENCIES = {
    "examples/isolated_glu/scaling_domain_controller.py",
    "examples/isolated_glu/scaling_domain_run.py",
    "examples/isolated_glu/scaling_domain_math.py",
    "examples/isolated_glu/scaling_domain_store.py",
    "examples/isolated_glu/precision.py",
    "src/modelforge/execution.py",
    "src/modelforge/glu32_io.py",
}


class ProtocolError(ValueError):
    """Terminal source/resource/transport/data protocol failure; never retry."""


def verify_source(
    repo: Path, config_path: Path, review_path: Path
) -> tuple[dict[str, Any], str]:
    config = read_json(config_path)
    review = read_json(review_path)
    if review.get("dimension_domain_full_protocol_approved") is not True:
        raise ValueError("Full dimension-domain protocol source approval required")
    if (
        config.get("policy") != POLICY
        or config.get("inventory_plan") != inventory_plan()
    ):
        raise ValueError("Frozen policy/inventory mismatch")
    pins = config.get("frozen_dependencies", {})
    if set(pins) != DEPENDENCIES or any(sha(repo / k) != h for k, h in pins.items()):
        raise ValueError("Exact dependency inventory/source mismatch")
    signed = review.get("source_sha256", {})
    if any(signed.get(k) != h for k, h in pins.items()) or signed.get(
        str(config_path.relative_to(repo))
    ) != sha(config_path):
        raise ValueError("Review does not bind full current source/preregistration")
    # Also bind every document/test present in the review, not only runtime sources.
    if any(sha(repo / k) != h for k, h in signed.items()):
        raise ValueError("Reviewed source/test/document bytes changed")

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, timeout=10
        ).strip()

    if (
        git("status", "--porcelain")
        or git("branch", "--show-current") != "discovery-lab"
    ):
        raise ValueError("Clean committed discovery-lab required")
    return config, git("rev-parse", "HEAD")


def expected_names() -> set[str]:
    names: set[str] = set()
    for p in range(4):
        for t in range(2):
            for channel in ("fp64", "fp32"):
                case = f"p{p}-t{t}-{channel}"
                names.add(case + "-context.npz")
                for s in range(2):
                    for x in range(64):
                        for kind in ("inputs", "responses"):
                            names.add(f"{case}-s{s}-x{x:02d}-{kind}.npz")
                for kind in ("reference", "fine", "coarse", "richardson"):
                    for h in range(8):
                        names.add(f"{case}-{kind}-h{h:02d}.npz")
                for kind in ("atoms", "reference", "fine", "coarse", "richardson"):
                    names.add(f"{case}-spectrum-{kind}.npz")
    return names


def validate_worker(
    output: Path, summary: dict[str, Any], head: str, config_sha: str, repo: Path
) -> dict[str, Any]:
    worker = output / "worker"
    path = worker / "worker-report.json"
    if path.stat().st_size > 128 * 1024 or sha(path) != summary.get(
        "worker_report_sha256"
    ):
        raise ValueError("Worker report commitment mismatch")
    report: dict[str, Any] = read_json(path)
    if (
        report.get("status") != "completed"
        or report.get("source_commit") != head
        or len(report.get("cases", [])) != 16
    ):
        raise ValueError("Incomplete source-bound worker result")
    for key in (
        "calls_reserved",
        "calls_attempted",
        "calls_returned",
        "calls_persisted",
    ):
        if report.get(key) != 1454080 or summary.get(key) != 1454080:
            raise ValueError("Complete1454080 vector counts required")
    marker = repo.parent / "experiments/isolated-glu-dimension-domain-v1-attempt.json"
    _ancestry(marker)
    if report.get("marker") != str(marker) or sha(marker) != report.get(
        "marker_sha256"
    ):
        raise ValueError("Global marker commitment mismatch")
    value = read_json(marker)
    if (
        value.get("policy") != POLICY
        or value.get("calls_reserved") != 1454080
        or value.get("source_commit") != head
        or value.get("config_sha256") != config_sha
        or value.get("output") != str(worker.absolute())
    ):
        raise ValueError("Global marker source/config/output binding mismatch")
    arrays = worker / "arrays"
    inventory = verify_manifest(arrays)
    if (
        inventory != report.get("artifact_inventory")
        or inventory["shards"] != 4704
        or sha(arrays / "manifest.jsonl") != report.get("artifact_manifest_sha256")
        or {p.name for p in arrays.iterdir()} != expected_names() | {"manifest.jsonl"}
    ):
        raise ValueError("Complete exact fixed artifact inventory required")
    if report.get("inventory_plan") != inventory_plan():
        raise ValueError("Combined planned inventory mismatch")
    return report


def run_protocol(
    repo: Path,
    output: Path,
    config_path: Path,
    review_path: Path,
    *,
    engine: Callable[..., dict[str, Any]] = run_json,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    started = clock()
    _ancestry(output)
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = {
        "status": "started",
        "worker_started": False,
        "diagnostic_completed": False,
        "automatic_retry": False,
        "private_gate": "closed",
    }
    path = output / "controller-report.json"
    save_json(path, report)

    def remaining() -> float:
        value = 180 - (clock() - started)
        if value <= 0:
            raise TimeoutError("Shared180s deadline expired")
        return value

    try:
        config, head = verify_source(repo, config_path, review_path)
        config_sha = sha(config_path)
        report.update(
            source_commit=head, config_sha256=config_sha, review_sha256=sha(review_path)
        )
        save_json(path, report)
        env = dict(os.environ)
        env.update(
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            PYTHONPATH=str(repo / "src"),
        )

        def call(
            command: list[str], payload: dict[str, Any], directory: str
        ) -> dict[str, Any]:
            result = engine(
                command,
                payload,
                cwd=output / directory,
                memory_limit_mb=512,
                timeout_seconds=remaining(),
                output_limit_bytes=2048,
                env=env,
            )
            report[directory + "_receipt"] = result
            save_json(path, report)
            if (
                result.get("process_monitor_scope") != "process_and_children"
                or result.get("memory_samples", 0) < 1
            ):
                raise ValueError("Pre-data/final worker tree monitoring unavailable")
            return result

        preflight = call(
            [
                sys.executable,
                "-c",
                "import time,json;time.sleep(.15);print(json.dumps({'synthetic_scope_preflight':True}))",
            ],
            {},
            "scope-engine",
        )
        report["scope_preflight"] = preflight
        save_json(path, report)
        transport = call(
            [sys.executable, str(repo / "examples/isolated_glu/scaling_domain_run.py")],
            {"operation": "synthetic_transport", "output": str(output / "transport")},
            "transport-engine",
        )
        report["transport_preflight"] = transport
        save_json(path, report)
        if transport.get("status") != "passed":
            raise ValueError("Synthetic transport incomplete before data")
        report["worker_started"] = True
        save_json(path, report)
        summary = call(
            [sys.executable, str(repo / "examples/isolated_glu/scaling_domain_run.py")],
            dict(
                output=str(output / "worker"),
                repo=str(repo),
                policy=POLICY,
                full_source_protocol_approved=True,
                source_commit=head,
                config_sha256=config_sha,
                scope_preflight=preflight,
                transport_preflight=transport,
            ),
            "worker-engine",
        )
        report["worker_receipt"] = summary
        save_json(path, report)
        result = validate_worker(output, summary, head, config_sha, repo)
        _, after = verify_source(repo, config_path, review_path)
        if after != head:
            raise ValueError("Source commit changed")
        # Includes retained synthetic transport; reserve another bounded final report.
        total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if total + 128 * 1024 > 1024**3:
            raise ValueError("Global artifact ceiling exceeded")
        remaining()
        report.update(
            status="completed",
            diagnostic_completed=True,
            worker_report=result,
            elapsed_seconds_before_final_report=clock() - started,
            artifact_bytes_before_final_report=total,
        )
        save_json(path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed",
            diagnostic_completed=False,
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
            counts_may_be_lower_bounds_if_worker_killed=bool(report["worker_started"]),
        )
        partial = output / "worker/progress.json"
        try:
            if partial.is_file() and partial.stat().st_size <= 128 * 1024:
                report["partial_query_counter_snapshot"] = read_json(partial)
        except Exception as partial_error:
            report["partial_counter_error"] = type(partial_error).__name__
        save_json(path, report)
        raise ProtocolError(str(exc)) from exc


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        result = run_protocol(
            Path(request["repo"]),
            Path(request["output"]),
            Path(request["config"]),
            Path(request["review"]),
        )
        print(json.dumps({"status": result["status"], "private_gate": "closed"}))
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}
            )
        )
