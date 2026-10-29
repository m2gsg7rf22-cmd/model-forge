"""Mandatory default outer TREE launcher for the one-shot numeric audit.

No actual run is permitted until independent full-source approval and clean freeze.
Fixed durable output consumes the default attempt even if preflight fails; never
retry under a different output name. No RNG, training, teacher or heldout calls.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable
import json

from modelforge.execution import run_json
from joint_response_audit_protocol import (
    ACTUAL,
    ACTUAL_COMMIT,
    PARENT,
    POLICY,
    THREAD_ENV as THREAD_ENV,
    OLD_MARKER_SHA,
    environment,
    verify_source,
)
from joint_response_audit_runtime import MARKER, layout, preflight, validate_transport
from joint_response_audit_storage import TraceReader
from joint_response_audit_jobs import audit_jobs
from joint_response_join import load_join
from variable_response_manifest import METADATA_CAP, _json
from variable_response_protocol import (
    ancestry,
    digest,
    read,
    save,
    marker,
    tree_ok,
)

OLDER_MARKERS = {
    "paired-row-acquisition-v1-attempt.json": "4d58c1d0e1f6832d14dbe473e6518c2ba11b7afecb73d76fdf2f8e12593b4b33",
    "variable-width-gate-development-v1-attempt.json": "88aedf0f73a740c4980875d190913efcb8b6f7ff0f226d47266e7b2325be45a0",
    "variable-response-view-v1-attempt.json": "cff83243898a1141eeaf35ebd0d8fe90364441f1c21e474efa725888abe7de70",
}


def disk_free(parent: Path) -> int:
    return shutil.disk_usage(parent).free


def prior_outputs(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Before numeric access: root JSON hashes, marker hashes and size-only inventory."""
    parent = repo.parent / "experiments"
    root = parent / ACTUAL
    ancestry(root)
    pins = cfg["input_bindings"]
    total = 0
    count = 0

    def sized(path: Path, cap: int) -> None:
        nonlocal total, count
        ancestry(path)
        if not path.is_file() or not 0 < path.stat().st_size <= cap:
            raise ValueError("prior completed output size/layout changed")
        total += path.stat().st_size
        count += 1

    expected = set(pins["root_files"]) | {"cases", "transport", "preflight-engine"}
    if {p.name for p in root.iterdir()} != expected:
        raise ValueError("exact prior root layout required")
    roots = {}
    for name, pin in pins["root_files"].items():
        path = root / name
        sized(path, 128 * 1024)
        if digest(path) != pin:
            raise ValueError("prior root metadata hash changed")
        roots[name] = read(path)
    if {p.name for p in (root / "cases").iterdir()} != set(pins["case_files"]):
        raise ValueError("exact six prior cases required")
    for job, files in pins["case_files"].items():
        case = root / "cases" / job
        ancestry(case)
        if {p.name for p in case.iterdir()} != set(files):
            raise ValueError("prior case file layout changed")
        for name in files:
            cap = (
                8 * 1024**2
                if name.endswith(".npz")
                else 4000 * 512
                if name == "progress.jsonl"
                else 128 * 1024
            )
            sized(case / name, cap)
    transport = root / "transport"
    ancestry(transport)
    if {p.name for p in transport.iterdir()} != {
        "schedule.npz",
        "state.npz",
        "manifest.json",
    }:
        raise ValueError("prior transport layout changed")
    for item in transport.iterdir():
        sized(item, 8 * 1024**2 if item.suffix == ".npz" else 4096)
    for engine in (root / "preflight-engine", parent / (ACTUAL + "-launch-engine")):
        ancestry(engine)
        if engine.exists() and (not engine.is_dir() or list(engine.iterdir())):
            raise ValueError("prior completed engine must be empty")
    if total != 186239324 or count != 237:
        raise ValueError(
            "prior independently audited exact final byte inventory changed"
        )
    old_marker = parent / "joint-response-development-v1-attempt.json"
    if digest(old_marker) != OLD_MARKER_SHA or old_marker.stat().st_size != 1209:
        raise ValueError("consumed original marker changed")
    for name, pin in OLDER_MARKERS.items():
        if digest(parent / name) != pin:
            raise ValueError("consumed ancestor marker changed")
    report = roots["controller-report.json"]
    binding = roots["input-binding.json"]
    if (
        report.get("status") != "completed"
        or report.get("source_commit") != ACTUAL_COMMIT
        or report.get("marker_consumed") is not True
        or report.get("case_driver_invocations") != 6
        or binding.get("widths_hypothesized") != POLICY["width_hypotheses"]
        or not tree_ok(roots["outer-monitor.json"].get("receipt", {}))
    ):
        raise ValueError("prior completed engineering/resource/width binding required")
    return binding


def _paths(repo: Path, output: Path) -> Path:
    ancestry(repo)
    ancestry(output)
    parent = repo.parent / "experiments"
    ancestry(parent)
    if (
        output.absolute() != parent.absolute() / POLICY["id"]
        or output.exists()
        or (parent / MARKER).exists()
        or (parent / MARKER).is_symlink()
    ):
        raise ValueError(
            "fresh fixed durable output/unconsumed marker required; no retry"
        )
    return parent


def _summary(case: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "job_id",
        "status",
        "normalization_agreed",
        "metrics_agreed",
        "saved_selected_step",
        "selected_step_agreed",
        "selection_ambiguous",
        "selection_certified",
        "physical_parameters_agreed",
        "all_predictions_agreed",
        "prediction_rows_checked",
        "prediction_rows_visited",
        "prediction_numeric_failure",
        "error_type",
        "result_already_committed",
        "committed_checkpoint_records",
        "uncommitted_tail",
        "failure_receipt_error_type",
        "reader_partial",
    )
    return {k: case[k] for k in keys if k in case}


def run_protocol(
    repo: Path,
    output: Path,
    config: Path,
    review: Path,
    *,
    engine: Callable[..., dict[str, Any]] = run_json,
) -> dict[str, Any]:
    started = time.monotonic()
    cfg, head = verify_source(repo, config, review)
    parent = _paths(repo, output)
    parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(
        status="started",
        source_commit=head,
        config_sha256=digest(config),
        review_sha256=digest(review),
        marker_consumed=False,
        scientific_reconstruction_success=False,
        numeric_consistency_agreed=False,
        automatic_retry=False,
        certificate=False,
        private_gate="closed",
        new_queries=0,
        rng_calls=0,
        optimizer_replays=0,
        cases=[dict(job_id=f"job{i:02d}", status="not_started") for i in range(6)],
    )
    save(output / "controller-report.json", report)
    view = None

    def remaining() -> float:
        value = 180 - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("shared180s elapsed")
        return value

    def check() -> None:
        remaining()
        free = disk_free(parent)
        if type(free) is not int or free < 2 * 1024**3:
            raise ValueError("actual2GiBfree disk required")
        report["last_free_bytes"] = free
        layout(output)

    def unchanged() -> None:
        _, now = verify_source(repo, config, review)
        if (
            now != head
            or digest(config) != report["config_sha256"]
            or digest(review) != report["review_sha256"]
        ):
            raise ValueError("source/config/review freeze changed")

    try:
        check()
        prior_outputs(repo, cfg)
        resource = engine(
            [
                sys.executable,
                str(repo / "examples/isolated_glu/joint_response_audit_controller.py"),
            ],
            dict(operation="preflight", output=str(output / "transport")),
            cwd=output / "preflight-engine",
            memory_limit_mb=512,
            timeout_seconds=min(remaining(), 30),
            output_limit_bytes=1024,
            env=environment(repo),
        )
        save(output / "preflight-receipt.json", resource)
        if resource.get("status") != "passed" or not tree_ok(resource):
            raise ValueError("actual preflight process-and-children receipt required")
        validate_transport(output / "transport", resource)
        check()
        unchanged()
        prior_outputs(repo, cfg)
        report["marker_sha256"] = marker(
            parent / MARKER,
            dict(
                id=POLICY["id"],
                source_commit=head,
                config_sha256=report["config_sha256"],
                review_sha256=report["review_sha256"],
                output=str(output.absolute()),
                policy=POLICY,
                preflight_sha256=digest(output / "preflight-receipt.json"),
            ),
        )
        report["marker_consumed"] = True
        save(output / "controller-report.json", report)
        source = parent / PARENT
        view = load_join(
            repo,
            source / "collection/public",
            source / "estimator",
            source / "collection/truth",
        )
        jobs = view.jobs
        if (
            len(jobs) != 6
            or [j.width_hypothesis for j in jobs] != POLICY["width_hypotheses"]
        ):
            raise ValueError("frozen six width hypotheses differ")
        readers = [
            TraceReader(
                parent / ACTUAL / "cases" / j.job_id,
                cfg["input_bindings"]["case_files"][j.job_id],
                j.width_hypothesis,
            )
            for j in jobs
        ]
        save(
            output / "input-binding.json",
            dict(
                source_commit=head,
                old_actual_source_commit=ACTUAL_COMMIT,
                input_bindings=cfg["input_bindings"],
                view=view.report(),
            ),
        )

        def progress(value: dict[str, Any]) -> None:
            save(output / "progress.json", value)
            check()

        result = audit_jobs(
            jobs, readers, output / "cases", check=check, progress=progress
        )
        report.update(
            cases=[_summary(c) for c in result["cases"]],
            completed_case_visits=result["completed_case_visits"],
            numeric_consistency_agreed=result["numeric_consistency_agreed"],
        )
        if result["status"] != "completed":
            raise ValueError(
                "all-six numeric audit interrupted; retain partial receipts"
            )
        report["view_completion"] = view.finish()
        if report["view_completion"].get("status") != "completed":
            raise ValueError("original response completion check failed")
        check()
        unchanged()
        prior_outputs(repo, cfg)
        if digest(parent / MARKER) != report["marker_sha256"]:
            raise ValueError("consumed audit marker changed")
        validate_transport(output / "transport", resource)
        save(
            output / "terminal-binding.json",
            dict(
                source_commit=head,
                marker_sha256=report["marker_sha256"],
                case_result_sha256={
                    j.job_id: digest(output / "cases" / j.job_id / "result.json")
                    for j in jobs
                },
                reader_receipts={
                    j.job_id: r.report() for j, r in zip(jobs, readers, strict=True)
                },
            ),
        )
        report.update(
            status="completed",
            elapsed_before_final_write_seconds=time.monotonic() - started,
        )
        save(output / "controller-report.json", report)
        check()
        return report
    except Exception as exc:
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
            marker_consumed=(parent / MARKER).exists(),
            elapsed_before_failure_write_seconds=time.monotonic() - started,
        )
        if view is not None:
            report["view_partial"] = view.report()
        save(output / "controller-report.json", report)
        raise


def launch(
    request: dict[str, Any], *, engine: Callable[..., dict[str, Any]] = run_json
) -> dict[str, Any]:
    if set(request) != {"repo", "output", "config", "review"} or any(
        not isinstance(v, str) for v in request.values()
    ):
        raise ValueError("exact default launcher path request required")
    repo, output, config, review = (
        Path(request[k]) for k in ("repo", "output", "config", "review")
    )
    verify_source(repo, config, review)
    _paths(repo, output)
    started = time.monotonic()
    result: dict[str, Any] = dict(
        status="failed",
        resource_receipt_unavailable=True,
        controller_counters_unknown=True,
        certificate=False,
        private_gate="closed",
        automatic_retry=False,
    )
    try:
        receipt = engine(
            [
                sys.executable,
                str(repo / "examples/isolated_glu/joint_response_audit_controller.py"),
            ],
            dict(request, operation="controlled_run"),
            cwd=output.parent / (output.name + "-launch-engine"),
            memory_limit_mb=512,
            timeout_seconds=180,
            output_limit_bytes=1024,
            env=environment(repo),
        )
        result.update(receipt=receipt, resource_receipt_unavailable=False)
        child = read(output / "controller-report.json")
        if (
            not tree_ok(receipt)
            or receipt.get("status") != "completed"
            or child.get("status") != "completed"
        ):
            raise ValueError("complete actual outer/controller receipt required")
        result.update(
            status="completed",
            controller_counters_unknown=False,
            controller_sha256=digest(output / "controller-report.json"),
            elapsed_before_outer_write_seconds=time.monotonic() - started,
        )
        save(output / "outer-monitor.json", result)
        result["retained_before_final_outer_write"] = layout(output)
        save(output / "outer-monitor.json", result)
        layout(output)
        return result
    except Exception as exc:
        result.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)[:200]
        )
        if output.exists():
            save(output / "outer-monitor.json", result)
        raise


if __name__ == "__main__":
    try:
        blob = sys.stdin.buffer.read(METADATA_CAP + 1)
        if len(blob) > METADATA_CAP:
            raise ValueError("request cap exceeded")
        request = _json(blob)
        operation = request.pop("operation", None)
        if operation == "preflight" and set(request) == {"output"}:
            result = preflight(Path(request["output"]))
        elif operation == "controlled_run" and set(request) == {
            "repo",
            "output",
            "config",
            "review",
        }:
            value = run_protocol(
                *(Path(request[k]) for k in ("repo", "output", "config", "review"))
            )
            result = {"status": value["status"]}
        elif operation is None:
            value = launch(request)
            result = {"status": value["status"]}
        else:
            raise ValueError("unsupported operation/schema")
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        print(
            json.dumps(
                dict(
                    ok=False,
                    status="failed",
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                )
            )
        )
        raise SystemExit(1)
