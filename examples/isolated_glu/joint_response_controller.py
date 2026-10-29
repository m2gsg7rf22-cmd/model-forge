"""One-shot joint calibration study: default outer TREE launcher is mandatory.

No teacher/evaluator/private model access. Conditional full independent source
approval, clean freeze and actual gates precede observations and registered RNG.
"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable

from modelforge.execution import run_json
from joint_response_arrays import read_typed
from joint_response_case import assemble_case
from joint_response_driver import run_case, _state
from joint_response_inventory import trace_specs
from joint_response_join import load_join
from joint_response_streams import prepare_inputs, stream_seed
from joint_response_trace import TraceStore, CHECKPOINTS
from joint_response_protocol import (
    MARKER,
    POLICY,
    THREAD_ENV as THREAD_ENV,
    PARENT,
    PCG64Stream,
    optimizer_binding,
    environment,
    verify_source,
)
from joint_response_runtime import layout, preflight, validate_transport
from variable_response_manifest import METADATA_CAP, _json, _read
from variable_response_protocol import (
    ancestry,
    canonical,
    digest,
    read,
    save,
    marker,
    tree_ok,
)


def disk_free(parent: Path) -> int:
    return shutil.disk_usage(parent).free


def prior_markers(repo: Path, parent: Path) -> None:
    audits = [
        (
            parent / "variable-response-view-v1-attempt.json",
            "cff83243898a1141eeaf35ebd0d8fe90364441f1c21e474efa725888abe7de70",
        )
    ]
    gate = read(
        repo
        / "experiments/evidence/variable-width-gate-development-v1/outcome-audit/outcome-audit.json"
    )
    if (
        gate.get("saved_outcome_audit_approved") is not True
        or gate.get("private_gate") != "closed"
    ):
        raise ValueError("prior gate audit must remain closed and approved")
    audits.append(
        (
            parent / "variable-width-gate-development-v1-attempt.json",
            gate["marker_sha256"],
        )
    )
    for path, pin in audits:
        if digest(path) != pin:
            raise ValueError("consumed prior marker changed or missing")


def _lines(
    path: Path, maximum: int, record_cap: int, count_cap: int
) -> list[dict[str, Any]]:
    blob = _read(path, maximum)
    if blob and not blob.endswith(b"\n"):
        raise ValueError("partial final JSONL record")
    lines = blob.splitlines()
    if len(lines) > count_cap or any(len(line) + 1 > record_cap for line in lines):
        raise ValueError("JSONL count/record cap exceeded")
    return [_json(line) for line in lines]


def validate_case(
    root: Path,
    outcome: dict[str, Any],
    width: int,
    binding: dict[str, str] | None = None,
    data_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Bind saved NEW trace/metrics/inputs, without RNG/optimization or teacher replay."""
    status = outcome.get("status")
    if canonical(read(root / "terminal.json")) != canonical(outcome):
        raise ValueError("case terminal binding changed")
    if status == "abstained":
        if {p.name for p in root.iterdir()} != {"terminal.json"} or outcome.get(
            "updates_completed"
        ) != 0:
            raise ValueError("abstained case must have no RNG/numeric artifacts")
        return {"terminal.json": digest(root / "terminal.json")}
    if (
        status not in ("completed", "numerical_failure")
        or binding is None
        or data_hashes is None
    ):
        raise ValueError("explicit bound case outcome required")
    specs = {s.name: s for s in trace_specs(width)}
    manifest = _lines(root / "manifest.jsonl", 31 * 4096, 4096, 31)
    names = set()
    pins = {}
    for record in manifest:
        if (
            set(record) != {"name", "bytes", "sha256"}
            or record["name"] not in specs
            or record["name"] in names
            or type(record["bytes"]) is not int
            or not 0 < record["bytes"] <= specs[record["name"]].encoded_bound
        ):
            raise ValueError("exact unique bounded trace manifest required")
        name = record["name"]
        path = root / (name + ".npz")
        ancestry(path)
        if (
            path.stat().st_size != record["bytes"]
            or digest(path, 8 * 1024**2) != record["sha256"]
        ):
            raise ValueError("saved trace hash/size changed")
        names.add(name)
        pins[path.name] = record["sha256"]
    actual = {p.name for p in root.iterdir() if p.suffix == ".npz"}
    if actual != {n + ".npz" for n in names}:
        raise ValueError("numeric trace/manifest inventory mismatch")
    if status == "completed" and names != set(specs) - {"failure-state"}:
        raise ValueError("completed trace inventory incomplete")
    if status == "numerical_failure" and (
        "failure-state" not in names or outcome.get("selected_step") is not None
    ):
        raise ValueError(
            "numerical prefix must preserve finite failure state and selectedNone"
        )
    progress = _lines(root / "progress.jsonl", 4000 * 512, 512, 4000)
    for i, row in enumerate(progress, 1):
        loss = row.get("sampled_loss_before_update")
        if (
            set(row) != {"step", "sampled_loss_before_update"}
            or type(row["step"]) is not int
            or row["step"] != i
            or not isinstance(loss, (int, float))
            or isinstance(loss, bool)
            or not math.isfinite(loss)
            or loss < 0
        ):
            raise ValueError("finite ordered sampled loss progress required")
    if (
        type(outcome.get("updates_completed")) is not int
        or len(progress) != outcome["updates_completed"]
    ):
        raise ValueError("case completed update count differs from saved progress")
    metrics = read(root / "metrics.json")
    journal = _lines(root / "checkpoint-metrics.jsonl", 17 * 4096, 4096, 17)
    intents = _lines(root / "checkpoint-intents.jsonl", 17 * 4096, 4096, 17)
    if (
        metrics.get("schema") != "joint-response-metrics-v1"
        or metrics.get("guard_is_consumed_calibration") is not True
        or canonical(metrics.get("checkpoints")) != canonical(journal)
        or metrics.get("checkpoint_metric_records") != len(journal)
        or metrics.get("metric_journal_bytes")
        != (root / "checkpoint-metrics.jsonl").stat().st_size
        or len(intents) != len(journal)
    ):
        raise ValueError("saved metric journal/intent/terminal binding mismatch")
    for i, (row, intent) in enumerate(zip(journal, intents, strict=True)):
        if (
            row.get("step") != CHECKPOINTS[i]
            or type(row.get("step")) is not int
            or intent != {"step": CHECKPOINTS[i]}
            or f"checkpoint-{CHECKPOINTS[i]:04d}" not in names
            or type(row.get("guard_mse")) not in (int, float)
            or not math.isfinite(row["guard_mse"])
            or row["guard_mse"] < 0
        ):
            raise ValueError("canonical finite checkpoint records required")
    if status == "completed":
        if len(journal) != 17 or len(progress) != 4000:
            raise ValueError("completed budget/checkpoints incomplete")
        best = min(journal, key=lambda x: x["guard_mse"])
        if (
            metrics.get("selected_step") != best["step"]
            or outcome.get("selected_step") != best["step"]
            or metrics.get("selected_guard_mse") != best["guard_mse"]
            or outcome.get("selected_guard_mse") != best["guard_mse"]
        ):
            raise ValueError("earliest finite guard selection binding mismatch")
    elif (
        metrics.get("selected_step") is not None
        or metrics.get("selected_guard_mse") is not None
    ):
        raise ValueError("failed case cannot select prefix")
    case = read(root / "case.json")
    if (
        case.get("schema") != "joint-response-case-v1"
        or case.get("job_id") != outcome["job_id"]
        or case.get("width_hypothesis") != width
        or case.get("width_is_hypothesis") is not True
        or case.get("guard_is_consumed_calibration") is not True
        or any(case.get(k) != v for k, v in data_hashes.items())
    ):
        raise ValueError("case decoded-observation/width metadata binding mismatch")
    rng = read(root / "rng.json")
    if (
        set(rng) != {"schema", "initial", "batches"}
        or rng["schema"] != "joint-response-rng-v1"
    ):
        raise ValueError("exact RNG metadata schema required")
    states = []
    for role in ("initial", "batches"):
        entry = rng[role]
        if (
            not isinstance(entry, dict)
            or set(entry) != {"seed", "before", "after"}
            or type(entry["seed"]) is not int
            or entry["seed"] != stream_seed(outcome["job_id"], role)
        ):
            raise ValueError("paired RNG seed metadata changed")
        for when in ("before", "after"):
            text = canonical(entry[when]).decode()
            _state(text)
            states.append(text)
    if hashlib.sha256(canonical(states)).hexdigest() != binding["states"]:
        raise ValueError("saved RNG state binding changed")
    for artifact, fields in [("initial", ("g", "u", "d")), ("schedule", ("indices",))]:
        saved = read_typed(root / (artifact + ".npz"), specs[artifact])
        for field in fields:
            key = field if artifact == "initial" else "schedule"
            if (
                hashlib.sha256(memoryview(saved[field]).cast("B")).hexdigest()
                != binding[key]
            ):
                raise ValueError(
                    "saved initial/schedule bytes differ from prepared inputs"
                )
    for path in root.iterdir():
        ancestry(path)
        if path.suffix != ".npz":
            pins[path.name] = digest(path, 2 * 1024**2)
    return pins


def run_protocol(
    repo: Path,
    output: Path,
    config: Path,
    review: Path,
    *,
    engine: Callable[..., dict[str, Any]] = run_json,
) -> dict[str, Any]:
    started = time.monotonic()
    _, head = verify_source(repo, config, review)
    ancestry(output)
    parent = repo.parent / "experiments"
    ancestry(parent)
    if output.absolute().parent != parent.absolute():
        raise ValueError("durable direct experiment child required")
    parent.mkdir(exist_ok=True)
    attempt = parent / MARKER
    if attempt.exists() or attempt.is_symlink():
        raise ValueError("consumed marker; no retry")
    prior_markers(repo, parent)
    output.mkdir(exist_ok=False)
    (output / "cases").mkdir()
    report: dict[str, Any] = dict(
        status="started",
        source_commit=head,
        marker_consumed=False,
        config_sha256=digest(config),
        review_sha256=digest(review),
        new_queries=0,
        case_driver_invocations=0,
        paired_binding_checks={str(i): "not_compared" for i in range(3)},
        certificate=False,
        private_gate="closed",
        scientific_reconstruction_success=False,
        automatic_retry=False,
        uncommitted_tail="unknown_after_interruption",
        cases=[
            dict(job_id=f"job{i:02d}", status="not_started", updates_completed=None)
            for i in range(6)
        ],
    )
    save(output / "controller-report.json", report)
    view = None

    def remaining() -> float:
        value = 180 - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("shared180s expired")
        return value

    def disk() -> None:
        value = disk_free(parent)
        if type(value) is not int or value < 2 * 1024**3:
            raise ValueError("actual2GiBfree disk required")
        report["last_free_bytes"] = value

    def unchanged() -> None:
        _, now = verify_source(repo, config, review)
        if (
            now != head
            or digest(config) != report["config_sha256"]
            or digest(review) != report["review_sha256"]
        ):
            raise ValueError("source/config/review freeze changed")

    try:
        disk()
        resource = engine(
            [
                sys.executable,
                str(repo / "examples/isolated_glu/joint_response_controller.py"),
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
            raise ValueError("real literal preflight TREE receipt required")
        validate_transport(output / "transport", resource)
        layout(output)
        disk()
        unchanged()
        marker_value = dict(
            id=POLICY["id"],
            source_commit=head,
            config_sha256=report["config_sha256"],
            review_sha256=report["review_sha256"],
            output=str(output.absolute()),
            policy=POLICY,
            preflight_sha256=digest(output / "preflight-receipt.json"),
        )
        report["marker_sha256"] = marker(attempt, marker_value)
        report["marker_consumed"] = True
        save(output / "controller-report.json", report)
        data = parent / PARENT
        view = load_join(
            repo,
            data / "collection/public",
            data / "estimator",
            data / "collection/truth",
        )
        jobs = view.jobs
        if (
            len(jobs) != 6
            or any(
                j.job_id != f"job{i:02d}"
                or j.n != 32
                or j.outputs != 32
                or j.batch_count != 34
                for i, j in enumerate(jobs)
            )
            or any(
                jobs[i].width_hypothesis != jobs[i + 1].width_hypothesis
                for i in (0, 2, 4)
            )
        ):
            raise ValueError("canonical six paired numeric jobs required")
        save(
            output / "input-binding.json",
            dict(
                view=view.report(),
                widths_hypothesized=[j.width_hypothesis for j in jobs],
                source_commit=head,
            ),
        )
        pairs: dict[int, dict[str, str]] = {}
        final_pins = {}
        for i, job in enumerate(jobs):
            remaining()
            disk()
            layout(output)
            report["cases"][i].update(
                status="running", width_hypothesis=job.width_hypothesis
            )
            save(output / "controller-report.json", report)
            case_root = output / "cases" / job.job_id
            try:
                case = assemble_case(job)
            except (FloatingPointError, OverflowError) as exc:
                case_root.mkdir()
                outcome = dict(
                    status="abstained",
                    job_id=job.job_id,
                    updates_completed=0,
                    selected_step=None,
                    reason=type(exc).__name__,
                    phase="normalization",
                    private_gate="closed",
                    certificate=False,
                )
                save(case_root / "terminal.json", outcome)
                final_pins[job.job_id] = validate_case(
                    case_root, outcome, job.width_hypothesis
                )
            except ValueError as exc:
                if str(exc) != "degenerate training RMS":
                    raise
                case_root.mkdir()
                outcome = dict(
                    status="abstained",
                    job_id=job.job_id,
                    updates_completed=0,
                    selected_step=None,
                    reason="degenerate_training_rms",
                    phase="normalization",
                    private_gate="closed",
                    certificate=False,
                )
                save(case_root / "terminal.json", outcome)
                final_pins[job.job_id] = validate_case(
                    case_root, outcome, job.width_hypothesis
                )
            else:
                remaining()
                inputs = prepare_inputs(job, case, PCG64Stream)
                binding = optimizer_binding(inputs)
                if i % 2 == 0:
                    pairs[i // 2] = binding
                elif i // 2 not in pairs:
                    report["paired_binding_checks"][str(i // 2)] = (
                        "unavailable_first_case_abstained"
                    )
                elif pairs[i // 2] != binding:
                    raise ValueError(
                        "paired normalized initialization/schedule/state mismatch"
                    )
                if i % 2 == 1 and i // 2 in pairs:
                    report["paired_binding_checks"][str(i // 2)] = "matched"
                hashes = dict(
                    input_array_sha256=hashlib.sha256(
                        memoryview(case.inputs).cast("B")
                    ).hexdigest(),
                    response_array_sha256=hashlib.sha256(
                        memoryview(case.responses).cast("B")
                    ).hexdigest(),
                )
                store = TraceStore(case_root, job.width_hypothesis)
                report["case_driver_invocations"] += 1
                save(output / "controller-report.json", report)
                outcome = run_case(job, case, inputs, store)
                final_pins[job.job_id] = validate_case(
                    case_root, outcome, job.width_hypothesis, binding, hashes
                )
                del store, inputs, case
            report["cases"][i] = outcome
            report["elapsed_before_progress_write_seconds"] = time.monotonic() - started
            save(
                output / "progress.json",
                dict(cases=report["cases"], completed_case_visits=i + 1),
            )
            save(output / "controller-report.json", report)
            layout(output)
            remaining()
        report["view_completion"] = view.finish()
        if report["view_completion"].get("status") != "completed":
            raise ValueError("response completion hash check failed")
        unchanged()
        prior_markers(repo, parent)
        if digest(attempt) != report["marker_sha256"]:
            raise ValueError("consumed marker changed")
        validate_transport(output / "transport", resource)
        disk()
        remaining()
        save(
            output / "terminal-binding.json",
            dict(
                case_files=final_pins,
                source_commit=head,
                marker_sha256=report["marker_sha256"],
                paired_optimizer_bindings=pairs,
            ),
        )
        report.update(
            status="completed",
            uncommitted_tail=False,
            completed_cases=sum(c["status"] == "completed" for c in report["cases"]),
            numerical_failures=sum(
                c["status"] == "numerical_failure" for c in report["cases"]
            ),
            abstentions=sum(c["status"] == "abstained" for c in report["cases"]),
            elapsed_before_final_write_seconds=time.monotonic() - started,
        )
        save(output / "controller-report.json", report)
        layout(output)
        return report
    except Exception as exc:
        for item in report["cases"]:
            if item.get("status") == "running":
                item.update(status="interrupted_or_uncommitted", updates_completed=None)
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc)[:500],
            marker_consumed=attempt.exists(),
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
    repo, output = Path(request["repo"]), Path(request["output"])
    ancestry(repo)
    ancestry(output)
    if (
        output.exists()
        or output.absolute().parent != (repo.parent / "experiments").absolute()
        or (output.parent / MARKER).exists()
        or (output.parent / MARKER).is_symlink()
    ):
        raise ValueError("fresh durable output/unconsumed marker required")
    result: dict[str, Any] = dict(
        status="failed",
        certificate=False,
        private_gate="closed",
        automatic_retry=False,
        controller_counters_unknown=True,
        resource_receipt_unavailable=True,
    )
    started = time.monotonic()
    try:
        receipt = engine(
            [
                sys.executable,
                str(repo / "examples/isolated_glu/joint_response_controller.py"),
            ],
            dict(request, operation="controlled_run"),
            cwd=output.parent / (output.name + "-launch-engine"),
            memory_limit_mb=512,
            timeout_seconds=180,
            output_limit_bytes=1024,
            env=environment(repo),
        )
        result.update(receipt=receipt, resource_receipt_unavailable=False)
        if output.exists():
            save(output / "outer-monitor.json", result)
        child = read(output / "controller-report.json")
        if (
            not tree_ok(receipt)
            or receipt.get("status") != "completed"
            or child.get("status") != "completed"
        ):
            raise ValueError("complete default outer TREE/controller receipt required")
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
            status="failed", error_type=type(exc).__name__, error=str(exc)[:500]
        )
        if output.exists():
            save(output / "outer-monitor.json", result)
        raise


if __name__ == "__main__":
    try:
        blob = sys.stdin.buffer.read(METADATA_CAP + 1)
        if len(blob) > METADATA_CAP:
            raise ValueError("request exceeds bounded cap")
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
            raise ValueError("unsupported request operation/schema")
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
