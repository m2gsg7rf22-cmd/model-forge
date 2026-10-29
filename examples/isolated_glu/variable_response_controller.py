"""One-shot read-only calibration validation. Full review/clean freeze precede IO.

No teacher factory, query, model, optimizer, RNG, private asset or heldout route.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import psutil
from modelforge.execution import run_json
from variable_response_inventory import inventory
from variable_response_manifest import (
    ENCODED_CAP,
    METADATA_CAP,
    PUBLIC_PINS,
    SOURCE,
    ArrayDescriptor,
    TranscriptManifest,
    _json,
    _read,
    load_manifest,
)
from variable_response_protocol import (
    MARKER,
    POLICY,
    DeliveryChain,
    ancestry,
    canonical,
    digest,
    marker,
    read,
    save,
    tree_ok,
)
from variable_response_view import ResponseView, ViewValidationError, _array, open_view

PARENT = "paired-row-acquisition-v1-20261028-07235ac"
PARENT_SOURCE = SOURCE
COMPONENT = "experiments/evidence/variable-response-adapter-v1/source-review.json"
COMPONENT_SHA = "eb12dadf7432583470afac56becad2c7d7d308caf0156a924e410cc893806543"
GATE_AUDIT = "experiments/evidence/variable-width-gate-development-v1/outcome-audit/outcome-audit.json"
GATE_AUDIT_SHA = "e03cb21debfd87e41e58175c3050beb88b64e8a632242846c57be9b81bfd4a26"
DEPENDENCIES = {
    "examples/isolated_glu/" + n + ".py"
    for n in (
        "variable_response_manifest",
        "variable_response_view",
        "variable_response_inventory",
        "variable_response_protocol",
        "variable_response_controller",
    )
} | {"src/modelforge/__init__.py", "src/modelforge/execution.py"}
REQUIRED_REVIEW = DEPENDENCIES | {
    "tests/test_variable_response_view.py",
    "tests/test_variable_response_protocol.py",
    "docs/glu-variable-width-response-view-design.md",
    "docs/glu-variable-width-response-adapter.md",
    "docs/glu-variable-width-response-read-protocol.md",
    "docs/superpowers/plans/2026-10-29-variable-response-adapter.md",
    "docs/superpowers/plans/2026-10-29-variable-response-read-protocol.md",
    COMPONENT,
    GATE_AUDIT,
    *(rel for _, rel, _ in PUBLIC_PINS),
}


def config_template(repo: Path) -> dict[str, Any]:
    return dict(
        id=POLICY["id"],
        policy=POLICY,
        inventory=inventory(),
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            psutil=psutil.__version__,
        ),
        frozen_dependencies={
            n: digest(repo / n, 2 * 1024**2) for n in sorted(DEPENDENCIES)
        },
        parent_directory=PARENT,
        parent_source=PARENT_SOURCE,
        component_review_sha256=COMPONENT_SHA,
        gate_outcome_audit_sha256=GATE_AUDIT_SHA,
        public_metadata={name: h for name, _, h in PUBLIC_PINS},
    )


def verify_source(repo: Path, config: Path, review: Path) -> tuple[dict[str, Any], str]:
    ancestry(repo)
    value, approval = read(config), read(review)
    if approval.get("full_variable_response_read_protocol_approved") is not True:
        raise ValueError("Complete independent read protocol approval required")
    if canonical(value) != canonical(config_template(repo)):
        raise ValueError(
            "Exact policy/runtime/inventory/input/dependency pins required"
        )
    relative = str(config.relative_to(repo))
    review.relative_to(repo)
    signed = approval.get("source_sha256")
    if not isinstance(signed, dict) or not REQUIRED_REVIEW | {relative} <= set(signed):
        raise ValueError("Incomplete reviewed source set")
    for name, h in signed.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or digest(repo / p, 2 * 1024**2) != h:
            raise ValueError("Reviewed source hash mismatch")
    if any(signed[n] != h for n, h in value["frozen_dependencies"].items()):
        raise ValueError("Review/runtime dependency mismatch")
    if (
        digest(repo / COMPONENT) != COMPONENT_SHA
        or digest(repo / GATE_AUDIT) != GATE_AUDIT_SHA
    ):
        raise ValueError("Prior component/outcome approval changed")
    component = read(repo / COMPONENT)
    if component.get("component_source_approved") is not True:
        raise ValueError("Prior component approval required")
    for name, h in component["source_sha256"].items():
        if signed.get(name) != h:
            raise ValueError("Prior signed component bytes changed")
    for name, rel, h in PUBLIC_PINS:
        if digest(repo / rel, 2 * 1024**2) != h:
            raise ValueError("Public metadata hash changed: " + name)

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, timeout=10
        ).strip()

    if (
        git("status", "--porcelain")
        or git("branch", "--show-current") != "discovery-lab"
    ):
        raise ValueError("Clean committed discovery-lab required")
    return value, git("rev-parse", "HEAD")


def synthetic_transport(output: Path) -> dict[str, Any]:
    ancestry(output)
    output.mkdir(exist_ok=False)
    records = []
    for field in ("inputs", "responses"):
        a = np.full((2112, 32), 0.0 if field == "inputs" else 1.0, dtype=np.float64)
        path = output / (field + ".npz")
        with path.open("xb") as f:
            fields: dict[str, Any] = {field: a}
            np.savez(f, **fields)
        d = ArrayDescriptor(path.name, field, (2112, 32), digest(path, ENCODED_CAP))
        restored = _array(_read(path, ENCODED_CAP), d)
        if not np.array_equal(a, restored):
            raise ValueError("Literal transport roundtrip failed")
        records.append(dict(**asdict(d), bytes=path.stat().st_size))
        del a, restored
    save(output / "manifest.json", dict(records=records), 4096)
    time.sleep(0.15)
    return dict(
        status="passed",
        manifest_sha256=digest(output / "manifest.json", 4096),
        shards=2,
        new_queries=0,
    )


def validate_transport(output: Path, receipt: dict[str, Any]) -> None:
    if (
        receipt.get("status") != "passed"
        or type(receipt.get("shards")) is not int
        or receipt["shards"] != 2
    ):
        raise ValueError("Literal transport incomplete")
    manifest = output / "manifest.json"
    if receipt.get("manifest_sha256") != digest(manifest, 4096):
        raise ValueError("Transport manifest changed")
    records = read(manifest, 4096).get("records")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("Exact transport inventory required")
    for record, field in zip(records, ("inputs", "responses"), strict=True):
        if (
            not isinstance(record, dict)
            or set(record) != {"name", "field", "shape", "digest", "bytes"}
            or record["name"] != field + ".npz"
            or record["field"] != field
            or record["shape"] != [2112, 32]
            or type(record["bytes"]) is not int
            or not 0 < record["bytes"] <= ENCODED_CAP
        ):
            raise ValueError("Transport descriptor mismatch")
        path = output / record["name"]
        if (
            path.stat().st_size != record["bytes"]
            or digest(path, ENCODED_CAP) != record["digest"]
        ):
            raise ValueError("Transport hash/size mismatch")
        a = _array(
            _read(path, ENCODED_CAP),
            ArrayDescriptor(path.name, field, (2112, 32), record["digest"]),
        )
        if not np.all(a == (0.0 if field == "inputs" else 1.0)):
            raise ValueError("Literal transport values changed")
        del a
    if {p.name for p in output.iterdir()} != {
        "inputs.npz",
        "responses.npz",
        "manifest.json",
    }:
        raise ValueError("Unexpected literal transport output")
    if (
        sum(p.stat().st_size for p in output.iterdir())
        > inventory()["transport_upper_bound"]
    ):
        raise ValueError("Literal transport exceeds bound")


def validate_view(value: dict[str, Any], manifest: TranscriptManifest) -> None:
    p = manifest.profile
    pairs = len(manifest.pairs)
    archives = 2 * pairs
    if (
        value.get("status") != "completed"
        or value.get("certificate") is not False
        or value.get("private_gate") != "closed"
        or value.get("error") is not None
    ):
        raise ValueError("Completed uncertified view required")
    for key in ("new_queries", "new_fits"):
        if type(value.get(key)) is not int or value[key] != 0:
            raise ValueError("No queries/fits permitted")
    for phase, arrays, metadata in [
        ("validation", archives, 4),
        ("delivery", archives, 0),
        ("completion", 0, 4),
    ]:
        c = value.get(phase, {})
        expected = dict(
            archives_attempted=archives,
            archives_read=archives,
            arrays_validated=arrays,
            metadata_attempted=metadata,
            metadata_read=metadata,
        )
        if any(type(c.get(k)) is not int or c[k] != v for k, v in expected.items()):
            raise ValueError("Native exact view pass counters required")
        if (
            type(c.get("bytes_read")) is not int
            or not 0 < c["bytes_read"] <= archives * ENCODED_CAP
        ):
            raise ValueError("Bounded complete read bytes required")
    for phase in ("validation", "completion"):
        n = value[phase].get("metadata_bytes")
        if type(n) is not int or not 0 < n <= 2 * 1024**2 + 3 * METADATA_CAP:
            raise ValueError("Bounded complete metadata byte count required")
    if value["validation"]["metadata_bytes"] != value["completion"]["metadata_bytes"]:
        raise ValueError("Public metadata byte counts differ")
    if (
        type(value["delivery"].get("metadata_bytes")) is not int
        or value["delivery"]["metadata_bytes"] != 0
    ):
        raise ValueError("No metadata read counted as delivery")
    if (
        len(
            {
                value[phase]["bytes_read"]
                for phase in ("validation", "delivery", "completion")
            }
        )
        != 1
    ):
        raise ValueError("Read pass byte counts differ")
    delivery = value["delivery"]
    for key, v in dict(
        pairs=pairs,
        rows=p.vector_rows,
        payload_bytes=p.vector_rows * (p.n + p.outputs) * 8,
    ).items():
        if type(delivery.get(key)) is not int or delivery[key] != v:
            raise ValueError("Delivered data counters mismatch")
    jobs = value.get("jobs", {})
    if set(jobs) != {f"job{i:02d}" for i in range(len(p.case_names))}:
        raise ValueError("All registered jobs required")
    for job in jobs.values():
        if (
            job.get("status") != "completed"
            or type(job.get("delivered_batches")) is not int
            or job["delivered_batches"] != 2 * p.probes
        ):
            raise ValueError("Incomplete delivered job")
    if (
        value.get("mapping")
        != {f"job{i:02d}": name for i, name in enumerate(p.case_names)}
        or value.get("metadata_sha256") != {x.name: x.digest for x in manifest.metadata}
        or type(value.get("allowed_archives")) is not int
        or value["allowed_archives"] != archives
        or type(value.get("raw_payload_bound")) is not int
        or value["raw_payload_bound"] != p.vector_rows * (p.n + p.outputs) * 8
    ):
        raise ValueError("View provenance mismatch")


def consume(
    view: ResponseView, output: Path, remaining: Callable[[], float]
) -> dict[str, Any]:
    count = view.jobs[0].batch_count
    chain = DeliveryChain(output / "delivery.jsonl", count)
    for job in view.jobs:
        for index, batch in enumerate(job.batches):
            remaining()
            if (
                batch.inputs.shape[1] != job.n
                or batch.responses.shape[1] != job.outputs
                or batch.inputs.flags.writeable
                or batch.responses.flags.writeable
                or not np.isfinite(batch.inputs).all()
                or not np.isfinite(batch.responses).all()
            ):
                raise ValueError("Immutable finite matching pair required")
            chain.append(
                job.job_id,
                index,
                batch.inputs.shape,
                batch.responses.shape,
                hashlib.sha256(batch.inputs.tobytes()).hexdigest(),
                hashlib.sha256(batch.responses.tobytes()).hexdigest(),
            )
            del batch
            save(
                output / "progress.json",
                dict(
                    status="delivering",
                    committed_pairs=chain.count,
                    chain_head=chain.head,
                    view=view.report(),
                    uncommitted_delivery="unknown_after_interruption",
                ),
            )
    result = view.finish()
    save(output / "view-report.json", result)
    return dict(view=result, delivery_chain=chain.verify())


def layout(output: Path) -> int:
    allowed = {
        "controller-report.json",
        "scope-receipt.json",
        "transport-receipt.json",
        "descriptor-seal.json",
        "progress.json",
        "view-report.json",
        "delivery.jsonl",
        "outer-monitor.json",
    }
    total = 0
    for p in output.iterdir():
        ancestry(p)
        if p.name in ("scope-engine", "transport-engine"):
            if not p.is_dir() or list(p.iterdir()):
                raise ValueError("Nonempty/unexpected engine directory")
        elif p.name == "transport":
            if not p.is_dir():
                raise ValueError("Transport directory required")
            total += sum(x.stat().st_size for x in p.iterdir())
        elif (
            p.name not in allowed or not p.is_file() or p.stat().st_size > METADATA_CAP
        ):
            raise ValueError("Unexpected/oversized protocol output")
        else:
            total += p.stat().st_size
    required = allowed - {"outer-monitor.json"}
    if not required <= {p.name for p in output.iterdir()}:
        raise ValueError("Required protocol receipt missing")
    if total > inventory()["new_artifacts_upper_bound"]:
        raise ValueError("Protocol output exceeds total inventory")
    return total


def _environment(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        PYTHONPATH=str(repo / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def run_protocol(
    repo: Path,
    output: Path,
    config: Path,
    review: Path,
    *,
    engine: Callable[..., dict[str, Any]] = run_json,
    _manifest_loader: Callable[[Path], TranscriptManifest] = load_manifest,
    _view_factory: Callable[[Path, TranscriptManifest], ResponseView] = open_view,
) -> dict[str, Any]:
    started = time.monotonic()
    _, head = verify_source(repo, config, review)
    ancestry(output)
    parent = repo.parent / "experiments"
    ancestry(parent)
    if output.absolute().parent != parent.absolute():
        raise ValueError("Durable direct experiment child required")
    parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    path = output / "controller-report.json"
    result: dict[str, Any] = dict(
        status="started",
        source_commit=head,
        marker_consumed=False,
        new_queries=0,
        new_fits=0,
        certificate=False,
        private_gate="closed",
        automatic_retry=False,
        uncommitted_delivery="unknown_after_interruption",
    )
    save(path, result)
    config_sha, review_sha = digest(config), digest(review)
    attempt = parent / MARKER
    view: ResponseView | None = None

    def remaining() -> float:
        seconds = 180 - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError("Shared180s expired")
        return seconds

    def unchanged() -> None:
        _, now = verify_source(repo, config, review)
        if now != head or digest(config) != config_sha or digest(review) != review_sha:
            raise ValueError("Source/config/review changed")

    try:
        if attempt.exists() or attempt.is_symlink():
            raise ValueError("Consumed global marker; no retry")
        scope = engine(
            [
                sys.executable,
                "-c",
                "import time,json;time.sleep(.15);print(json.dumps({'status':'passed'}))",
            ],
            {},
            cwd=output / "scope-engine",
            memory_limit_mb=512,
            timeout_seconds=remaining(),
            output_limit_bytes=2048,
            env=_environment(repo),
        )
        result["scope_receipt"] = scope
        save(output / "scope-receipt.json", scope)
        save(path, result)
        if not tree_ok(scope) or scope.get("status") != "passed":
            raise ValueError("Actual TREE preflight required")
        result["free_bytes_before"] = shutil.disk_usage(parent).free
        save(path, result)
        if (
            type(result["free_bytes_before"]) is not int
            or result["free_bytes_before"] < 2 * 1024**3
        ):
            raise ValueError("At least2GiB actual free disk required")
        transport = engine(
            [
                sys.executable,
                str(repo / "examples/isolated_glu/variable_response_controller.py"),
            ],
            dict(operation="transport", output=str(output / "transport")),
            cwd=output / "transport-engine",
            memory_limit_mb=512,
            timeout_seconds=remaining(),
            output_limit_bytes=2048,
            env=_environment(repo),
        )
        result["transport_receipt"] = transport
        save(output / "transport-receipt.json", transport)
        save(path, result)
        if not tree_ok(transport):
            raise ValueError("Actual TREE transport required")
        validate_transport(output / "transport", transport)
        unchanged()
        m = _manifest_loader(repo)  # Only public metadata, never a source transcript.
        seal = dict(
            pairs=[asdict(p) for p in m.pairs],
            public_metadata={x.name: x.digest for x in m.metadata},
        )
        if len(canonical(seal)) > inventory()["descriptor_seal_upper_bound"]:
            raise ValueError("Descriptor seal exceeds scalar inventory")
        save(output / "descriptor-seal.json", seal)
        value = dict(
            id=POLICY["id"],
            source_commit=head,
            output=str(output.absolute()),
            config_sha256=config_sha,
            review_sha256=review_sha,
            descriptor_sha256=digest(output / "descriptor-seal.json"),
            policy=POLICY,
        )
        result["free_bytes_before_marker"] = shutil.disk_usage(parent).free
        save(path, result)
        if (
            type(result["free_bytes_before_marker"]) is not int
            or result["free_bytes_before_marker"] < 2 * 1024**3
        ):
            raise ValueError("At least2GiB free disk still required before marker/data")
        result["marker_sha256"] = marker(attempt, value)
        result["marker_consumed"] = True
        save(path, result)
        view = _view_factory(parent / PARENT / "collection/truth", m)
        result.update(consume(view, output, remaining))
        validate_view(result["view"], m)
        if result["delivery_chain"]["pairs"] != len(m.pairs):
            raise ValueError("Delivery commitment count mismatch")
        unchanged()
        if digest(attempt) != result["marker_sha256"]:
            raise ValueError("Consumed marker changed")
        if digest(output / "descriptor-seal.json") != value["descriptor_sha256"]:
            raise ValueError("Descriptor seal changed after marker")
        if digest(output / "delivery.jsonl") != result["delivery_chain"]["sha256"]:
            raise ValueError("Saved delivery chain changed")
        if canonical(read(output / "view-report.json")) != canonical(result["view"]):
            raise ValueError("Saved view receipt changed")
        progress = read(output / "progress.json")
        if (
            type(progress.get("committed_pairs")) is not int
            or progress["committed_pairs"] != result["delivery_chain"]["pairs"]
            or progress.get("chain_head") != result["delivery_chain"]["head"]
        ):
            raise ValueError("Saved progress prefix mismatch")
        validate_transport(output / "transport", transport)
        result["free_bytes_after"] = shutil.disk_usage(parent).free
        result["elapsed_before_final_write_seconds"] = time.monotonic() - started
        remaining()
        result["status"] = "completed"
        result["uncommitted_delivery"] = False
        save(path, result)
        result["retained_bytes_before_final_report"] = layout(output)
        save(path, result)
        return result
    except Exception as exc:
        result.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)[:500]
        )
        result["marker_consumed"] = attempt.exists()
        if isinstance(exc, ViewValidationError):
            result["view"] = exc.report
        elif view is not None:
            result["view"] = view.report()
        save(path, result)
        raise


def launch(
    request: dict[str, Any], *, engine: Callable[..., dict[str, Any]] = run_json
) -> dict[str, Any]:
    if set(request) != {"repo", "output", "config", "review"} or any(
        not isinstance(x, str) for x in request.values()
    ):
        raise ValueError("Exact default launcher paths required")
    repo, output = Path(request["repo"]), Path(request["output"])
    ancestry(output)
    if (
        output.exists()
        or output.absolute().parent != (repo.parent / "experiments").absolute()
    ):
        raise ValueError("Fresh durable direct output required; no retry")
    if (output.parent / MARKER).exists():
        raise ValueError("Consumed global marker; no retry")
    result: dict[str, Any] = dict(
        status="failed",
        certificate=False,
        private_gate="closed",
        automatic_retry=False,
        controller_counters_unknown=True,
        resource_receipt_unavailable=True,
    )
    try:
        receipt = engine(
            [
                sys.executable,
                str(repo / "examples/isolated_glu/variable_response_controller.py"),
            ],
            dict(request, operation="controlled_run"),
            cwd=output.parent / (output.name + "-launch-engine"),
            memory_limit_mb=512,
            timeout_seconds=180,
            output_limit_bytes=2048,
            env=_environment(repo),
        )
        result["receipt"] = receipt
        result["resource_receipt_unavailable"] = False
        if output.exists():
            save(
                output / "outer-monitor.json", result
            )  # Preserve returned receipt before validation.
        child = read(output / "controller-report.json")
        if (
            not tree_ok(receipt)
            or receipt.get("status") != "completed"
            or child.get("status") != "completed"
        ):
            raise ValueError("Outer TREE or controller incomplete")
        result.update(
            status="completed",
            controller_counters_unknown=False,
            controller_sha256=digest(output / "controller-report.json"),
        )
        save(output / "outer-monitor.json", result)
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
            raise ValueError("Bounded request required")
        request = _json(blob)
        operation = request.pop("operation", None)
        if operation == "transport" and set(request) == {"output"}:
            result = synthetic_transport(Path(request["output"]))
        elif operation == "controlled_run" and set(request) == {
            "repo",
            "output",
            "config",
            "review",
        }:
            result = run_protocol(
                *(Path(request[k]) for k in ("repo", "output", "config", "review"))
            )
        elif operation is None:
            result = launch(request)
        else:
            raise ValueError("Unknown or invalid read-only operation")
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "manifest_sha256",
                        "shards",
                        "new_queries",
                        "new_fits",
                    )
                    if k in result
                }
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                dict(
                    status="failed", error_type=type(exc).__name__, error=str(exc)[:500]
                )
            )
        )
