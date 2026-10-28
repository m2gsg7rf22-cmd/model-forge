"""Source-reviewed one-attempt variable-width finite-subspace protocol.

This is an engineering controller, not a mathematical reconstruction certificate.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from modelforge.execution import run_json
from fresh_rank_common import (
    MARKER as MARKER,
    POLICY,
    canonical,
    read_arrays,
    read_json,
    records,
    save_json,
    sha,
    tree_ok,
    verify_manifest,
)
from fresh_rank_protocol_inventory import inventory_plan, profile_schemas, worst_schemas
from fresh_rank_view import FDView
from fresh_rank_estimation import registered_view
from fresh_rank_evaluation import validate_public_seal
from scaling_domain_store import _ancestry, ShardStore

DEPENDENCIES = {
    "examples/isolated_glu/" + name + ".py"
    for name in (
        "fresh_rank_common",
        "fresh_rank_collection",
        "fresh_rank_estimation",
        "fresh_rank_evaluation",
        "fresh_rank_controller",
        "fresh_rank_protocol_inventory",
        "fresh_rank_inventory",
        "fresh_rank_view",
        "rank_selection",
        "scaling_domain_math",
        "scaling_domain_store",
        "precision",
    )
} | {
    "src/modelforge/" + name + ".py"
    for name in (
        "__init__",
        "execution",
        "glu32_io",
        "glu32",
        "antipodal",
    )
}
REQUIRED_REVIEW = DEPENDENCIES | {
    "tests/test_glu_fresh_rank_collection.py",
    "tests/test_glu_fresh_rank_stages.py",
    "tests/test_glu_fresh_rank_controller.py",
    "docs/glu-fresh-rank-protocol.md",
    "docs/glu-rank-fresh-width-validation-draft.md",
    "docs/superpowers/plans/2026-10-28-fresh-width-protocol.md",
}

STORE_PATHS = dict(
    truth="collection/truth",
    public="collection/public/arrays",
    estimator="estimator/arrays",
    evaluator="evaluation/arrays",
    transport="transport/arrays",
)
METADATA_PATHS = {
    "controller-report.json",
    "collection/collection-report.json",
    "collection/registration.json",
    "collection/query-progress.json",
    "collection/seed-commitment.json",
    "collection/truth-commitment.json",
    "collection/public/cases.json",
    "estimator/choices-sealed.json",
    "estimator/estimator-report.json",
    "evaluation/evaluation-report.json",
}


def exact_layout(root: Path) -> None:
    expected = set(METADATA_PATHS)
    for directory in STORE_PATHS.values():
        expected.add(directory + "/manifest.jsonl")
        expected.update(directory + "/" + name for name in records(root / directory))
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    allowed_dirs = {
        str(parent)
        for name in expected
        for parent in Path(name).parents
        if str(parent) != "."
    }
    allowed_dirs.update(
        stage + "-engine"
        for stage in ("scope", "transport", "collection", "estimation", "evaluation")
    )
    dirs = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir()}
    if actual != expected or not dirs <= allowed_dirs:
        raise ValueError("Exact final artifact layout required")


class ProtocolError(ValueError):
    """Terminal, possibly partial; the global attempt is never retried."""


def signed_path(repo: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Signed path must stay inside repository")
    result = repo / path
    _ancestry(result)
    return result


def verify_source(
    repo: Path, config_path: Path, review_path: Path
) -> tuple[dict[str, Any], str]:
    config, review = read_json(config_path), read_json(review_path)
    if review.get("full_fresh_protocol_approved") is not True:
        raise ValueError("Full independent source/protocol approval required")
    if (
        config.get("policy") != POLICY
        or config.get("inventory_plan") != inventory_plan()
        or config.get("runtime")
        != dict(python=platform.python_version(), numpy=np.__version__)
    ):
        raise ValueError("Exact policy/inventory/runtime mismatch")
    pins, signed = (
        config.get("frozen_dependencies", {}),
        review.get("source_sha256", {}),
    )
    config_name = str(config_path.relative_to(repo))
    if (
        set(pins) != DEPENDENCIES
        or not (REQUIRED_REVIEW | {config_name}) <= set(signed)
        or any(signed.get(k) != h for k, h in pins.items())
        or signed.get(config_name) != sha(config_path)
    ):
        raise ValueError("Review must bind complete runtime/preregistration/tests/docs")
    if any(sha(signed_path(repo, k)) != h for k, h in signed.items()):
        raise ValueError("Reviewed bytes changed")

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


def synthetic_transport(output: Path) -> dict[str, Any]:
    """Literal zeros only: seven maximum archive families, no RNG or teacher."""
    output.mkdir(exist_ok=False)
    plan = inventory_plan()
    store = ShardStore(
        output / "arrays", total_limit_bytes=plan["preflight_bytes_upper_bound"]
    )
    specs = worst_schemas()
    for spec in specs:
        fields = {k: np.zeros(shape) for k, shape in spec.fields.items()}
        descriptor = store.save(spec.kind + ".npz", **fields)
        restored = read_arrays(
            store.root / descriptor["name"], dict(spec.fields), descriptor["sha256"]
        )
        if any(not np.array_equal(restored[k], a) for k, a in fields.items()):
            raise ValueError("Maximum literal transport mismatch")
    actual = verify_manifest(store.root)
    total = sum(p.stat().st_size for p in store.root.iterdir())
    if actual["shards"] != 7 or total > plan["preflight_bytes_upper_bound"]:
        raise ValueError("Maximum transport inventory/cap mismatch")
    time.sleep(0.15)
    return dict(
        status="passed",
        shards=7,
        bytes=total,
        new_queries=0,
        manifest_sha256=sha(store.root / "manifest.jsonl"),
    )


def check_counts(report: dict[str, Any], source: str) -> None:
    if (
        report.get("status") != "completed"
        or report.get("source_commit") != source
        or any(
            type(report.get(k)) is not int or report[k] != 545280
            for k in (
                "calls_reserved",
                "calls_attempted",
                "calls_returned",
                "calls_persisted",
            )
        )
        or type(report.get("callback_batches")) is not int
        or report["callback_batches"] != 768
        or report.get("certificate") is not False
        or report.get("private_gate") != "closed"
    ):
        raise ValueError("Complete source-bound collection counts required")


def stage_report(
    root: Path, stage: str, receipt: dict[str, Any], source: str
) -> dict[str, Any]:
    path = root / stage / (stage + "-report.json")
    if sha(path) != receipt.get("report_sha256"):
        raise ValueError("Stage report digest mismatch")
    report = read_json(path)
    if report.get("status") != "completed" or report.get("source_commit") != source:
        raise ValueError("Incomplete/wrong-source stage report")
    return report


def public_boundary(
    root: Path, collection: dict[str, Any], est: dict[str, Any]
) -> dict[str, Any]:
    directory = root / "estimator"
    if sha(directory / "choices-sealed.json") != est.get("seal_sha256"):
        raise ValueError("Complete public seal required")
    seal = read_json(directory / "choices-sealed.json")
    inp = seal["input_commitments"]
    if (
        inp["cases_sha256"] != collection["public_cases_sha256"]
        or inp["manifest_sha256"] != collection["public_manifest_sha256"]
    ):
        raise ValueError("Seal did not bind original public collection")
    view = FDView(
        root / "collection/public", inp["cases_sha256"], inp["manifest_sha256"]
    )
    registered_view(view)
    if len(view.records) != 144 or set(inp["used_input_sha256"]) != set(view.records):
        raise ValueError("All144publicinputs required")
    for name, digest in inp["used_input_sha256"].items():
        if sha(view.arrays / name) != digest or view.records[name]["sha256"] != digest:
            raise ValueError("Public source changed before reveal")
    validate_public_seal(directory, view, seal, est)
    return seal


def exact_store(
    root: Path, limits: dict[str, Any], schemas: dict[str, dict[str, tuple[int, ...]]]
) -> None:
    rec = records(root)
    actual = verify_manifest(root)
    if (
        set(rec) != set(schemas)
        or actual["shards"] != limits["shards"]
        or any(v["bytes"] > limits["shard_limit_bytes"] for v in rec.values())
        or (root / "manifest.jsonl").stat().st_size
        > limits["manifest_bytes_upper_bound"]
        or sum(v["bytes"] for v in rec.values())
        + (root / "manifest.jsonl").stat().st_size
        > limits["bytes_upper_bound"]
    ):
        raise ValueError("Exact store inventory/caps required")
    for name, schema in schemas.items():
        read_arrays(root / name, schema, rec[name]["sha256"])


def validate_final(
    root: Path,
    collection: dict[str, Any],
    est: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    # Privileged schema inspection occurs only after all public choices sealed.
    public_boundary(root, collection, est)
    mapping = read_json(root / "collection/registration.json")["mapping"]
    stores: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {
        k: {} for k in inventory_plan()["stores"]
    }
    for i, teacher in enumerate(mapping):
        schema = {
            s.kind: dict(s.fields)
            for s in profile_schemas(teacher["n"], teacher["width"])
        }
        for channel in ("fp64", "fp32"):
            case = teacher["id"] + "-" + channel
            stores["truth"][case + "-context.npz"] = schema["context"]
            for step in range(2):
                for probe in range(16):
                    prefix = f"{case}-s{step}-x{probe:02d}"
                    for kind in ("inputs", "responses"):
                        stores["truth"][prefix + "-" + kind + ".npz"] = schema[kind]
            for j in range(2):
                stores["truth"][f"{case}-reference-h{j:02d}.npz"] = schema["reference"]
                for kind in ("fine", "coarse", "richardson"):
                    stores["public"][f"{case}-{kind}-h{j:02d}.npz"] = schema[
                        "observations"
                    ]
            index = 2 * i + (channel == "fp32")
            for kind in ("bases", "choices"):
                stores["estimator"][f"case{index:02d}-{kind}.npz"] = schema[kind]
            stores["evaluator"][f"case{index:02d}-evaluation.npz"] = schema[
                "evaluation"
            ]
    for name, path in dict(
        truth="collection/truth",
        public="collection/public/arrays",
        estimator="estimator/arrays",
        evaluator="evaluation/arrays",
    ).items():
        exact_store(root / path, inventory_plan()["stores"][name], stores[name])
    if (
        len(evaluation.get("cases", [])) != 24
        or evaluation.get("certificate") is not False
        or evaluation.get("new_queries") != 0
        or evaluation.get("new_fits") != 0
        or evaluation.get("private_gate") != "closed"
    ):
        raise ValueError("Complete non-certificate evaluation required")


def validate_transport(root: Path, receipt: dict[str, Any]) -> None:
    specs = worst_schemas()
    plan = inventory_plan()
    if (
        receipt.get("shards") != 7
        or receipt.get("new_queries") != 0
        or receipt.get("manifest_sha256") != sha(root / "arrays/manifest.jsonl")
    ):
        raise ValueError("Pre-data transport commitment mismatch")
    exact_store(
        root / "arrays",
        dict(
            shards=7,
            shard_limit_bytes=plan["largest_shard_upper_bound"],
            manifest_bytes_upper_bound=7 * 188,
            bytes_upper_bound=plan["preflight_bytes_upper_bound"],
        ),
        {s.kind + ".npz": dict(s.fields) for s in specs},
    )


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
    report: dict[str, Any] = dict(
        status="started",
        collection_started=False,
        evaluation_started=False,
        automatic_retry=False,
        private_gate="closed",
        certificate=False,
    )
    path = output / "controller-report.json"
    save_json(path, report)

    def remaining() -> float:
        seconds = 180 - (clock() - started)
        if seconds <= 0:
            raise TimeoutError("Shared180s deadline expired")
        return seconds

    try:
        _, head = verify_source(repo, config_path, review_path)
        config_sha, review_sha = sha(config_path), sha(review_path)
        report.update(
            source_commit=head, config_sha256=config_sha, review_sha256=review_sha
        )
        env = dict(os.environ)
        env.update(
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            PYTHONPATH=str(repo / "src"),
        )

        def unchanged() -> None:
            _, current = verify_source(repo, config_path, review_path)
            if (
                current != head
                or sha(config_path) != config_sha
                or sha(review_path) != review_sha
            ):
                raise ValueError("Frozen source/config/review changed")

        def call(
            stage: str, command: list[str], payload: dict[str, Any]
        ) -> dict[str, Any]:
            report["active_stage"] = stage
            save_json(path, report)
            try:
                result = engine(
                    command,
                    payload,
                    cwd=output / (stage + "-engine"),
                    memory_limit_mb=512,
                    timeout_seconds=remaining(),
                    output_limit_bytes=2048,
                    env=env,
                )
            except Exception:
                report["unreceived_worker_counters_unknown"] = report[
                    "collection_started"
                ]
                raise
            report[stage + "_receipt"] = result
            save_json(path, report)
            if not tree_ok(result) or result.get("status") not in (
                "completed",
                "passed",
            ):
                raise ValueError("Stage failed or TREE monitoring unavailable")
            return result

        scope = call(
            "scope",
            [
                sys.executable,
                "-c",
                "import time,json;time.sleep(.15);print(json.dumps({'status':'passed'}))",
            ],
            {},
        )
        transport = call(
            "transport",
            [
                sys.executable,
                str(repo / "examples/isolated_glu/fresh_rank_controller.py"),
            ],
            dict(operation="synthetic_transport", output=str(output / "transport")),
        )
        validate_transport(output / "transport", transport)
        plan = inventory_plan()
        if shutil.disk_usage(output).free < plan["minimum_free_disk_bytes"]:
            raise ValueError("At least2GiB actual free disk required")
        expected_parent = repo.parent / "experiments"
        _ancestry(expected_parent)
        expected_parent.mkdir(exist_ok=True)
        if output.absolute().parent != expected_parent.absolute():
            raise ValueError(
                "Durable output must be direct child of experiment directory outside Git"
            )
        unchanged()
        marker = expected_parent / MARKER
        _ancestry(marker)
        with marker.open("xb") as f:
            data = canonical(
                dict(
                    policy=POLICY,
                    source_commit=head,
                    config_sha256=config_sha,
                    output=str(output.absolute()),
                )
            )
            if f.write(data) != len(data):
                raise OSError("Partial global marker; no retry")
            f.flush()
            os.fsync(f.fileno())
        marker_sha = sha(marker)
        report.update(
            marker=str(marker), marker_sha256=marker_sha, collection_started=True
        )
        save_json(path, report)
        receipt = call(
            "collection",
            [
                sys.executable,
                str(repo / "examples/isolated_glu/fresh_rank_collection.py"),
            ],
            dict(
                root=str(output),
                repo=str(repo),
                policy=POLICY,
                inventory_plan=plan,
                full_source_protocol_approved=True,
                source_commit=head,
                config_sha256=config_sha,
                marker_sha256=marker_sha,
                scope_preflight=scope,
                transport_preflight=transport,
            ),
        )
        collection = stage_report(output, "collection", receipt, head)
        check_counts(collection, head)
        if (
            collection["truth_inventory"]["shards"] != 1608
            or collection["public_inventory"]["shards"] != 144
        ):
            raise ValueError("Incomplete collection inventory")
        # Only hash receipts; do not parse secret/truth manifests before ALL choices seal.
        bound = {
            name: collection[field]
            for name, field in (
                ("truth-commitment.json", "truth_commitment_sha256"),
                ("seed-commitment.json", "seed_commitment_sha256"),
            )
        }
        for name, digest in bound.items():
            if sha(output / "collection" / name) != digest:
                raise ValueError("Collection commitment changed")
        unchanged()
        receipt = call(
            "estimation",
            [
                sys.executable,
                str(repo / "examples/isolated_glu/fresh_rank_estimation.py"),
            ],
            dict(
                public_root=str(output / "collection/public"),
                output=str(output / "estimator"),
                cases_sha256=collection["public_cases_sha256"],
                manifest_sha256=collection["public_manifest_sha256"],
                source_commit=head,
                store_limits=plan["stores"]["estimator"],
            ),
        )
        est = stage_report(output, "estimator", receipt, head)
        public_boundary(output, collection, est)
        for name, digest in bound.items():
            if sha(output / "collection" / name) != digest:
                raise ValueError("Pre-seal commitment changed")
        unchanged()
        report["evaluation_started"] = True
        save_json(path, report)
        receipt = call(
            "evaluation",
            [
                sys.executable,
                str(repo / "examples/isolated_glu/fresh_rank_evaluation.py"),
            ],
            dict(
                root=str(output),
                source_commit=head,
                store_limits=plan["stores"]["evaluator"],
                seal_sha256=est["seal_sha256"],
                estimator_report_sha256=sha(output / "estimator/estimator-report.json"),
                truth_commitment_sha256=collection["truth_commitment_sha256"],
                seed_commitment_sha256=collection["seed_commitment_sha256"],
            ),
        )
        evaluation = stage_report(output, "evaluation", receipt, head)
        validate_final(output, collection, est, evaluation)
        validate_transport(output / "transport", transport)
        unchanged()
        if sha(marker) != marker_sha:
            raise ValueError("Global marker changed")
        for name, digest in bound.items():
            if sha(output / "collection" / name) != digest:
                raise ValueError("Collection commitment changed at finish")
        exact_layout(output)
        files = [p for p in output.rglob("*") if p.is_file()]
        for p in output.rglob("*"):
            _ancestry(p)
        metadata = [
            p for p in files if p.suffix != ".npz" and p.name != "manifest.jsonl"
        ]
        if len(metadata) > 16 or any(p.stat().st_size > 128 * 1024 for p in metadata):
            raise ValueError("Global metadata cap exceeded")
        total = sum(p.stat().st_size for p in files)
        if total + plan["metadata_reserve_bytes"] >= 1024**3:
            raise ValueError("Global1GiB artifact ceiling exceeded")
        remaining()
        report.update(
            status="completed",
            elapsed_seconds_before_final_report=clock() - started,
            artifact_bytes_before_final_report=total,
            rule_passes=evaluation["rule_passes"],
            collection_report_sha256=sha(output / "collection/collection-report.json"),
            estimator_report_sha256=sha(output / "estimator/estimator-report.json"),
            evaluation_report_sha256=sha(output / "evaluation/evaluation-report.json"),
        )
        save_json(path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
            collection_may_be_partial=report["collection_started"],
        )
        save_json(path, report)
        raise ProtocolError(str(exc)) from exc


def launch(
    request: dict[str, Any], *, engine: Callable[..., dict[str, Any]] = run_json
) -> dict[str, Any]:
    """Outer monitor covers controller plus worker children, including imports.

    Launcher host RSS is excluded. Polled RSS is not an OS memory reservation.
    The outer receipt is written only after the monitored controller terminates.
    """
    output = Path(request["output"])
    _ancestry(output)
    if output.exists():
        raise ProtocolError("Existing attempt output preserved; no retry/overwrite")
    env = dict(os.environ)
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        PYTHONPATH=str(Path(request["repo"]) / "src"),
    )
    report: dict[str, Any] = dict(
        status="failed",
        private_gate="closed",
        certificate=False,
        automatic_retry=False,
        controller_counters_unknown=True,
    )
    try:
        receipt = engine(
            [sys.executable, str(Path(__file__).absolute())],
            dict(request, operation="controlled_run"),
            cwd=output.parent / (output.name + "-launch-engine"),
            memory_limit_mb=512,
            timeout_seconds=180,
            output_limit_bytes=2048,
            env=env,
        )
        report["receipt"] = receipt
        if not tree_ok(receipt) or receipt.get("status") != "completed":
            raise ProtocolError("Outer controller TREE monitoring or completion failed")
        controller = output / "controller-report.json"
        if read_json(controller).get("status") != "completed":
            raise ProtocolError("Inner controller incomplete")
        report.update(
            status="completed",
            controller_sha256=sha(controller),
            controller_counters_unknown=False,
        )
        save_json(output / "outer-monitor.json", report)
        return report
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)[:1000]
        )
        if output.exists():
            save_json(output / "outer-monitor.json", report)
        raise ProtocolError(str(exc)) from exc


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        if request.get("operation") == "synthetic_transport":
            result = synthetic_transport(Path(request["output"]))
        elif request.get("operation") == "controlled_run":
            result = run_protocol(
                Path(request["repo"]),
                Path(request["output"]),
                Path(request["config"]),
                Path(request["review"]),
            )
        else:
            result = launch(request)
        print(
            json.dumps(
                dict(
                    status=result["status"],
                    private_gate="closed",
                    **(
                        {
                            k: result[k]
                            for k in (
                                "shards",
                                "bytes",
                                "new_queries",
                                "manifest_sha256",
                            )
                        }
                        if request.get("operation") == "synthetic_transport"
                        else {}
                    ),
                )
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
