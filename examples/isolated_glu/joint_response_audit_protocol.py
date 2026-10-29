"""Frozen source/runtime/input contract for independent numeric audit v1.

Approval gates precede numeric artifact reads. No RNG, optimizer or teacher.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import psutil
from joint_response_audit_inventory import inventory
from joint_response_audit_runtime import MARKER as MARKER
from joint_response_join import RANK_PINS
from variable_response_manifest import PUBLIC_PINS
from variable_response_protocol import ancestry, canonical, digest, read

ACTUAL = "joint-response-development-v1-20261029-585eb97"
ACTUAL_COMMIT = "585eb97db907d6a07463e072b1bc690857594bd3"
PARENT = "paired-row-acquisition-v1-20261028-07235ac"
AUDIT = "experiments/evidence/joint-response-development-v1/outcome-audit/outcome-audit.json"
AUDIT_SHA = "8724353ed385a28349fc862127c78352f2e5f87f718e3befb9f90de5acb52714"
OLD_REVIEW = "experiments/evidence/joint-response-development-v1/source-review/source-review.json"
OLD_REVIEW_SHA = "0a83d99bc4e05c4034e4cccd4148d28c8be352a4d1978ee1c85576a5522e9c4d"
OLD_PREREG = "experiments/preregistrations/joint-response-development-v1.json"
OLD_PREREG_SHA = "a57d0a7b2ce5ecf83f99e8d1c79f675a48e4963960e230fd864a33b8fb68d146"
OLD_MARKER_SHA = "4573ee6e1f72316be5fa03235383553e5eb46686c781fa1e9206bf9936945213"
THREAD_ENV = {
    k: "1"
    for k in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}
POLICY: dict[str, Any] = dict(
    id="joint-response-numeric-audit-v1",
    jobs=6,
    checkpoints=[0, *range(256, 4000, 256), 4000],
    width_hypotheses=[96, 96, 48, 48, 16, 16],
    n=32,
    outputs=32,
    probes=17,
    rows_per_pair=2112,
    prediction_rows=8192,
    forward_chunk_rows=4096,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-10,
    near_tie="gap<=tau_i+tau_j; uncertified; retain saved winner",
    timeout_seconds=180,
    memory_limit_mb=512,
    blas_threads=1,
    output_limit_bytes=2048,
    per_engine_output_limit_bytes=1024,
    min_free_bytes=2 * 1024**3,
    new_queries=0,
    rng_calls=0,
    optimizer_replays=0,
    guard_is_consumed_calibration=True,
    certificate=False,
    private_gate="closed",
)
COMPONENTS = {
    "experiments/evidence/joint-response-numeric-audit-design-v1/design-review.json": "feb1728295d1c2e4eee01446383711e3551d2cc57c0d88c247c49c8a71d6de6b",
    "experiments/evidence/joint-response-numeric-audit-components-v1/review.json": "90f42ada4784f77a88b04292e931e7b431bcc06348d4661b15437f794ed6b355",
    "experiments/evidence/joint-response-numeric-audit-case-inventory-v1/review.json": "f01318ca313b825c88bdfbb258b60755bbac9250b27d3ea24ad79d6ff2acc58e",
    "experiments/evidence/joint-response-numeric-audit-storage-jobs-v1/review.json": "7307d32c9b8cce6c0e47f309b537c367053acf64ebba016dc8cb5d3fc51c40ef",
}
RUNTIME = (
    {
        f"examples/isolated_glu/joint_response_audit_{n}.py"
        for n in (
            "math",
            "projection",
            "case",
            "inventory",
            "storage",
            "jobs",
            "runtime",
            "protocol",
            "controller",
        )
    }
    | {
        f"examples/isolated_glu/{n}.py"
        for n in (
            "joint_response_join",
            "joint_response_arrays",
            "joint_response_inventory",
            "joint_response_runtime",
            "variable_gate_view",
            "row_acquisition_view",
            "fresh_rank_common",
            "scaling_domain_math",
            "scaling_domain_store",
            "variable_response_manifest",
            "variable_response_view",
            "variable_response_protocol",
        )
    }
    | {
        f"src/modelforge/{n}.py"
        for n in ("__init__", "execution", "glu32_io", "glu32", "antipodal")
    }
)
NEW_REVIEW = (
    {
        f"tests/test_joint_response_audit_{n}.py"
        for n in (
            "math",
            "projection",
            "case",
            "inventory",
            "storage",
            "jobs",
            "runtime",
            "protocol",
            "controller",
        )
    }
    | {
        f"docs/glu-joint-response-numeric-audit{suffix}.md"
        for suffix in (
            "-design",
            "-components",
            "-case-inventory",
            "-storage-jobs",
            "-protocol",
        )
    }
    | {
        "docs/superpowers/plans/2026-10-29-joint-response-numeric-audit.md",
        AUDIT,
        OLD_REVIEW,
        OLD_PREREG,
    }
    | set(COMPONENTS)
)


def environment(repo: Path) -> dict[str, str]:
    return dict(
        os.environ,
        **THREAD_ENV,
        PYTHONPATH=str(repo / "src") + os.pathsep + str(repo / "examples/isolated_glu"),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )


def input_bindings(audit: dict[str, Any], actual: Path) -> dict[str, Any]:
    """Pure parsing; authority comes from pinned audit digest in config_template."""
    if (
        audit.get("saved_engineering_receipt_audit_approved") is not True
        or audit.get("numeric_metric_audit_approved") is not False
        or audit.get("actual_source_commit") != ACTUAL_COMMIT
        or audit.get("private_gate") != "closed"
        or audit.get("certificate") is not False
    ):
        raise ValueError("approved closed engineering receipt audit required")
    pins = audit.get("artifact_sha256")
    if not isinstance(pins, dict):
        raise ValueError("complete artifact hash map required")
    rootfiles = (
        "controller-report.json",
        "outer-monitor.json",
        "preflight-receipt.json",
        "input-binding.json",
        "terminal-binding.json",
        "progress.json",
    )
    roots = {name: pins[str(actual / name)] for name in rootfiles}
    cases = {
        f"job{i:02d}": {
            Path(p).name: h
            for p, h in pins.items()
            if Path(p).parent == actual / "cases" / f"job{i:02d}"
        }
        for i in range(6)
    }
    numeric = (
        {"initial", "schedule", "split-indices", "selected-physical"}
        | {f"checkpoint-{step:04d}" for step in POLICY["checkpoints"]}
        | {f"prediction-{i:02d}" for i in range(9)}
    )
    metadata = {
        "case.json",
        "metrics.json",
        "terminal.json",
        "rng.json",
        "manifest.jsonl",
        "checkpoint-intents.jsonl",
        "checkpoint-metrics.jsonl",
        "progress.jsonl",
    }
    expected = {n + ".npz" for n in numeric} | metadata
    if any(set(group) != expected for group in cases.values()):
        raise ValueError("exact38file completed case inventory required")
    for group in (roots, *cases.values()):
        if not group or any(
            not isinstance(h, str)
            or len(h) != 64
            or any(c not in "0123456789abcdef" for c in h)
            for h in group.values()
        ):
            raise ValueError("native SHA256 artifact bindings required")
    return dict(root_files=roots, case_files=cases)


def required_review(repo: Path, config_relative: str) -> set[str]:
    return (
        set(read(repo / OLD_REVIEW)["source_sha256"])
        | RUNTIME
        | NEW_REVIEW
        | {config_relative}
    )


def config_template(repo: Path) -> dict[str, Any]:
    if (
        digest(repo / AUDIT) != AUDIT_SHA
        or digest(repo / OLD_REVIEW) != OLD_REVIEW_SHA
        or digest(repo / OLD_PREREG) != OLD_PREREG_SHA
    ):
        raise ValueError("prior source/audit/prereg pins changed")
    audit = read(repo / AUDIT)
    actual = repo.parent / "experiments" / ACTUAL
    return dict(
        id=POLICY["id"],
        policy=POLICY,
        inventory=inventory(),
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            psutil=psutil.__version__,
            machine=platform.machine(),
            executable=sys.executable,
        ),
        required_environment=THREAD_ENV,
        frozen_dependencies={n: digest(repo / n, 2 * 1024**2) for n in sorted(RUNTIME)},
        component_reviews=COMPONENTS,
        old_engineering_audit_sha256=AUDIT_SHA,
        old_source_review_sha256=OLD_REVIEW_SHA,
        old_prereg_sha256=OLD_PREREG_SHA,
        old_source_commit=ACTUAL_COMMIT,
        old_marker_sha256=OLD_MARKER_SHA,
        actual_directory=ACTUAL,
        parent_directory=PARENT,
        rank_metadata=dict(RANK_PINS),
        response_metadata={n: h for n, _, h in PUBLIC_PINS},
        input_bindings=input_bindings(audit, actual),
    )


def verify_source(repo: Path, config: Path, review: Path) -> tuple[dict[str, Any], str]:
    ancestry(repo)
    ancestry(config)
    ancestry(review)
    approval = read(review)
    if (
        approval.get("full_joint_response_numeric_audit_protocol_approved") is not True
        or approval.get("execution_after_gates_approved") is not True
        or approval.get("private_gate") != "closed"
        or approval.get("certificate") is not False
    ):
        raise ValueError(
            "complete independent numeric-audit source/conditional execution approval required"
        )
    relative = str(config.relative_to(repo))
    review.relative_to(repo)
    value = read(config)
    if canonical(value) != canonical(config_template(repo)):
        raise ValueError("exact frozen source/runtime/input/inventory policy required")
    if any(os.environ.get(k) != v for k, v in THREAD_ENV.items()):
        raise ValueError("BLAS1 environment required before numeric imports")
    signed = approval.get("source_sha256")
    if not isinstance(signed, dict) or set(signed) != required_review(repo, relative):
        raise ValueError("exact full reviewed source set required")
    for n, h in signed.items():
        p = Path(n)
        if p.is_absolute() or ".." in p.parts or digest(repo / p, 2 * 1024**2) != h:
            raise ValueError("reviewed source changed")
    old = read(repo / OLD_REVIEW)
    for n, h in old["source_sha256"].items():
        if signed.get(n) != h:
            raise ValueError("original frozen study source changed")
    for n, h in COMPONENTS.items():
        if signed.get(n) != h:
            raise ValueError("component review changed")
        component = read(repo / n)
        if component.get("private_gate") != "closed":
            raise ValueError("component private gate changed")
        if component.get("design_approved") is True:
            if (
                signed.get("docs/glu-joint-response-numeric-audit-design.md")
                != component["design_sha256"]
            ):
                raise ValueError("design changed")
        elif component.get("component_source_approved") is not True:
            raise ValueError("component not source approved")
        for source, pin in component.get("source_sha256", {}).items():
            if signed.get(source) != pin:
                raise ValueError("reviewed component bytes changed")

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, timeout=10
        ).strip()

    if (
        git("status", "--porcelain")
        or git("branch", "--show-current") != "discovery-lab"
    ):
        raise ValueError("clean committed discovery-lab required")
    git("merge-base", "--is-ancestor", ACTUAL_COMMIT, "HEAD")
    return value, git("rev-parse", "HEAD")
