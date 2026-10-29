"""Full-protocol source contract and PCG64 wrapper; no default execution.

Actual use still requires complete independent full-source approval, clean freeze,
resource/transport/disk gates and a distinct exclusive attempt marker.
"""

from __future__ import annotations
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import psutil
from joint_response_inventory import inventory
from joint_response_join import RANK_PINS
from joint_response_streams import OptimizerInputs, stream_seed
from variable_response_manifest import PUBLIC_PINS
from variable_response_protocol import canonical, read, digest, ancestry

MARKER = "joint-response-development-v1-attempt.json"
POLICY = dict(
    id="joint-response-development-v1",
    jobs=6,
    n=32,
    outputs=32,
    probes=17,
    steps=2,
    rows_per_pair=2112,
    train_rows=54912,
    guard_rows=16896,
    guard_centers=[3, 7, 11, 15],
    max_width_hypothesis=96,
    adam_steps=4000,
    batch_size=256,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    learning_rate_start=0.01,
    learning_rate_end=0.0001,
    checkpoint_steps=[0, *range(256, 4000, 256), 4000],
    prediction_chunk_rows=8192,
    timeout_seconds=180,
    memory_limit_mb=512,
    blas_threads=1,
    output_limit_bytes=2048,
    per_engine_output_limit_bytes=1024,
    min_free_bytes=2 * 1024**3,
    new_queries=0,
    guard_is_consumed_calibration=True,
    certificate=False,
    private_gate="closed",
)
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
PARENT = "paired-row-acquisition-v1-20261028-07235ac"
COMPONENT_PINS = {
    "experiments/evidence/joint-response-development-design-v1/review.json": "88f61e33bdc8cc187c26d9712fa81866e4e5af8d21fc5b92064d78edeef45712",
    "experiments/evidence/joint-response-kernel-v1/review.json": "7ae8a8f444955535a1296221ca22c611f47a27eca5c4557abd5b85d62b3ebdee",
    "experiments/evidence/joint-response-join-inventory-v1/review.json": "9a9841aed4ab3519672d5cd56f239fa5e002edfa16e074ada0e7fc9a0928de9e",
    "experiments/evidence/joint-response-case-v1/review.json": "fcd9efb7cbc9250d7ac7d9486916e4a6185664449c09d8fa7673d70684df228f",
    "experiments/evidence/joint-response-trace-v1/review.json": "7fdec3e671a30169cb878afd014fb8a1e34927508952b11ae1f32e724f17cad1",
    "experiments/evidence/joint-response-stream-v1/review.json": "bc35a672c281a4a606685afa7c090019fdbbdb7f92c3cb9e772d462dbe07b6db",
    "experiments/evidence/joint-response-driver-v1/review.json": "229acca6c31c5623a10f29219a6391d30832d5f3e3b7281d952b6f3e8d71a711",
}
EVIDENCE_PINS = {
    "experiments/evidence/variable-response-view-v1/source-review/source-review.json": "903c2ea2b4be8080c4d3ba6bf2a064f36ea4fa87691b14168f16cb51e73990b9",
    "experiments/evidence/variable-response-view-v1/outcome-audit/outcome-audit.json": "14573ecea3251d205b3974387d7f1def81ab731eeb1ccffda0841116db6bf2cc",
    "experiments/evidence/variable-width-gate-development-v1/outcome-audit/outcome-audit.json": "e03cb21debfd87e41e58175c3050beb88b64e8a632242846c57be9b81bfd4a26",
}
DEPENDENCIES = {
    "examples/isolated_glu/" + n + ".py"
    for n in (
        "joint_response_kernel",
        "joint_response_join",
        "joint_response_inventory",
        "joint_response_case",
        "joint_response_streams",
        "joint_response_trace",
        "joint_response_driver",
        "joint_response_arrays",
        "joint_response_protocol",
        "joint_response_runtime",
        "joint_response_controller",
        "variable_gate_view",
        "row_acquisition_view",
        "fresh_rank_common",
        "scaling_domain_math",
        "scaling_domain_store",
        "variable_response_manifest",
        "variable_response_view",
        "variable_response_protocol",
    )
} | {
    "src/modelforge/" + n + ".py"
    for n in ("__init__", "execution", "glu32_io", "glu32", "antipodal")
}
REQUIRED_REVIEW = (
    DEPENDENCIES
    | set(COMPONENT_PINS)
    | set(EVIDENCE_PINS)
    | {rel for _, rel, _ in PUBLIC_PINS}
    | {
        "tests/test_joint_response_" + name + ".py"
        for name in (
            "arrays",
            "kernel",
            "join",
            "inventory",
            "case",
            "streams",
            "trace",
            "driver",
            "protocol",
            "runtime",
            "controller",
        )
    }
    | {
        "docs/" + name + ".md"
        for name in (
            "glu-joint-response-development-design",
            "glu-joint-response-development-protocol",
            "glu-joint-response-kernel",
            "glu-joint-response-join-inventory",
            "glu-joint-response-case-trace",
            "glu-joint-response-driver",
        )
    }
    | {"docs/superpowers/plans/2026-10-29-joint-response-development.md"}
)


class PCG64Stream:
    """Pinned actual backend. Instantiate ONLY after caller source/input/attempt gates."""

    def __init__(self, seed: int):
        if type(seed) is not int or not 0 <= seed < 2**256:
            raise ValueError("native SHA256 domain seed required")
        self._generator = np.random.Generator(np.random.PCG64(seed))

    def state(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._generator.bit_generator.state)

    def normal(self, shape: tuple[int, int]) -> NDArray[np.float64]:
        return self._generator.standard_normal(shape, dtype=np.float64)

    def indices(self, high: int, shape: tuple[int, int]) -> NDArray[np.int64]:
        return self._generator.integers(0, high, size=shape, dtype=np.int64)


def optimizer_binding(inputs: OptimizerInputs) -> dict[str, str]:
    values = {
        name: hashlib.sha256(
            memoryview(getattr(inputs.initial, name)).cast("B")
        ).hexdigest()
        for name in ("g", "u", "d")
    }
    values["schedule"] = hashlib.sha256(
        memoryview(inputs.schedule).cast("B")
    ).hexdigest()
    values["states"] = hashlib.sha256(
        canonical(
            [
                inputs.initial_state_before,
                inputs.initial_state_after,
                inputs.batch_state_before,
                inputs.batch_state_after,
            ]
        )
    ).hexdigest()
    return values


def environment(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(THREAD_ENV)
    env.update(
        PYTHONPATH=str(repo / "src") + os.pathsep + str(repo / "examples/isolated_glu"),
        PYTHONNOUSERSITE="1",
    )
    return env


def config_template(repo: Path) -> dict[str, Any]:
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
        frozen_dependencies={
            n: digest(repo / n, 2 * 1024**2) for n in sorted(DEPENDENCIES)
        },
        component_reviews=COMPONENT_PINS,
        prior_evidence=EVIDENCE_PINS,
        rank_metadata=dict(RANK_PINS),
        response_metadata={n: h for n, _, h in PUBLIC_PINS},
        paired_seeds={
            f"group{i:02d}": {
                role: stream_seed(f"job{2 * i:02d}", role)
                for role in ("initial", "batches")
            }
            for i in range(3)
        },
        parent_directory=PARENT,
        parent_source_commit="07235acd0ecaa576d57b37eefb2b4e6fc3aa0c6a",
    )


def verify_source(repo: Path, config: Path, review: Path) -> tuple[dict[str, Any], str]:
    ancestry(repo)
    ancestry(config)
    ancestry(review)
    approval = read(review)
    if (
        approval.get("full_joint_response_development_protocol_approved") is not True
        or approval.get("registered_execution_after_gates_approved") is not True
        or approval.get("private_gate") != "closed"
        or approval.get("certificate") is not False
    ):
        raise ValueError(
            "complete independent conditional execution/source approval required"
        )
    # This gate precedes config/source/observation access, including metadata pins.
    config_relative = str(config.relative_to(repo))
    review.relative_to(repo)
    value = read(config)
    if canonical(value) != canonical(config_template(repo)):
        raise ValueError(
            "exact frozen policy/runtime/inventory/dependency/input/seed contract required"
        )
    if any(os.environ.get(k) != v for k, v in THREAD_ENV.items()):
        raise ValueError("BLAS1 environment must precede NumPy worker import")
    signed = approval.get("source_sha256")
    required = REQUIRED_REVIEW | {config_relative}
    if not isinstance(signed, dict) or set(signed) != required:
        raise ValueError("exact complete reviewed source set required")
    for name, h in signed.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or digest(repo / p, 2 * 1024**2) != h:
            raise ValueError("reviewed source changed")
    for name, h in {
        **COMPONENT_PINS,
        **EVIDENCE_PINS,
        **{rel: h for _, rel, h in PUBLIC_PINS},
    }.items():
        if signed.get(name) != h:
            raise ValueError("prior component/evidence/input pin changed")
    for name, h in COMPONENT_PINS.items():
        component = read(repo / name)
        if component.get("private_gate") != "closed":
            raise ValueError("prior component gate changed")
        if "development-design-v1" in name:
            target = "docs/glu-joint-response-development-design.md"
            if component.get("design_approved") is not True or signed[
                target
            ] != component.get("design_sha256"):
                raise ValueError("approved design changed")
        else:
            if component.get("component_source_approved") is not True:
                raise ValueError("prior component not approved")
            for relative, pin in component["source_sha256"].items():
                if signed.get(relative) != pin:
                    raise ValueError("signed component bytes changed")

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, timeout=10
        ).strip()

    if (
        git("status", "--porcelain")
        or git("branch", "--show-current") != "discovery-lab"
    ):
        raise ValueError("clean committed discovery-lab required")
    return value, git("rev-parse", "HEAD")
