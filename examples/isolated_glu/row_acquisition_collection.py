"""Private collection stage. Registered generation is behind full controller gates."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import numpy as np
from numpy.typing import NDArray
from precision import forward, reference_weights
from scaling_domain_math import (
    analytic_hessian,
    curvature_from_responses,
    four_corner_inputs,
    richardson,
    symmetric_atoms,
)
from row_acquisition_common import (
    MARKER as MARKER,
    POLICY as POLICY,
    ShardStore as ShardStore,
    collection_gate,
    create_store,
    save_json,
    sha,
    verify_manifest,
)
from row_acquisition_inventory import WIDTHS, inventory_plan

from fresh_rank_collection import RawCounter as RawCounter

Array = NDArray[np.float64]


def validate_context(ctx: dict[str, Array]) -> tuple[int, int, int]:
    if set(ctx) != {"g", "u", "d", "probes", "random_directions"}:
        raise ValueError("Exact supplied context fields required")
    for a in ctx.values():
        if (
            not isinstance(a, np.ndarray)
            or a.dtype != np.float64
            or a.ndim != 2
            or not a.size
            or a.nbytes > 8 * 1024**2
            or not np.isfinite(a).all()
        ):
            raise ValueError("Bounded finite native context required")
    m, n = ctx["g"].shape
    if (
        not 1 <= m <= 96
        or not 1 <= n <= 32
        or ctx["u"].shape != (m, n)
        or ctx["d"].shape != (n, m)
        or not 1 <= len(ctx["probes"]) <= 17
        or ctx["probes"].shape[1] != n
        or ctx["random_directions"].shape != (64, n)
    ):
        raise ValueError("Incompatible width96 supplied context")
    return n, m, len(ctx["probes"])


def collect_case(
    truth: ShardStore,
    public: ShardStore,
    ledger: RawCounter,
    case: str,
    channel: str,
    ctx: dict[str, Array],
    callback: Callable[[Array], Array],
) -> None:
    n, _, count = validate_context(ctx)
    if channel not in ("fp64", "fp32_compute"):
        raise ValueError("Compute channel required")
    shapes = {k: a.shape for k, a in ctx.items()}
    snapshots = {k: a.tobytes() for k, a in ctx.items()}
    g, u, d, x = ctx["g"], ctx["u"], ctx["d"], ctx["probes"]
    gr, ur, dr = reference_weights(g, u, d, channel)
    truth.save(
        case + "-context.npz",
        **ctx,
        reference_g=gr,
        reference_u=ur,
        reference_d=dr,
        atoms=symmetric_atoms(gr, ur),
        preactivations=x @ gr.T,
    )
    for a in ctx.values():
        a.setflags(write=False)
    href = analytic_hessian(gr, ur, dr, x)
    for start in range(0, count, 8):
        truth.save(
            f"{case}-reference-h{start // 8:02d}.npz", hessian=href[start : start + 8]
        )
    estimates = []
    p = n * (n + 1) // 2
    for step_index, step in enumerate((0.3, 0.6)):
        h = np.empty_like(href)
        for i, point in enumerate(x):
            inputs = four_corner_inputs(point, step).reshape(-1, n)
            y = ledger.query(
                truth, f"{case}-s{step_index}-x{i:02d}", inputs, callback, n
            )
            if set(ctx) != set(snapshots) or any(
                not isinstance(a, np.ndarray)
                or a.shape != shapes[k]
                or a.dtype != np.float64
                or a.tobytes() != snapshots[k]
                for k, a in ctx.items()
            ):
                ledger.terminal = True
                save_json(
                    ledger.path,
                    dict(status="failed", reason="context mutation", **ledger.counts),
                )
                raise ValueError("Teacher/probe context mutation")
            h[i] = curvature_from_responses(y.reshape(4, p, n), n, step)
        estimates.append(h)
    for kind, h in [
        ("fine", estimates[0]),
        ("coarse", estimates[1]),
        ("richardson", richardson(*estimates)),
    ]:
        for start in range(0, count, 8):
            public.save(
                f"{case}-{kind}-h{start // 8:02d}.npz", hessian=h[start : start + 8]
            )


def assignment(seed: bytes) -> list[dict[str, Any]]:
    if type(seed) is not bytes or len(seed) != 32:
        raise ValueError("256-bit seed required")
    ordered = sorted(
        WIDTHS,
        key=lambda m: hashlib.sha256(
            seed + f"/row-acquisition-v1/assignment/{m}".encode()
        ).digest(),
    )
    return [
        dict(id=f"c{i:02d}", n=32, width=m, kind="generic")
        for i, m in enumerate(ordered)
    ]


def contexts(
    secret: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], dict[str, Array]]]:
    seed = bytes.fromhex(secret["seed"])
    for teacher in secret["mapping"]:
        digest = hashlib.sha256(
            seed + b"/row-acquisition-v1/teacher/" + teacher["id"].encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big") | (1 << 255))
        n, m = teacher["n"], teacher["width"]
        g = rng.normal(size=(m, n)) * 0.2
        u = rng.normal(size=(m, n)) * 0.2
        d = rng.normal(size=(n, m)) * 0.2
        yield (
            teacher,
            dict(
                g=g,
                u=u,
                d=d,
                probes=rng.normal(size=(17, n)) * 4,
                random_directions=rng.normal(size=(64, n)),
            ),
        )


def build(request: dict[str, Any]) -> dict[str, Any]:
    root = collection_gate(request)
    plan = inventory_plan()
    if request.get("inventory_plan") != plan:
        raise ValueError("Exact reviewed inventory required")
    if shutil.disk_usage(root).free < plan["minimum_free_disk_bytes"]:
        raise ValueError("At least2GiB free required")
    output = root / "collection"
    output.mkdir(exist_ok=False)
    public_root = output / "public"
    public_root.mkdir()
    report: dict[str, Any] = dict(
        status="started",
        source_commit=request["source_commit"],
        private_gate="closed",
        certificate=False,
        new_fits=0,
    )
    save_json(output / "collection-report.json", report)
    ledger = RawCounter(output / "query-progress.json", 430848)
    try:
        # Marker and all source/resource/inventory gates precede entropy or teachers.
        secret: dict[str, Any] = dict(
            seed=os.urandom(32).hex(), nonce=os.urandom(32).hex()
        )
        secret["mapping"] = assignment(bytes.fromhex(secret["seed"]))
        save_json(output / "registration.json", secret)
        save_json(
            output / "seed-commitment.json",
            dict(
                sha256=sha(output / "registration.json"),
                mapping_withheld=True,
                source_commit=request["source_commit"],
            ),
        )
        truth = create_store(output / "truth", plan["stores"]["truth"])
        public = create_store(public_root / "arrays", plan["stores"]["public"])
        cases = []
        for teacher, ctx in contexts(secret):
            for suffix, channel in [("fp64", "fp64"), ("fp32", "fp32_compute")]:
                case = teacher["id"] + "-" + suffix
                collect_case(
                    truth,
                    public,
                    ledger,
                    case,
                    channel,
                    ctx,
                    lambda xx: forward(ctx["g"], ctx["u"], ctx["d"], xx, channel),
                )
                cases.append(
                    dict(
                        name=case,
                        n=teacher["n"],
                        outputs=teacher["n"],
                        probes=17,
                        compute=channel,
                    )
                )
        save_json(
            public_root / "cases.json",
            dict(
                schema="modelforge-row-acquisition-observations-v1",
                steps=[0.3, 0.6],
                cases=cases,
            ),
        )
        actual_truth = verify_manifest(truth.root)
        actual_public = verify_manifest(public.root)
        if (
            actual_truth["shards"] != 432
            or actual_public["shards"] != 54
            or any(
                ledger.counts[k] != 430848
                for k in (
                    "calls_reserved",
                    "calls_attempted",
                    "calls_returned",
                    "calls_persisted",
                )
            )
            or ledger.counts["callback_batches"] != 204
        ):
            raise ValueError("Incomplete registered collection counts")
        commitment = dict(
            registration_sha256=sha(output / "registration.json"),
            truth_manifest_sha256=sha(truth.root / "manifest.jsonl"),
            public_manifest_sha256=sha(public.root / "manifest.jsonl"),
            public_cases_sha256=sha(public_root / "cases.json"),
        )
        save_json(output / "truth-commitment.json", commitment)
        report.update(
            status="completed",
            truth_inventory=actual_truth,
            public_inventory=actual_public,
            truth_commitment_sha256=sha(output / "truth-commitment.json"),
            seed_commitment_sha256=sha(output / "seed-commitment.json"),
            public_cases_sha256=commitment["public_cases_sha256"],
            public_manifest_sha256=commitment["public_manifest_sha256"],
            **ledger.counts,
        )
        save_json(output / "collection-report.json", report)
        return report
    except Exception as exc:
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc)[:500],
            **ledger.counts,
        )
        save_json(output / "collection-report.json", report)
        raise


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        result = build(request)
        print(
            json.dumps(
                dict(
                    status=result["status"],
                    report_sha256=sha(
                        Path(request["root"]) / "collection/collection-report.json"
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
