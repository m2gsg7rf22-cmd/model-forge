"""One source-reviewed zero-query analysis of already PUBLIC dimension data."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any
import numpy as np
from modelforge.glu32_io import read_json
from rank_analysis import Inputs, SPECS, Array, run_views, save_json
from scaling_domain_store import ShardStore, _ancestry, sha, verify_manifest

MARKER = "noise-aware-rank-analysis-v1-attempt.json"
LEGACY = "dimension-domain-v1-20261026-01b45d9"
MANIFEST_SHA = "9c2f1b8ae520024f19a10e2011f7dadbee18315302a69ab3d3efd2f6ca31a609"
POLICY = dict(
    id="noise-aware-rank-analysis-v1",
    cases=16,
    rules=["strict", "gap", "stable"],
    strict_rank_rtol=1e-10,
    gap_ratio=10,
    stability_rms_sine=0.01,
    resolution_multiplier=64,
    new_reference_queries=0,
    new_fits=0,
    memory_limit_mb=512,
    timeout_seconds=180,
    blas_threads=1,
    worker_output_limit_bytes=2048,
    minimum_free_disk_bytes=1024**3,
    blind_claim=False,
    certificate=False,
    private_gate="closed",
)


def check_legacy_commitments(repo: Path) -> Path:
    root = repo.parent / "experiments" / LEGACY
    # Byte hashes only: do not parse legacy reports containing truth/width/ranks.
    pins = {
        root
        / "controller-report.json": "ec13f57abb5253522be6d9ea591392578c03a59dbeb62d711eea40a71e764dcf",
        root
        / "worker/worker-report.json": "914b731a06cba7469ba90f9d65fe5d47c2f2e698fde9f8ce68a6cdf6ac5c102f",
        root / "worker/arrays/manifest.jsonl": MANIFEST_SHA,
        repo
        / "experiments/evidence/isolated-glu-dimension-domain-v1/outcome-audit.json": "14a79ddcb737ce90bede8a364e79b6967f3d1b11523f4c5b06e6f980f5fe3e35",
    }
    for path, digest in pins.items():
        _ancestry(path)
        if not path.is_file() or sha(path) != digest:
            raise ValueError("Legacy byte commitment mismatch")
    return root / "worker/arrays"


def synthetic_transport(root: Path, *, free_bytes: int | None = None) -> dict[str, Any]:
    available = (
        shutil.disk_usage(root.parent).free if free_bytes is None else free_bytes
    )
    if type(available) is not int or available < 1024**3:
        raise ValueError("At least1GiB actual free disk required before inputs")
    store = ShardStore(root)
    basis: dict[str, Array] = {
        f"{kind}_basis": np.eye(528) for kind in ("fine", "coarse", "richardson")
    }
    basis.update(
        {f"{kind}_spectrum": np.ones(528) for kind in ("fine", "coarse", "richardson")}
    )
    choices: dict[str, Array] = {
        f"{kind}_projector": np.eye(528) for kind in ("strict", "gap", "stable")
    }
    choices.update(
        log_gaps=np.ones((3, 527)),
        gap_ratios=np.ones((3, 527)),
        eligible=np.ones((3, 527)),
        resolution_clipped=np.zeros((3, 527)),
        stability=np.ones(527),
        resolution_floor=np.ones(1),
        ranks=np.ones(3),
    )
    for name, fields in [("basis", basis), ("choices", choices)]:
        store.save(name + ".npz", **fields)
        with np.load(root / (name + ".npz"), allow_pickle=False) as loaded:
            if set(loaded.files) != set(fields) or any(
                not np.array_equal(loaded[k], a) for k, a in fields.items()
            ):
                raise ValueError("Worst synthetic transport roundtrip failed")
    return dict(
        status="passed",
        **verify_manifest(root),
        free_disk_bytes_before=available,
        registered_data_opened=False,
        new_reference_queries=0,
        new_fits=0,
    )


def build(request: dict[str, Any]) -> dict[str, Any]:
    output = Path(request["output"]).absolute()
    _ancestry(output)
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(
        status="started",
        registered_data_opened=False,
        new_reference_queries=0,
        new_fits=0,
        private_gate="closed",
        certificate=False,
        automatic_retry=False,
    )
    path = output / "worker-report.json"
    save_json(path, report)
    try:
        if (
            request.get("policy") != POLICY
            or request.get("full_source_protocol_approved") is not True
        ):
            raise ValueError("Exact full-source protocol approval required")
        for key in ("scope_preflight", "transport_preflight"):
            receipt = request.get(key, {})
            if (
                receipt.get("process_monitor_scope") != "process_and_children"
                or receipt.get("memory_samples", 0) < 1
            ):
                raise ValueError("Pre-data tree monitoring required")
        if request["transport_preflight"].get("status") != "passed":
            raise ValueError("Pre-data worst-archive transport required")
        if shutil.disk_usage(output).free < 1024**3:
            raise ValueError("At least1GiB free disk required before inputs")
        repo = Path(request["repo"]).absolute()
        source = check_legacy_commitments(repo)
        marker = repo.parent / "experiments" / MARKER
        _ancestry(marker)
        with marker.open("x") as f:
            json.dump(
                dict(
                    policy=POLICY,
                    source_commit=request["source_commit"],
                    config_sha256=request["config_sha256"],
                    output=str(output),
                    automatic_retry=False,
                    new_reference_queries=0,
                ),
                f,
                allow_nan=False,
            )
        report.update(
            source_commit=request["source_commit"],
            marker=str(marker),
            marker_sha256=sha(marker),
        )
        save_json(path, report)
        report["registered_data_opened"] = True
        save_json(path, report)
        reader = Inputs(source, MANIFEST_SHA, SPECS)
        result = run_views(reader, ShardStore(output / "arrays"), output, SPECS)
        if result["artifact_inventory"]["shards"] != 48:
            raise ValueError("Exact48 output shards required")
        check_legacy_commitments(repo)
        report.update(
            result, artifact_manifest_sha256=sha(output / "arrays/manifest.jsonl")
        )
        save_json(path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)[:1000]
        )
        save_json(path, report)
        raise


if __name__ == "__main__":
    request = (
        read_json(Path(sys.argv[1])) if len(sys.argv) == 2 else json.load(sys.stdin)
    )
    try:
        if request.get("operation") == "synthetic_transport":
            result = synthetic_transport(Path(request["output"]))
        else:
            result = build(request)
            result = dict(
                status=result["status"],
                worker_report_sha256=sha(
                    Path(request["output"]) / "worker-report.json"
                ),
            )
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        print(
            json.dumps(
                dict(ok=False, error_type=type(exc).__name__, error=str(exc)[:500])
            )
        )
