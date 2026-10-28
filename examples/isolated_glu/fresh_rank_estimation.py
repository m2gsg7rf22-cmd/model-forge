"""Public-only estimation process. Never import private factory or inventory."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from fresh_rank_view import FDView
from rank_selection import select
from fresh_rank_common import (
    ShardStore,
    create_store,
    save_json,
    sha,
    verify_manifest,
)

RULES = ("strict", "gap", "stable")


def registered_view(view: FDView) -> None:
    expected = []
    for i in range(12):
        n = 4 << (i // 3)
        for suffix, compute in [("fp64", "fp64"), ("fp32", "fp32_compute")]:
            expected.append(
                dict(
                    name=f"c{i:02d}-{suffix}",
                    n=n,
                    outputs=n,
                    probes=16,
                    compute=compute,
                )
            )
    if [asdict(c) for c in view.cases] != expected:
        raise ValueError("Exact24case/12pairedteacher/16probe public geometry required")


def estimate(
    view: FDView, store: ShardStore, output: Path, *, registered: bool = True
) -> dict[str, Any]:
    if registered:
        registered_view(view)
    selections = []
    for index, case in enumerate(view.cases):
        obs = view.packed(case.name)
        result = select(obs["fine"], obs["coarse"], obs["richardson"])
        p = result["p"]
        v = result["bases"]["richardson"]
        bases = store.save(
            f"case{index:02d}-bases.npz",
            **{k + "_basis": a for k, a in result["bases"].items()},
            **{k + "_spectrum": a for k, a in result["spectra"].items()},
        )
        fields = {
            k + "_projector": np.zeros((p, p))
            if result["choices"][k]["rank"] is None
            else v[: result["choices"][k]["rank"]].T @ v[: result["choices"][k]["rank"]]
            for k in RULES
        }
        fields.update(
            {
                k: np.asarray(a, dtype=np.float64).reshape(1)
                if np.ndim(a) == 0
                else np.asarray(a, dtype=np.float64)
                for k, a in result["diagnostics"].items()
            }
        )
        fields["ranks"] = np.array(
            [
                -1
                if result["choices"][k]["rank"] is None
                else result["choices"][k]["rank"]
                for k in RULES
            ],
            dtype=np.float64,
        )
        choices = store.save(f"case{index:02d}-choices.npz", **fields)
        selections.append(
            dict(
                case=case.name,
                index=index,
                meta=asdict(case),
                choices=result["choices"],
                basis=bases,
                selected=choices,
            )
        )
    inputs = view.verify_unchanged()
    inventory = verify_manifest(store.root)
    if inventory["shards"] != 2 * len(view.cases):
        raise ValueError("Incomplete estimator artifacts")
    path = output / "choices-sealed.json"
    if path.exists():
        raise ValueError("Choices seal already exists; no retry")
    seal = dict(
        phase="all_public_choices_complete",
        selections=selections,
        input_commitments=inputs,
        artifact_manifest_sha256=sha(store.root / "manifest.jsonl"),
        certificate=False,
        new_queries=0,
        new_fits=0,
    )
    save_json(path, seal)
    return dict(
        status="completed",
        selections=selections,
        seal_sha256=sha(path),
        input_commitments=inputs,
        artifact_inventory=inventory,
        artifact_manifest_sha256=seal["artifact_manifest_sha256"],
        new_queries=0,
        new_fits=0,
        certificate=False,
        private_gate="closed",
    )


def build(request: dict[str, Any]) -> dict[str, Any]:
    output = Path(request["output"])
    output.mkdir(exist_ok=False)
    path = output / "estimator-report.json"
    report: dict[str, Any] = dict(
        status="started",
        private_gate="closed",
        certificate=False,
        new_queries=0,
        new_fits=0,
    )
    save_json(path, report)
    try:
        view = FDView(
            Path(request["public_root"]),
            request["cases_sha256"],
            request["manifest_sha256"],
        )
        report.update(
            estimate(
                view, create_store(output / "arrays", request["store_limits"]), output
            ),
            source_commit=request["source_commit"],
        )
        save_json(path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)[:500]
        )
        save_json(path, report)
        raise


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        r = build(request)
        print(
            json.dumps(
                dict(
                    status=r["status"],
                    report_sha256=sha(
                        Path(request["output"]) / "estimator-report.json"
                    ),
                    seal_sha256=r["seal_sha256"],
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
