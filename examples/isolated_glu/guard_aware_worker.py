"""Reviewed saved-input adapter; launch only through guard_aware_controller."""

from __future__ import annotations
from collections.abc import Callable
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from modelforge.glu32_io import raw, read_json, read_arrays

spec = importlib.util.spec_from_file_location(
    "reviewed_step", Path(__file__).with_name("guard_aware_step.py")
)
assert spec and spec.loader
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def read_data(
    paths: dict[str, Path],
    config: dict[str, Any],
    inventory_sink: Callable[[dict[str, str]], None] | None = None,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    names = {
        "candidate",
        "compact_metadata",
        "full_fit_report",
        "observations",
        "manifest",
        "original_manifest",
    }
    if set(paths) != names or set(config.get("inputs", {})) != names:
        raise ValueError("Exact step input inventory required")
    before = {k: sha(p) for k, p in paths.items()}
    if inventory_sink is not None:
        inventory_sink(before)
    if before != config["inputs"]:
        raise ValueError("Step input commitments changed")
    compact = read_json(paths["compact_metadata"])
    meta = read_json(paths["manifest"])
    original = read_json(paths["original_manifest"])
    if (
        compact.get("method") != "isolated-glu-fp32-guarded-gate-step-v1"
        or compact.get("candidate_built") is not True
        or compact.get("candidate_sha256") != before["candidate"]
        or compact.get("forward_raw_sha256") != before["observations"]
        or compact.get("forward_manifest_sha256") != before["manifest"]
        or compact.get("calibration_manifest_sha256") != before["original_manifest"]
        or compact.get("full_fit_report_sha256") != before["full_fit_report"]
        or compact.get("full_fit_details_preserved_externally") is not True
        or original.get("channel") != "fp32_compute"
    ):
        raise ValueError("Saved candidate/raw calibration provenance mismatch")
    expected = {
        "method": "isolated-glu-fp32-guarded-forward-polish-v1",
        "channel": "fp32_compute",
        "train_seed": 882030,
        "guard_seed": 882031,
        "scales": [0.01, 1.0, 10.0, 100.0],
        "group_rows": 256,
        "vector_calls": 2048,
        "baseline_sha256": "b2d5fdd4cf8be40954562c2ce9393cd5b9a3c3cbd3cd8cbfa9bc18728d1de41e",
        "raw_sha256": before["observations"],
    }
    if meta != expected:
        raise ValueError("Guard calibration manifest changed")
    observed = read_arrays(
        paths["observations"],
        {name: (1024, 32) for name in ["train_x", "train_y", "guard_x", "guard_y"]},
        before["observations"],
    )
    for name, seed in [("train", 882030), ("guard", 882031)]:
        rng = np.random.default_rng(seed)
        expected_x = np.vstack(
            [rng.normal(size=(256, 32)) * s for s in [0.01, 1.0, 10.0, 100.0]]
        )
        if not np.array_equal(expected_x, observed[name + "_x"]):
            raise ValueError("Reused input seed mismatch")
    candidate = read_arrays(
        paths["candidate"],
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        before["candidate"],
    )
    return before, candidate, observed


def build(
    paths: dict[str, Path],
    output: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    if (
        preflight.get("process_monitor_scope") != "process_and_children"
        or preflight.get("memory_samples", 0) < 1
    ):
        raise ValueError("Successful controller pre-data scope preflight required")
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = {
        "method": step.METHOD,
        "status": "started",
        "candidate_built": False,
        "private_gate": "closed",
    }
    step.write_json(output / "worker-report.json", report)
    failure: Exception | None = None

    def record_inputs(before: dict[str, str]) -> None:
        report["inputs_before"] = before
        step.write_json(output / "worker-report.json", report)

    try:
        before, candidate, observed = read_data(paths, config, record_inputs)
        changed, details = step.run_step(
            output / "step",
            candidate["gates"],
            candidate["couplings"],
            observed["train_x"],
            observed["train_y"],
            observed["guard_x"],
            observed["guard_y"],
            rows=256,
        )
        report.update(
            status="completed",
            changed=changed is not None,
            candidate_built=changed is not None,
            new_target_queries=0,
            reused_calibration_vectors=2048,
            candidate_updates=len(details["accepted"]),
            trials=len(details["trials"]),
            original_full_fit_report_read_mode="hash_only",
            step_report_sha256=sha(output / "step/report.json"),
            artifact_manifest_sha256=sha(output / "step/artifact-manifest.json"),
        )
        if changed is not None:
            report["candidate_sha256"] = sha(output / "step/candidate.npz")
    except Exception as exc:
        failure = exc
        report.update(
            status="failed",
            candidate_built=False,
            changed=False,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    finally:
        if "inputs_before" in report:
            report["inputs_unchanged"] = None
            try:
                after = {k: sha(p) for k, p in paths.items()}
                report.update(
                    inputs_after=after,
                    inputs_unchanged=report["inputs_before"] == after,
                )
                if not report["inputs_unchanged"]:
                    raise ValueError("Input commitments changed during attempt")
            except Exception as check:
                report.update(
                    status="failed",
                    candidate_built=False,
                    changed=False,
                    input_check_error=str(check),
                )
                if failure is None:
                    failure = check
                    report.update(error_type=type(check).__name__, error=str(check))
        else:
            report["inputs_unchanged"] = None
        step.write_json(output / "worker-report.json", report)
    if failure is not None:
        raise failure
    return {
        "method": step.METHOD,
        "status": report["status"],
        "changed": report["changed"],
        "worker_report_sha256": sha(output / "worker-report.json"),
        "actual_target_queries": 0,
        "candidate_updates": report["candidate_updates"],
        "private_gate": "closed",
    }


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        result = build(
            {k: Path(v) for k, v in request["paths"].items()},
            Path(request["output"]),
            request["config"],
            request["preflight"],
        )
    except Exception as exc:
        result = {"ok": False, "error": str(exc)[:500], "private_gate": "closed"}
    print(json.dumps(result, allow_nan=False))
