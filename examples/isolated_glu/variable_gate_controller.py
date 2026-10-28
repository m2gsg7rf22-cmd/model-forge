"""One-attempt public candidate development; default outer TREE monitor required.

No teacher factory, query callback, truth labels, private model or heldout access.
Full source/preregistration approval and resource gates precede registered fitting.
"""

from __future__ import annotations
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
from modelforge.execution import run_json
from modelforge.glu32_io import read_arrays
from scaling_domain_store import ShardStore, _ancestry, verify_manifest
from variable_gate_common import (
    POLICY,
    MARKER,
    canonical,
    digest,
    read_json,
    save_json,
    records,
    seed_map,
    tree_ok,
    fsynced_marker,
    TraceStore,
    read_trace,
    integer,
    decode_state,
)
from variable_gate_inventory import (
    inventory,
    trace_names,
    transport_schemas,
)
from variable_gate_view import load_bundle, GateJob
from variable_gate_search import search_job, GateResult, losses

DEPENDENCIES = {
    "examples/isolated_glu/" + n + ".py"
    for n in (
        "variable_gate_inventory",
        "variable_gate_common",
        "variable_gate_view",
        "variable_gate_search",
        "variable_gate_controller",
        "row_acquisition_view",
        "fresh_rank_common",
        "scaling_domain_math",
        "scaling_domain_store",
    )
} | {
    "src/modelforge/" + n + ".py"
    for n in ("__init__", "execution", "glu32_io", "glu32", "antipodal")
}
REQUIRED_REVIEW = DEPENDENCIES | {
    "tests/test_variable_gate_view.py",
    "tests/test_variable_gate_search.py",
    "tests/test_variable_gate_protocol.py",
    "docs/glu-variable-width-gate-development-design.md",
    "docs/glu-variable-width-gate-adapter.md",
    "docs/glu-variable-width-gate-search.md",
    "docs/glu-variable-width-gate-development-protocol.md",
    "docs/superpowers/plans/2026-10-28-variable-width-gate-development.md",
}
PARENT_NAME = "paired-row-acquisition-v1-20261028-07235ac"
PARENT_SOURCE = "07235acd0ecaa576d57b37eefb2b4e6fc3aa0c6a"
PARENT_AUDIT = "363e46db8c437b40e435ef9d3873e6f834d5a877fb688136d024d7d05aa553c7"
INPUT_PINS = dict(
    cases_sha256="2d51bfe26130a1eebb054d71f1817a7e246005e9ea5a6f5824edc04cebc64cd6",
    manifest_sha256="ffe53561cd35a5cd6de6d27b7537a98a05ab804b3c49477654ded3723b705e7f",
    seal_sha256="f436992bc2f3b24fd593a373c2127235124626a58c82f95a5626594af2f76901",
    report_sha256="bf546c416a65e4d46b6fbd544cf0d19b130c17cf2f799a55f2cd365c5a51d2ec",
)
REQUIRED_REVIEW |= {
    "experiments/evidence/paired-row-acquisition-v1/outcome-audit/outcome-audit.json",
    *{
        "experiments/evidence/paired-row-acquisition-v1/actual/" + name
        for name in (
            "collection/public/cases.json",
            "collection/collection-report.json",
            "estimator/choices-sealed.json",
            "estimator/estimator-report.json",
        )
    },
}


def verify_source(
    repo: Path, config_path: Path, review_path: Path
) -> tuple[dict[str, Any], str]:
    config = read_json(config_path)
    review = read_json(review_path)
    if review.get("full_variable_gate_protocol_approved") is not True:
        raise ValueError("Full independent source/protocol approval required")
    if (
        canonical(config.get("policy")) != canonical(POLICY)
        or canonical(config.get("inventory")) != canonical(inventory())
        or config.get("runtime")
        != dict(python=platform.python_version(), numpy=np.__version__)
        or canonical(config.get("seeds")) != canonical(seed_map())
        or config.get("parent_directory") != PARENT_NAME
        or config.get("parent_source_commit") != PARENT_SOURCE
        or config.get("parent_outcome_audit_sha256") != PARENT_AUDIT
        or canonical(config.get("input_commitments")) != canonical(INPUT_PINS)
    ):
        raise ValueError("Exact policy/inventory/seeds/runtime required")
    signed = review.get("source_sha256", {})
    pins = config.get("frozen_dependencies", {})
    config_name = str(config_path.relative_to(repo))
    review_path.relative_to(repo)
    if (
        set(pins) != DEPENDENCIES
        or not REQUIRED_REVIEW | {config_name} <= set(signed)
        or any(signed.get(n) != h for n, h in pins.items())
        or signed.get(config_name) != digest(config_path, 128 * 1024)
    ):
        raise ValueError("Complete signed dependency/config/review set required")
    for name, h in signed.items():
        p = Path(name)
        if (
            p.is_absolute()
            or ".." in p.parts
            or not p.parts
            or digest(repo / p, 2 * 1024**2) != h
        ):
            raise ValueError("Reviewed source bytes changed")

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
    output.mkdir(exist_ok=False)
    p = inventory()
    store = ShardStore(
        output / "arrays",
        shard_limit_bytes=p["largest_shard_upper_bound"],
        total_limit_bytes=p["preflight_bytes_upper_bound"],
    )
    for name, fields in transport_schemas().items():
        arrays = {k: np.zeros(s, dtype=np.float64) for k, s in fields.items()}
        rec = store.save(name + ".npz", **arrays)
        restored = read_arrays(store.root / rec["name"], fields, rec["sha256"])
        if any(not np.array_equal(a, restored[k]) for k, a in arrays.items()):
            raise ValueError("Maximum literal transport mismatch")
    time.sleep(0.15)
    return dict(
        status="passed",
        manifest_sha256=digest(store.root / "manifest.jsonl"),
        shards=len(store.names),
        new_queries=0,
    )


def validate_transport(root: Path, receipt: dict[str, Any]) -> None:
    p = inventory()
    rec = records(root / "arrays")
    specs = transport_schemas()
    if (
        receipt.get("status") != "passed"
        or type(receipt.get("shards")) is not int
        or receipt["shards"] != len(specs)
        or receipt.get("manifest_sha256") != digest(root / "arrays/manifest.jsonl")
        or set(rec) != {n + ".npz" for n in specs}
    ):
        raise ValueError("Transport seal/inventory mismatch")
    verify_manifest(root / "arrays")
    for name, fields in specs.items():
        read_arrays(
            root / "arrays" / (name + ".npz"), fields, rec[name + ".npz"]["sha256"]
        )
    if (
        sum(x.stat().st_size for x in (root / "arrays").iterdir())
        > p["preflight_bytes_upper_bound"]
    ):
        raise ValueError("Transport byte cap exceeded")


def expected_case_names(job: str, result: GateResult) -> set[str]:
    names = {
        n
        for n in trace_names(job)
        if not n.startswith(job + "-r")
        or n in (job + "-rng-before", job + "-rng-after")
    }
    rounds = result.report.get("rounds")
    if not isinstance(rounds, list) or len(rounds) > 4:
        raise ValueError("Bounded round reports required")
    for i, row in enumerate(rounds):
        if type(row.get("round")) is not int or row["round"] != i:
            raise ValueError("Canonical rescue round order required")
        prefix = f"{job}-r{i}-"
        if row.get("status") == "unexpected_component_geometry":
            if i != len(rounds) - 1 or result.status != "unexpected_component_geometry":
                raise ValueError("Geometry stop metadata mismatch")
            names.update((prefix + "input", prefix + "geometry"))
        else:
            names.update(n for n in trace_names(job) if n.startswith(prefix))
    return names


def validate_case(
    job: GateJob, result: GateResult, store: TraceStore
) -> dict[str, Any]:
    inv = records(store.store.root)
    actual = {n[:-4] for n in inv if n.startswith(job.job_id + "-")}
    if actual != expected_case_names(job.job_id, result):
        raise ValueError("Incomplete/extraneous case trace")
    loaded = {}
    for name in sorted(actual):
        arrays = read_trace(
            store.store.root, name, job.n, job.rank, descriptor=inv[name + ".npz"]
        )
        if "-chunk" in name and (
            arrays["complete"][0] != 1
            or arrays["count"][0] != 8
            or arrays["attempted"][0] != 8
        ):
            raise ValueError("Completed case contains partial chunk")
        loaded[name] = arrays

    def get(suffix: str) -> dict[str, Any]:
        name = job.job_id + "-" + suffix
        return loaded[name]

    def refine_chunks(prefix: str, chunks: int, aggregate: dict[str, Any]) -> None:
        rows = [get(f"{prefix}-refine-chunk{i:03d}") for i in range(chunks)]
        if not np.array_equal(
            np.concatenate([a["refined"] for a in rows]), aggregate["refined"]
        ):
            raise ValueError("Refinement chunks disagree with aggregate")
        if not np.allclose(
            np.concatenate([a["losses"] for a in rows]),
            aggregate["losses"],
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError("Chunk losses disagree with aggregate")

    initial = get("initial-refine")
    pre = get("initial-pre-refine")
    if any(
        not np.array_equal(pre[k], initial[k])
        for k in ("starts", "optimized", "before_losses")
    ):
        raise ValueError("Initial pretrace disagrees with aggregate")
    refine_chunks("initial", 32, initial)
    if not np.array_equal(get("initial-merge")["losses"], initial["losses"]):
        raise ValueError("Initial merge losses disagree")
    if not np.array_equal(get("basis")["basis"], job.basis):
        raise ValueError("Saved basis disagrees with actual optimizer job")
    final = get("final")
    count = integer(final["count"], 0, 607)
    target = job.rank // 2
    if result.candidates.shape != (count, job.n) or not np.array_equal(
        result.candidates, final["directions"][:count]
    ):
        raise ValueError("Candidate/final trace mismatch")
    known = get("initial-refine")["refined"]
    merge = get("initial-merge")
    accepted = [int(i) for i in merge["order"] if merge["decision"][int(i)] == 2]
    known = known[accepted]
    if type(result.report.get("base_count")) is not int or result.report[
        "base_count"
    ] != len(known):
        raise ValueError("Base count disagrees with merge")
    for i, row in enumerate(result.report["rounds"]):
        before = get(f"r{i}-input")["known"]
        if (
            not np.array_equal(before, known)
            or type(row.get("input_count")) is not int
            or row["input_count"] != len(known)
        ):
            raise ValueError("Rescue input/count mismatch")
        geometry = get(f"r{i}-geometry")["geometry"]
        g = row.get("geometry", {})
        if (
            type(g.get("component_rank")) is not int
            or type(g.get("residual_dimension")) is not int
            or g.get("component_rank") != int(geometry[0])
            or g.get("residual_dimension") != int(geometry[1])
        ):
            raise ValueError("Geometry report mismatch")
        saved_geometry = get(f"r{i}-geometry")
        if not np.array_equal(saved_geometry["known"], before):
            raise ValueError("Geometry known directions changed")
        if row.get("status") == "unexpected_component_geometry":
            if geometry[0] == 2 * len(known) and geometry[1] == job.rank - 2 * len(
                known
            ):
                raise ValueError("False geometry failure")
            break
        if geometry[0] != 2 * len(known) or geometry[1] != job.rank - 2 * len(known):
            raise ValueError("Unexpected geometry proceeded")
        aggregate = get(f"r{i}-refine")
        pre = get(f"r{i}-pre-refine")
        if any(
            not np.array_equal(pre[k], aggregate[k])
            for k in ("starts", "before_losses")
        ):
            raise ValueError("Rescue pretrace disagrees")
        refine_chunks(f"r{i}", 64, aggregate)
        mixtures = get(f"r{i}-mixtures")
        eig = get(f"r{i}-eigen")
        mixrows = [get(f"r{i}-mix-pre{c:03d}") for c in range(32)]
        eigenrows = [get(f"r{i}-eigen-chunk{c:03d}") for c in range(32)]
        if any(
            not np.array_equal(np.concatenate([a[k] for a in mixrows]), mixtures[k])
            for k in ("raw", "normalized")
        ):
            raise ValueError("Mixture chunks disagree with aggregate")
        if any(
            not np.array_equal(np.concatenate([a[k] for a in eigenrows]), eig[k])
            for k in ("values", "vectors", "indices")
        ):
            raise ValueError("Eigen chunks disagree with aggregate")
        if not np.array_equal(
            np.concatenate([a["starts"] for a in eigenrows]), aggregate["starts"]
        ):
            raise ValueError("Eigen starts disagree with refinement")
        if not np.array_equal(get(f"r{i}-merge")["losses"], aggregate["losses"]):
            raise ValueError("Rescue merge losses disagree")
        ref = aggregate["refined"]
        mt = get(f"r{i}-merge")
        new = [int(k) for k in mt["order"] if mt["decision"][int(k)] == 2]
        known = np.concatenate((known, ref[new]))
        if type(row.get("retained_count")) is not int or row["retained_count"] != len(
            known
        ):
            raise ValueError("Round count disagrees with merge")
    if not np.array_equal(known, result.candidates):
        raise ValueError("Final candidates disagree with full merge trajectory")
    if (
        result.report.get("certificate") is not False
        or result.report.get("coverage_certified") is not False
        or result.report.get("private_gate") != "closed"
        or type(result.report.get("target_hypothesis")) is not int
        or result.report["target_hypothesis"] != target
        or type(result.report.get("final_count")) is not int
        or result.report["final_count"] != count
    ):
        raise ValueError("Uncertified count metadata required")
    status = (
        "zero_candidates"
        if count == 0
        else "overshoot"
        if count > target
        else "candidate_target"
        if count == target
        else "unexpected_component_geometry"
        if result.report["rounds"]
        and result.report["rounds"][-1].get("status") == "unexpected_component_geometry"
        else "rounds_exhausted"
    )
    if result.status != status or (
        status == "rounds_exhausted" and len(result.report["rounds"]) != 4
    ):
        raise ValueError("Status contradicts terminal trajectory")
    if result.report.get("job_id") != job.job_id:
        raise ValueError("Opaque report job mismatch")
    for phase in ("before", "after"):
        saved = get("rng-" + phase)
        states = {
            key: decode_state(saved[key], saved[key + "_length"])
            for key in ("initial", "rescue")
        }
        if canonical(result.report.get("rng_" + phase)) != canonical(states):
            raise ValueError("Report RNG states disagree with trace")
    if integer(get("view-residuals")["count"], 0, 607) != count:
        raise ValueError("View diagnostics count mismatch")
    maximum = result.report.get("max_retained_loss")
    if (
        count
        and (
            type(maximum) is not float
            or maximum != float(final["losses"][:count].max())
        )
    ) or (not count and maximum is not None):
        raise ValueError("Report maximum loss mismatch")
    recomputed = losses(result.candidates, job.basis)
    if count and (
        not np.allclose(
            np.linalg.norm(result.candidates, axis=1), 1, atol=1e-10, rtol=0
        )
        or np.any(recomputed > 1e-8)
        or not np.allclose(recomputed, final["losses"][:count], atol=1e-14, rtol=1e-12)
    ):
        raise ValueError("Final numerical geometry mismatch")
    cos = np.abs(result.candidates @ result.candidates.T)
    if count:
        np.fill_diagonal(cos, 0)
        if np.any(cos >= 0.9995) or not np.allclose(
            cos, final["cosines"], atol=1e-13, rtol=0
        ):
            raise ValueError("Final nonduplicate geometry mismatch")
    return dict(
        job_id=job.job_id,
        status="completed",
        candidate_status=status,
        rank=job.rank,
        target_hypothesis=target,
        final_count=count,
        rounds=len(result.report["rounds"]),
        certificate=False,
    )


def view_diagnostics(
    job: GateJob,
    result: GateResult,
    store: TraceStore,
    est: Path,
    provenance: dict[str, Any],
) -> None:
    index = 2 * int(job.job_id[-2:]) + 1
    name = f"case{index:02d}-bases.npz"
    p = 528
    schema: dict[str, tuple[int, ...]] = {
        k + "_basis": (p, p) for k in ("fine", "coarse", "richardson")
    }
    schema.update({k + "_spectrum": (p,) for k in ("fine", "coarse", "richardson")})
    arrays = read_arrays(
        est / "arrays" / name, schema, provenance["estimator_sha256"][name]
    )
    count = len(result.candidates)
    values = (
        np.stack(
            [
                losses(result.candidates, arrays[k + "_basis"][: job.rank])
                for k in ("fine", "coarse", "richardson")
            ]
        )
        if count
        else np.zeros((3, 1))
    )
    store.write(
        job.job_id + "-view-residuals",
        dict(losses=values, count=np.array([count], dtype=np.float64)),
    )


def final_layout(output: Path, store: TraceStore) -> int:
    rec = records(store.store.root)
    verify_manifest(store.store.root)
    p = inventory()
    if (
        len(rec) > p["trace_shards"]
        or (store.store.root / "manifest.jsonl").stat().st_size
        > p["manifest_bytes_upper_bound"]
    ):
        raise ValueError("Trace record/manifest cap exceeded")
    total = sum(x.stat().st_size for x in output.rglob("*") if x.is_file())
    for x in output.rglob("*"):
        _ancestry(x)
    allowed = {
        "controller-report.json",
        "adapter-report.json",
        "trace-progress.json",
        "seed-map.json",
        "outer-monitor.json",
    } | {f"reports/job{i:02d}.json" for i in range(6)}
    expected = (
        allowed - {"outer-monitor.json"}
        | {"trace/manifest.jsonl", "transport/arrays/manifest.jsonl"}
        | {"trace/" + n for n in rec}
        | {"transport/arrays/" + n for n in records(output / "transport/arrays")}
    )
    actual = {str(x.relative_to(output)) for x in output.rglob("*") if x.is_file()}
    if actual != expected:
        raise ValueError("Exact final file layout required")
    allowed_dirs = {"scope-engine", "transport-engine"} | {
        str(parent)
        for name in expected
        for parent in Path(name).parents
        if str(parent) != "."
    }
    if {
        str(x.relative_to(output)) for x in output.rglob("*") if x.is_dir()
    } - allowed_dirs:
        raise ValueError("Unexpected final directory")
    meta = [x for x in output.rglob("*") if x.is_file() and x.suffix == ".json"]
    if (
        len(meta) > 32
        or any(x.stat().st_size > 128 * 1024 for x in meta)
        or total + p["parent_retained_bytes_upper_bound"] + p["metadata_reserve_bytes"]
        >= 1024**3
    ):
        raise ValueError("Combined artifact/metadata ceiling exceeded")
    return total


def streams(job_id: str) -> tuple[np.random.Generator, np.random.Generator]:
    values = seed_map()[job_id]
    return np.random.default_rng(values["initial"]), np.random.default_rng(
        values["rescue"]
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
    config, head = verify_source(repo, config_path, review_path)
    _ancestry(output)
    expected_parent = repo.parent / "experiments"
    _ancestry(expected_parent)
    if output.absolute().parent != expected_parent.absolute():
        raise ValueError("Durable direct experiment child required")
    expected_parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(
        status="started",
        source_commit=head,
        certificate=False,
        private_gate="closed",
        new_queries=0,
        automatic_retry=False,
        registered_rng_started=False,
        cases=[dict(job_id=f"job{i:02d}", status="not_started") for i in range(6)],
    )
    path = output / "controller-report.json"
    save_json(path, report)
    config_sha = digest(config_path, 128 * 1024)
    review_sha = digest(review_path, 128 * 1024)

    def unchanged() -> None:
        _, current = verify_source(repo, config_path, review_path)
        if (
            current != head
            or digest(config_path, 128 * 1024) != config_sha
            or digest(review_path, 128 * 1024) != review_sha
        ):
            raise ValueError("Source/config/review changed")

    def remaining() -> float:
        seconds = 180 - (clock() - started)
        if seconds <= 0:
            raise TimeoutError("Shared180s deadline expired")
        return seconds

    env = dict(os.environ)
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        PYTHONPATH=str(repo / "src"),
    )
    try:
        parent = repo.parent / "experiments"
        _ancestry(parent)
        parent.mkdir(exist_ok=True)
        if output.absolute().parent != parent.absolute():
            raise ValueError("Durable direct experiment child required")
        marker = parent / MARKER
        if marker.exists() or marker.is_symlink():
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
            env=env,
        )
        report["scope_receipt"] = scope
        save_json(path, report)
        if not tree_ok(scope) or scope.get("status") != "passed":
            raise ValueError("Predata TREE monitoring required")
        transport = engine(
            [sys.executable, str(Path(__file__).absolute())],
            dict(operation="transport", output=str(output / "transport")),
            cwd=output / "transport-engine",
            memory_limit_mb=512,
            timeout_seconds=remaining(),
            output_limit_bytes=2048,
            env=env,
        )
        report["transport_receipt"] = transport
        save_json(path, report)
        if not tree_ok(transport):
            raise ValueError("Transport TREE monitoring required")
        validate_transport(output / "transport", transport)
        if shutil.disk_usage(output).free < inventory()["minimum_free_disk_bytes"]:
            raise ValueError("At least2GiB free disk required")
        unchanged()
        report.update(
            scope_receipt=scope,
            transport_receipt=transport,
            config_sha256=config_sha,
            review_sha256=review_sha,
        )
        report["marker_sha256"] = fsynced_marker(
            marker,
            dict(
                policy=POLICY,
                source_commit=head,
                config_sha256=config_sha,
                review_sha256=review_sha,
                output=str(output.absolute()),
                seeds=seed_map(),
                input_commitments=config["input_commitments"],
            ),
        )
        save_json(path, report)
        data = parent / config["parent_directory"]
        public = data / "collection/public"
        est = data / "estimator"
        bundle = load_bundle(public, est, config["input_commitments"])
        if len(bundle.jobs) != 6 or any(j.n != 32 for j in bundle.jobs):
            raise ValueError("Exact6 n32 jobs required")
        save_json(output / "adapter-report.json", bundle.provenance)
        save_json(output / "seed-map.json", dict(seeds=seed_map(), source_commit=head))
        (output / "reports").mkdir()
        store = TraceStore(
            output / "trace",
            output / "trace-progress.json",
            {j.job_id: (j.n, j.rank) for j in bundle.jobs},
        )
        for i, job in enumerate(bundle.jobs):
            remaining()
            unchanged()
            report["cases"][i].update(status="started")
            report["active_job"] = job.job_id
            save_json(path, report)
            store.write(job.job_id + "-basis", dict(basis=job.basis))
            report["registered_rng_started"] = True
            save_json(path, report)
            initial_rng, rescue_rng = streams(job.job_id)
            result = search_job(job, initial_rng, rescue_rng, store)
            view_diagnostics(job, result, store, est, bundle.provenance)
            summary = validate_case(job, result, store)
            job_report = output / "reports" / (job.job_id + ".json")
            save_json(
                job_report,
                dict(
                    result.report,
                    candidate_status=result.status,
                    source_commit=head,
                    new_queries=0,
                ),
            )
            summary["report_sha256"] = digest(job_report, 128 * 1024)
            report["cases"][i] = summary
            save_json(path, report)
        # Revalidate parent public bindings and own output hashes after every job is terminal.
        load_bundle(public, est, config["input_commitments"])
        unchanged()
        remaining()
        validate_transport(output / "transport", transport)
        if digest(marker, 128 * 1024) != report["marker_sha256"]:
            raise ValueError("Global marker changed")
        total = final_layout(output, store)
        report.update(
            status="completed",
            active_job=None,
            retained_bytes_before_final_report=total,
            elapsed_seconds_before_final_report=clock() - started,
            candidate_target_jobs=sum(
                c["candidate_status"] == "candidate_target" for c in report["cases"]
            ),
            method_promotion=False,
            trace_manifest_sha256=digest(store.store.root / "manifest.jsonl", 1024**2),
        )
        save_json(path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc)[:800],
            uncommitted_chunk="unknown",
            unstarted_cases="not_started",
        )
        for case in report["cases"]:
            if case["status"] == "started":
                case["status"] = "failed_or_partial"
        save_json(path, report)
        raise


def launch(
    request: dict[str, Any], *, engine: Callable[..., dict[str, Any]] = run_json
) -> dict[str, Any]:
    output = Path(request["output"])
    _ancestry(output)
    if output.exists():
        raise ValueError("Existing output preserved; no retry")
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
        certificate=False,
        private_gate="closed",
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
        controller = read_json(output / "controller-report.json")
        if (
            not tree_ok(receipt)
            or receipt.get("status") != "completed"
            or controller.get("status") != "completed"
        ):
            raise ValueError("Outer TREE or controller incomplete")
        report.update(
            status="completed",
            controller_sha256=digest(output / "controller-report.json", 128 * 1024),
            controller_counters_unknown=False,
        )
        save_json(output / "outer-monitor.json", report)
        return report
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)[:800]
        )
        if output.exists():
            save_json(output / "outer-monitor.json", report)
        raise


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        if request.get("operation") == "transport":
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
                {
                    k: result[k]
                    for k in ("status", "manifest_sha256", "shards", "new_queries")
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
