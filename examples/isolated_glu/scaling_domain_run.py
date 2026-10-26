"""Sequential supplied-array dimension diagnostic; no fitting or hidden labels.

Registered data generation is separate and must be behind reviewed controller gates.
Every signed row is evaluated, grouped sign-major into one4p-row batch per probe/step.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray

from precision import forward as forward, reference_weights
from scaling_domain_math import (
    analytic_hessian,
    curvature_from_responses,
    direction_residuals,
    four_corner_inputs,
    richardson,
    spectrum,
    symmetric_atoms,
    svec,
)
from scaling_domain_store import ShardStore, verify_manifest as verify_manifest, NAME

Array = NDArray[np.float64]
PROFILES = ((4, 8, 4), (8, 16, 8), (16, 32, 16), (32, 64, 32))
CHANNELS = ("fp64", "fp32_compute")
STEPS = (0.3, 0.6)
EXPECTED_CALLS = 1454080


def save_json(path: Path, report: dict[str, Any]) -> None:
    encoded = (json.dumps(report, allow_nan=False, sort_keys=True) + "\n").encode()
    if len(encoded) > 128 * 1024:
        raise ValueError("Bounded128KiB report required")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f:
        if f.write(encoded) != len(encoded):
            raise OSError("Short progress write")
    temporary.replace(path)


def inventory_plan() -> dict[str, int]:
    """Conservative fixed-shape byte inventory; no arrays/RNG/scientific data.

    Per NPZ array4096header/ZIP bytes, per manifest record exact maximum-schema bytes,1MiB metadata
    and8MiB synthetic transport allowance. Runtime actual bytes remain hard-capped.
    """
    total = calls = shards = largest = 0
    for n, m, o in PROFILES:
        p = n * (n + 1) // 2
        # Context: original + rounded factors, probes/random controls, atoms, z.
        context = (
            8 * (2 * (2 * m * n + o * m) + 128 * n + 2 * m * p + 64 * m) + 10 * 4096
        )
        raw_input = 8 * 4 * p * n + 4096
        raw_output = 8 * 4 * p * o + 4096
        hessian = 8 * 8 * o * n * n + 4096
        spectral = 8 * (p * p + p + m + 64) + 4 * 4096
        per_case = (
            context + 128 * (raw_input + raw_output) + 32 * hessian + 5 * spectral
        )
        per_shards = 1 + 256 + 32 + 5
        total += 4 * per_case
        calls += 4 * 64 * 2 * 4 * p
        shards += 4 * per_shards
        largest = max(largest, context, raw_input, raw_output, hessian, spectral)
    max_record = len(
        (
            json.dumps(
                {"name": "x" * 80 + ".npz", "bytes": 8 * 1024**2, "sha256": "0" * 64},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )
    manifest = shards * max_record
    return {
        "cases": 16,
        "shards": shards,
        "vector_calls": calls,
        "largest_shard_upper_bound": largest,
        "manifest_bytes_upper_bound": manifest,
        "bytes_upper_bound": total + manifest + 1024**2 + 8 * 1024**2,
    }


class QueryLedger:
    """One attempted stream; persist inputs and intent before invoking callback.

    Failed/unknown calls consume attempted reservation. External kill may leave
    counters as lower bounds; never repair/retry. Reserved!=executed!=returned.
    """

    def __init__(self, path: Path, reserved: int) -> None:
        if type(reserved) is not int or reserved < 1 or reserved > EXPECTED_CALLS:
            raise ValueError("Bounded positive vector reservation required")
        self.path = path
        self.counts = dict(
            calls_reserved=reserved,
            calls_attempted=0,
            calls_returned=0,
            calls_persisted=0,
        )
        self.terminal = False
        with path.open("x") as f:
            f.write(json.dumps({"status": "reserved", **self.counts}) + "\n")

    def query(
        self,
        store: ShardStore,
        name: str,
        x: Array,
        callback: Callable[[Array], Array],
        outputs: int,
    ) -> Array:
        if (
            not isinstance(x, np.ndarray)
            or x.dtype != np.float64
            or x.ndim != 2
            or not 1 <= len(x) <= 2112
            or not 1 <= x.shape[1] <= 32
            or x.nbytes > store.shard_limit_bytes - 4096
        ):
            raise ValueError("Bounded nativeFP64 query input required before copy")
        if (
            self.terminal
            or self.counts["calls_attempted"] + len(x) > self.counts["calls_reserved"]
        ):
            raise ValueError("Query ledger terminal or reservation exceeded; no retry")
        try:
            xx = np.array(x, dtype=np.float64, copy=True)
            store.save(name + "-inputs.npz", inputs=xx)
            original = xx.tobytes()
            xx.setflags(write=False)
            self.counts["calls_attempted"] += len(xx)
            save_json(self.path, {"status": "attempting", "batch": name, **self.counts})
            returned = callback(xx)
            self.counts["calls_returned"] += len(xx)
            # Bound native schema BEFORE any copy/allocation or silent complex cast.
            if (
                not isinstance(returned, np.ndarray)
                or returned.dtype != np.float64
                or returned.shape != (len(xx), outputs)
                or not 1 <= outputs <= 32
                or returned.nbytes > store.shard_limit_bytes - 4096
            ):
                raise ValueError(
                    "Invalid returned response shape/nativeFP64 schema before copy; raw not persisted"
                )
            # Detach immediately so later callback buffer reuse cannot alter evidence.
            y = np.array(returned, dtype=np.float64, copy=True)
            store.save(name + "-responses.npz", responses=y)
            self.counts["calls_persisted"] += len(xx)
            save_json(self.path, {"status": "persisted", "batch": name, **self.counts})
            if xx.tobytes() != original:
                raise ValueError("Query input mutation; saved original inputs retained")
            return y
        except Exception as exc:
            self.terminal = True
            try:
                save_json(
                    self.path,
                    {
                        "status": "failed",
                        "batch": name,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        **self.counts,
                    },
                )
            except Exception:
                pass  # Keep the original failure; final controller must fail too.
            raise


def _relative(estimate: Array, reference: Array) -> float | None:
    scale = float(max(np.max(np.abs(estimate)), np.max(np.abs(reference))))
    if scale == 0:
        return None
    a, b = estimate / scale, reference / scale
    if np.any((estimate != 0) & (a == 0)) or np.any((reference != 0) & (b == 0)):
        raise ValueError("Relative metric normalization underflow")

    def norm(x: Array) -> float:
        maximum = float(np.max(np.abs(x)))
        return 0.0 if maximum == 0 else float(maximum * np.linalg.norm(x / maximum))

    denominator = norm(b)
    if denominator == 0:
        return None
    numerator = norm(a - b)
    value = numerator / denominator
    if numerator > 0 and value == 0:
        raise ValueError("Relative ratio underflow")
    if not np.isfinite(value):
        raise ValueError("Nonfinite relative metric")
    return value


def _validate_context(
    g: Array, u: Array, d: Array, probes: Array, random_directions: Array
) -> None:
    for a in (g, u, d, probes, random_directions):
        if not isinstance(a, np.ndarray) or a.dtype != np.float64 or a.ndim != 2:
            raise ValueError("Explicit nativeFP64 matrices required")
        if a.size == 0 or a.nbytes > 8 * 1024**2 or not np.isfinite(a).all():
            raise ValueError("Bounded nonempty finite context required")
    m, n = g.shape
    if (
        not 1 <= n <= 32
        or not 1 <= m <= 64
        or u.shape != g.shape
        or d.shape[1] != m
        or not 1 <= len(d) <= 32
        or probes.shape[1] != n
        or not 1 <= len(probes) <= 64
        or random_directions.shape[1] != n
        or not 1 <= len(random_directions) <= 64
    ):
        raise ValueError("Incompatible bounded context shapes")


def run_case(
    store: ShardStore,
    ledger: QueryLedger,
    case: str,
    *,
    g: Array,
    u: Array,
    d: Array,
    probes: Array,
    random_directions: Array,
    channel: str,
    query: Callable[[Array], Array],
) -> dict[str, Any]:
    """One supplied-array privileged diagnostic; no registered RNG or learner."""
    if channel not in CHANNELS or not NAME.fullmatch(case + ".npz") or len(case) > 20:
        raise ValueError("Fixed channel/simple case identifier required")
    _validate_context(g, u, d, probes, random_directions)
    context_inputs = (g, u, d, probes, random_directions)
    context_bytes = tuple(a.tobytes() for a in context_inputs)
    gr, ur, dr = reference_weights(g, u, d, channel)
    n = g.shape[1]
    p = n * (n + 1) // 2
    atoms = symmetric_atoms(gr, ur)
    store.save(
        case + "-context.npz",
        g=g,
        u=u,
        d=d,
        reference_g=gr,
        reference_u=ur,
        reference_d=dr,
        probes=probes,
        random_directions=random_directions,
        atoms=atoms,
        preactivations=probes @ gr.T,
    )
    h_ref = analytic_hessian(gr, ur, dr, probes)
    estimates = []
    for step_index, step in enumerate(STEPS):
        h = np.empty_like(h_ref)
        for row, x in enumerate(probes):
            q = four_corner_inputs(x, step)
            y = ledger.query(
                store,
                f"{case}-s{step_index}-x{row:02d}",
                q.reshape(-1, n),
                query,
                len(d),
            )
            if any(
                a.tobytes() != b
                for a, b in zip(context_inputs, context_bytes, strict=True)
            ):
                ledger.terminal = True
                save_json(
                    ledger.path,
                    {"status": "failed", "reason": "context mutation", **ledger.counts},
                )
                raise ValueError(
                    "Privileged context mutation; original context shard retained"
                )
            h[row] = curvature_from_responses(y.reshape(4, p, len(d)), n, step)
        estimates.append(h)
    values = {
        "reference": h_ref,
        "fine": estimates[0],
        "coarse": estimates[1],
        "richardson": richardson(*estimates),
    }
    spaces: dict[str, Any] = {}
    reference_projector: Array | None = None
    for kind, value in values.items():
        for start in range(0, len(value), 8):
            store.save(
                f"{case}-{kind}-h{start // 8:02d}.npz", hessian=value[start : start + 8]
            )
    for kind, rows in [
        ("atoms", atoms),
        *[(k, svec(v).reshape(-1, p)) for k, v in values.items()],
    ]:
        spec = spectrum(rows)
        basis = spec["basis"]
        projector = basis.T @ basis
        gates = direction_residuals(basis, gr)
        controls = direction_residuals(basis, random_directions)
        store.save(
            f"{case}-spectrum-{kind}.npz",
            singular_values=spec["singular_values"],
            projector=projector,
            gate_residuals=gates,
            random_residuals=controls,
        )
        item: dict[str, Any] = {
            "rank": spec["rank"],
            "gap": spec["gap"],
            "gap_reason": spec["gap_reason"],
            "full_space": spec["rank"] == p,
            "gate_residual_min": float(gates.min()),
            "gate_residual_max": float(gates.max()),
            "random_residual_min": float(controls.min()),
            "random_residual_max": float(controls.max()),
        }
        if kind == "reference":
            reference_projector = projector
        if kind in ("fine", "coarse", "richardson"):
            assert reference_projector is not None
            item["hessian_relative_error"] = _relative(values[kind], h_ref)
            item["hessian_error_null_reason"] = (
                "zero_reference" if item["hessian_relative_error"] is None else None
            )
            item["reference_projector_frobenius_distance"] = float(
                np.linalg.norm(projector - reference_projector)
            )
        spaces[kind] = item
    z = probes @ gr.T
    return {
        "case": case,
        "channel": channel,
        "n": n,
        "width": len(g),
        "outputs": len(d),
        "p": p,
        "spaces": spaces,
        "preactivation_mean": float(z.mean()),
        "preactivation_std": float(z.std()),
        "preactivation_min": float(z.min()),
        "preactivation_max": float(z.max()),
        "privileged_diagnostics_only": True,
        "recovery_claim": False,
        "private_gate": "closed",
    }


POLICY = {
    "id": "isolated-glu-dimension-domain-v1",
    "profiles": [list(p) for p in PROFILES],
    "teacher_seeds": [[684031 + 100 * j, 684032 + 100 * j] for j in range(4)],
    "probe_seeds": [684033 + 100 * j for j in range(4)],
    "random_direction_seeds": [684034 + 100 * j for j in range(4)],
    "weight_std": 0.2,
    "probe_std": 4,
    "probe_count": 64,
    "channels": list(CHANNELS),
    "steps": list(STEPS),
    "rank_rtol": 1e-10,
    "vector_calls": EXPECTED_CALLS,
    "batching": "sign_major_4p_per_probe_step_no_dedup",
    "timeout_seconds": 180,
    "memory_limit_mb": 512,
    "blas_threads": 1,
    "worker_output_limit_bytes": 2048,
}


def registered_contexts() -> Any:
    """Called ONLY after full-source/controller/resource gates + global marker.

    Explicitly privileged PUBLIC owned diagnostic; no learner receives weights.
    """
    for j, (n, m, o) in enumerate(PROFILES):
        probes = np.random.default_rng(684033 + 100 * j).normal(size=(64, n)) * 4
        directions = np.random.default_rng(684034 + 100 * j).normal(size=(64, n))
        for teacher in range(2):
            rng = np.random.default_rng(684031 + 100 * j + teacher)
            g = rng.normal(size=(m, n)) * 0.2
            u = rng.normal(size=(m, n)) * 0.2
            d = rng.normal(size=(o, m)) * 0.2
            for channel in CHANNELS:
                case = f"p{j}-t{teacher}-" + ("fp64" if channel == "fp64" else "fp32")
                yield (
                    case,
                    channel,
                    dict(g=g, u=u, d=d, probes=probes, random_directions=directions),
                )


def build(request: dict[str, Any]) -> dict[str, Any]:
    """Controller-only registered worker. Invalid gates fail before RNG or data."""
    from scaling_domain_store import _ancestry, sha

    output = Path(request["output"])
    _ancestry(output)
    output.mkdir(exist_ok=False)
    path = output / "worker-report.json"
    report: dict[str, Any] = {
        "status": "started",
        "cases": [],
        "private_gate": "closed",
        "new_fits": 0,
        "full_map_recovery_verified": False,
        "calls_reserved": None,
    }
    ledger: QueryLedger | None = None
    try:
        if (
            request.get("policy") != POLICY
            or request.get("full_source_protocol_approved") is not True
        ):
            raise ValueError(
                "Exact frozen full-source protocol approval required before generation"
            )
        for key in ("scope_preflight", "transport_preflight"):
            receipt = request.get(key, {})
            if (
                receipt.get("process_monitor_scope") != "process_and_children"
                or receipt.get("memory_samples", 0) < 1
            ):
                raise ValueError("Pre-data process-and-children monitoring required")
        if request["transport_preflight"].get("status") != "passed":
            raise ValueError("Pre-data transport incomplete")
        inventory = inventory_plan()
        if (
            inventory["vector_calls"] != EXPECTED_CALLS
            or inventory["shards"] != 4704
            or inventory["bytes_upper_bound"] > 1024**3
            or inventory["manifest_bytes_upper_bound"] > 1024**2
            or inventory["largest_shard_upper_bound"] > 8 * 1024**2
        ):
            raise ValueError("Full combined artifact inventory exceeds frozen budget")
        repo = Path(request["repo"]).absolute()
        marker = (
            repo.parent / "experiments/isolated-glu-dimension-domain-v1-attempt.json"
        )
        _ancestry(marker)
        # Canonical root is global across output directories. Any partial marker blocks reuse.
        with marker.open("x") as f:
            json.dump(
                {
                    "policy": POLICY,
                    "calls_reserved": EXPECTED_CALLS,
                    "source_commit": request["source_commit"],
                    "config_sha256": request["config_sha256"],
                    "output": str(output.absolute()),
                    "automatic_retry": False,
                },
                f,
                allow_nan=False,
            )
        report.update(
            marker=str(marker),
            marker_sha256=sha(marker),
            calls_reserved=EXPECTED_CALLS,
            source_commit=request["source_commit"],
            inventory_plan=inventory,
        )
        save_json(path, report)
        ledger = QueryLedger(output / "progress.json", EXPECTED_CALLS)
        store = ShardStore(output / "arrays")
        for case, channel, context in registered_contexts():

            def query(x: Array) -> Array:
                return forward(context["g"], context["u"], context["d"], x, channel)

            result = run_case(
                store, ledger, case, query=query, channel=channel, **context
            )
            report["cases"].append(result)
            report.update(ledger.counts)
            save_json(path, report)
        inventory_actual = verify_manifest(store.root)
        if (
            len(report["cases"]) != 16
            or inventory_actual["shards"] != 4704
            or any(v != EXPECTED_CALLS for v in ledger.counts.values())
        ):
            raise ValueError("Incomplete fixed diagnostic cases/calls/artifacts")
        report.update(
            status="completed",
            artifact_manifest_sha256=sha(store.root / "manifest.jsonl"),
            artifact_inventory=inventory_actual,
            **ledger.counts,
        )
        save_json(path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)[:1000]
        )
        if ledger is not None:
            report.update(ledger.counts)
        try:
            save_json(path, report)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    import sys
    from scaling_domain_store import sha

    request = json.load(sys.stdin)
    try:
        if request.get("operation") == "synthetic_transport":
            from scaling_domain_store import synthetic_preflight

            result = synthetic_preflight(Path(request["output"]))
            print(json.dumps(result, allow_nan=False))
        else:
            result = build(request)
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "worker_report_sha256": sha(
                            Path(request["output"]) / "worker-report.json"
                        ),
                        **{
                            k: result[k]
                            for k in [
                                "calls_reserved",
                                "calls_attempted",
                                "calls_returned",
                                "calls_persisted",
                            ]
                        },
                    }
                )
            )
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}
            )
        )
