"""Trace-only extension of the frozen cone calculation; no actual fit/target access."""

from __future__ import annotations
from collections.abc import Callable
from copy import deepcopy
import importlib.util
from pathlib import Path
import re
from typing import Any
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Sink = Callable[[dict[str, Any]], None]
spec = importlib.util.spec_from_file_location(
    "frozen_trace_cone", Path(__file__).with_name("guard_aware_direction.py")
)
assert spec and spec.loader
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
TOL = frozen.TOL
RANK_TOL = frozen.RANK_TOL
GEOMETRY_TOL = frozen.GEOMETRY_TOL
certificate = frozen.certificate
certificate_details = frozen.certificate_details
exact_row_rank = frozen.exact_row_rank


class TraceWriteError(OSError):
    """An observer write failed; never continue computing other subsets."""


class TraceError(ValueError):
    def __init__(self, message: str, trace: dict[str, Any]):
        super().__init__(message)
        self.trace = trace


def _emit(trace: dict[str, Any], sink: Sink | None, event: dict[str, Any]) -> None:
    ordinal = trace["events_attempted"]
    trace["events_attempted"] += 1
    if sink is not None:
        payload = dict(
            event,
            ordinal=ordinal,
            proposal_id=trace["proposal_id"],
            stage=trace["stage"],
        )
        try:
            sink(deepcopy(payload))
        except Exception as exc:
            raise TraceWriteError(str(exc)) from exc
        trace["events_delivered"] += 1


def traced_direction(
    gate: Array,
    d: Array,
    k: Array,
    a: Array,
    *,
    proposal_id: str = "synthetic",
    event_sink: Sink | None = None,
) -> tuple[Array, dict[str, Any]]:
    trace: dict[str, Any] = {
        "schema_version": 1,
        "proposal_id": proposal_id,
        "stage": "input_validation",
        "subsets": [],
        "linear_solves_attempted": 0,
        "events_attempted": 0,
        "events_delivered": 0,
        "journal_enabled": event_sink is not None,
    }
    try:
        if not isinstance(proposal_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}", proposal_id
        ):
            raise ValueError("Invalid bounded proposal id")
        _emit(trace, event_sink, {"kind": "proposal_started"})
        direction, info = _compute(gate, d, k, a, trace, event_sink)
        trace["result"] = {
            name: value for name, value in info.items() if name != "subsets"
        }
        trace.update(stage="completed", terminal_status="completed")
        _emit(
            trace,
            event_sink,
            {"kind": "proposal_completed", "selected_mask": info["selected_mask"]},
        )
        return direction, trace
    except Exception as exc:
        trace.update(
            terminal_status="failed",
            failure_kind="trace_write"
            if isinstance(exc, TraceWriteError)
            else "computation",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        if trace["subsets"] and trace["subsets"][-1]["status"] == "started":
            trace["subsets"][-1].update(status="aborted", error=str(exc))
        if not isinstance(exc, TraceWriteError):
            try:
                _emit(
                    trace,
                    event_sink,
                    {
                        "kind": "proposal_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            except TraceWriteError as write:
                trace.update(failure_kind="trace_write", journal_error=str(write))
        raise TraceError(str(exc), trace) from exc


def _compute(
    gate: Array, d: Array, k: Array, a: Array, trace: dict[str, Any], sink: Sink | None
) -> tuple[Array, dict[str, Any]]:
    n = len(gate) if gate.ndim == 1 else 0
    if (
        not 1 <= n <= 32
        or d.shape != (n,)
        or k.shape != (n, n)
        or a.ndim != 2
        or a.shape[1] != n
        or len(a) > 8
        or any(
            v.dtype != np.float64 or not np.isfinite(v).all() for v in [gate, d, k, a]
        )
        or not np.array_equal(k, k.T)
    ):
        raise ValueError(
            "Finite FP64 gate/d/K/A required within32dimensions/eight normals"
        )
    trace["inputs"] = {
        "gate": gate.tolist(),
        "gradient": d.tolist(),
        "gram": k.tolist(),
        "constraints": a.tolist(),
    }
    trace["stage"] = "gate_and_row_norms"
    gate_norm = float(np.linalg.norm(gate))
    row_norms = np.linalg.norm(a, axis=1)
    if (
        gate_norm <= 0
        or not np.isfinite(gate_norm)
        or np.any(row_norms <= 0)
        or not np.isfinite(row_norms).all()
    ):
        raise ValueError("Nonzero finite gate/normals required")
    trace["stage"] = "gram_eigensystem"
    eig = np.linalg.eigvalsh(k)
    if not np.isfinite(eig).all() or eig[-1] <= 0 or eig[0] <= RANK_TOL * eig[-1]:
        raise ValueError(
            "K must be strictly well-conditioned SPD; no ridge/pseudoinverse"
        )
    beta = float(eig[-1])
    kn = k / beta
    dn = d / beta
    bn = a / row_norms[:, None]
    if not np.isfinite(dn).all():
        raise ValueError("Nonfinite scaled gradient")
    trace["normalization"] = {
        "objective_scale": beta,
        "row_norms": row_norms.tolist(),
        "gram_eigenvalues": eig.tolist(),
        "scaled_gram": kn.tolist(),
        "scaled_gradient": dn.tolist(),
        "normalized_constraints": bn.tolist(),
    }
    records: list[dict[str, Any]] = trace["subsets"]
    best: tuple[float, int, Array, Array, dict[str, float]] | None = None
    for mask in range(1 << len(a)):
        active = [i for i in range(len(a)) if mask & (1 << i)]
        rows = bn[active]
        record: dict[str, Any] = {
            "mask": mask,
            "active": active,
            "status": "started",
            "solution": None,
            "normalized_duals": None,
            "solution_finite": None,
        }
        records.append(record)
        trace["stage"] = "subset_geometry"
        _emit(trace, sink, {"kind": "subset_started", "mask": mask, "active": active})
        if active:
            sv = np.linalg.svd(rows, compute_uv=False)
            record.update(
                singular_values=sv.tolist(), singular_ratio=float(sv[-1] / sv[0])
            )
            if len(active) > n or sv[-1] <= GEOMETRY_TOL * sv[0]:
                # A numerically discarded but exactly independent support can
                # hide a large geometric error behind tiny KKT residuals.
                # Reject globally, even if another approximate candidate passed.
                rank = exact_row_rank(a[active]) if len(active) <= n else n
                if len(active) <= n and rank == len(active):
                    record.update(
                        exact_original_rank=rank,
                        reason="exact_independent_near_dependency",
                    )
                    raise ValueError(
                        "Exactly independent near-dependent active normals; no numerical optimum claimed"
                    )
                record.update(
                    status="dependent_active_rows",
                    singular_values=sv.tolist(),
                    exact_original_rank=rank if len(active) <= n else None,
                    reason="exact_rational_dependency"
                    if len(active) <= n
                    else "more_rows_than_dimensions",
                )
                _emit(trace, sink, {"kind": "subset_completed", "record": record})
                continue
        try:
            trace["stage"] = "subset_solve"
            if active:
                matrix = np.block(
                    [[2 * kn, rows.T], [rows, np.zeros((len(active), len(active)))]]
                )
                trace["linear_solves_attempted"] += 1
                solved = np.linalg.solve(
                    matrix, np.concatenate([-dn, np.zeros(len(active))])
                )
                z = solved[:n]
                mu = np.zeros(len(a))
                mu[active] = solved[n:]
            else:
                trace["linear_solves_attempted"] += 1
                z = np.linalg.solve(2 * kn, -dn)
                mu = np.zeros(len(a))
            finite = bool(np.isfinite(z).all() and np.isfinite(mu).all())
            record["solution_finite"] = finite
            if finite:
                record.update(solution=z.tolist(), normalized_duals=mu.tolist())
            else:
                record["nonfinite_solution_ieee754_hex"] = (
                    np.concatenate([z, mu]).astype("<f8", copy=False).tobytes().hex()
                )
                record["nonfinite_solution_encoding"] = {
                    "dtype": "<f8",
                    "layout": "z_then_mu",
                    "z_count": n,
                    "mu_count": len(a),
                }
            if not np.isfinite(z).all() or not np.isfinite(mu).all():
                raise ValueError("Nonfinite subset solution")
            checks = certificate(z, mu, dn, kn, bn)
            objective = float(dn @ z + z @ kn @ z)
            if not np.isfinite(objective):
                raise ValueError("Nonfinite subset objective")
            passed = all(v <= TOL for v in checks.values())
            record.update(
                status="numerical_kkt_passed" if passed else "numerical_kkt_rejected",
                objective=objective,
                checks=checks,
            )
            if passed and (best is None or objective < best[0]):
                best = (objective, mask, z.copy(), mu.copy(), checks)
        except (ValueError, np.linalg.LinAlgError) as exc:
            record.update(status="numerical_solve_failed", error=str(exc))
        _emit(trace, sink, {"kind": "subset_completed", "record": record})
    trace["stage"] = "selection"
    if best is None:
        raise ValueError(
            "No numerical KKT candidate; near-dependent subset skips may lose completeness"
        )
    trace["stage"] = "postclip"
    _, mask, z, mu, checks = best
    dual = beta * mu / row_norms
    length = float(np.linalg.norm(z))
    radius = 0.05 * gate_norm
    if not np.isfinite(dual).all() or not np.isfinite(length):
        raise ValueError("Nonfinite original dual or direction norm")
    factor = 1.0 if length == 0 else min(1.0, radius / length)
    clipped = z * factor
    pre = float(d @ z + z @ k @ z)
    post = float(d @ clipped + clipped @ k @ clipped)
    post_values = bn @ clipped
    post_primal = float(
        np.max(
            np.maximum(0.0, post_values) / np.maximum(1.0, np.abs(post_values)),
            initial=0.0,
        )
    )
    normalized_details = certificate_details(z, mu, dn, kn, bn)
    original_details = certificate_details(z, dual, d, k, a)
    if not np.isfinite([pre, post, post_primal]).all() or post_primal > TOL:
        raise ValueError("Invalid postclip objective or numerical cone feasibility")
    return clipped, {
        "unclipped": z.tolist(),
        "normalized_duals": mu.tolist(),
        "original_duals": dual.tolist(),
        "objective_scale": beta,
        "constraint_row_norms": row_norms.tolist(),
        "gram_eigenvalues": eig.tolist(),
        "numerical_kkt_checked_preclip": True,
        "preclip_checks": checks,
        "preclip_normalized_details": normalized_details,
        "preclip_original_details_informational": original_details,
        "numerical_kkt_tolerance": TOL,
        "gram_condition_tolerance": RANK_TOL,
        "independent_normal_geometry_tolerance": GEOMETRY_TOL,
        "selected_mask": mask,
        "subsets_considered": len(records),
        "subsets": records,
        "dependent_subsets": sum(
            v["status"] == "dependent_active_rows" for v in records
        ),
        "preclip_norm": length,
        "clip_radius": radius,
        "clip_factor": factor,
        "preclip_objective": pre,
        "postclip_objective": post,
        "score": -post,
        "postclip_normalized_primal_violation": post_primal,
        "postclip_kkt_claimed": False,
        "trust_region_minimizer_claimed": False,
        "exact_numerical_optimality_claimed": False,
        "target_gradients_used": False,
    }
