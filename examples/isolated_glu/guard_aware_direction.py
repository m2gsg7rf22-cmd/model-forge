"""Bounded synthetic-stage cone QP; no target access or real-data adapter.

Cost has NO half: q(z)=d.T z+z.T K z. Numerical KKT checks are
scaled residual checks, not exact optimality/feasibility certificates.
"""

from __future__ import annotations
from typing import Any
from fractions import Fraction
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
TOL = 1e-10
RANK_TOL = 1e-12
GEOMETRY_TOL = (
    1e-5  # sqrt(TOL): conservative independent-normal domain, not an error bound.
)


def certificate_details(
    z: Array, dual: Array, d: Array, k: Array, a: Array
) -> dict[str, Any]:
    with np.errstate(over="ignore", invalid="ignore"):
        az = a @ z
        force = 2 * k @ z
        normal = a.T @ dual
        station = force + d + normal
        complement = dual * az
    if any(not np.isfinite(v).all() for v in [az, force, normal, station, complement]):
        raise ValueError("Nonfinite numerical KKT arithmetic")
    # Componentwise floors avoid a large unrelated component hiding a
    # violation. The caller passes normalized rows/objective for acceptance.
    scales = {
        "primal": np.maximum(1.0, np.abs(az)),
        "dual": np.maximum(1.0, np.abs(dual)),
        "stationarity": np.maximum.reduce(
            [np.ones_like(d), np.abs(d), np.abs(force), np.abs(normal)]
        ),
        "complementarity": np.maximum.reduce(
            [np.ones_like(dual), np.abs(dual), np.abs(az)]
        ),
    }
    residuals = {
        "primal": np.maximum(0.0, az),
        "dual": np.maximum(0.0, -dual),
        "stationarity": np.abs(station),
        "complementarity": np.abs(complement),
    }
    ratios = {
        name: float(np.max(values / scales[name], initial=0.0))
        for name, values in residuals.items()
    }
    if not np.isfinite(list(ratios.values())).all():
        raise ValueError("Nonfinite numerical KKT scale or ratio")
    return {
        "ratios": ratios,
        "raw_constraint_values": az.tolist(),
        "raw_duals": dual.tolist(),
        "raw_stationarity": station.tolist(),
        "raw_complementarity": complement.tolist(),
        "component_scales": {name: value.tolist() for name, value in scales.items()},
    }


def certificate(
    z: Array, dual: Array, d: Array, k: Array, a: Array
) -> dict[str, float]:
    values: dict[str, float] = certificate_details(z, dual, d, k, a)["ratios"]
    return values


def exact_row_rank(rows: Array) -> int:
    """Rank of original binary64 coefficients as exact rationals (<=8x32).

    Used only when SVD flags a possible dependency, never to solve the QP.
    This distinguishes an exact redundant support from an ill-conditioned
    independent support that must fail the entire numerical problem.
    """
    matrix = [[Fraction(float(value)) for value in row] for row in rows]
    rank = 0
    for column in range(rows.shape[1]):
        pivot = next((i for i in range(rank, len(matrix)) if matrix[i][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        divisor = matrix[rank][column]
        matrix[rank] = [value / divisor for value in matrix[rank]]
        for i in range(rank + 1, len(matrix)):
            multiplier = matrix[i][column]
            if multiplier:
                matrix[i] = [
                    value - multiplier * base
                    for value, base in zip(matrix[i], matrix[rank])
                ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def cone_direction(
    gate: Array, d: Array, k: Array, a: Array
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
    gate_norm = float(np.linalg.norm(gate))
    row_norms = np.linalg.norm(a, axis=1)
    if (
        gate_norm <= 0
        or not np.isfinite(gate_norm)
        or np.any(row_norms <= 0)
        or not np.isfinite(row_norms).all()
    ):
        raise ValueError("Nonzero finite gate/normals required")
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
    records: list[dict[str, Any]] = []
    best: tuple[float, int, Array, Array, dict[str, float]] | None = None
    for mask in range(1 << len(a)):
        active = [i for i in range(len(a)) if mask & (1 << i)]
        rows = bn[active]
        record: dict[str, Any] = {"mask": mask, "active": active}
        if active:
            sv = np.linalg.svd(rows, compute_uv=False)
            if len(active) > n or sv[-1] <= GEOMETRY_TOL * sv[0]:
                # A numerically discarded but exactly independent support can
                # hide a large geometric error behind tiny KKT residuals.
                # Reject globally, even if another approximate candidate passed.
                rank = exact_row_rank(a[active]) if len(active) <= n else n
                if len(active) <= n and rank == len(active):
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
                records.append(record)
                continue
        try:
            if active:
                matrix = np.block(
                    [[2 * kn, rows.T], [rows, np.zeros((len(active), len(active)))]]
                )
                solved = np.linalg.solve(
                    matrix, np.concatenate([-dn, np.zeros(len(active))])
                )
                z = solved[:n]
                mu = np.zeros(len(a))
                mu[active] = solved[n:]
            else:
                z = np.linalg.solve(2 * kn, -dn)
                mu = np.zeros(len(a))
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
        records.append(record)
    if best is None:
        raise ValueError(
            "No numerical KKT candidate; near-dependent subset skips may lose completeness"
        )
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
