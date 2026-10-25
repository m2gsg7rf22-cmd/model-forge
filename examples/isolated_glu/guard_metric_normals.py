"""Synthetic known-candidate guard metric derivatives, not target gradients.

Physical reference and complete candidate error enter UNWEIGHTED. A first-order
plane never certifies nonlinear finite-step retention. No fitter/target access.
"""

from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
NEAR_MAX_RELATIVE_TOLERANCE = 1e-12
MAX_LOGICAL_PLANES = 8
spec = importlib.util.spec_from_file_location(
    "frozen_metric_gradient", Path(__file__).with_name("gate_gradient.py")
)
assert spec and spec.loader
gradient = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gradient)


def integer(value: int, minimum: int, maximum: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and minimum <= value <= maximum
    )


def metric_group(
    gate: Array,
    coupling: Array,
    x: Array,
    reference: Array,
    error: Array,
    group_index: int,
    row_offset: int,
) -> tuple[Array, dict[str, Any]]:
    if gate.ndim != 1 or coupling.ndim != 2:
        raise ValueError("Single known gate/coupling required")
    # Reuse the frozen finite/profile check; this is known computation only.
    gradient.forward(gate[None, :], coupling[None, :, :], x)
    if (
        not integer(group_index, 0, 3)
        or not integer(row_offset, 0, 4095)
        or row_offset + len(x) > 4096
        or reference.shape != (len(x), coupling.shape[0])
        or error.shape != reference.shape
        or any(
            v.dtype != np.float64 or not np.isfinite(v).all()
            for v in [reference, error]
        )
    ):
        raise ValueError(
            "Finite physical FP64 reference/error and bounded provenance required"
        )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        denominator = float(np.sum(reference * reference))
        error_squared = float(np.sum(error * error))
    if denominator <= 0 or not np.isfinite([denominator, error_squared]).all():
        raise ValueError("Invalid physical reference/error squared norm")
    maximum = float(np.max(np.abs(error)))
    if maximum > 0 and error_squared == 0:
        raise ValueError("Nonzero physical residual squared norm underflow")
    if maximum == 0:
        raise ValueError(
            "Degenerate zero maximum: sign(0) cannot represent both Jacobian signs"
        )
    threshold = maximum * (1 - NEAR_MAX_RELATIVE_TOLERANCE)
    coordinates = np.argwhere(np.abs(error) >= threshold)
    if 1 + len(coordinates) > MAX_LOGICAL_PLANES:
        raise ValueError("Group logical plane budget exceeded; no truncation")
    t = x @ gate
    sigmoid = 0.5 * (1 + np.tanh(0.5 * t))
    prime = sigmoid + t * sigmoid * (1 - sigmoid)
    branch = x @ coupling.T
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        relative_gradient = (
            2 * (x.T @ (prime * np.sum(error * branch, axis=1))) / denominator
        )
        relative_squared = error_squared / denominator
    if error_squared > 0 and relative_squared == 0:
        raise ValueError("Nonzero physical squared-relative metric underflow")
    if not np.isfinite(relative_squared) or not np.isfinite(relative_gradient).all():
        raise ValueError("Nonfinite guard derivative arithmetic")
    vectors: list[Array] = []
    records: list[dict[str, Any]] = []

    def retain(normal: Array, record: dict[str, Any]) -> None:
        if not np.isfinite(normal).all():
            raise ValueError("Nonfinite guard normal")
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or (length == 0 and np.any(normal != 0)):
            raise ValueError("Overflow/underflow guard normal norm")
        matrix_row = None if length == 0 else len(vectors)
        if matrix_row is not None:
            vectors.append(normal.copy())
        records.append(
            dict(
                record,
                logical_index=len(records),
                matrix_row=matrix_row,
                normal=normal.tolist(),
                normal_norm=length,
                omitted_reason="computed_FP64_zero" if length == 0 else None,
            )
        )

    retain(
        relative_gradient,
        {
            "metric": "relative_squared",
            "group": group_index,
            "row": None,
            "output": None,
            "exact_maximum": None,
            "signed_residual": None,
        },
    )
    for row, output in coordinates:
        signed = float(error[row, output])
        sign = 1.0 if signed > 0 else -1.0
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            normal = sign * prime[row] * branch[row, output] * x[row]
        retain(
            normal,
            {
                "metric": "maximum_absolute",
                "group": group_index,
                "row": row_offset + int(row),
                "local_row": int(row),
                "output": int(output),
                "exact_maximum": abs(signed) == maximum,
                "signed_residual": signed,
                "sign": sign,
            },
        )
    matrix = (
        np.vstack(vectors) if vectors else np.empty((0, len(gate)), dtype=np.float64)
    )
    return matrix, {
        "group": group_index,
        "row_offset": row_offset,
        "rows": len(x),
        "reference_squared_norm": denominator,
        "error_squared_norm": error_squared,
        "relative_squared": relative_squared,
        "relative": float(np.sqrt(relative_squared)),
        "maximum_absolute": maximum,
        "near_max_threshold": threshold,
        "near_max_relative_tolerance": NEAR_MAX_RELATIVE_TOLERANCE,
        "logical_plane_count": len(records),
        "matrix_plane_count": len(matrix),
        "records": records,
        "near_planes_are_conservative_not_exact_derivative": True,
        "zero_normals_imply_finite_retention": False,
    }


def guard_normals(
    g: Array, c: Array, x: Array, y: Array, rows: int, unit: int
) -> tuple[Array, dict[str, Any]]:
    prediction = gradient.forward(g, c, x)
    if (
        not integer(rows, 1, 1024)
        or len(x) != 4 * rows
        or not integer(unit, 0, len(g) - 1)
        or y.shape != prediction.shape
        or y.dtype != np.float64
        or not np.isfinite(y).all()
    ):
        raise ValueError(
            "Four complete physical guard groups and valid selected unit required"
        )
    error = prediction - y
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite full physical residual")
    matrices: list[Array] = []
    records: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    matrix_offset = 0
    for group in range(4):
        block = slice(group * rows, (group + 1) * rows)
        matrix, report = metric_group(
            g[unit], c[unit], x[block], y[block], error[block], group, group * rows
        )
        if len(records) + len(report["records"]) > MAX_LOGICAL_PLANES:
            raise ValueError(
                "Full profile logical plane budget exceeded; ties cannot be truncated or omitted"
            )
        for original in report["records"]:
            record = dict(original)
            record["logical_index"] = len(records)
            if record["matrix_row"] is not None:
                record["matrix_row"] += matrix_offset
            records.append(record)
        matrix_offset += len(matrix)
        matrices.append(matrix)
        groups.append({k: v for k, v in report.items() if k != "records"})
    combined = np.vstack(matrices)
    return combined, {
        "unit": unit,
        "logical_plane_count": len(records),
        "matrix_plane_count": len(combined),
        "records": records,
        "groups": groups,
        "error_convention": "physical_candidate_minus_physical_reference",
        "first_order_current_metrics_only": True,
        "current_initial_bounds_are_not_different_derivative_points": True,
        "finite_forward_retention_proven": False,
        "private_gate": "closed",
    }
