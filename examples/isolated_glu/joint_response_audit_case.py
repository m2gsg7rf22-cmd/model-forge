"""One-case independent saved-candidate audit; no IO/RNG/fit or teacher.

Caller must obtain full-protocol/source/resource authorization before supplying
actual observations. Literal fixture seam permits the same bounded arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
from joint_response_join import JointJob
from joint_response_audit_math import (
    forward,
    metrics,
    rms,
    scalar_agrees,
    array_agrees,
    selection,
)

Array = NDArray[np.float64]
Indices = NDArray[np.int64]
CHECKPOINTS = (0, *range(256, 4000, 256), 4000)
GUARD = (3, 7, 11, 15)


def _hash(a: Array) -> str:
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()


def _matrix(a: Array, shape: tuple[int, int]) -> None:
    if (
        not isinstance(a, np.ndarray)
        or a.dtype != np.dtype("float64")
        or not a.dtype.isnative
        or a.shape != shape
        or not np.isfinite(a).all()
    ):
        raise ValueError("exact finite native case array required")


def _immutable(a: Array) -> Array:
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def _indices(probes: int, rows: int) -> tuple[Indices, Indices]:
    indices = np.arange(2 * probes * rows, dtype=np.int64)
    guard = np.isin((indices // rows) % probes, GUARD)
    return tuple(
        np.frombuffer(a.tobytes(), dtype=np.int64)
        for a in (indices[~guard], indices[guard])
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class Observations:
    job_id: str
    n: int
    outputs: int
    width: int
    probes: int
    rows: int
    x: Array
    y: Array
    train: Indices
    guard: Indices
    sx: float
    sy: float

    def metadata(self) -> dict[str, Any]:
        return dict(
            schema="joint-response-case-v1",
            job_id=self.job_id,
            n=self.n,
            outputs=self.outputs,
            width_hypothesis=self.width,
            width_is_hypothesis=True,
            sx=self.sx,
            sy=self.sy,
            train_rows=len(self.train),
            guard_rows=len(self.guard),
            guard_centers=list(GUARD),
            guard_is_consumed_calibration=True,
            input_array_sha256=_hash(self.x),
            response_array_sha256=_hash(self.y),
            new_queries=0,
            certificate=False,
            private_gate="closed",
        )


def _raw_blocks(a: Array, indices: Indices) -> Iterator[Array]:
    for start in range(0, len(indices), 4096):
        yield a[indices[start : start + 4096]]


def _collect(job: JointJob, probes: int, rows: int) -> Observations:
    if (
        re.fullmatch("job0[0-5]", job.job_id) is None
        or any(
            type(v) is not int
            for v in (
                job.n,
                job.outputs,
                job.width_hypothesis,
                job.batch_count,
                probes,
                rows,
            )
        )
        or not 1 <= job.n <= 32
        or not 1 <= job.outputs <= 32
        or not 1 <= job.width_hypothesis <= 96
        or not 4 <= probes <= 17
        or not 1 <= rows <= 2112
        or job.batch_count != 2 * probes
    ):
        raise ValueError("bounded case profile required")
    x = np.empty((2 * probes * rows, job.n), dtype=np.float64)
    y = np.empty((len(x), job.outputs), dtype=np.float64)
    batches = iter(job.batches)
    for index in range(2 * probes):
        try:
            batch = next(batches)
        except StopIteration as exc:
            raise ValueError("missing response batch") from exc
        _matrix(batch.inputs, (rows, job.n))
        _matrix(batch.responses, (rows, job.outputs))
        x[index * rows : (index + 1) * rows] = batch.inputs
        y[index * rows : (index + 1) * rows] = batch.responses
    try:
        next(batches)
    except StopIteration:
        pass
    else:
        raise ValueError("extra response batch")
    train, guard = _indices(probes, rows)
    sx = rms(_raw_blocks(x, train))
    sy = rms(_raw_blocks(y, train))
    if min(sx, sy) <= 1e-12:
        raise ValueError("degenerate independent training RMS")
    return Observations(
        job.job_id,
        job.n,
        job.outputs,
        job.width_hypothesis,
        probes,
        rows,
        _immutable(x),
        _immutable(y),
        train,
        guard,
        sx,
        sy,
    )


def collect_case(job: JointJob) -> Observations:
    if job.n != 32 or job.outputs != 32 or job.batch_count != 34:
        raise ValueError("registered32/32/34 case required")
    return _collect(job, 17, 2112)


def _collect_fixture(job: JointJob, *, probes: int, rows: int) -> Observations:
    return _collect(job, probes, rows)


def _parameters(p: dict[str, Array], data: Observations) -> None:
    if set(p) != {"g", "u", "d"}:
        raise ValueError("exact G/U/D candidate required")
    for name, shape in [
        ("g", (data.width, data.n)),
        ("u", (data.width, data.n)),
        ("d", (data.outputs, data.width)),
    ]:
        _matrix(p[name], shape)


def _metric(
    data: Observations, p: dict[str, Array], indices: Indices
) -> dict[str, float | int | None]:
    def blocks() -> Iterator[tuple[Array, Array]]:
        for start in range(0, len(indices), 4096):
            take = indices[start : start + 4096]
            with np.errstate(
                over="raise", invalid="raise", divide="raise", under="raise"
            ):
                x = data.x[take] / data.sx
                y = data.y[take] / data.sy
            yield forward(x, p["g"], p["u"], p["d"]), y

    return metrics(blocks())


def _number(value: Any, *, positive: bool = False) -> float:
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError("finite native nonnegative scalar required")
    return float(value)


def audit_case(
    *,
    data: Observations,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    selected_step: int,
    checkpoint: Callable[[int], dict[str, Array]],
    physical: dict[str, Array],
    predictions: Iterator[Array],
    saved_split: dict[str, Indices],
    prediction_rows: int = 8192,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Audit all17candidates; disagreements persist, no replacement model is fitted."""
    if type(prediction_rows) is not int or not 1 <= prediction_rows <= 8192:
        raise ValueError("bounded prediction shard rows required")
    if (
        type(selected_step) is not int
        or selected_step not in CHECKPOINTS
        or len(records) != 17
    ):
        raise ValueError("exact selected/checkpoint budget required")
    if (
        any(
            type(v) is not int
            for v in (data.n, data.outputs, data.width, data.probes, data.rows)
        )
        or not 1 <= data.n <= 32
        or not 1 <= data.outputs <= 32
        or not 1 <= data.width <= 96
        or not 4 <= data.probes <= 17
        or not 1 <= data.rows <= 2112
        or re.fullmatch("job0[0-5]", data.job_id) is None
    ):
        raise ValueError("bounded independent observations required")
    _matrix(data.x, (2 * data.probes * data.rows, data.n))
    _matrix(data.y, (len(data.x), data.outputs))
    if data.x.flags.writeable or data.y.flags.writeable:
        raise ValueError("immutable observation snapshots required")
    train, guard = _indices(data.probes, data.rows)
    if set(saved_split) != {"train", "guard"}:
        raise ValueError("exact split fields required")
    for key, expected, actual in [
        ("train", train, data.train),
        ("guard", guard, data.guard),
    ]:
        for a in (actual, saved_split[key]):
            if (
                not isinstance(a, np.ndarray)
                or a.dtype != np.dtype("int64")
                or not a.dtype.isnative
                or not np.array_equal(a, expected)
            ):
                raise ValueError("exact native whole-center split required")
    # Independently recompute scaling from X/Y, rather than trusting forged objects.
    sx = rms(_raw_blocks(data.x, train))
    sy = rms(_raw_blocks(data.y, train))
    if min(sx, sy) <= 1e-12 or data.sx != sx or data.sy != sy:
        raise ValueError("independent scale binding differs")
    expected_metadata = data.metadata()
    if set(metadata) != set(expected_metadata):
        raise ValueError("exact original case metadata required")
    for key, value in expected_metadata.items():
        if key in ("sx", "sy"):
            continue
        if type(metadata[key]) is not type(value) or metadata[key] != value:
            raise ValueError("case data/identity binding differs")
    reported_scales = {k: _number(metadata[k], positive=True) for k in ("sx", "sy")}
    if min(reported_scales.values()) <= 1e-12:
        raise ValueError("degenerate saved training RMS")
    normalization = all(
        scalar_agrees(v, reported_scales[k]) for k, v in [("sx", sx), ("sy", sy)]
    )
    best = math.inf
    saved_best = None
    for step, row in zip(CHECKPOINTS, records, strict=True):
        if (
            set(row)
            != {
                "step",
                "train_half_mse",
                "guard_mse",
                "guard_relative_error",
                "selected_at_checkpoint",
                "selected_step_so_far",
            }
            or type(row["step"]) is not int
            or row["step"] != step
        ):
            raise ValueError("exact ordered checkpoint metadata required")
        _number(row["train_half_mse"])
        loss = _number(row["guard_mse"])
        if row["guard_relative_error"] is not None:
            _number(row["guard_relative_error"])
        chosen = loss < best
        if chosen:
            best = loss
            saved_best = step
        if (
            type(row["selected_at_checkpoint"]) is not bool
            or row["selected_at_checkpoint"] != chosen
            or type(row["selected_step_so_far"]) is not int
            or row["selected_step_so_far"] != saved_best
        ):
            raise ValueError("saved selection metadata differs from earliest minimum")
    if saved_best != selected_step:
        raise ValueError("saved selected step differs from metrics")
    checked = []
    selected = None
    losses = []
    failed = False
    for step, row in zip(CHECKPOINTS, records, strict=True):
        p = checkpoint(step)
        _parameters(p, data)
        if step == selected_step:
            selected = {k: _immutable(v) for k, v in p.items()}
        result: dict[str, Any] = {"step": step, "status": "checked"}
        try:
            tr = _metric(data, p, train)
            gu = _metric(data, p, guard)
            comparisons = {
                "train_half_mse": scalar_agrees(
                    _number(tr["half_mse"]), row["train_half_mse"]
                ),
                "guard_mse": scalar_agrees(_number(gu["mse"]), row["guard_mse"]),
            }
            relative = gu["relative_error"]
            reported = row["guard_relative_error"]
            comparisons["guard_relative_error"] = (
                (relative is None and reported is None)
                if relative is None or reported is None
                else scalar_agrees(float(relative), reported)
            )
            result.update(
                recomputed={
                    "train_half_mse": tr["half_mse"],
                    "guard_mse": gu["mse"],
                    "guard_relative_error": relative,
                },
                saved={k: row[k] for k in comparisons},
                agreements=comparisons,
                metrics_agreed=all(comparisons.values()),
            )
            losses.append(_number(gu["mse"]))
        except (FloatingPointError, OverflowError) as exc:
            result.update(
                status="numerical_failure",
                error_type=type(exc).__name__,
                metrics_agreed=False,
            )
            failed = True
        checked.append(result)
        if emit is not None:
            emit(result)
    choice = (
        None if failed else selection(losses, [float(r["guard_mse"]) for r in records])
    )
    ambiguous = False if choice is None else bool(choice["ambiguous"])
    step_agrees = False
    selection_compatible = False
    if choice is not None:
        index = choice["index"]
        competitors = choice["ambiguous_with"]
        if type(index) is not int or not isinstance(competitors, list):
            raise ValueError("invalid independent selection result")
        step_agrees = CHECKPOINTS[index] == selected_step
        selection_compatible = (
            step_agrees or CHECKPOINTS.index(selected_step) in competitors
        )
    if selected is None:
        raise ValueError("saved selected checkpoint missing")
    _parameters(physical, data)
    physical_agrees = False
    predictions_agree = True
    normalized_predictions_agree = True
    physical_predictions_agree = True
    normalized_physical_agree = True
    rows_checked = 0
    rows_visited = 0
    parameter_agreements: dict[str, bool] = {}
    prediction_numeric_failure = False
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            converted = {
                "g": selected["g"] / sx,
                "u": selected["u"] / sx,
                "d": selected["d"] * sy,
            }
        parameter_agreements = {
            k: array_agrees(converted[k], physical[k]) for k in ("g", "u", "d")
        }
        physical_agrees = all(parameter_agreements.values())
    except (FloatingPointError, OverflowError):
        failed = True
    for start in range(0, len(data.x), prediction_rows):
        rows = min(prediction_rows, len(data.x) - start)
        try:
            saved = next(predictions)
        except StopIteration as exc:
            raise ValueError("missing saved prediction shard") from exc
        _matrix(saved, (rows, data.outputs))
        try:
            independent = np.empty_like(saved)
            normalized = np.empty_like(saved)
            for inner in range(0, rows, 4096):
                stop = min(inner + 4096, rows)
                independent[inner:stop] = forward(
                    data.x[start + inner : start + stop],
                    physical["g"],
                    physical["u"],
                    physical["d"],
                )
                # Bind predictions directly to selected normalized computation;
                # coefficient tolerances alone can hide scale-amplified changes.
                with np.errstate(
                    over="raise", invalid="raise", divide="raise", under="raise"
                ):
                    normalized[inner:stop] = (
                        forward(
                            data.x[start + inner : start + stop] / sx,
                            selected["g"],
                            selected["u"],
                            selected["d"],
                        )
                        * sy
                    )
            normalized_predictions_agree = bool(
                array_agrees(normalized, saved) and normalized_predictions_agree
            )
            physical_predictions_agree = bool(
                array_agrees(independent, saved) and physical_predictions_agree
            )
            normalized_physical_agree = bool(
                array_agrees(normalized, independent) and normalized_physical_agree
            )
            predictions_agree = (
                normalized_predictions_agree
                and physical_predictions_agree
                and normalized_physical_agree
            )
            rows_checked += rows
        except (FloatingPointError, OverflowError):
            prediction_numeric_failure = True
            normalized_predictions_agree = False
            physical_predictions_agree = False
            normalized_physical_agree = False
            predictions_agree = False
            failed = True
        rows_visited += rows
    try:
        next(predictions)
    except StopIteration:
        pass
    else:
        raise ValueError("extra saved prediction shard")
    metric_agreement = all(r["metrics_agreed"] for r in checked)
    agreed = (
        normalization
        and metric_agreement
        and selection_compatible
        and physical_agrees
        and predictions_agree
    )
    status = (
        "numerical_failure"
        if failed
        else "disagreed"
        if not agreed
        else "ambiguous"
        if ambiguous
        else "agreed"
    )
    return dict(
        status=status,
        job_id=data.job_id,
        width_hypothesis=data.width,
        width_is_hypothesis=True,
        checkpoints=checked,
        normalization_agreed=normalization,
        metrics_agreed=metric_agreement,
        saved_selected_step=selected_step,
        recomputed_selection=choice,
        selected_step_agreed=step_agrees,
        selection_ambiguous=ambiguous,
        selection_certified=agreed and step_agrees and not failed and not ambiguous,
        physical_parameters_agreed=physical_agrees,
        parameter_agreements=parameter_agreements,
        all_predictions_agreed=predictions_agree,
        normalized_predictions_agreed=normalized_predictions_agree,
        physical_predictions_agreed=physical_predictions_agree,
        normalized_physical_forward_agreed=normalized_physical_agree,
        prediction_rows_checked=rows_checked,
        prediction_rows_visited=rows_visited,
        prediction_numeric_failure=prediction_numeric_failure,
        guard_is_consumed_calibration=True,
        certificate=False,
        private_gate="closed",
        new_queries=0,
        rng_calls=0,
        optimizer_replays=0,
    )
