"""One-case integration only; actual use awaits full protocol/source approval.

No RNG, teacher, heldout or observation reader. The caller supplies an authorized
case and prepared streams after all global gates. Engineering completion never
certifies reconstruction or calibration generalization.
"""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from joint_response_case import CaseArrays
from joint_response_join import JointJob
from joint_response_kernel import (
    Checkpoint,
    Step,
    NumericalFailure,
    Parameters,
    fit,
    physical_parameters,
    predict,
)
from joint_response_streams import OptimizerInputs, stream_seed
from joint_response_trace import CHECKPOINTS, TraceArray, TraceStore
from scaling_domain_store import _ancestry


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate state key")
        result[key] = value
    return result


def _state(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or len(text.encode()) > 4096:
        raise ValueError("bounded canonical PCG64 state required")
    value = json.loads(text, object_pairs_hook=_unique)
    if (
        not isinstance(value, dict)
        or set(value) != {"bit_generator", "state", "has_uint32", "uinteger"}
        or value["bit_generator"] != "PCG64"
        or not isinstance(value["state"], dict)
        or set(value["state"]) != {"state", "inc"}
        or any(
            type(v) is not int or not 0 <= v < 2**128 for v in value["state"].values()
        )
        or value["state"]["inc"] % 2 != 1
        or type(value["has_uint32"]) is not int
        or value["has_uint32"] not in (0, 1)
        or type(value["uinteger"]) is not int
        or not 0 <= value["uinteger"] < 2**32
        or json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        != text
    ):
        raise ValueError("exact canonical native PCG64 state schema required")
    return value


def _params(p: Parameters, prefix: str = "") -> dict[str, TraceArray]:
    return {prefix + "g": p.g, prefix + "u": p.u, prefix + "d": p.d}


def _moments(p: Parameters, m: Parameters, v: Parameters) -> dict[str, TraceArray]:
    return {**_params(p), **_params(m, "m_"), **_params(v, "v_")}


class _Journal:
    """17bounded metric records. Included in the existing per-case1MiB cap.

    Four128KiBmetadata+31x4096manifest+17x4096intent+17x4096journal
    is790528bytes<1MiB. All record/write byte counts are checked; no retry.
    """

    def __init__(self, root: Path):
        self.path = root / "checkpoint-metrics.jsonl"
        _ancestry(self.path)
        with self.path.open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        self.size = 0
        self.count = 0

    def append(self, value: dict[str, Any]) -> None:
        blob = (
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            + b"\n"
        )
        _ancestry(self.path)
        if (
            self.count >= 17
            or len(blob) > 4096
            or self.size + len(blob) > 17 * 4096
            or self.path.stat().st_size != self.size
        ):
            raise ValueError("metric journal count/byte/layout cap exceeded")
        with self.path.open("ab") as stream:
            if stream.write(blob) != len(blob):
                raise OSError("short metric journal write")
            stream.flush()
            os.fsync(stream.fileno())
        self.size += len(blob)
        self.count += 1


def _run(
    job: JointJob, case: CaseArrays, opt: OptimizerInputs, store: TraceStore
) -> dict[str, Any]:
    if (
        job.n != 32
        or job.outputs != 32
        or job.batch_count != 34
        or type(job.width_hypothesis) is not int
        or not 1 <= job.width_hypothesis <= 96
        or case.inputs.shape != (71808, 32)
        or case.responses.shape != (71808, 32)
        or case.train_x.shape != (54912, 32)
        or case.train_y.shape != (54912, 32)
        or case.guard_x.shape != (16896, 32)
        or case.guard_y.shape != (16896, 32)
        or not all(math.isfinite(v) and v > 1e-12 for v in (case.sx, case.sy))
        or opt.initial.g.shape != (job.width_hypothesis, 32)
        or opt.initial.u.shape != (job.width_hypothesis, 32)
        or opt.initial.d.shape != (32, job.width_hypothesis)
        or store.specs["initial"].fields[0].shape != opt.initial.g.shape
        or store.names
        or store.metadata_names
        or store.completed_updates
    ):
        raise ValueError("fresh registered32/32/34 case/optimizer/trace required")
    rng: dict[str, Any] = {"schema": "joint-response-rng-v1"}
    for role, before, after in (
        ("initial", opt.initial_state_before, opt.initial_state_after),
        ("batches", opt.batch_state_before, opt.batch_state_after),
    ):
        b, a = _state(before), _state(after)
        if b["state"]["inc"] != a["state"]["inc"]:
            raise ValueError("PCG64 stream increment changed")
        rng[role] = dict(seed=stream_seed(job.job_id, role), before=b, after=a)
    # State validation precedes all trace content. No random generator is created.
    journal = _Journal(store.root)
    store.metadata("rng.json", rng)
    store.metadata(
        "case.json",
        dict(
            schema="joint-response-case-v1",
            job_id=job.job_id,
            n=32,
            outputs=32,
            width_hypothesis=job.width_hypothesis,
            width_is_hypothesis=True,
            sx=case.sx,
            sy=case.sy,
            train_rows=54912,
            guard_rows=16896,
            guard_centers=[3, 7, 11, 15],
            guard_is_consumed_calibration=True,
            input_array_sha256=hashlib.sha256(
                memoryview(case.inputs).cast("B")
            ).hexdigest(),
            response_array_sha256=hashlib.sha256(
                memoryview(case.responses).cast("B")
            ).hexdigest(),
            new_queries=0,
            certificate=False,
            private_gate="closed",
        ),
    )
    store.save("initial", _params(opt.initial))
    store.save("schedule", {"indices": opt.schedule})
    store.save(
        "split-indices", {"train": case.train_indices, "guard": case.guard_indices}
    )
    records: list[dict[str, Any]] = []
    selected_step: int | None = None
    selected_mse = math.inf

    def observer(event: Step | Checkpoint) -> None:
        nonlocal selected_step, selected_mse
        if isinstance(event, Step):
            store.progress(event.step, event.sampled_loss_before_update)
            return
        if (
            not isinstance(event, Checkpoint)
            or len(records) >= 17
            or event.step != CHECKPOINTS[len(records)]
        ):
            raise ValueError("canonical metric checkpoint required")
        chosen = event.guard_mse < selected_mse
        if chosen:
            selected_step, selected_mse = event.step, event.guard_mse
        record = dict(
            step=event.step,
            train_half_mse=event.train_loss,
            guard_mse=event.guard_mse,
            guard_relative_error=event.guard_relative_error,
            selected_at_checkpoint=chosen,
            selected_step_so_far=selected_step,
        )
        store.intent(event.step)
        store.save(
            f"checkpoint-{event.step:04d}",
            _moments(event.parameters, event.first_moment, event.second_moment),
        )
        journal.append(record)
        records.append(record)

    def failure(error: NumericalFailure, phase: str) -> dict[str, Any]:
        if error.updates_completed != store.completed_updates:
            raise ValueError("numeric failure/progress count mismatch")
        store.save(
            "failure-state",
            _moments(error.parameters, error.first_moment, error.second_moment),
        )
        store.metadata(
            "metrics.json",
            dict(
                schema="joint-response-metrics-v1",
                checkpoints=records,
                selected_step=None,
                selected_guard_mse=None,
                guard_is_consumed_calibration=True,
                checkpoint_metric_records=journal.count,
                metric_journal_bytes=journal.size,
                failure_phase=phase,
                updates_completed=error.updates_completed,
            ),
        )
        outcome = dict(
            status="numerical_failure",
            job_id=job.job_id,
            updates_completed=error.updates_completed,
            selected_step=None,
            selected_guard_mse=None,
            failure_phase=phase,
            last_parameter_checkpoint=max((r["step"] for r in records), default=None),
            certificate=False,
            private_gate="closed",
        )
        store.metadata("terminal.json", outcome)
        return outcome

    try:
        result = fit(
            opt.initial,
            case.train_x,
            case.train_y,
            case.guard_x,
            case.guard_y,
            opt.schedule,
            observer,
        )
    except NumericalFailure as exc:
        return failure(exc, "optimizer_or_metric")
    if (
        result.updates_completed != 4000
        or len(records) != 17
        or selected_step != result.selected_step
        or tuple(c.step for c in result.checkpoints) != CHECKPOINTS
        or result.selected
        is not result.checkpoints[CHECKPOINTS.index(result.selected_step)].parameters
    ):
        raise ValueError("kernel result/committed trace selection mismatch")
    try:
        physical = physical_parameters(result.selected, case.sx, case.sy)
        store.save("selected-physical", _params(physical))
        for start in range(0, 71808, 8192):
            values = predict(physical, case.inputs[start : start + 8192])
            store.save(f"prediction-{start // 8192:02d}", {"predictions": values})
    except (FloatingPointError, OverflowError) as exc:
        final = result.checkpoints[-1]
        error = NumericalFailure(
            4000, final.parameters, final.first_moment, final.second_moment
        )
        error.__cause__ = exc
        return failure(error, "physical_export_or_candidate_prediction")
    store.metadata(
        "metrics.json",
        dict(
            schema="joint-response-metrics-v1",
            checkpoints=records,
            selected_step=selected_step,
            selected_guard_mse=selected_mse,
            initial=records[0],
            final=records[-1],
            selected=records[CHECKPOINTS.index(result.selected_step)],
            guard_is_consumed_calibration=True,
            checkpoint_metric_records=journal.count,
            metric_journal_bytes=journal.size,
            updates_completed=4000,
        ),
    )
    outcome = dict(
        status="completed",
        job_id=job.job_id,
        updates_completed=4000,
        selected_step=selected_step,
        selected_guard_mse=selected_mse,
        predicted_calibration_rows=71808,
        quality_success_label=False,
        certificate=False,
        private_gate="closed",
    )
    store.metadata("terminal.json", outcome)
    return outcome


def run_case(
    job: JointJob, case: CaseArrays, opt: OptimizerInputs, store: TraceStore
) -> dict[str, Any]:
    """All actual access/RNG is caller-gated. Only NumericFailure becomes a case outcome.

    IO/schema/source/callback errors propagate globally, mark the trace failed and
    cannot be retried. Global source/provenance/resource/marker gates are absent
    here; this component is NOT authorization to run the actual experiment.
    """
    try:
        return _run(job, case, opt, store)
    except Exception:
        store.failed = True
        raise
