"""All-six saved-candidate audit integration; not a protocol or launcher.

Call only after full-source/resource/disk/marker gates. Does not create RNG,
query a teacher, fit, replay optimization or authorize its own actual data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from joint_response_join import JointJob
from joint_response_audit_case import Observations, Array, collect_case, audit_case
from joint_response_audit_storage import TraceReader, AuditStore
from variable_response_protocol import ancestry


def _run_jobs(
    jobs: Sequence[JointJob],
    readers: Sequence[TraceReader],
    root: Path,
    *,
    collect: Callable[[JointJob], Observations],
    prediction_rows: int,
    check: Callable[[], None],
    progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    if (
        len(jobs) != 6
        or len(readers) != 6
        or any(j.job_id != f"job{i:02d}" for i, j in enumerate(jobs))
    ):
        raise ValueError("canonical six audit jobs/readers required")
    ancestry(root)
    root.mkdir(exist_ok=False)
    report: dict[str, Any] = dict(
        status="running",
        cases=[dict(job_id=f"job{i:02d}", status="not_started") for i in range(6)],
        completed_case_visits=0,
        numeric_consistency_agreed=False,
        scientific_reconstruction_success=False,
        certificate=False,
        private_gate="closed",
        new_queries=0,
        rng_calls=0,
        optimizer_replays=0,
        automatic_retry=False,
    )
    for i, (job, reader) in enumerate(zip(jobs, readers, strict=True)):
        store = None
        try:
            check()
            report["cases"][i].update(status="running")
            store = AuditStore(root / job.job_id)
            metadata = reader.metadata()
            data = collect(job)
            check()
            if (
                data.job_id != job.job_id
                or data.width != reader.profile.width
                or data.n != reader.profile.n
                or data.outputs != reader.profile.outputs
                or data.probes != reader.profile.probes
                or data.rows != reader.profile.rows
                or prediction_rows != reader.profile.prediction_rows
            ):
                raise ValueError("independent case/reader profile differs")

            def checkpoint(step: int) -> dict[str, Array]:
                check()
                return reader.checkpoint(step)

            def predictions() -> Iterator[Array]:
                for value in reader.predictions():
                    check()
                    yield value

            def emit(row: dict[str, Any]) -> None:
                check()
                assert store is not None
                store.append(row)

            outcome = audit_case(
                data=data,
                **metadata,
                checkpoint=checkpoint,
                predictions=predictions(),
                prediction_rows=prediction_rows,
                emit=emit,
            )
            del data, metadata
            check()
            outcome["trace_reader"] = reader.finish()
            check()
            store.complete(outcome)
            report["cases"][i] = outcome
            report["completed_case_visits"] = i + 1
            progress(
                dict(
                    completed_case_visits=i + 1,
                    cases=[
                        dict(job_id=c["job_id"], status=c["status"])
                        for c in report["cases"]
                    ],
                )
            )
        except Exception as exc:
            failure = dict(
                job_id=job.job_id,
                status="failed",
                error_type=type(exc).__name__,
                reader_partial=reader.report(),
                result_already_committed=False,
                committed_checkpoint_records=0,
                uncommitted_tail="unknown",
            )
            if store is not None:
                failure.update(
                    result_already_committed=store.done,
                    committed_checkpoint_records=len(store.rows),
                    uncommitted_tail=store.uncommitted_tail,
                )
                if not store.done:
                    try:
                        store.fail(type(exc).__name__)
                    except Exception as receipt_exc:
                        failure["failure_receipt_error_type"] = type(
                            receipt_exc
                        ).__name__
            report["cases"][i] = failure
            report.update(
                status="failed",
                error_type=type(exc).__name__,
                numeric_consistency_agreed=False,
            )
            return report
    report.update(
        status="completed",
        numeric_consistency_agreed=all(
            c["status"] == "agreed" for c in report["cases"]
        ),
    )
    return report


def audit_jobs(
    jobs: Sequence[JointJob],
    readers: Sequence[TraceReader],
    root: Path,
    *,
    check: Callable[[], None],
    progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    if any(
        j.n != 32
        or j.outputs != 32
        or j.batch_count != 34
        or r.profile.n != 32
        or r.profile.outputs != 32
        or r.profile.probes != 17
        or r.profile.rows != 2112
        or r.profile.prediction_rows != 8192
        for j, r in zip(jobs, readers, strict=True)
    ):
        raise ValueError("registered32/32/34 independent audit profile required")
    return _run_jobs(
        jobs,
        readers,
        root,
        collect=collect_case,
        prediction_rows=8192,
        check=check,
        progress=progress,
    )
