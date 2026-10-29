"""Single-use streamed calibration view. No experiment/solver/query permission.

The internal literal fixture seam shares validation with the fixed production
profile. Opaque job separation is an API boundary, not an OS sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any, Iterator
import zipfile

import numpy as np
from numpy.typing import NDArray

from variable_response_manifest import (
    ENCODED_CAP,
    HEADER_CAP,
    METADATA_CAP,
    ORIGINAL,
    PUBLIC_PINS,
    REGISTERED,
    ArrayDescriptor,
    PairDescriptor,
    TranscriptManifest,
    _compile,
    _hash,
    _read,
)

Array = NDArray[np.float64]


def _array(blob: bytes, descriptor: ArrayDescriptor) -> Array:
    """Check the declared allocation BEFORE decoding a bounded native NPY."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            members = z.infolist()
            if (
                len(members) != 1
                or members[0].filename != descriptor.field + ".npy"
                or members[0].flag_bits & 1
                or members[0].compress_type != zipfile.ZIP_STORED
                or not 0 < members[0].file_size <= ENCODED_CAP
            ):
                raise ValueError("Exact single stored bounded NPY member required")
            member = z.read(members[0])
        stream = io.BytesIO(member)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=HEADER_CAP
            )
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                stream, max_header_size=HEADER_CAP
            )
        else:
            raise ValueError("NPY version outside frozen format")
        if (
            shape != descriptor.shape
            or dtype != np.dtype(np.float64)
            or not dtype.isnative
        ):
            raise ValueError("Native FP64 exact transcript schema required")
        expected = descriptor.shape[0] * descriptor.shape[1] * 8
        if len(member) != stream.tell() + expected:
            raise ValueError("Exact NPY payload length required")
        a = np.frombuffer(member, dtype=np.float64, offset=stream.tell()).reshape(
            shape, order="F" if fortran else "C"
        )
        if not np.isfinite(a).all():
            raise ValueError("Finite transcript required")
        return a
    except (OSError, zipfile.BadZipFile, EOFError, KeyError, OverflowError) as exc:
        raise ValueError("Invalid bounded transcript archive") from exc


@dataclass(frozen=True, slots=True)
class ResponseBatch:
    inputs: Array
    responses: Array

    def __post_init__(self) -> None:
        for name in ("inputs", "responses"):
            a = getattr(self, name)
            if (
                not isinstance(a, np.ndarray)
                or a.ndim != 2
                or a.dtype != np.float64
                or not a.size
                or a.nbytes > ENCODED_CAP
                or not np.isfinite(a).all()
            ):
                raise ValueError("Bounded native finite response pair required")
            object.__setattr__(
                self,
                name,
                np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape),
            )
        if len(self.inputs) != len(self.responses):
            raise ValueError("Paired row counts required")


@dataclass(frozen=True, slots=True)
class ResponseJob:
    job_id: str
    n: int
    outputs: int
    batch_count: int
    batches: Iterator[ResponseBatch]


def _counter() -> dict[str, int]:
    return dict(
        archives_attempted=0,
        archives_read=0,
        bytes_read=0,
        arrays_validated=0,
        metadata_attempted=0,
        metadata_read=0,
        metadata_bytes=0,
    )


class ViewValidationError(ValueError):
    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


class ResponseView:
    """Parent-owned provenance/counters. Only jobs go to a future numeric learner.

    Constructor validates the entire whitelist before returning jobs. Validation
    stores no arrays; a delivered consumer may retain arrays under its own cap.
    Read bytes are lower bounds after a failed filesystem read, never invented.
    """

    def __init__(self, root: Path, manifest: TranscriptManifest) -> None:
        self._root = root
        self._manifest = manifest
        self._status = "validating"
        self._phase = "validation"
        self._counts: dict[str, dict[str, int]] = {
            k: _counter() for k in ("validation", "delivery", "completion")
        }
        self._counts["delivery"].update(pairs=0, rows=0, payload_bytes=0)
        self._job_status: dict[str, dict[str, Any]] = {
            f"job{i:02d}": dict(status="not_started", delivered_batches=0)
            for i in range(len(manifest.profile.case_names))
        }
        self._next_job = 0
        self._error: str | None = None
        try:
            self._verify_manifest()
            for pair in manifest.pairs:
                x = self._decode(pair.inputs)
                y = self._decode(pair.responses)
                del x, y
        except Exception as exc:
            self._status = "failed"
            self._error = type(exc).__name__ + ": " + str(exc)[:300]
            raise ViewValidationError(str(exc), self.report()) from exc
        self._status = "ready"
        p = manifest.profile
        self._jobs = tuple(
            ResponseJob(
                f"job{i:02d}", p.n, p.outputs, p.probes * 2, self._stream(f"job{i:02d}")
            )
            for i in range(len(p.case_names))
        )

    def _metadata(self, path: Path, cap: int) -> bytes:
        c = self._counts[self._phase]
        c["metadata_attempted"] += 1
        blob = _read(path, cap)
        c["metadata_read"] += 1
        c["metadata_bytes"] += len(blob)
        return blob

    def _verify_manifest(self) -> None:
        m = self._manifest
        pins = {p.name: (p.path, p.digest) for p in m.metadata}
        expected = _compile(
            pins, m.profile, m.committed_collection, _reader=self._metadata
        )
        if expected != m:
            raise ValueError(
                "Transcript manifest differs from pinned public commitments"
            )

    def _blob(self, d: ArrayDescriptor) -> bytes:
        c = self._counts[self._phase]
        c["archives_attempted"] += 1
        blob = _read(self._root / d.name, ENCODED_CAP)
        c["archives_read"] += 1
        c["bytes_read"] += len(blob)
        if _hash(blob) != d.digest:
            raise ValueError("Transcript hash mismatch")
        return blob

    def _decode(self, d: ArrayDescriptor) -> Array:
        a = _array(self._blob(d), d)
        self._counts[self._phase]["arrays_validated"] += 1
        return a

    @property
    def jobs(self) -> tuple[ResponseJob, ...]:
        return self._jobs

    def _stream(self, job: str) -> Iterator[ResponseBatch]:
        if self._status not in ("ready", "delivering"):
            raise ValueError("View terminal; no delivery retry")
        try:
            if job != f"job{self._next_job:02d}":
                raise ValueError("Registered case order required")
            self._phase = "delivery"
            for pair in self._manifest.pairs:
                if pair.job_id != job:
                    continue
                if self._status not in ("ready", "delivering"):
                    raise ValueError("View terminal during delivery")
                yield self._deliver(pair)

        except Exception as exc:
            self._status = "failed"
            self._job_status[job]["status"] = "failed"
            self._error = type(exc).__name__ + ": " + str(exc)[:300]
            raise

    def _deliver(self, pair: PairDescriptor) -> ResponseBatch:
        # This frame dies before the generator yields: no retained delivered array.
        x = self._decode(pair.inputs)
        y = self._decode(pair.responses)
        batch = ResponseBatch(x, y)
        c = self._counts["delivery"]
        c["pairs"] += 1
        c["rows"] += len(batch.inputs)
        c["payload_bytes"] += batch.inputs.nbytes + batch.responses.nbytes
        state = self._job_status[pair.job_id]
        state["delivered_batches"] += 1
        last = state["delivered_batches"] == 2 * self._manifest.profile.probes
        state["status"] = "completed" if last else "partial"
        self._status = "delivering"
        if last:
            self._next_job += 1
        return batch

    def finish(self) -> dict[str, Any]:
        if self._status not in ("ready", "delivering"):
            raise ValueError("View terminal; completion cannot retry")
        if any(v["status"] != "completed" for v in self._job_status.values()):
            raise ValueError("Partial jobs: not every batch delivered")
        try:
            self._phase = "completion"
            self._verify_manifest()
            for pair in self._manifest.pairs:
                self._blob(pair.inputs)
                self._blob(pair.responses)
            self._status = "completed"
        except Exception as exc:
            self._status = "failed"
            self._error = type(exc).__name__ + ": " + str(exc)[:300]
            raise
        return self.report()

    def report(self) -> dict[str, Any]:
        p = self._manifest.profile
        result = dict(
            status=self._status,
            jobs=self._job_status,
            **self._counts,
            source_commit="07235acd0ecaa576d57b37eefb2b4e6fc3aa0c6a",
            mapping={f"job{i:02d}": name for i, name in enumerate(p.case_names)},
            metadata_sha256={x.name: x.digest for x in self._manifest.metadata},
            allowed_archives=len(self._manifest.pairs) * 2,
            raw_payload_bound=p.vector_rows * (p.n + p.outputs) * 8,
            read_bytes_are_lower_bounds_on_io_failure=True,
            new_queries=0,
            new_fits=0,
            certificate=False,
            private_gate="closed",
            error=self._error,
        )
        blob = json.dumps(result, allow_nan=False, sort_keys=True).encode()
        if len(blob) > METADATA_CAP:
            raise ValueError("Adapter report exceeds bounded cap")
        value: dict[str, Any] = json.loads(blob)
        return value


def _open_fixture_view(root: Path, manifest: TranscriptManifest) -> ResponseView:
    """Literal-only test seam; never called by the registered entry point."""
    return ResponseView(root, manifest)


def open_view(root: Path, manifest: TranscriptManifest) -> ResponseView:
    """Fixed registered profile. Call only under a separately approved read run."""
    wanted = tuple((name, h) for name, _, h in PUBLIC_PINS)
    if (
        manifest.profile != REGISTERED
        or manifest.committed_collection != ORIGINAL
        or tuple((p.name, p.digest) for p in manifest.metadata) != wanted
    ):
        raise ValueError("Only registered pinned profile is allowed")
    return ResponseView(root, manifest)
