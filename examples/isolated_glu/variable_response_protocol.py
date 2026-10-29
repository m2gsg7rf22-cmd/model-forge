"""Bounded read-only receipts/marker/chain, without source transcript access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from variable_response_manifest import METADATA_CAP, _json, _read, _safe

MARKER = "variable-response-view-v1-attempt.json"
POLICY = dict(
    id="variable-response-view-v1",
    jobs=6,
    n=32,
    outputs=32,
    probes=17,
    steps=2,
    rows_per_pair=2112,
    timeout_seconds=180,
    memory_limit_mb=512,
    blas_threads=1,
    output_limit_bytes=2048,
    min_free_bytes=2 * 1024**3,
    new_queries=0,
    new_fits=0,
    certificate=False,
    private_gate="closed",
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def digest(path: Path, cap: int = METADATA_CAP) -> str:
    return hashlib.sha256(_read(path, cap)).hexdigest()


def read(path: Path, cap: int = METADATA_CAP) -> dict[str, Any]:
    return _json(_read(path, cap))


def ancestry(path: Path) -> None:
    if ".." in path.parts or any(x.is_symlink() for x in (path, *path.parents)):
        raise ValueError("Symlink/aliased output forbidden")


def _encoded(value: Any, cap: int) -> bytes:
    blob = canonical(value)
    if len(blob) > cap:
        raise ValueError("JSON exceeds bounded cap")
    return blob


def save(path: Path, value: Any, cap: int = METADATA_CAP) -> None:
    blob = _encoded(value, cap)
    ancestry(path)
    if path.exists():
        _safe(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("xb") as f:
            if f.write(blob) != len(blob):
                raise OSError("Short receipt write")
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    except Exception:
        # A failed partial write is preserved; never claim it committed.
        raise


def marker(path: Path, value: Any) -> str:
    blob = _encoded(value, METADATA_CAP)
    ancestry(path)
    with path.open("xb") as f:
        if f.write(blob) != len(blob):
            raise OSError("Short marker; no retry")
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(blob).hexdigest()


def tree_ok(value: dict[str, Any]) -> bool:
    return (
        value.get("process_monitor_scope") == "process_and_children"
        and type(value.get("memory_samples")) is int
        and value["memory_samples"] > 0
        and type(value.get("peak_sampled_rss_bytes")) is int
        and 0 < value["peak_sampled_rss_bytes"] <= 512 * 1024**2
        and type(value.get("memory_limit_mb")) is int
        and value["memory_limit_mb"] == 512
    )


class DeliveryChain:
    def __init__(self, path: Path, batches_per_job: int = 34) -> None:
        if type(batches_per_job) is not int or not 1 <= batches_per_job <= 34:
            raise ValueError("Bounded batch count required")
        ancestry(path)
        with path.open("xb"):
            pass
        self.path = path
        self.batches_per_job = batches_per_job
        self.head = "0" * 64
        self.count = 0
        self.bytes = 0
        self.failed = False

    def append(
        self,
        job: str,
        index: int,
        inputs_shape: tuple[int, ...],
        responses_shape: tuple[int, ...],
        inputs_sha256: str,
        responses_sha256: str,
    ) -> None:
        if self.failed:
            raise ValueError("Terminal chain; no append retry")
        try:
            record: dict[str, Any] = dict(
                job_id=job,
                index=index,
                inputs_shape=list(inputs_shape),
                responses_shape=list(responses_shape),
                inputs_sha256=inputs_sha256,
                responses_sha256=responses_sha256,
                previous=self.head,
            )
            self._validate(record, self.count)
            record["chain"] = hashlib.sha256(canonical(record)).hexdigest()
            blob = _encoded(record, 511) + b"\n"
            _safe(self.path)
            if (
                self.path.stat().st_size != self.bytes
                or self.bytes + len(blob) > METADATA_CAP
            ):
                raise ValueError("Chain size changed or exceeded cap")
            with self.path.open("ab") as f:
                if f.write(blob) != len(blob):
                    raise OSError("Short chain write; unknown partial delivery")
                f.flush()
                os.fsync(f.fileno())
            self.count += 1
            self.bytes += len(blob)
            self.head = record["chain"]
        except Exception:
            self.failed = True
            raise

    def _validate(self, value: dict[str, Any], ordinal: int) -> None:
        if (
            set(value)
            != {
                "job_id",
                "index",
                "inputs_shape",
                "responses_shape",
                "inputs_sha256",
                "responses_sha256",
                "previous",
            }
            or ordinal >= 204
            or value["job_id"] != f"job{ordinal // self.batches_per_job:02d}"
            or type(value["index"]) is not int
            or value["index"] != ordinal % self.batches_per_job
        ):
            raise ValueError("Exact ordered chain record required")
        for key in ("inputs_sha256", "responses_sha256", "previous"):
            if (
                not isinstance(value[key], str)
                or re.fullmatch("[0-9a-f]{64}", value[key]) is None
            ):
                raise ValueError("SHA256 chain fields required")
        for key in ("inputs_shape", "responses_shape"):
            shape = value[key]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(x) is not int for x in shape)
                or not 1 <= shape[0] <= 2112
                or not 1 <= shape[1] <= 32
            ):
                raise ValueError("Bounded native shape required")
        if value["inputs_shape"][0] != value["responses_shape"][0]:
            raise ValueError("Paired rows required")

    def verify(self) -> dict[str, Any]:
        blob = _read(self.path, METADATA_CAP)
        head = "0" * 64
        lines = blob.splitlines()
        for index, line in enumerate(lines):
            if len(line) + 1 > 512:
                raise ValueError("Bounded chain line required")
            record = _json(line)
            claimed = record.pop("chain", None)
            self._validate(record, index)
            if (
                record["previous"] != head
                or claimed != hashlib.sha256(canonical(record)).hexdigest()
            ):
                raise ValueError("Chain order/commitment mismatch")
            head = claimed
        if len(lines) != self.count or len(blob) != self.bytes or head != self.head:
            raise ValueError("Chain receipt state mismatch")
        return dict(
            pairs=self.count,
            head=head,
            bytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(),
        )
