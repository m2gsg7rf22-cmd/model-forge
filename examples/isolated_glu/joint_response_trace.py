"""Exclusive bounded per-case trace IO; no RNG, observation reads or fitting.

This is not an approved experiment launcher. Future controller must bind metadata,
initial/schedule provenance, full source/runtime/attempt gates and global budgets.
"""

from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from joint_response_inventory import trace_specs
from scaling_domain_store import _ancestry

TraceArray = NDArray[np.float64] | NDArray[np.int64]
CHECKPOINTS = (0, *range(256, 4000, 256), 4000)
METADATA_NAMES = {"rng.json", "case.json", "metrics.json", "terminal.json"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            h.update(chunk)
    return h.hexdigest()


def _json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )


class TraceStore:
    """One immutable numeric file per registered name; any error is terminal.

    Closed numeric/metadata files and checkpoint intents/manifests are fsynced.
    Step progress is closed per event, without an fsync per Adam update. No atomic
    host-power-loss or adversarial concurrent-writer guarantee is asserted.
    """

    def __init__(self, root: Path, width: int):
        self.specs = {s.name: s for s in trace_specs(width)}
        self.root = root.absolute()
        _ancestry(self.root)
        self.root.mkdir(exist_ok=False)
        self.names: set[str] = set()
        self.metadata_names: set[str] = set()
        self.sizes: dict[str, int] = {}
        self.reserved_bytes = 0
        self.metadata_bytes = 0
        self.progress_bytes = 0
        self.completed_updates = 0
        self.next_checkpoint = 0
        self.pending_intent: int | None = None
        self.failed = False
        self.finished = False
        for name in ("manifest.jsonl", "progress.jsonl", "checkpoint-intents.jsonl"):
            self._exclusive(self.root / name, b"")
            self.sizes[name] = 0

    def _ready(self) -> None:
        if self.failed or self.finished:
            raise ValueError("trace terminal; no retry")
        _ancestry(self.root)

    def _exclusive(self, path: Path, blob: bytes) -> None:
        _ancestry(path)
        with path.open("xb") as stream:
            if stream.write(blob) != len(blob):
                raise OSError("short trace write; no retry")
            stream.flush()
            os.fsync(stream.fileno())

    def _append(
        self, name: str, blob: bytes, cap: int, *, progress: bool = False
    ) -> None:
        path = self.root / name
        _ancestry(path)
        if not path.is_file() or path.stat().st_size != self.sizes[name]:
            raise ValueError("trace append file changed")
        new_size = self.sizes[name] + len(blob)
        if new_size > cap or (
            not progress and self.metadata_bytes + len(blob) > 1024**2
        ):
            raise ValueError("trace record/metadata aggregate cap exceeded")
        with path.open("ab") as stream:
            if stream.write(blob) != len(blob):
                raise OSError("short trace write; no retry")
            stream.flush()
            if not progress:
                os.fsync(stream.fileno())
        self.sizes[name] = new_size
        if progress:
            self.progress_bytes = new_size
        else:
            self.metadata_bytes += len(blob)

    def _initialized(self) -> bool:
        return {"initial", "schedule", "split-indices"} <= self.names and {
            "rng.json",
            "case.json",
        } <= self.metadata_names

    def _validate(self, name: str, arrays: dict[str, TraceArray]) -> None:
        spec = self.specs[name]
        if set(arrays) != {f.name for f in spec.fields}:
            raise ValueError("exact trace fields required")
        for field in spec.fields:
            a = arrays[field.name]
            if (
                not isinstance(a, np.ndarray)
                or a.dtype != np.dtype(field.dtype)
                or not a.dtype.isnative
                or a.shape != field.shape
                or not np.isfinite(a).all()
            ):
                raise ValueError("exact finite native trace schema required")
        if name == "schedule" and (
            np.any(arrays["indices"] < 0) or np.any(arrays["indices"] >= 54912)
        ):
            raise ValueError("schedule index outside training rows")
        if name == "split-indices":
            indices = np.arange(71808, dtype=np.int64)
            guard = np.isin((indices // 2112) % 17, (3, 7, 11, 15))
            if not np.array_equal(
                arrays["train"], indices[~guard]
            ) or not np.array_equal(arrays["guard"], indices[guard]):
                raise ValueError("canonical whole-center split required")

    def save(self, name: str, arrays: dict[str, TraceArray]) -> dict[str, Any]:
        try:
            self._ready()
            if name not in self.specs:
                raise ValueError("unregistered shard name")
            if name in self.names:
                raise FileExistsError(name)
            if name.startswith("checkpoint-"):
                step = int(name.removeprefix("checkpoint-"))
                if self.pending_intent != step or not self._initialized():
                    raise ValueError(
                        "checkpoint requires persisted matching intent and initialization"
                    )
            if name == "selected-physical" or name.startswith("prediction-"):
                if (
                    self.completed_updates != 4000
                    or "checkpoint-4000" not in self.names
                    or "failure-state" in self.names
                ):
                    raise ValueError(
                        "selected output requires completed fit, never failed prefix"
                    )
                if (
                    name.startswith("prediction-")
                    and "selected-physical" not in self.names
                ):
                    raise ValueError(
                        "predictions require selected physical parameters first"
                    )
            if (
                name in ("initial", "schedule", "split-indices")
                and self.completed_updates
            ):
                raise ValueError("initial trace must precede updates")
            self._validate(name, arrays)
            spec = self.specs[name]
            if self.reserved_bytes + spec.encoded_bound > sum(
                s.encoded_bound for s in self.specs.values()
            ):
                raise ValueError("numeric trace aggregate cap exceeded")
            self.reserved_bytes += spec.encoded_bound
            self.names.add(name)  # Reservation survives failed writes; never retry.
            stream = io.BytesIO()
            np.savez(stream, allow_pickle=False, **cast(dict[str, Any], arrays))
            blob = stream.getvalue()
            if len(blob) > spec.encoded_bound:
                raise ValueError("encoded shard cap exceeded")
            path = self.root / (name + ".npz")
            self._exclusive(path, blob)
            receipt = dict(
                name=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()
            )
            record = _json(receipt)
            if len(record) > 4096:
                raise ValueError("manifest record cap exceeded")
            self._append("manifest.jsonl", record, 31 * 4096)
            if name.startswith("checkpoint-"):
                self.pending_intent = None
                self.next_checkpoint += 1
            return receipt
        except Exception:
            self.failed = True
            raise

    def metadata(self, name: str, value: dict[str, Any]) -> None:
        try:
            self._ready()
            if name not in METADATA_NAMES:
                raise ValueError("unregistered metadata name")
            if name in self.metadata_names:
                raise FileExistsError(name)
            if name == "terminal.json":
                status = value.get("status")
                if status not in ("completed", "numerical_failure", "partial"):
                    raise ValueError("explicit terminal status required")
                if status == "completed" and (
                    not {"rng.json", "case.json", "metrics.json"} <= self.metadata_names
                    or self.names != set(self.specs) - {"failure-state"}
                    or self.completed_updates != 4000
                ):
                    raise ValueError("complete trace inventory required for success")
                if status == "numerical_failure" and "failure-state" not in self.names:
                    raise ValueError("numerical failure requires finite saved state")
            blob = _json(value)
            if len(blob) > 128 * 1024 or self.metadata_bytes + len(blob) > 1024**2:
                raise ValueError("bounded case metadata required")
            self.metadata_bytes += len(blob)
            self.metadata_names.add(name)
            self._exclusive(self.root / name, blob)
            if name == "terminal.json":
                self.finished = True
        except Exception:
            self.failed = True
            raise

    def intent(self, step: int) -> None:
        try:
            self._ready()
            if (
                not self._initialized()
                or self.pending_intent is not None
                or self.next_checkpoint >= len(CHECKPOINTS)
                or type(step) is not int
                or step != CHECKPOINTS[self.next_checkpoint]
                or step != self.completed_updates
            ):
                raise ValueError("canonical checkpoint intent order required")
            self._append("checkpoint-intents.jsonl", _json(dict(step=step)), 17 * 4096)
            self.pending_intent = step
        except Exception:
            self.failed = True
            raise

    def progress(self, step: int, sampled_loss: float) -> None:
        try:
            self._ready()
            if (
                not self._initialized()
                or self.pending_intent is not None
                or "failure-state" in self.names
                or "checkpoint-0000" not in self.names
                or type(step) is not int
                or step != self.completed_updates + 1
                or step > 4000
                or isinstance(sampled_loss, bool)
                or not math.isfinite(sampled_loss)
                or sampled_loss < 0
                or (
                    self.completed_updates in CHECKPOINTS
                    and f"checkpoint-{self.completed_updates:04d}" not in self.names
                )
            ):
                raise ValueError(
                    "ordered finite step progress requires committed previous checkpoints"
                )
            blob = _json(dict(step=step, sampled_loss_before_update=sampled_loss))
            if len(blob) > 512:
                raise ValueError("progress record cap exceeded")
            self._append("progress.jsonl", blob, 4000 * 512, progress=True)
            self.completed_updates = step
        except Exception:
            self.failed = True
            raise
