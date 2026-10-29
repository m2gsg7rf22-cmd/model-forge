"""Hash-bound saved-candidate readers and bounded prefix journal, never a launcher.

Actual use requires the future full reviewed protocol and exclusive audit marker.
No original observations, RNG, schedules or optimizer moments are decoded here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any
from collections.abc import Iterator

from joint_response_arrays import decode_typed
from joint_response_audit_projection import decode_checkpoint
from joint_response_audit_case import CHECKPOINTS, Array
from joint_response_inventory import Field, Shard
from variable_response_manifest import _json, _read
from variable_response_protocol import ancestry, canonical, save


@dataclass(frozen=True)
class _Profile:
    width: int
    n: int = 32
    outputs: int = 32
    probes: int = 17
    rows: int = 2112
    prediction_rows: int = 8192

    def specs(self) -> dict[str, Shard]:
        if (
            any(
                type(v) is not int
                for v in (
                    self.width,
                    self.n,
                    self.outputs,
                    self.probes,
                    self.rows,
                    self.prediction_rows,
                )
            )
            or not 1 <= self.width <= 96
            or not 1 <= self.n <= 32
            or not 1 <= self.outputs <= 32
            or not 4 <= self.probes <= 17
            or not 1 <= self.rows <= 2112
            or not 1 <= self.prediction_rows <= 8192
        ):
            raise ValueError("bounded exact reader profile required")
        params = tuple(
            Field(k, "float64", shape)
            for k, shape in [
                ("g", (self.width, self.n)),
                ("u", (self.width, self.n)),
                ("d", (self.outputs, self.width)),
            ]
        )
        states = tuple(
            Field(prefix + f.name, f.dtype, f.shape)
            for prefix in ("", "m_", "v_")
            for f in params
        )
        total = 2 * self.probes * self.rows
        guard = 2 * sum(i < self.probes for i in (3, 7, 11, 15)) * self.rows
        specs = {
            f"checkpoint-{s:04d}": Shard(f"checkpoint-{s:04d}", states)
            for s in CHECKPOINTS
        }
        specs["selected-physical"] = Shard("selected-physical", params)
        specs["split-indices"] = Shard(
            "split-indices",
            (
                Field("train", "int64", (total - guard,)),
                Field("guard", "int64", (guard,)),
            ),
        )
        for start in range(0, total, self.prediction_rows):
            name = f"prediction-{start // self.prediction_rows:02d}"
            specs[name] = Shard(
                name,
                (
                    Field(
                        "predictions",
                        "float64",
                        (min(self.prediction_rows, total - start), self.outputs),
                    ),
                ),
            )
        return specs


class TraceReader:
    def __init__(
        self,
        root: Path,
        pins: dict[str, str],
        width: int,
        *,
        _profile: _Profile | None = None,
    ):
        self.root = root
        ancestry(root)
        self.profile = _Profile(width) if _profile is None else _profile
        self.specs = self.profile.specs()
        required = {n + ".npz" for n in self.specs} | {
            "case.json",
            "metrics.json",
            "checkpoint-metrics.jsonl",
        }
        allowed = required | {
            "initial.npz",
            "schedule.npz",
            "rng.json",
            "terminal.json",
            "manifest.jsonl",
            "checkpoint-intents.jsonl",
            "progress.jsonl",
        }
        if (
            not isinstance(pins, dict)
            or not required <= set(pins) <= allowed
            or any(
                not isinstance(h, str) or re.fullmatch("[0-9a-f]{64}", h) is None
                for h in pins.values()
            )
        ):
            raise ValueError("complete exact saved case pin whitelist required")
        self.pins = {n: pins[n] for n in required}
        self.used: set[str] = set()
        self.failed = False
        self.done = False
        self.has_metadata = False
        self.reads = 0
        self.read_attempts = 0
        self.bytes_read = 0
        self.numeric_decodes = 0
        self.numeric_decode_attempts = 0
        self.checkpoint_count = 0
        self.prediction_count = 0
        self.completion_hashes = 0
        self.completion_bytes = 0

    def _active(self) -> None:
        if self.failed or self.done:
            raise ValueError("terminal reader; no retry")

    def _blob(self, name: str, cap: int) -> bytes:
        self._active()
        try:
            if name not in self.pins or name in self.used:
                raise ValueError("unregistered or repeated case read")
            self.read_attempts += 1
            blob = _read(self.root / name, cap)
            self.bytes_read += len(blob)
            if hashlib.sha256(blob).hexdigest() != self.pins[name]:
                raise ValueError("case hash changed before decode")
            self.used.add(name)
            self.reads += 1
            return blob
        except Exception:
            self.failed = True
            raise

    def _numeric(self, name: str) -> dict[str, Any]:
        spec = self.specs[name]
        blob = self._blob(name + ".npz", spec.encoded_bound)
        self.numeric_decode_attempts += 1
        try:
            result = decode_typed(blob, spec)
            self.numeric_decodes += 1
            return result
        except Exception:
            self.failed = True
            raise

    def metadata(self) -> dict[str, Any]:
        self._active()
        try:
            if self.has_metadata or self.checkpoint_count:
                raise ValueError("metadata first and once required")
            case = _json(self._blob("case.json", 128 * 1024))
            metrics = _json(self._blob("metrics.json", 128 * 1024))
            blob = self._blob("checkpoint-metrics.jsonl", 17 * 4096)
            lines = blob.splitlines()
            if (
                not blob.endswith(b"\n")
                or len(lines) != 17
                or any(len(line) + 1 > 4096 for line in lines)
            ):
                raise ValueError("exact bounded checkpoint journal required")
            records = [_json(line) for line in lines]
            expected = {
                "schema",
                "checkpoints",
                "initial",
                "final",
                "selected",
                "selected_step",
                "selected_guard_mse",
                "guard_is_consumed_calibration",
                "updates_completed",
                "checkpoint_metric_records",
                "metric_journal_bytes",
            }
            if (
                set(metrics) != expected
                or metrics["schema"] != "joint-response-metrics-v1"
                or metrics["guard_is_consumed_calibration"] is not True
                or type(metrics["updates_completed"]) is not int
                or metrics["updates_completed"] != 4000
                or type(metrics["checkpoint_metric_records"]) is not int
                or metrics["checkpoint_metric_records"] != 17
                or type(metrics["metric_journal_bytes"]) is not int
                or metrics["metric_journal_bytes"] != len(blob)
                or canonical(metrics["checkpoints"]) != canonical(records)
                or canonical(metrics["initial"]) != canonical(records[0])
                or canonical(metrics["final"]) != canonical(records[-1])
            ):
                raise ValueError("exact saved metrics/journal binding required")
            step = metrics["selected_step"]
            if (
                type(step) is not int
                or step not in CHECKPOINTS
                or canonical(metrics["selected"])
                != canonical(records[CHECKPOINTS.index(step)])
                or type(metrics["selected_guard_mse"]) not in (float, int)
                or metrics["selected_guard_mse"]
                != records[CHECKPOINTS.index(step)].get("guard_mse")
            ):
                raise ValueError("selected metadata differs from checkpoint")
            physical = self._numeric("selected-physical")
            split = self._numeric("split-indices")
            self.has_metadata = True
            return dict(
                metadata=case,
                records=records,
                selected_step=step,
                physical=physical,
                saved_split=split,
            )
        except Exception:
            self.failed = True
            raise

    def checkpoint(self, step: int) -> dict[str, Array]:
        self._active()
        try:
            if (
                not self.has_metadata
                or self.checkpoint_count >= 17
                or type(step) is not int
                or step != CHECKPOINTS[self.checkpoint_count]
            ):
                raise ValueError("exact checkpoint read order required")
            name = f"checkpoint-{step:04d}"
            blob = self._blob(name + ".npz", self.specs[name].encoded_bound)
            self.numeric_decode_attempts += 1
            result = decode_checkpoint(
                blob, self.profile.width, n=self.profile.n, outputs=self.profile.outputs
            )
            self.numeric_decodes += 1
            self.checkpoint_count += 1
            return result
        except Exception:
            self.failed = True
            raise

    def predictions(self) -> Iterator[Array]:
        self._active()
        try:
            if self.checkpoint_count != 17:
                raise ValueError("all checkpoints precede predictions")
            total = 2 * self.profile.probes * self.profile.rows
            for start in range(0, total, self.profile.prediction_rows):
                index = start // self.profile.prediction_rows
                if index != self.prediction_count:
                    raise ValueError("exact one-use prediction order required")
                value = self._numeric(f"prediction-{index:02d}")["predictions"]
                self.prediction_count += 1
                yield value
        except Exception:
            self.failed = True
            raise

    def report(self) -> dict[str, Any]:
        return dict(
            reads=self.reads,
            read_attempts=self.read_attempts,
            bytes_read=self.bytes_read,
            numeric_decodes=self.numeric_decodes,
            numeric_decode_attempts=self.numeric_decode_attempts,
            checkpoints=self.checkpoint_count,
            prediction_shards=self.prediction_count,
            completion_hashes=self.completion_hashes,
            completion_bytes=self.completion_bytes,
            read_bytes_are_lower_bounds_on_io_failure=True,
            completion_bytes_are_lower_bounds_on_io_failure=True,
            failed=self.failed,
            status="completed" if self.done else "failed" if self.failed else "partial",
            certificate=False,
            private_gate="closed",
        )

    def finish(self) -> dict[str, Any]:
        self._active()
        try:
            if self.used != set(self.pins) or self.checkpoint_count != 17:
                raise ValueError("all registered reads must finish first")
            # Rehash only already admitted outputs, streaming without numeric decode.
            for name in sorted(self.used):
                path = self.root / name
                ancestry(path)
                cap = (
                    self.specs[name[:-4]].encoded_bound
                    if name.endswith(".npz")
                    else 17 * 4096
                    if name.endswith(".jsonl")
                    else 128 * 1024
                )
                if not path.is_file() or not 0 < path.stat().st_size <= cap:
                    raise ValueError("completion file cap changed")
                h = hashlib.sha256()
                seen = 0
                with path.open("rb") as f:
                    while chunk := f.read(min(1024 * 1024, cap - seen + 1)):
                        seen += len(chunk)
                        self.completion_bytes += len(chunk)
                        if seen > cap:
                            raise ValueError("completion stream cap exceeded")
                        h.update(chunk)
                if h.hexdigest() != self.pins[name]:
                    raise ValueError("completion hash changed")
                self.completion_hashes += 1
            self.done = True
            return self.report()
        except Exception:
            self.failed = True
            raise


def _fixture_reader(
    root: Path,
    pins: dict[str, str],
    *,
    width: int,
    n: int,
    outputs: int,
    probes: int,
    rows: int,
    prediction_rows: int,
) -> TraceReader:
    return TraceReader(
        root,
        pins,
        width,
        _profile=_Profile(width, n, outputs, probes, rows, prediction_rows),
    )


class AuditStore:
    def __init__(self, root: Path):
        ancestry(root)
        root.mkdir(exist_ok=False)
        self.root = root
        self.path = root / "checkpoint-metrics.jsonl"
        self.rows: list[dict[str, Any]] = []
        self.bytes = 0
        self.failed = False
        self.done = False
        self.uncommitted_tail = False
        with self.path.open("xb") as f:
            f.flush()
            os.fsync(f.fileno())

    def append(self, row: dict[str, Any]) -> None:
        if self.failed or self.done:
            raise ValueError("terminal journal; no retry")
        try:
            if (
                len(self.rows) >= 17
                or type(row.get("step")) is not int
                or row["step"] != CHECKPOINTS[len(self.rows)]
            ):
                raise ValueError("exact checkpoint append order required")
            blob = canonical(row) + b"\n"
            ancestry(self.path)
            if (
                len(blob) > 4096
                or self.path.stat().st_size != self.bytes
                or self.bytes + len(blob) > 17 * 4096
            ):
                raise ValueError("journal layout/size cap exceeded")
            with self.path.open("ab") as f:
                self.uncommitted_tail = True
                if f.write(blob) != len(blob):
                    raise OSError("short audit journal write")
                f.flush()
                os.fsync(f.fileno())
            self.rows.append(_json(blob))
            self.bytes += len(blob)
            self.uncommitted_tail = False
        except Exception:
            self.failed = True
            raise

    def complete(self, result: dict[str, Any]) -> None:
        if self.failed or self.done:
            raise ValueError("terminal store")
        try:
            if (
                len(self.rows) != 17
                or canonical(result.get("checkpoints")) != canonical(self.rows)
                or result.get("status")
                not in ("agreed", "ambiguous", "disagreed", "numerical_failure")
                or result.get("certificate") is not False
                or result.get("private_gate") != "closed"
            ):
                raise ValueError("complete result must bind all17journalrecords")
            actual_journal = _read(self.path, 17 * 4096)
            expected_journal = b"".join(canonical(row) + b"\n" for row in self.rows)
            if len(actual_journal) != self.bytes or actual_journal != expected_journal:
                raise ValueError(
                    "saved journal differs from committed checkpoint prefix"
                )
            save(self.root / "result.json", result, 128 * 1024)
            self.done = True
        except Exception:
            self.failed = True
            raise

    def fail(self, error_type: str) -> None:
        if self.done:
            raise ValueError("completed result cannot be replaced")
        if (self.root / "result.json").exists() or (
            self.root / "result.json.tmp"
        ).exists():
            raise ValueError("failed receipt cannot be retried")
        self.failed = True
        save(
            self.root / "result.json",
            dict(
                status="failed",
                error_type=str(error_type)[:128],
                committed_checkpoint_records=len(self.rows),
                journal_bytes=self.bytes,
                uncommitted_tail=self.uncommitted_tail,
                certificate=False,
                private_gate="closed",
            ),
            128 * 1024,
        )
