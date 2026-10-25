"""Bounded proposal artifacts/journals; no actual model fitting or target access."""

from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge.glu32_io import raw

Array = NDArray[np.float64]


def load(name: str) -> Any:
    s = importlib.util.spec_from_file_location(
        "frozen_store_" + name, Path(__file__).with_name(name + ".py")
    )
    assert s and s.loader
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


traced = load("guard_aware_trace")
transport = load("evidence_transport")


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def reject_constant(value: str) -> None:
    raise ValueError("Nonfinite journal constant: " + value)


def inspect_journal(path: Path, proposal_id: str) -> dict[str, Any]:
    data = raw(path)
    lines = data.splitlines(keepends=True)
    count = 0
    last = None
    terminal = False
    truncated = False
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if index != len(lines) - 1:
                raise ValueError("Unexpected incomplete interior journal line")
            truncated = True
            break
        event = json.loads(line, parse_constant=reject_constant)
        if (
            terminal
            or not isinstance(event, dict)
            or event.get("ordinal") != count
            or event.get("proposal_id") != proposal_id
        ):
            raise ValueError("Journal ordering/identity/terminal invariant failed")
        last = event.get("kind")
        count += 1
        if count > 514:
            raise ValueError("Journal exceeds bounded514events")
        terminal = last in {"proposal_completed", "proposal_failed"}
    return {
        "records": count,
        "truncated_tail": truncated,
        "closed_with_terminal": terminal and not truncated,
        "successful": last == "proposal_completed" and terminal and not truncated,
    }


class TraceStore:
    def __init__(self, root: Path):
        if any(p.is_symlink() for p in [root, *root.parents]):
            raise ValueError("No symlink trace ancestors")
        root.mkdir(exist_ok=False)
        self.root = root
        self.entries: list[dict[str, Any]] = []
        self.failed = False
        self.closed = False
        (root / "proposals").mkdir()
        (root / "attempt.json").write_text(
            '{"automatic_retry":false,"max_proposals":72,"max_subset_solves":18432}\n'
        )

    def checkpoint(self) -> None:
        value = {
            "schema_version": 1,
            "proposals": self.entries,
            "failed": self.failed,
            "closed": self.closed,
            "private_gate": "closed",
        }
        temporary = self.root / "manifest.tmp"
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.root / "manifest.json")

    def run_proposal(
        self, proposal_id: str, g: Array, d: Array, k: Array, a: Array
    ) -> tuple[Array, dict[str, Any]]:
        if self.closed or self.failed:
            raise ValueError("Closed or failed evidence attempt; no retry")
        if (
            len(self.entries) >= 72
            or not isinstance(proposal_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", proposal_id)
        ):
            self.failed = True
            self.checkpoint()
            raise ValueError("Invalid proposal id/budget")
        output = self.root / "proposals" / proposal_id
        try:
            output.mkdir(exist_ok=False)
        except OSError:
            self.failed = True
            self.checkpoint()
            raise
        values = [g, d, k, a]
        if any(
            not isinstance(v, np.ndarray)
            or v.dtype.kind not in "fiu"
            or v.nbytes > 32768
            for v in values
        ):
            self.failed = True
            self.checkpoint()
            raise ValueError("Raw problem arrays outside bounded numeric domain")
        try:
            np.savez(output / "problem.npz", gate=g, gradient=d, gram=k, constraints=a)
        except Exception:
            self.failed = True
            self.checkpoint()
            raise
        trace: dict[str, Any]
        failure: Exception | None = None
        direction = np.empty(0)
        try:
            with (output / "events.jsonl").open("x", buffering=1) as log:

                def sink(event: dict[str, Any]) -> None:
                    log.write(json.dumps(event, allow_nan=False) + "\n")
                    log.flush()

                direction, trace = traced.traced_direction(
                    g, d, k, a, proposal_id=proposal_id, event_sink=sink
                )
        except traced.TraceError as exc:
            trace = exc.trace
            failure = exc
            self.failed = True
        except Exception:
            self.failed = True
            self.checkpoint()
            raise
        try:
            encoded = (json.dumps(trace, indent=2, allow_nan=False) + "\n").encode()
            (output / "trace.json").write_bytes(encoded)
            journal = inspect_journal(output / "events.jsonl", proposal_id)
            if not journal["closed_with_terminal"] or not journal["successful"]:
                self.failed = True
            self.entries.append(
                {
                    "proposal_id": proposal_id,
                    "trace_file": str(output.relative_to(self.root) / "trace.json"),
                    "trace_sha256": sha(output / "trace.json"),
                    "trace_bytes": len(encoded),
                    "problem_sha256": sha(output / "problem.npz"),
                    "journal_sha256": sha(output / "events.jsonl"),
                    "journal": journal,
                    "terminal_status": trace["terminal_status"],
                    "subset_count": len(trace["subsets"]),
                    "linear_solves_attempted": trace["linear_solves_attempted"],
                }
            )
            self.checkpoint()
        except Exception:
            self.failed = True
            self.checkpoint()
            raise
        if failure is not None:
            raise failure
        if self.failed:
            raise ValueError("Incomplete proposal journal; attempt failed")
        return direction, trace

    def finalize(
        self, arrays: dict[str, Array], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        if self.closed:
            raise ValueError("Evidence already finalized; no retry")
        self.closed = True
        gates = arrays.get("trial_gates")
        if (
            gates is None
            or gates.ndim != 2
            or len(gates) > 64
            or not 1 <= gates.shape[1] <= 32
        ):
            self.failed = True
            self.checkpoint()
            raise ValueError("Atmost64bounded trial gate rows required")
        solves = sum(v["linear_solves_attempted"] for v in self.entries)
        if solves > 18432:
            self.failed = True
            self.checkpoint()
            raise ValueError("Subset solve budget exceeded")
        self.checkpoint()
        try:
            summary = transport.save_evidence(
                self.root / "evidence",
                arrays,
                {
                    "manifest_file": "../manifest.json",
                    "manifest_sha256": sha(self.root / "manifest.json"),
                    "proposal_count": len(self.entries),
                    "status": "failed" if self.failed else "completed",
                    "caller_metadata": metadata,
                },
            )
            receipt = {
                "method": "guard-aware-step-evidence-v1",
                "status": "failed" if self.failed else "completed",
                "proposal_count": len(self.entries),
                "subset_count": sum(v["subset_count"] for v in self.entries),
                "subset_linear_solves_attempted": solves,
                "trial_count": len(gates),
                "manifest_sha256": sha(self.root / "manifest.json"),
                "numeric_sha256": summary["evidence_sha256"],
                "summary_sha256": sha(self.root / "evidence/summary.json"),
                "counter_scope": "evidence_helper_only_not_caller_fit",
                "actual_target_queries": 0,
                "actual_candidate_updates": 0,
                "private_gate": "closed",
            }
            (self.root / "receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            return receipt
        except Exception:
            self.failed = True
            self.checkpoint()
            raise


def synthetic_transport(output: Path) -> dict[str, Any]:
    store = TraceStore(output)
    k = np.diag(np.linspace(1.0, 2.0, 32))
    a = np.eye(8, 32)
    for i in range(72):
        store.run_proposal(
            "control-" + str(i),
            np.full(32, 100.0 + i),
            -np.arange(1.0, 33.0) / (10 + i),
            k,
            a,
        )
    x = np.ones((1024, 32))
    y = np.full((1024, 32), 0.5)
    w = np.linspace(0.01, 0.02, 1024)
    weighted = y * w[:, None]
    arrays = {
        "trial_gates": np.zeros((64, 32)),
        "physical_train_x": x,
        "physical_train_y": y,
        "weights": w,
        "weighted_target": weighted,
        "initial_weighted_residual": weighted.copy(),
        "final_weighted_residual": weighted.copy(),
        "physical_guard_x": x.copy(),
        "physical_guard_y": y.copy(),
    }
    return store.finalize(
        arrays,
        {
            "synthetic_transport_only": True,
            "actual_saved_model_data": False,
            "actual_fits": 0,
        },
    )
