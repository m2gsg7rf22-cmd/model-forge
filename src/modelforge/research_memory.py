"""Bounded, crash-recoverable research checkpoints; never executes queued work."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any
import uuid

import psutil

from .execution import _stop

MAX_STATE_BYTES = 2 * 1024 * 1024
MAX_LEDGER_BYTES = 64 * 1024 * 1024
STATUSES = {"queued", "running", "completed", "failed", "interrupted"}


class MemoryError(ValueError):
    """Invalid or inconsistent persisted evidence."""


def _capture_command(command: Sequence[str], timeout_seconds: float = 10) -> str:
    """Capture bounded Git text without Windows pipe-reader cleanup threads."""
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
        process = subprocess.Popen(
            list(command), stdin=subprocess.DEVNULL, stdout=output, stderr=error
        )
        deadline = time.monotonic() + timeout_seconds
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise MemoryError("Git verification exceeded its time budget")
                if (
                    os.fstat(output.fileno()).st_size + os.fstat(error.fileno()).st_size
                    > MAX_STATE_BYTES
                ):
                    raise MemoryError("Git verification exceeded its output budget")
                time.sleep(0.01)
        finally:
            _stop(process)
        if (
            os.fstat(output.fileno()).st_size + os.fstat(error.fileno()).st_size
            > MAX_STATE_BYTES
        ):
            raise MemoryError("Git verification exceeded its output budget")
        if process.returncode:
            raise MemoryError("Git verification failed: " + " ".join(command[-2:]))
        output.seek(0)
        return output.read(MAX_STATE_BYTES).decode("utf-8", errors="replace").strip()


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise MemoryError("Value is not bounded finite JSON") from exc


def process_identity(pid: int | None = None) -> dict[str, Any]:
    process = psutil.Process(os.getpid() if pid is None else pid)
    return {"pid": process.pid, "create_time": process.create_time()}


def identity_alive(identity: dict[str, Any]) -> bool:
    try:
        return bool(
            abs(psutil.Process(identity["pid"]).create_time() - identity["create_time"])
            < 0.001
        )
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # Insufficient evidence of death: never steal this owner.


def _identity(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and type(value.get("pid")) is int
        and 0 < value["pid"] <= 2**31 - 1
        and type(value.get("create_time")) in (int, float)
        and 0 < value["create_time"] < float("inf")
    )


def validate_state(state: Any) -> dict[str, Any]:
    if (
        not isinstance(state, dict)
        or type(state.get("schema_version")) is not int
        or state.get("schema_version") != 1
    ):
        raise MemoryError("Expected research state schema_version 1")
    if len(canonical(state)) > MAX_STATE_BYTES:
        raise MemoryError("State exceeds size limit")
    base = state.get("checkpoint_base_commit")
    if not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40}", base):
        raise MemoryError("checkpoint_base_commit must identify a prior Git commit")
    head = state.get("ledger_head")
    if head is not None and (
        not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{64}", head)
    ):
        raise MemoryError("Invalid ledger_head")
    gate = state.get("private_model_gate")
    if not isinstance(gate, dict) or gate.get("status") != "closed":
        raise MemoryError(
            "Private model gate must remain closed pending independent verification"
        )
    tasks = state.get("tasks")
    if not isinstance(tasks, list) or len(tasks) > 1000:
        raise MemoryError("Expected at most 1000 tasks")
    ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise MemoryError("Task must be an object")
        name = task.get("id")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", name)
            or name in ids
        ):
            raise MemoryError("Task IDs must be unique bounded identifiers")
        ids.add(name)
        if not isinstance(task.get("status"), str) or task["status"] not in STATUSES:
            raise MemoryError("Invalid task status")
        if type(task.get("priority")) is not int or not 0 <= task["priority"] <= 100000:
            raise MemoryError("Invalid task priority")
        deps = task.get("dependencies")
        if (
            not isinstance(deps, list)
            or len(deps) > 1000
            or any(not isinstance(d, str) for d in deps)
        ):
            raise MemoryError("Invalid task dependencies")
        owner = task.get("owner")
        if owner is not None and not _identity(owner):
            raise MemoryError("Invalid task process identity")
        if task["status"] == "running" and owner is None:
            raise MemoryError("Running task requires process identity")
        requires_gate = task.get("requires_private_model_gate", False)
        if type(requires_gate) is not bool:
            raise MemoryError("Invalid private-model gate requirement")
        if (requires_gate or name == "private_luna_model") and task["status"] in {
            "running",
            "completed",
        }:
            raise MemoryError(
                "Private model task cannot run while its verification gate is closed"
            )
    graph = {t["id"]: t["dependencies"] for t in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise MemoryError("Cyclic task dependencies")
        if name in visited:
            return
        visiting.add(name)
        for dep in graph[name]:
            if dep not in ids:
                raise MemoryError("Unknown task dependency")
            visit(dep)
        visiting.remove(name)
        visited.add(name)

    try:
        for name in graph:
            visit(name)
    except RecursionError as exc:
        raise MemoryError("Dependency graph too deep") from exc
    return state


def _read_json(path: Path, limit: int) -> Any:
    if path.is_symlink() or path.stat().st_size > limit:
        raise MemoryError("Refusing oversized or symlink input")
    try:
        return json.loads(path.read_bytes())
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise MemoryError("Invalid JSON input") from exc


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


class RepositoryLock:
    """Fail-fast mkdir lock. A live or unreadable owner is never displaced."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".modelforge" / "research.lock"
        self.token = uuid.uuid4().hex

    def __enter__(self) -> RepositoryLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                self.path.mkdir()
            except FileExistsError:
                try:
                    owner = _read_json(self.path / "owner.json", 4096)
                except (OSError, MemoryError):
                    raise MemoryError(
                        "Research lock has no trustworthy owner; manual inspection required"
                    ) from None
                if not isinstance(owner, dict) or not _identity(owner):
                    raise MemoryError("Invalid lock owner; manual inspection required")
                if identity_alive(owner):
                    raise MemoryError("Research checkpoint is locked by a live process")
                # Serialize stale cleanup with a claim inside the existing lock.
                try:
                    claim = os.open(
                        self.path / "recovery.claim",
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                except FileExistsError:
                    raise MemoryError(
                        "Another process is recovering the stale lock"
                    ) from None
                os.close(claim)
                confirmed = _read_json(self.path / "owner.json", 4096)
                if confirmed != owner or identity_alive(confirmed):
                    (self.path / "recovery.claim").unlink(missing_ok=True)
                    raise MemoryError(
                        "Lock owner changed during recovery; retry after inspection"
                    )
                stale = self.path.with_name("stale-" + uuid.uuid4().hex)
                os.rename(self.path, stale)
                shutil.rmtree(stale)
                continue
            _atomic_json(
                self.path / "owner.json", {**process_identity(), "token": self.token}
            )
            return self
        raise MemoryError("Unable to acquire checkpoint lock")

    def __exit__(self, *args: Any) -> None:
        owner = _read_json(self.path / "owner.json", 4096)
        if owner.get("token") != self.token:
            raise MemoryError("Research lock ownership changed")
        shutil.rmtree(self.path)


class ResearchMemory:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        for directory in (self.root / ".modelforge", self.root / "experiments"):
            if directory.is_symlink():
                raise MemoryError("Research directories must not be symlinks")
        self.state_path = self.root / ".modelforge" / "research_state.json"
        self.ledger_path = self.root / "experiments" / "ledger.jsonl"

    def verify_manifest(self) -> dict[str, str]:
        """Verify pinned mission documents; hashes detect edits, not authorship."""
        path = self.root / ".modelforge" / "mission_manifest.json"
        if not path.exists():
            raise MemoryError("Required immutable mission manifest is missing")
        manifest = _read_json(path, 65536)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise MemoryError("Invalid mission manifest")
        documents = manifest.get("immutable_documents", manifest.get("documents"))
        if not isinstance(documents, dict) or not 1 <= len(documents) <= 100:
            raise MemoryError("Invalid mission document mapping")
        verified: dict[str, str] = {}
        for relative, digest in documents.items():
            if (
                not isinstance(relative, str)
                or len(relative) > 300
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise MemoryError("Invalid mission document digest")
            item = Path(relative)
            if item.is_absolute() or ".." in item.parts or "\\" in relative:
                raise MemoryError("Mission document must be repository-relative")
            target = self.root / item
            if (
                target.is_symlink()
                or not target.resolve().is_relative_to(self.root)
                or target.stat().st_size > MAX_STATE_BYTES
            ):
                raise MemoryError("Unsafe or oversized mission document")
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != digest:
                raise MemoryError("Immutable mission document changed: " + relative)
            verified[relative] = actual
        return verified

    def git_info(self, state: dict[str, Any]) -> dict[str, Any]:
        def git(*args: str) -> str:
            return _capture_command(["git", "-C", str(self.root), *args])

        current = git("rev-parse", "HEAD")
        try:
            git("merge-base", "--is-ancestor", state["checkpoint_base_commit"], current)
        except MemoryError:
            if git("rev-parse", "--is-shallow-repository") == "true":
                raise MemoryError(
                    "Cannot verify checkpoint ancestry in a shallow checkout; "
                    "fetch the full Git history (git fetch --unshallow) and retry"
                ) from None
            raise
        return {
            "current_commit": current,
            "checkpoint_base_commit": state["checkpoint_base_commit"],
            "working_tree_changes": git("status", "--porcelain"),
            "recent_commits": git("log", "-5", "--format=%H %s"),
        }

    def _load_locked(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self.verify_manifest()
        state = validate_state(_read_json(self.state_path, MAX_STATE_BYTES))
        records: list[dict[str, Any]] = []
        previous: str | None = None
        if self.ledger_path.exists():
            if (
                self.ledger_path.is_symlink()
                or self.ledger_path.stat().st_size > MAX_LEDGER_BYTES
            ):
                raise MemoryError("Ledger exceeds size limit or is a symlink")
            raw = self.ledger_path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise MemoryError(
                    "Incomplete ledger append; preserve and inspect before recovery"
                )
            for line in raw.splitlines():
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise MemoryError("Ledger record must be an object")
                    digest = record.pop("hash")
                    if (
                        record["schema_version"] != 1
                        or record["sequence"] != len(records) + 1
                        or record["previous_hash"] != previous
                        or hashlib.sha256(canonical(record)).hexdigest() != digest
                    ):
                        raise MemoryError("Ledger hash chain mismatch")
                    recorded_state = validate_state(record["state"])
                    if recorded_state["ledger_head"] != previous:
                        raise MemoryError("Ledger state predecessor mismatch")
                    record["hash"] = digest
                    records.append(record)
                    previous = digest
                except (KeyError, TypeError, ValueError, UnicodeError) as exc:
                    raise MemoryError("Invalid ledger record") from exc
                if len(records) > 10000:
                    raise MemoryError("Too many ledger records")
        head = state["ledger_head"]
        hashes = [record["hash"] for record in records]
        if head is not None and head not in hashes:
            raise MemoryError("State references absent ledger evidence")
        start = 0 if head is None else hashes.index(head) + 1
        if head is not None:
            expected = dict(records[start - 1]["state"], ledger_head=head)
            if canonical(expected) != canonical(state):
                raise MemoryError("State differs from its committed ledger snapshot")
        # A fully flushed append is the transaction commit; restore its checkpoint.
        if start < len(records):
            state = dict(records[-1]["state"], ledger_head=records[-1]["hash"])
            self.git_info(state)
            _atomic_json(self.state_path, state)
        self.git_info(state)
        return state, records

    def _checkpoint_locked(
        self,
        state: dict[str, Any],
        event: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        validate_state(state)
        self.git_info(state)
        if len(canonical(event)) > 65536:
            raise MemoryError("Event exceeds size limit")
        record = {
            "schema_version": 1,
            "sequence": len(records) + 1,
            "previous_hash": state["ledger_head"],
            "event": event,
            "state": state,
            "recorded_at_unix": time.time(),
        }
        digest = hashlib.sha256(canonical(record)).hexdigest()
        data = canonical(dict(record, hash=digest)) + b"\n"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        size = self.ledger_path.stat().st_size if self.ledger_path.exists() else 0
        if size + len(data) > MAX_LEDGER_BYTES or len(records) >= 10000:
            raise MemoryError("Ledger capacity reached")
        with self.ledger_path.open("ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(self.ledger_path.parent)
        updated = dict(state, ledger_head=digest)
        _atomic_json(self.state_path, updated)
        return updated

    def checkpoint(
        self, state: dict[str, Any], event: dict[str, Any]
    ) -> dict[str, Any]:
        with RepositoryLock(self.root):
            current, records = self._load_locked()
            if state.get("ledger_head") != current["ledger_head"]:
                raise MemoryError("Stale checkpoint; reload before changing state")
            return self._checkpoint_locked(state, event, records)

    def resume(self) -> dict[str, Any]:
        with RepositoryLock(self.root):
            state, records = self._load_locked()
            interrupted = []
            for task in state["tasks"]:
                if task["status"] == "running" and not identity_alive(task["owner"]):
                    task["status"] = "interrupted"
                    task["owner"] = None
                    interrupted.append(task["id"])
            if interrupted:
                state = self._checkpoint_locked(
                    state,
                    {"kind": "recover_interrupted", "tasks": interrupted},
                    records,
                )
            completed = {t["id"] for t in state["tasks"] if t["status"] == "completed"}
            eligible = sorted(
                (
                    t
                    for t in state["tasks"]
                    if t["status"] in {"queued", "interrupted"}
                    and not t.get("requires_private_model_gate", False)
                    and t["id"] != "private_luna_model"
                    and set(t["dependencies"]) <= completed
                ),
                key=lambda t: (t["priority"], t["id"]),
            )
            return {
                "state": state,
                "git": self.git_info(state),
                "eligible_tasks": eligible,
                "mission_documents": self.verify_manifest(),
                "execution": "report_only",
                "private_model_gate": "closed",
            }


def require_private_model_gate(state: dict[str, Any]) -> None:
    validate_state(state)
    raise MemoryError(
        "Private model creation remains blocked until independent platform/reconstruction verification"
    )
