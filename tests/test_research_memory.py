from __future__ import annotations

import json
import hashlib

import psutil
from pathlib import Path
import subprocess
from typing import Any

import pytest

from modelforge import research_memory as rm


@pytest.fixture
def repo(tmp_path: Path) -> tuple[rm.ResearchMemory, dict[str, Any]]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "base",
    )
    state: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_base_commit": git("rev-parse", "HEAD"),
        "ledger_head": None,
        "private_model_gate": {"status": "closed"},
        "tasks": [
            {
                "id": "first",
                "status": "queued",
                "priority": 0,
                "dependencies": [],
                "owner": None,
            },
            {
                "id": "second",
                "status": "queued",
                "priority": 1,
                "dependencies": ["first"],
                "owner": None,
            },
        ],
    }
    memory = rm.ResearchMemory(tmp_path)
    rm._atomic_json(memory.state_path, state)
    mission = memory.root / "mission.md"
    mission.write_text("immutable source")
    rm._atomic_json(
        memory.root / ".modelforge/mission_manifest.json",
        {
            "schema_version": 1,
            "immutable_documents": {
                "mission.md": hashlib.sha256(mission.read_bytes()).hexdigest()
            },
        },
    )
    return memory, state


def test_resume_reports_eligible_without_execution(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    state["tasks"][0]["command"] = "touch unsafe"
    rm._atomic_json(memory.state_path, state)
    report = memory.resume()
    assert [t["id"] for t in report["eligible_tasks"]] == ["first"]
    assert report["execution"] == "report_only"
    assert not (memory.root / "unsafe").exists()


def test_checkpoint_recovers_after_append_before_state_crash(
    repo: tuple[rm.ResearchMemory, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, state = repo
    state["tasks"][0]["status"] = "completed"
    original = rm._atomic_json

    def crash(path: Path, value: Any) -> None:
        if path == memory.state_path:
            raise OSError("simulated crash")
        original(path, value)

    with monkeypatch.context() as context:
        context.setattr(rm, "_atomic_json", crash)
        with pytest.raises(OSError, match="simulated crash"):
            memory.checkpoint(state, {"kind": "completed"})
    result = memory.resume()
    assert result["state"]["tasks"][0]["status"] == "completed"
    assert result["state"]["ledger_head"]
    assert result["eligible_tasks"][0]["id"] == "second"
    assert len(memory.ledger_path.read_text().splitlines()) == 1


def test_tampered_and_torn_ledger_rejected(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    memory.checkpoint(state, {"kind": "test"})
    original = memory.ledger_path.read_bytes()
    memory.ledger_path.write_bytes(original[:-1])
    with pytest.raises(rm.MemoryError, match="Incomplete"):
        memory.resume()
    memory.ledger_path.write_bytes(original.replace(b'"test"', b'"evil"'))
    with pytest.raises(rm.MemoryError, match="Invalid ledger"):
        memory.resume()


def test_state_tamper_and_stale_writer_rejected(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    memory.checkpoint(state, {"kind": "test"})
    with pytest.raises(rm.MemoryError, match="Stale"):
        memory.checkpoint(state, {"kind": "stale"})
    saved = json.loads(memory.state_path.read_text())
    saved["tasks"][0]["status"] = "completed"
    rm._atomic_json(memory.state_path, saved)
    with pytest.raises(rm.MemoryError, match="differs"):
        memory.resume()


def test_live_lock_never_stolen(repo: tuple[rm.ResearchMemory, dict[str, Any]]) -> None:
    memory, _ = repo
    with rm.RepositoryLock(memory.root):
        with pytest.raises(rm.MemoryError, match="live process"):
            memory.resume()
    assert not (memory.root / ".modelforge/research.lock").exists()


def test_pid_reuse_recovers_lock_and_task(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    reused = rm.process_identity()
    reused["create_time"] -= 100
    state["tasks"][0].update(status="running", owner=reused)
    rm._atomic_json(memory.state_path, state)
    lock = memory.root / ".modelforge/research.lock"
    lock.mkdir()
    rm._atomic_json(lock / "owner.json", reused)
    report = memory.resume()
    assert report["state"]["tasks"][0]["status"] == "interrupted"
    assert (
        json.loads(memory.ledger_path.read_text())["event"]["kind"]
        == "recover_interrupted"
    )
    assert memory.resume()["state"]["ledger_head"] == report["state"]["ledger_head"]


def test_unknown_owner_lock_conservatively_blocks(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, _ = repo
    (memory.root / ".modelforge/research.lock").mkdir()
    with pytest.raises(rm.MemoryError, match="manual inspection"):
        memory.resume()


def test_git_base_can_be_ancestor_but_not_unknown(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    subprocess.run(
        [
            "git",
            "-C",
            str(memory.root),
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@x",
            "commit",
            "--allow-empty",
            "-qm",
            "child",
        ],
        check=True,
    )
    report = memory.resume()
    assert report["git"]["current_commit"] != state["checkpoint_base_commit"]
    state["checkpoint_base_commit"] = "0" * 40
    rm._atomic_json(memory.state_path, state)
    with pytest.raises(rm.MemoryError, match="Git verification failed"):
        memory.resume()


@pytest.mark.parametrize(
    "mutation", ["gate", "cycle", "missing_dep", "duplicate", "no_owner"]
)
def test_invalid_state_rejected(
    repo: tuple[rm.ResearchMemory, dict[str, Any]], mutation: str
) -> None:
    _, state = repo
    if mutation == "gate":
        state["private_model_gate"] = {
            "status": "open",
            "independent_proof": {"approved": True},
        }
    elif mutation == "cycle":
        state["tasks"][0]["dependencies"] = ["second"]
    elif mutation == "missing_dep":
        state["tasks"][0]["dependencies"] = ["absent"]
    elif mutation == "duplicate":
        state["tasks"].append(state["tasks"][0])
    else:
        state["tasks"][0]["status"] = "running"
    with pytest.raises(rm.MemoryError):
        rm.validate_state(state)


def test_private_gate_closed_even_with_claimed_proof(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    _, state = repo
    state["private_model_gate"]["independent_proof"] = {"self_claim": "verified"}
    with pytest.raises(rm.MemoryError, match="remains blocked"):
        rm.require_private_model_gate(state)


def test_symlink_input_rejected(repo: tuple[rm.ResearchMemory, dict[str, Any]]) -> None:
    memory, _ = repo
    copy = memory.root / "other.json"
    copy.write_bytes(memory.state_path.read_bytes())
    memory.state_path.unlink()
    try:
        memory.state_path.symlink_to(copy)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    with pytest.raises(rm.MemoryError, match="symlink"):
        memory.resume()


def test_manifest_verified_before_recovery(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    mission = memory.root / "mission.md"
    mission.write_text("immutable source")
    digest = hashlib.sha256(mission.read_bytes()).hexdigest()
    rm._atomic_json(
        memory.root / ".modelforge/mission_manifest.json",
        {"schema_version": 1, "documents": {"mission.md": digest}},
    )
    assert memory.resume()["mission_documents"] == {"mission.md": digest}
    before = memory.state_path.read_bytes()
    mission.write_text("changed source")
    with pytest.raises(rm.MemoryError, match="Immutable mission document changed"):
        memory.resume()
    assert memory.state_path.read_bytes() == before
    assert not memory.ledger_path.exists()


def test_running_live_owner_preserved(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    state["tasks"][0].update(status="running", owner=rm.process_identity())
    rm._atomic_json(memory.state_path, state)
    report = memory.resume()
    assert report["state"]["tasks"][0]["status"] == "running"
    assert report["eligible_tasks"] == []
    assert not memory.ledger_path.exists()


def test_manifest_path_traversal_rejected(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, _ = repo
    rm._atomic_json(
        memory.root / ".modelforge/mission_manifest.json",
        {"schema_version": 1, "documents": {"../external": "0" * 64}},
    )
    with pytest.raises(rm.MemoryError, match="repository-relative"):
        memory.resume()


def test_cli_structured_error_on_git_timeout(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib.util
    import sys

    memory, _ = repo
    path = Path(__file__).parents[1] / "scripts/resume_research.py"
    spec = importlib.util.spec_from_file_location("research_resume_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(path), "--root", str(memory.root)])

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise rm.MemoryError("Git verification exceeded its time budget")

    monkeypatch.setattr(rm, "_capture_command", fail)
    assert module.main() == 1
    assert "error" in json.loads(capsys.readouterr().err)


def test_missing_manifest_fails_closed(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, _ = repo
    (memory.root / ".modelforge/mission_manifest.json").unlink()
    with pytest.raises(rm.MemoryError, match="manifest is missing"):
        memory.resume()


def test_private_task_not_eligible_even_with_completed_prerequisites(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    state["tasks"][0]["status"] = "completed"
    state["tasks"][1].update(id="private_luna_model", requires_private_model_gate=True)
    rm._atomic_json(memory.state_path, state)
    assert memory.resume()["eligible_tasks"] == []
    state["tasks"][1].update(status="running", owner=rm.process_identity())
    with pytest.raises(rm.MemoryError, match="cannot run"):
        rm.validate_state(state)
    state["tasks"][1]["status"] = "completed"
    with pytest.raises(rm.MemoryError, match="cannot run"):
        rm.validate_state(state)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", True, "schema_version"),
        (
            "checkpoint_base_commit",
            1234567890123456789012345678901234567890,
            "prior Git commit",
        ),
        ("ledger_head", "invalid-digest", "ledger_head"),
        ("tasks", {}, "at most 1000"),
    ],
)
def test_malformed_checkpoint_header_rejected(
    repo: tuple[rm.ResearchMemory, dict[str, Any]], field: str, value: Any, error: str
) -> None:
    memory, state = repo
    state[field] = value
    rm._atomic_json(memory.state_path, state)
    with pytest.raises(rm.MemoryError, match=error):
        memory.resume()
    assert not memory.ledger_path.exists()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("status", ["queued"], "task status"),
        ("priority", True, "task priority"),
        ("dependencies", [None], "task dependencies"),
        ("owner", {"pid": 2**63, "create_time": 123}, "process identity"),
        ("requires_private_model_gate", "false", "gate requirement"),
    ],
)
def test_malformed_task_metadata_never_enters_queue(
    repo: tuple[rm.ResearchMemory, dict[str, Any]], field: str, value: Any, error: str
) -> None:
    memory, state = repo
    state["tasks"][0][field] = value
    rm._atomic_json(memory.state_path, state)
    with pytest.raises(rm.MemoryError, match=error):
        memory.resume()
    assert not memory.ledger_path.exists()


def test_nonfinite_event_and_resource_limits_preserve_checkpoint(
    repo: tuple[rm.ResearchMemory, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, state = repo
    before = memory.state_path.read_bytes()
    events: list[tuple[dict[str, Any], str]] = [
        ({"loss": float("nan")}, "finite JSON"),
        ({"payload": "x" * 65537}, "Event exceeds size"),
    ]
    for event, error in events:
        with pytest.raises(rm.MemoryError, match=error):
            memory.checkpoint(state, event)
    with monkeypatch.context() as context:
        context.setattr(rm, "MAX_LEDGER_BYTES", 100)
        with pytest.raises(rm.MemoryError, match="Ledger capacity"):
            memory.checkpoint(state, {"kind": "bounded"})
    assert memory.state_path.read_bytes() == before
    assert not memory.ledger_path.exists()
    state["unbounded_note"] = "x" * rm.MAX_STATE_BYTES
    with pytest.raises(rm.MemoryError, match="State exceeds size"):
        rm.validate_state(state)


def test_deleted_ledger_is_not_silently_recreated(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    memory.checkpoint(state, {"kind": "evidence"})
    before = memory.state_path.read_bytes()
    memory.ledger_path.unlink()
    with pytest.raises(rm.MemoryError, match="absent ledger evidence"):
        memory.resume()
    assert memory.state_path.read_bytes() == before
    assert not memory.ledger_path.exists()


def test_corrupt_json_and_nonobject_ledger_preserved(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, _ = repo
    before = memory.state_path.read_bytes()
    memory.state_path.write_bytes(b'{"incomplete":')
    with pytest.raises(rm.MemoryError, match="Invalid JSON"):
        memory.resume()
    memory.state_path.write_bytes(before)
    memory.ledger_path.parent.mkdir()
    memory.ledger_path.write_bytes(b"[]\n")
    with pytest.raises(rm.MemoryError, match="Invalid ledger record"):
        memory.resume()
    assert memory.ledger_path.read_bytes() == b"[]\n"
    assert memory.state_path.read_bytes() == before


def test_hash_valid_but_inconsistent_ledger_predecessor_rejected(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, state = repo
    memory.checkpoint(state, {"kind": "evidence"})
    record = json.loads(memory.ledger_path.read_text())
    record.pop("hash")
    record["state"]["ledger_head"] = "0" * 64
    record["hash"] = hashlib.sha256(rm.canonical(record)).hexdigest()
    memory.ledger_path.write_bytes(rm.canonical(record) + b"\n")
    with pytest.raises(rm.MemoryError, match="Invalid ledger record"):
        memory.resume()


@pytest.mark.parametrize(
    "manifest,error",
    [
        ({"schema_version": 2}, "Invalid mission manifest"),
        ({"schema_version": 1, "immutable_documents": {}}, "document mapping"),
        (
            {"schema_version": 1, "immutable_documents": {"mission.md": "unverified"}},
            "document digest",
        ),
    ],
)
def test_malformed_manifest_rejected(
    repo: tuple[rm.ResearchMemory, dict[str, Any]], manifest: dict[str, Any], error: str
) -> None:
    memory, _ = repo
    rm._atomic_json(memory.root / ".modelforge/mission_manifest.json", manifest)
    with pytest.raises(rm.MemoryError, match=error):
        memory.resume()


def test_stale_lock_recovery_claim_and_invalid_owner_block(
    repo: tuple[rm.ResearchMemory, dict[str, Any]],
) -> None:
    memory, _ = repo
    lock = memory.root / ".modelforge/research.lock"
    lock.mkdir()
    rm._atomic_json(lock / "owner.json", {"pid": "untrusted"})
    with pytest.raises(rm.MemoryError, match="Invalid lock owner"):
        memory.resume()
    stale_identity = rm.process_identity()
    stale_identity["create_time"] -= 100
    rm._atomic_json(lock / "owner.json", stale_identity)
    (lock / "recovery.claim").write_text("another recovery in progress")
    with pytest.raises(rm.MemoryError, match="Another process is recovering"):
        memory.resume()
    assert (lock / "recovery.claim").read_text() == "another recovery in progress"


@pytest.mark.parametrize("access_denied,expected", [(False, False), (True, True)])
def test_process_disappearance_vs_insufficient_permission(
    monkeypatch: pytest.MonkeyPatch, access_denied: bool, expected: bool
) -> None:
    identity = rm.process_identity()

    def inaccessible_process(pid: int) -> Any:
        if access_denied:
            raise psutil.AccessDenied(pid)
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", inaccessible_process)
    assert rm.identity_alive(identity) is expected


def test_shallow_checkout_reports_missing_history_without_mutation(
    repo: tuple[rm.ResearchMemory, dict[str, Any]], tmp_path: Path
) -> None:
    memory, state = repo
    memory.checkpoint(state, {"kind": "ancestry fixture"})
    subprocess.run(["git", "-C", str(memory.root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(memory.root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "checkpoint",
        ],
        check=True,
    )
    shallow = tmp_path.parent / (tmp_path.name + "-shallow")
    subprocess.run(
        ["git", "clone", "--depth", "1", memory.root.as_uri(), str(shallow)],
        check=True,
        capture_output=True,
    )
    before = (shallow / ".modelforge/research_state.json").read_bytes()
    with pytest.raises(rm.MemoryError, match="shallow.*history"):
        rm.ResearchMemory(shallow).resume()
    assert (shallow / ".modelforge/research_state.json").read_bytes() == before


def test_capture_does_not_wait_for_descendant_inherited_output() -> None:
    """A descendant keeping stdout open must not extend parent cleanup indefinitely."""
    import sys
    import time

    code = (
        "import subprocess,sys;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(2)']);"
        "print('parent-result',flush=True)"
    )
    started = time.monotonic()
    result = rm._capture_command([sys.executable, "-c", code], timeout_seconds=0.5)
    assert result == "parent-result"
    assert time.monotonic() - started < 1.5


def test_capture_timeout_remains_bounded_and_fails_closed() -> None:
    import sys
    import time

    started = time.monotonic()
    with pytest.raises(rm.MemoryError, match="time budget"):
        rm._capture_command(
            [sys.executable, "-c", "import time;time.sleep(10)"], timeout_seconds=0.1
        )
    assert time.monotonic() - started < 2
