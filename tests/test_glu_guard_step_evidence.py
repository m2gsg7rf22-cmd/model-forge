import importlib.util
from pathlib import Path
import json
import sys
import numpy as np
import pytest
from modelforge.execution import run_json


def source():
    p = Path(__file__).parents[1] / "examples/isolated_glu/guard_step_evidence.py"
    assert p.exists(), "Bounded proposal evidence store is absent"
    return p


def load():
    s = importlib.util.spec_from_file_location("guard_step_evidence", source())
    assert s and s.loader
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_proposal_journal_hashes_failed_attempt_and_no_retry(tmp_path):
    m = load()
    store = m.TraceStore(tmp_path / "study")
    g = np.array([100.0, 0.0])
    d = np.array([-6.0, -2.0])
    k = np.array([[2.0, 1.0], [1.0, 2.0]])
    store.run_proposal("screen-0", g, d, k, np.eye(2))
    with pytest.raises(m.traced.TraceError):
        store.run_proposal("screen-1", g, d, k, np.array([[1.0, 0.0], [-1.0, 1e-11]]))
    receipt = store.finalize({"trial_gates": np.empty((0, 2))}, {"status": "success"})
    assert receipt["status"] == "failed" and receipt["proposal_count"] == 2
    manifest = json.loads((tmp_path / "study/manifest.json").read_text())
    assert manifest["proposals"][0]["journal"]["successful"] is True
    assert manifest["proposals"][1]["terminal_status"] == "failed"
    with pytest.raises(ValueError):
        store.run_proposal("screen-2", g, d, k, np.eye(2))
    with pytest.raises(FileExistsError):
        m.TraceStore(tmp_path / "study")


def test_truncated_journal_is_reported_without_repair(tmp_path):
    m = load()
    p = tmp_path / "journal.jsonl"
    p.write_text(
        json.dumps({"ordinal": 0, "proposal_id": "p", "kind": "proposal_started"})
        + "\n"
        + "{bad"
    )
    original = p.read_bytes()
    info = m.inspect_journal(p, "p")
    assert info["truncated_tail"] and not info["successful"] and info["records"] == 1
    assert p.read_bytes() == original


def test_worst72_by256_raw_solutions_and64trials_fit_compact_receipt(tmp_path):
    p = source()
    code = f"""import importlib.util,json
from pathlib import Path
s=importlib.util.spec_from_file_location('evidence',{str(p)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps(m.synthetic_transport(Path('study'))))
"""
    receipt = run_json(
        [sys.executable, "-c", code],
        {},
        cwd=tmp_path,
        memory_limit_mb=512,
        timeout_seconds=60,
        output_limit_bytes=2048,
    )
    assert (
        receipt["proposal_count"] == 72
        and receipt["subset_count"] == 18432
        and receipt["trial_count"] == 64
    )
    manifest = json.loads((tmp_path / "study/manifest.json").read_text())
    assert sum(v["trace_bytes"] for v in manifest["proposals"]) > 2_000_000
    first = json.loads((tmp_path / "study/proposals/control-0/trace.json").read_text())
    assert len(first["subsets"]) == 256 and all(
        v["solution"] is not None and len(v["normalized_duals"]) == 8
        for v in first["subsets"]
    )
    with np.load(
        tmp_path / "study/evidence/numeric-evidence.npz", allow_pickle=False
    ) as arrays:
        assert arrays["trial_gates"].shape == (64, 32)
        assert arrays["physical_train_y"].shape == (1024, 32)
        assert arrays["weighted_target"].shape == (1024, 32)
    assert len(json.dumps(receipt)) < 2048


def test_problem_write_failure_blocks_attempt(tmp_path, monkeypatch):
    m = load()
    store = m.TraceStore(tmp_path / "study")

    def fail(*args, **kwargs):
        raise OSError("injected problem write failure")

    monkeypatch.setattr(m.np, "savez", fail)
    with pytest.raises(OSError):
        store.run_proposal("p", np.ones(2), -np.ones(2), np.eye(2), np.eye(2))
    assert store.failed
    assert json.loads((store.root / "manifest.json").read_text())["failed"]
    with pytest.raises(ValueError):
        store.run_proposal("next", np.ones(2), -np.ones(2), np.eye(2), np.eye(2))


@pytest.mark.parametrize("failure_stage", ["transport", "receipt"])
def test_final_artifact_write_failure_blocks_attempt(
    tmp_path, monkeypatch, failure_stage
):
    m = load()
    store = m.TraceStore(tmp_path / "study")
    if failure_stage == "transport":

        def fail(*args, **kwargs):
            raise OSError("injected numeric evidence failure")

        monkeypatch.setattr(m.transport, "save_evidence", fail)
    else:
        original = Path.write_text

        def fail(path, *args, **kwargs):
            if path.name == "receipt.json":
                raise OSError("injected receipt failure")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError):
        store.finalize({"trial_gates": np.empty((0, 2))}, {})
    assert store.closed and store.failed
    assert json.loads((store.root / "manifest.json").read_text())["failed"]
    with pytest.raises(ValueError):
        store.finalize({"trial_gates": np.empty((0, 2))}, {})


@pytest.mark.parametrize("proposal_id", [None, 1, "../bad"])
def test_invalid_proposal_identity_fails_closed(tmp_path, proposal_id):
    m = load()
    store = m.TraceStore(tmp_path / "study")
    with pytest.raises(ValueError):
        store.run_proposal(proposal_id, np.ones(2), -np.ones(2), np.eye(2), np.eye(2))
    assert store.failed
    assert json.loads((store.root / "manifest.json").read_text())["failed"]
