import importlib.util
import json
from pathlib import Path
import pytest


def load():
    path = (
        Path(__file__).parents[1]
        / "examples/isolated_glu/guard_aware_eval_controller.py"
    )
    assert path.exists(), "Distinct evaluation controller absent"
    spec = importlib.util.spec_from_file_location("eval_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scope", ["process_only_children_unavailable", "none"])
def test_failed_preflight_never_launches_evaluation_worker(
    tmp_path, monkeypatch, scope
):
    m = load()
    for name in ["config", "review"]:
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(m, "verify_source", lambda *a: ({}, "synthetic-head"))
    calls = []

    def engine(*args, **kwargs):
        calls.append(1)
        return {"process_monitor_scope": scope, "memory_samples": 1}

    with pytest.raises(m.ProtocolError):
        m.run_protocol(
            tmp_path,
            tmp_path / "attempt",
            {},
            tmp_path / "config",
            tmp_path / "review",
            engine=engine,
        )
    r = json.loads((tmp_path / "attempt/controller-report.json").read_text())
    assert len(calls) == 1 and not r["worker_started"]
    assert r["quality_criterion_met"] is None and not r["evaluation_completed"]


def test_expired_shared_deadline_prevents_worker_and_reservation(tmp_path, monkeypatch):
    m = load()
    for name in ["config", "review"]:
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(m, "verify_source", lambda *a: ({}, "synthetic-head"))
    calls = []
    values = iter([0.0, 0.0, 181.0])

    def engine(*args, **kwargs):
        calls.append(1)
        return {"process_monitor_scope": "process_and_children", "memory_samples": 2}

    with pytest.raises(m.ProtocolError):
        m.run_protocol(
            tmp_path,
            tmp_path / "attempt",
            {},
            tmp_path / "config",
            tmp_path / "review",
            engine=engine,
            clock=lambda: next(values, 181.0),
        )
    assert len(calls) == 1


def test_unreviewed_evaluation_cannot_reach_preflight_or_reference(
    tmp_path, monkeypatch
):
    m = load()

    def denied(*args):
        raise ValueError("Exact source approval required")

    monkeypatch.setattr(m, "verify_source", denied)

    def forbidden(*args, **kwargs):
        pytest.fail("Unreviewed protocol reached subprocess")

    with pytest.raises(m.ProtocolError):
        m.run_protocol(
            tmp_path,
            tmp_path / "attempt",
            {},
            tmp_path / "config",
            tmp_path / "review",
            engine=forbidden,
        )
