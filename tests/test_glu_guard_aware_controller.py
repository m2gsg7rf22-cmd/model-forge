import importlib.util
import json
from pathlib import Path
import pytest


def load():
    path = Path(__file__).parents[1] / "examples/isolated_glu/guard_aware_controller.py"
    assert path.exists(), "Pre-data monitored controller absent"
    spec = importlib.util.spec_from_file_location("step_control", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("scope", ["process_only_children_unavailable", "none"])
def test_bad_preflight_never_starts_data_worker(tmp_path, monkeypatch, scope):
    m = load()
    (tmp_path / "prereg").write_text("{}")
    (tmp_path / "review").write_text("{}")
    monkeypatch.setattr(
        m, "verify_source", lambda *a: ({"timeout_seconds": 180}, "synthetic-head")
    )
    calls = []

    def engine(*args, **kwargs):
        calls.append((args, kwargs))
        return {"process_monitor_scope": scope, "memory_samples": 3}

    with pytest.raises(m.ProtocolError):
        m.run_protocol(
            tmp_path,
            tmp_path / "attempt",
            {},
            tmp_path / "prereg",
            tmp_path / "review",
            engine=engine,
        )
    assert len(calls) == 1
    report = json.loads((tmp_path / "attempt/controller-report.json").read_text())
    assert report["status"] == "failed" and not report["worker_started"]
    assert not report["evaluation_eligible"]


def test_expired_budget_after_preflight_never_starts_worker(tmp_path, monkeypatch):
    m = load()
    (tmp_path / "prereg").write_text("{}")
    (tmp_path / "review").write_text("{}")
    monkeypatch.setattr(
        m, "verify_source", lambda *a: ({"timeout_seconds": 180}, "synthetic-head")
    )
    calls = []
    times = iter([0.0, 0.0, 181.0])

    def engine(*args, **kwargs):
        calls.append(1)
        return {"process_monitor_scope": "process_and_children", "memory_samples": 2}

    with pytest.raises(m.ProtocolError):
        m.run_protocol(
            tmp_path,
            tmp_path / "attempt",
            {},
            tmp_path / "prereg",
            tmp_path / "review",
            engine=engine,
            clock=lambda: next(times),
        )
    assert len(calls) == 1
    assert not json.loads((tmp_path / "attempt/controller-report.json").read_text())[
        "worker_started"
    ]


def test_source_review_gate_precedes_preflight_and_data(tmp_path, monkeypatch):
    m = load()
    (tmp_path / "prereg").write_text("{}")
    (tmp_path / "review").write_text("{}")

    def denied(*args):
        raise ValueError("Unreviewed sources")

    monkeypatch.setattr(m, "verify_source", denied)

    def forbidden(*args, **kwargs):
        pytest.fail("No source approval, no subprocess")

    with pytest.raises(m.ProtocolError):
        m.run_protocol(
            tmp_path,
            tmp_path / "attempt",
            {},
            tmp_path / "prereg",
            tmp_path / "review",
            engine=forbidden,
        )
    assert not json.loads((tmp_path / "attempt/controller-report.json").read_text())[
        "worker_started"
    ]
