import importlib.util
import json
from pathlib import Path

import pytest


def load(monkeypatch):
    root = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(root))
    p = root / "scaling_domain_controller.py"
    assert p.exists(), "Full dimension controller absent"
    spec = importlib.util.spec_from_file_location("scaling_domain_controller", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize(
    "failure", ["source", "scope", "transport", "worker_scope", "deadline"]
)
def test_gates_block_data_or_completion_without_real_worker(
    tmp_path, monkeypatch, failure
):
    m = load(monkeypatch)
    calls = []
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "review.json").write_text("{}")

    def verify(*a):
        if failure == "source":
            raise ValueError("injected source rejection")
        return {"policy": m.POLICY}, "synthetic-not-actual"

    monkeypatch.setattr(m, "verify_source", verify)

    def engine(*args, **kwargs):
        calls.append((args, kwargs))
        index = len(calls)
        assert kwargs["memory_limit_mb"] == 512 and kwargs["timeout_seconds"] <= 180
        assert kwargs["output_limit_bytes"] == 2048
        if index == 1:
            return dict(
                process_monitor_scope="bad"
                if failure == "scope"
                else "process_and_children",
                memory_samples=1,
            )
        if index == 2:
            return dict(
                status="failed" if failure == "transport" else "passed",
                process_monitor_scope="process_and_children",
                memory_samples=1,
            )
        return dict(
            status="completed",
            process_monitor_scope="process_only_children_unavailable",
            memory_samples=1,
        )

    clock_values = iter([0.0, 181.0])
    clock = (
        (lambda: next(clock_values, 181.0)) if failure == "deadline" else lambda: 0.0
    )
    with pytest.raises(m.ProtocolError):
        m.run_protocol(
            tmp_path,
            tmp_path / "attempt",
            tmp_path / "config.json",
            tmp_path / "review.json",
            engine=engine,
            clock=clock,
        )
    report = json.loads((tmp_path / "attempt/controller-report.json").read_text())
    assert report["status"] == "failed" and report["diagnostic_completed"] is False
    assert (
        len(calls)
        == {"source": 0, "scope": 1, "transport": 2, "worker_scope": 3, "deadline": 0}[
            failure
        ]
    )
    assert report["worker_started"] == (failure == "worker_scope")


def test_review_must_bind_complete_protocol_not_component(tmp_path, monkeypatch):
    m = load(monkeypatch)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "review.json").write_text(
        json.dumps(
            {
                "approved": True,
                "pure_math_core_approved": True,
                "artifact_transport_component_approved": True,
            }
        )
    )
    with pytest.raises(ValueError, match="ull"):
        m.verify_source(tmp_path, tmp_path / "config.json", tmp_path / "review.json")


def test_rejected_resource_receipt_is_preserved(tmp_path, monkeypatch):
    m = load(monkeypatch)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "review.json").write_text("{}")
    monkeypatch.setattr(
        m, "verify_source", lambda *a: ({"policy": m.POLICY}, "synthetic")
    )
    receipt = dict(
        process_monitor_scope="process_only_children_unavailable",
        memory_samples=2,
        process_peak_rss_bytes=12345,
    )
    with pytest.raises(m.ProtocolError):
        m.run_protocol(
            tmp_path,
            tmp_path / "attempt",
            tmp_path / "config.json",
            tmp_path / "review.json",
            engine=lambda *a, **k: receipt,
            clock=lambda: 0.0,
        )
    r = json.loads((tmp_path / "attempt/controller-report.json").read_text())
    assert r["scope-engine_receipt"] == receipt


def test_postworker_source_change_cannot_be_completion(tmp_path, monkeypatch):
    m = load(monkeypatch)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "review.json").write_text("{}")
    heads = iter(["synthetic-before", "synthetic-after"])
    monkeypatch.setattr(
        m, "verify_source", lambda *a: ({"policy": m.POLICY}, next(heads))
    )
    monkeypatch.setattr(m, "validate_worker", lambda *a: {"synthetic_only": True})
    with pytest.raises(m.ProtocolError, match="commit changed"):
        m.run_protocol(
            tmp_path,
            tmp_path / "attempt",
            tmp_path / "config.json",
            tmp_path / "review.json",
            engine=lambda *a, **k: dict(
                status="passed",
                process_monitor_scope="process_and_children",
                memory_samples=1,
            ),
            clock=lambda: 0.0,
        )
    r = json.loads((tmp_path / "attempt/controller-report.json").read_text())
    assert r["worker_started"] and not r["diagnostic_completed"]


@pytest.mark.parametrize("change", ["none", "dependency", "reviewed_document", "dirty"])
def test_actual_source_binding_with_synthetic_files(tmp_path, monkeypatch, change):
    m = load(monkeypatch)
    for name in m.DEPENDENCIES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic source, never executable")
    doc = tmp_path / "protocol.md"
    doc.write_text("synthetic protocol")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                policy=m.POLICY,
                inventory_plan=m.inventory_plan(),
                frozen_dependencies={
                    name: m.sha(tmp_path / name) for name in m.DEPENDENCIES
                },
            )
        )
    )
    signed = {name: m.sha(tmp_path / name) for name in m.DEPENDENCIES}
    signed["config.json"] = m.sha(config)
    signed["protocol.md"] = m.sha(doc)
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            dict(dimension_domain_full_protocol_approved=True, source_sha256=signed)
        )
    )

    def git(args, **kwargs):
        if args[1:] == ["status", "--porcelain"]:
            return "dirty" if change == "dirty" else ""
        if args[1:] == ["branch", "--show-current"]:
            return "discovery-lab"
        return "synthetic-head"

    monkeypatch.setattr(m.subprocess, "check_output", git)
    if change == "dependency":
        (tmp_path / next(iter(m.DEPENDENCIES))).write_text("changed")
    elif change == "reviewed_document":
        doc.write_text("changed")
    if change == "none":
        assert m.verify_source(tmp_path, config, review)[1] == "synthetic-head"
    else:
        with pytest.raises(ValueError):
            m.verify_source(tmp_path, config, review)


def test_complete_expected_names_match_fixed_shape_inventory(monkeypatch):
    m = load(monkeypatch)
    names = m.expected_names()
    assert len(names) == 4704
    assert max(map(len, names)) <= 84
    assert "p3-t1-fp32-spectrum-richardson.npz" in names
