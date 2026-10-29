import importlib
from pathlib import Path

import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_runtime")


def test_literal_transport_typed_roundtrip_and_hash_tamper(m, tmp_path):
    root = tmp_path / "transport"
    receipt = m.transport(root)
    m.validate_transport(root, receipt, require_memory=False)
    assert set(p.name for p in root.iterdir()) == {
        "schedule.npz",
        "state.npz",
        "manifest.json",
    }
    p = root / "schedule.npz"
    with p.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError):
        m.validate_transport(root, receipt, require_memory=False)


def test_scalar_memory_exercise_excludes_runtime_allowance_without_allocating(m):
    from joint_response_inventory import memory_plan

    plan = m.allocation_plan()
    assert "runtime_allocator_and_monitor_allowance" not in plan
    assert sum(plan.values()) == sum(memory_plan().values()) - 96 * 1024**2
    assert max(plan.values()) < 512 * 1024**2


def test_layout_accepts_only_registered_partial_output_and_rejects_unbounded_files(
    m, tmp_path
):
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "controller-report.json").write_text("{}")
    case = root / "cases/job00"
    case.mkdir(parents=True)
    (case / "progress.jsonl").write_bytes(b"")
    (case / "checkpoint-metrics.jsonl").write_bytes(b"{}\n")
    v = m.layout(root)
    assert v["retained_bytes"] == 5 and v["combined_retained_bytes"] == 671428088
    (case / "context.json").write_text("{}")
    with pytest.raises(ValueError):
        m.layout(root)
    (case / "context.json").unlink()
    with (case / "progress.jsonl").open("wb") as stream:
        stream.truncate(4000 * 512 + 1)
    with pytest.raises(ValueError):
        m.layout(root)


def test_layout_rejects_symlink_and_noncanonical_case(m, tmp_path):
    root = tmp_path / "attempt"
    (root / "cases").mkdir(parents=True)
    (root / "cases/teacher").mkdir()
    with pytest.raises(ValueError):
        m.layout(root)
    (root / "cases/teacher").rmdir()
    (root / "controller-report.json").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        m.layout(root)


def test_transport_without_memory_exercise_cannot_pass_real_resource_gate(m, tmp_path):
    receipt = m.transport(tmp_path / "transport")
    with pytest.raises(ValueError):
        m.validate_transport(tmp_path / "transport", receipt, require_memory=True)
