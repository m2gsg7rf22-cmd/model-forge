import dataclasses
import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return tuple(
        importlib.import_module(n)
        for n in (
            "joint_response_audit_jobs",
            "joint_response_audit_storage",
            "joint_response_audit_case",
        )
    )


def setup(modules, tmp_path):
    from test_joint_response_audit_storage import build
    from joint_response_join import JointJob
    from variable_response_view import ResponseBatch

    _, storage, _ = modules
    jobs = []
    readers = []
    for i in range(6):
        reader, data, pins, root = build(storage, tmp_path / f"input{i}")
        data = dataclasses.replace(data, job_id=f"job{i:02d}")
        (root / "case.json").write_text(json.dumps(data.metadata()))
        pins["case.json"] = hashlib.sha256(
            (root / "case.json").read_bytes()
        ).hexdigest()
        reader = storage._fixture_reader(
            root, pins, width=1, n=1, outputs=1, probes=4, rows=2, prediction_rows=8
        )
        batches = [
            ResponseBatch(data.x[k : k + 2], data.y[k : k + 2]) for k in range(0, 16, 2)
        ]
        jobs.append(JointJob(f"job{i:02d}", 1, 1, 1, 8, iter(batches)))
        readers.append(reader)
    return jobs, readers


def run(modules, jobs, readers, root, **kwargs):
    m, _, case = modules
    return m._run_jobs(
        jobs,
        readers,
        root,
        collect=lambda j: case._collect_fixture(j, probes=4, rows=2),
        prediction_rows=8,
        check=kwargs.pop("check", lambda: None),
        progress=kwargs.pop("progress", lambda v: None),
        **kwargs,
    )


def test_all_six_cases_all102_checkpoints_and_partial_hooks(modules, tmp_path):
    jobs, readers = setup(modules, tmp_path)
    updates = []
    calls = []
    a = run(
        modules,
        jobs,
        readers,
        tmp_path / "output",
        progress=updates.append,
        check=lambda: calls.append(True),
    )
    assert a["status"] == "completed" and a["completed_case_visits"] == 6
    assert (
        a["numeric_consistency_agreed"] and not a["scientific_reconstruction_success"]
    )
    assert len(updates) == 6 and len(calls) > 6
    assert sum(len(r["checkpoints"]) for r in a["cases"]) == 102
    assert all(p.done for p in readers)
    assert all(
        (tmp_path / "output" / f"job{i:02d}/result.json").is_file() for i in range(6)
    )
    assert a["private_gate"] == "closed" and not a["certificate"]


def test_case_failure_preserves_prefix_and_later_not_started(
    modules, tmp_path, monkeypatch
):
    jobs, readers = setup(modules, tmp_path)
    original = readers[2].checkpoint

    def interrupted(step):
        if step == 512:
            raise OSError("literal injected interruption")
        return original(step)

    monkeypatch.setattr(readers[2], "checkpoint", interrupted)
    a = run(modules, jobs, readers, tmp_path / "output")
    assert a["status"] == "failed" and a["completed_case_visits"] == 2
    assert [r["status"] for r in a["cases"]] == [
        "agreed",
        "agreed",
        "failed",
        "not_started",
        "not_started",
        "not_started",
    ]
    partial = json.loads((tmp_path / "output/job02/result.json").read_text())
    assert (
        partial["committed_checkpoint_records"] == 2 and partial["status"] == "failed"
    )
    assert not a["numeric_consistency_agreed"]


def test_canonical_six_case_order_before_any_reads(modules, tmp_path):
    jobs, readers = setup(modules, tmp_path)
    with pytest.raises(ValueError):
        run(modules, jobs[::-1], readers, tmp_path / "output")
    assert not (tmp_path / "output").exists() and all(r.reads == 0 for r in readers)


def test_no_resume_or_retry_of_existing_output(modules, tmp_path):
    jobs, readers = setup(modules, tmp_path)
    (tmp_path / "output").mkdir()
    with pytest.raises(FileExistsError):
        run(modules, jobs, readers, tmp_path / "output")
    assert all(r.reads == 0 for r in readers)
