import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return tuple(
        importlib.import_module(n)
        for n in (
            "joint_response_join",
            "variable_gate_view",
            "variable_response_manifest",
        )
    )


def public_fixture(modules):
    j, g, r = modules
    jobs = []
    for i in range(6):
        basis = np.zeros((2, 528))
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        jobs.append(g.GateJob(f"job{i:02d}", basis))
    mapping = {
        f"job{i:02d}": f"c{i // 2:02d}-" + ("fp64" if i % 2 == 0 else "fp32") + "-p17"
        for i in range(6)
    }
    bundle = g.GateBundle(
        tuple(jobs), {"mapping": mapping, "commitments": dict(j.RANK_PINS)}
    )
    pairs = []
    for i in range(6):
        case = r.REGISTERED.case_names[i]
        for k in range(34):
            fields = [
                r.ArrayDescriptor(
                    f"{case}-s{k // 17}-x{k % 17:02d}-{field}.npz",
                    field,
                    (2112, 32),
                    ("1" if field == "inputs" else "2") * 64,
                )
                for field in ("inputs", "responses")
            ]
            pairs.append(r.PairDescriptor(f"job{i:02d}", k, *fields))
    return bundle, r.TranscriptManifest(r.ORIGINAL, r.REGISTERED, (), tuple(pairs))


def test_contract_exposes_only_rank_hypotheses_without_basis(modules):
    j, _, _ = modules
    bundle, manifest = public_fixture(modules)
    c = j.compile_contract(bundle, manifest)
    assert c.width_hypotheses == (1, 1, 1, 1, 1, 1)
    assert c.paired_input_commitments == 102
    assert set(c.__dataclass_fields__) == {
        "width_hypotheses",
        "paired_input_commitments",
    }


@pytest.mark.parametrize(
    "kind",
    [
        "mapping",
        "order",
        "paired_rank",
        "paired_input",
        "shape",
        "name",
        "index",
        "count",
        "profile",
    ],
)
def test_join_rejects_before_numeric_jobs(modules, kind):
    j, g, _ = modules
    b, m = public_fixture(modules)
    if kind == "mapping":
        b = replace(b, provenance={"mapping": {"job00": "teacher-width-1"}})
    if kind == "order":
        b = replace(b, jobs=b.jobs[::-1])
    if kind == "paired_rank":
        basis = np.zeros((4, 528))
        basis[:4, :4] = np.eye(4)
        b = replace(b, jobs=(b.jobs[0], g.GateJob("job01", basis), *b.jobs[2:]))
    pairs = list(m.pairs)
    if kind in ("paired_input", "shape", "name"):
        values = {
            "paired_input": {"digest": "3" * 64},
            "shape": {"shape": (2111, 32)},
            "name": {"name": "context.npz"},
        }[kind]
        pairs[34] = replace(pairs[34], inputs=replace(pairs[34].inputs, **values))
    if kind == "index":
        pairs[0] = replace(pairs[0], index=1)
    if kind == "count":
        pairs.pop()
    if kind == "profile":
        m = replace(m, committed_collection="/wrong")
    m = replace(m, pairs=tuple(pairs))
    with pytest.raises(ValueError):
        j.compile_contract(b, m)


def test_loader_validates_both_sources_before_returning_numeric_jobs(
    modules, monkeypatch, tmp_path
):
    j, _, _ = modules
    b, m = public_fixture(modules)
    events = []
    monkeypatch.setattr(
        j, "load_manifest", lambda repository: events.append("metadata") or m
    )
    monkeypatch.setattr(
        j, "load_bundle", lambda *args: events.append("all_rank_archives") or b
    )
    from variable_response_view import ResponseJob

    class View:
        def __init__(self):
            self.jobs = tuple(
                ResponseJob(f"job{i:02d}", 32, 32, 34, iter(())) for i in range(6)
            )

        def report(self):
            return {"status": "ready"}

        def finish(self):
            return {"status": "completed"}

    monkeypatch.setattr(
        j, "open_view", lambda *args: events.append("all_response_archives") or View()
    )
    result = j.load_join(tmp_path, tmp_path, tmp_path, tmp_path)
    assert events == ["metadata", "all_rank_archives", "all_response_archives"]
    assert len(result.jobs) == 6
    for job in result.jobs:
        assert set(job.__dataclass_fields__) == {
            "job_id",
            "n",
            "outputs",
            "width_hypothesis",
            "batch_count",
            "batches",
        }
        assert job.width_hypothesis == 1
    assert result.finish()["status"] == "completed"


def test_incompatible_pair_abstains_before_response_open(
    modules, monkeypatch, tmp_path
):
    j, _, _ = modules
    b, m = public_fixture(modules)
    pairs = list(m.pairs)
    pairs[34] = replace(pairs[34], inputs=replace(pairs[34].inputs, digest="3" * 64))
    monkeypatch.setattr(j, "load_manifest", lambda r: replace(m, pairs=tuple(pairs)))
    monkeypatch.setattr(j, "load_bundle", lambda *args: b)

    def forbidden(*args):
        raise AssertionError("response opened before join gate")

    monkeypatch.setattr(j, "open_view", forbidden)
    with pytest.raises(ValueError):
        j.load_join(tmp_path, tmp_path, tmp_path, tmp_path)
