import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_case")


def fixture_job(guard_scale=1.0):
    from joint_response_join import JointJob
    from variable_response_view import ResponseBatch

    batches = []
    for index in range(8):
        value = guard_scale if index % 4 == 3 else 1.0
        batches.append(
            ResponseBatch(np.full((2, 2), value), np.full((2, 1), 2 * value))
        )
    return JointJob("job00", 2, 1, 1, 8, iter(batches))


def test_literal_assembly_split_train_only_scales_and_no_alias(m):
    a = m._assemble_fixture(fixture_job(1e6), probes=4, rows=2)
    b = m._assemble_fixture(fixture_job(1.0), probes=4, rows=2)
    assert a.inputs.shape == (16, 2) and a.responses.shape == (16, 1)
    assert a.sx == b.sx == 1.0 and a.sy == b.sy == 2.0
    np.testing.assert_array_equal(
        a.train_indices, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
    )
    np.testing.assert_array_equal(a.guard_indices, [6, 7, 14, 15])
    np.testing.assert_array_equal(a.train_x, b.train_x)
    np.testing.assert_array_equal(a.train_y, b.train_y)
    np.testing.assert_array_equal(a.guard_x, np.full((4, 2), 1e6))
    assert not set(a.train_indices) & set(a.guard_indices)
    for v in (
        a.inputs,
        a.responses,
        a.train_x,
        a.train_y,
        a.guard_x,
        a.guard_y,
        a.train_indices,
        a.guard_indices,
    ):
        with pytest.raises(ValueError):
            v.setflags(write=True)


def test_production_assembly_rejects_fixture_profile_before_iterator(m):
    class Forbidden:
        def __iter__(self):
            raise AssertionError("iterator touched before schema gate")

    from joint_response_join import JointJob

    with pytest.raises(ValueError):
        m.assemble_case(JointJob("job00", 2, 1, 1, 8, Forbidden()))


@pytest.mark.parametrize(
    "fault", ["short", "long", "shape", "zero_scale", "width", "job"]
)
def test_literal_assembly_rejects_partial_extra_schema_and_scale(m, fault):
    from joint_response_join import JointJob
    from variable_response_view import ResponseBatch

    job = fixture_job()
    batches = list(job.batches)
    if fault == "short":
        batches.pop()
    if fault == "long":
        batches.append(batches[-1])
    if fault == "shape":
        batches[0] = ResponseBatch(np.ones((1, 2)), np.ones((1, 1)))
    if fault == "zero_scale":
        batches = [ResponseBatch(np.zeros((2, 2)), np.ones((2, 1))) for _ in range(8)]
    width = 0 if fault == "width" else 1
    name = "teacher" if fault == "job" else "job00"
    with pytest.raises(ValueError):
        m._assemble_fixture(
            JointJob(name, 2, 1, width, 8, iter(batches)), probes=4, rows=2
        )
