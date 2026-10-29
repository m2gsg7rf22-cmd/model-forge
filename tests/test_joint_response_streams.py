import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_streams")


class LiteralStream:
    def __init__(self, seed):
        self.seed = seed
        self.calls = 0

    def state(self):
        return dict(
            bit_generator="PCG64",
            state={"state": self.calls, "inc": 1},
            has_uint32=0,
            uinteger=0,
        )

    def normal(self, shape):
        self.calls += 1
        return np.full(shape, float(self.calls))

    def indices(self, high, shape):
        self.calls += 1
        return np.zeros(shape, dtype=np.int64)


def test_shared_paired_seed_domains_and_injected_literal_schedule(m):
    assert m.stream_seed("job00", "initial") == m.stream_seed("job01", "initial")
    assert m.stream_seed("job00", "batches") == m.stream_seed("job01", "batches")
    assert (
        len(
            {
                m.stream_seed(f"job{i:02d}", role)
                for i in (0, 2, 4)
                for role in ("initial", "batches")
            }
        )
        == 6
    )
    calls = []

    def factory(seed):
        calls.append(seed)
        return LiteralStream(seed)

    a = m._prepare_fixture("job00", 2, 1, 1, 4, factory)
    b = m._prepare_fixture("job01", 2, 1, 1, 4, factory)
    assert calls[:2] == calls[2:]
    np.testing.assert_array_equal(a.initial.g, np.full((1, 2), 1 / np.sqrt(2)))
    np.testing.assert_array_equal(a.initial.u, np.full((1, 2), 2 / np.sqrt(2)))
    np.testing.assert_array_equal(a.initial.d, np.full((1, 1), 3.0))
    np.testing.assert_array_equal(a.initial.g, b.initial.g)
    np.testing.assert_array_equal(a.schedule, b.schedule)
    assert a.schedule.shape == (4000, 256) and a.schedule.dtype == np.int64
    assert json.loads(a.initial_state_before)["state"]["state"] == 0
    assert json.loads(a.initial_state_after)["state"]["state"] == 3
    assert json.loads(a.batch_state_after)["state"]["state"] == 1
    with pytest.raises(ValueError):
        a.schedule.setflags(write=True)


@pytest.mark.parametrize(
    "job,role", [("teacher", "initial"), ("job06", "initial"), ("job00", "rescue")]
)
def test_stream_domains_are_exact(m, job, role):
    with pytest.raises(ValueError):
        m.stream_seed(job, role)


def test_production_profile_gate_precedes_factory(m):
    from joint_response_join import JointJob

    def forbidden(seed):
        raise AssertionError("RNG before profile gate")

    with pytest.raises(ValueError):
        m.prepare_inputs(JointJob("job00", 2, 1, 1, 8, iter(())), None, forbidden)


def test_injected_bad_state_or_schedule_rejects(m):
    class BadState(LiteralStream):
        def state(self):
            return {"teacher": "forbidden"}

    with pytest.raises(ValueError):
        m._prepare_fixture("job00", 2, 1, 1, 4, BadState)

    class BadSchedule(LiteralStream):
        def indices(self, high, shape):
            return np.full(shape, high, dtype=np.int64)

    with pytest.raises(ValueError):
        m._prepare_fixture("job00", 2, 1, 1, 4, BadSchedule)


def test_factory_must_return_two_distinct_stream_objects(m):
    shared = LiteralStream(0)
    with pytest.raises(ValueError):
        m._prepare_fixture("job00", 2, 1, 1, 4, lambda seed: shared)
    assert shared.calls == 0
