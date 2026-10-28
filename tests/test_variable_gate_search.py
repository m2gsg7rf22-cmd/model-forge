import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("variable_gate_search")


class Sink:
    def __init__(self):
        self.records = {}

    def write(self, name, arrays):
        assert name not in self.records
        assert all(
            a.dtype == np.float64 and a.size and np.isfinite(a).all()
            for a in arrays.values()
        )
        self.records[name] = {k: v.copy() for k, v in arrays.items()}


def job():
    from variable_gate_view import GateJob

    return GateJob("job00", np.eye(6)[[0, 1, 3, 4]])


def test_quartic_fixture_identity(m):
    # Existing limitation as a regression assertion, not a research experiment.
    b = np.eye(3)[:2]
    t = 0.01
    x = np.array([[np.cos(t), np.sin(t)]])
    assert m.losses(x, b)[0] == pytest.approx(np.sin(t) ** 4, rel=1e-12)


def test_merge_sign_duplicates_threshold_and_no_truncation(m):
    x = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    selected, trace = m.merge_candidates(
        np.empty((0, 3)), x, np.array([0.0, 0.0, 1e-8, 1e-8 + 1e-12])
    )
    assert len(selected) == 2
    assert sorted(trace["decision"].tolist()) == [0.0, 1.0, 2.0, 2.0]
    assert sorted(trace["order"].tolist()) == list(range(4))


@pytest.mark.parametrize(
    "mode,status,count",
    [
        ("zero", "zero_candidates", 0),
        ("exact", "candidate_target", 2),
        ("overshoot", "overshoot", 3),
    ],
)
def test_initial_stops_without_rescue(m, mode, status, count):
    x = {
        "zero": np.array([[0.0, 0, 1.0]]),
        "exact": np.eye(3)[:2],
        "overshoot": np.array([[1.0, 0, 0], [0, 1.0, 0], [2**-0.5, 2**-0.5, 0]]),
    }[mode]
    sink = Sink()

    def init(*args):
        return x.copy(), x.copy()

    def rescue(*args):
        raise AssertionError("rescue must not run")

    r = m._search_fixture(
        job(),
        np.random.default_rng(1),
        np.random.default_rng(2),
        sink,
        initializer=init,
        rescuer=rescue,
        refiner=lambda x, b, steps: x.copy(),
    )
    assert r.status == status and len(r.candidates) == count
    assert r.report["certificate"] is False and r.report["target_hypothesis"] == 2
    assert r.report["rounds"] == []
    assert sink.records["job00-final"]["count"][0] == count
    assert {"job00-rng-before", "job00-rng-after"} <= sink.records.keys()


def test_geometry_failure_is_scientific(m):
    sink = Sink()

    def rescue(*args):
        return None, {"component_rank": 0, "residual_dimension": 4}

    r = m._search_fixture(
        job(),
        np.random.default_rng(1),
        np.random.default_rng(2),
        sink,
        initializer=lambda *args: (np.eye(3)[:1], np.eye(3)[:1]),
        rescuer=rescue,
        refiner=lambda x, b, steps: x.copy(),
    )
    assert r.status == "unexpected_component_geometry" and len(r.candidates) == 1


def test_four_round_exhaustion_preserves_trace(m):
    sink = Sink()
    calls = []

    def rescue(known, basis, rng, sink, prefix):
        calls.append(prefix)
        return np.eye(3)[:1], {"component_rank": 2, "residual_dimension": 2}

    r = m._search_fixture(
        job(),
        np.random.default_rng(1),
        np.random.default_rng(2),
        sink,
        initializer=lambda *args: (np.eye(3)[:1], np.eye(3)[:1]),
        rescuer=rescue,
        refiner=lambda x, b, steps: x.copy(),
    )
    assert r.status == "rounds_exhausted" and len(calls) == 4
    assert len(r.report["rounds"]) == 4
    assert all(f"job00-r{i}-merge" in sink.records for i in range(4))


def test_rescue_overshoot_is_not_clipped(m):
    sink = Sink()

    def rescue(*args):
        return np.array([[0.0, 1.0, 0], [2**-0.5, 2**-0.5, 0]]), {
            "component_rank": 2,
            "residual_dimension": 2,
        }

    r = m._search_fixture(
        job(),
        np.random.default_rng(1),
        np.random.default_rng(2),
        sink,
        initializer=lambda *args: (np.eye(3)[:1], np.eye(3)[:1]),
        rescuer=rescue,
        refiner=lambda x, b, steps: x.copy(),
    )
    assert (
        r.status == "overshoot"
        and len(r.candidates) == 3
        and len(r.report["rounds"]) == 1
    )


def test_loss_is_recomputed_from_directions(m):
    sink = Sink()
    r = m._search_fixture(
        job(),
        np.random.default_rng(1),
        np.random.default_rng(2),
        sink,
        initializer=lambda *args: (np.eye(3)[:1], np.eye(3)[:1]),
        rescuer=lambda *args: (None, {"component_rank": 0, "residual_dimension": 4}),
        refiner=lambda x, b, steps: np.array([0.0, 0, 1.0]),
    )
    assert r.status == "zero_candidates"
    assert sink.records["job00-initial-refine"]["losses"][0] == 1


def test_nonfinite_and_sink_failure_propagate(m):
    with pytest.raises(FloatingPointError):
        m._search_fixture(
            job(),
            np.random.default_rng(1),
            np.random.default_rng(2),
            Sink(),
            initializer=lambda *args: (np.ones((1, 3)), np.full((1, 3), np.nan)),
            refiner=lambda x, b, steps: x,
        )

    class Broken:
        def write(self, *args):
            raise OSError("fixture disk failure")

    rng = np.random.default_rng(1)
    before = str(rng.bit_generator.state)
    with pytest.raises(OSError):
        m.search_job(job(), rng, np.random.default_rng(2), Broken())
    assert str(rng.bit_generator.state) == before


def test_rng_must_be_distinct_and_supported(m):
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        m.search_job(job(), rng, rng, Sink())
    with pytest.raises(ValueError):
        m.search_job(job(), rng, np.random.Generator(np.random.MT19937(2)), Sink())


def test_small_adam_and_refine_are_actual_numerics(m):
    b = np.eye(3)[:2]
    x = np.array([[2**-0.5, 2**-0.5]])
    y = m._adam(x.copy(), b, steps=2)
    assert y.shape == x.shape and np.allclose(np.linalg.norm(y, axis=1), 1)
    from modelforge.glu32 import refine

    z = m._refine(x[0], b, 12)
    np.testing.assert_array_equal(z, refine(x[0], b, steps=12)[0])
    assert m.losses(z[None, :], b)[0] <= m.losses(x, b)[0]


def test_actual_rescue_uses_injected_stream_and_records_raw_eigen(m):
    a = Sink()
    b = Sink()
    j = job()
    known = np.eye(3)[:1]
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    x, g = m._rescue(known, j.basis, rng1, a, "fixture")
    y, h = m._rescue(known, j.basis, rng2, b, "fixture")
    assert g["component_rank"] == 2 and g["residual_dimension"] == 2
    assert x.shape == (512, 3) and np.allclose(np.linalg.norm(x, axis=1), 1)
    np.testing.assert_array_equal(x, y)
    assert rng1.bit_generator.state == rng2.bit_generator.state
    assert a.records["fixture-mixtures"]["raw"].shape == (256, 2)
    assert a.records["fixture-eigen"]["vectors"].shape == (256, 3, 3)
    for name in a.records:
        for k, v in a.records[name].items():
            np.testing.assert_array_equal(v, b.records[name][k])


def test_rng_state_trace_is_exact_json(m):
    import json

    initial = np.random.default_rng(1)
    rescue = np.random.default_rng(2)
    a = m._states(initial, rescue)
    for name, rng in [("initial", initial), ("rescue", rescue)]:
        n = int(a[name + "_length"][0])
        decoded = json.loads(bytes(a[name][:n].astype(np.uint8)).decode("ascii"))
        assert decoded == rng.bit_generator.state
        assert np.all(a[name][n:] == 0)


def test_trace_cannot_mutate_kernel_storage(m):
    x = np.ones((1, 2), dtype=np.float64)

    class Mutation:
        def write(self, name, arrays):
            with pytest.raises(ValueError):
                arrays["x"].setflags(write=True)

    m._emit(Mutation(), "literal", x=x)
    np.testing.assert_array_equal(x, np.ones((1, 2)))


def test_adam_matches_independent_quartic_gradient(m):
    # For this literal span, the ambient loss is y**4. Its sphere gradient
    # supplies an independent reference for both fixed Adam steps.
    x = np.array([[0.6, 0.8]], dtype=np.float64)
    expected = x.copy()
    moment = np.zeros_like(x)
    variance = np.zeros_like(x)
    for t in (1, 2):
        gradient = np.array([[0.0, 4 * expected[0, 1] ** 3]])
        gradient -= np.sum(gradient * expected, axis=1, keepdims=True) * expected
        moment = 0.9 * moment + 0.1 * gradient
        variance = 0.999 * variance + 0.001 * gradient * gradient
        expected -= (
            0.02 * (moment / (1 - 0.9**t)) / (np.sqrt(variance / (1 - 0.999**t)) + 1e-8)
        )
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(
        m._adam(x, np.eye(3)[:2], steps=2), expected, rtol=0, atol=2e-15
    )


def test_refine_failure_preserves_pretrace_and_completed_output(m):
    sink = Sink()
    calls = 0

    def fail_second(x, b, steps):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture refine failure")
        return x.copy()

    with pytest.raises(RuntimeError, match="fixture refine"):
        m._search_fixture(
            job(),
            np.random.default_rng(1),
            np.random.default_rng(2),
            sink,
            initializer=lambda *args: (np.eye(3)[:2], np.eye(3)[:2]),
            refiner=fail_second,
        )
    assert "job00-initial-pre-refine" in sink.records
    a = sink.records["job00-initial-refine-chunk000"]
    assert a["count"][0] == 1 and a["attempted"][0] == 2 and a["complete"][0] == 0
    np.testing.assert_array_equal(a["refined"][0], np.eye(3)[0])


def test_eigen_failure_preserves_raw_chunk_and_completed_output(m, monkeypatch):
    sink = Sink()
    calls = 0
    original = np.linalg.eigh

    def fail_second(x):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture eigen failure")
        return original(x)

    monkeypatch.setattr(np.linalg, "eigh", fail_second)
    with pytest.raises(RuntimeError, match="fixture eigen"):
        m._rescue(np.eye(3)[:1], job().basis, np.random.default_rng(7), sink, "fixture")
    assert sink.records["fixture-mix-pre000"]["raw"].shape == (8, 2)
    a = sink.records["fixture-eigen-chunk000"]
    assert a["count"][0] == 1 and a["attempted"][0] == 2 and a["complete"][0] == 0


def test_partial_trace_failure_does_not_mask_original_refine_error(m):
    class Broken(Sink):
        def write(self, name, arrays):
            if "refine-chunk" in name:
                raise OSError("fixture partial flush failure")
            super().write(name, arrays)

    def bad(*args):
        raise RuntimeError("original numerical failure")

    with pytest.raises(RuntimeError, match="original numerical failure"):
        m._search_fixture(
            job(),
            np.random.default_rng(1),
            np.random.default_rng(2),
            Broken(),
            initializer=lambda *args: (np.eye(3)[:1], np.eye(3)[:1]),
            refiner=bad,
        )


def test_fixed_production_path_on_tiny_literal_basis(m):
    from variable_gate_view import GateJob

    sink = Sink()
    initial = np.random.default_rng(11)
    rescue = np.random.default_rng(12)
    before = rescue.bit_generator.state
    r = m.search_job(GateJob("job00", np.eye(3)[:2]), initial, rescue, sink)
    assert r.status == "candidate_target" and r.candidates.shape == (1, 2)
    assert r.report["coverage_certified"] is False and r.report["rounds"] == []
    assert rescue.bit_generator.state == before
    chunks = [sink.records[f"job00-initial-refine-chunk{i:03d}"] for i in range(32)]
    assert all(a["complete"][0] == 1 and a["count"][0] == 8 for a in chunks)
    np.testing.assert_array_equal(
        np.concatenate([a["refined"] for a in chunks]),
        sink.records["job00-initial-refine"]["refined"],
    )
    np.testing.assert_allclose(
        m.losses(r.candidates, np.eye(3)[:2]),
        sink.records["job00-final"]["losses"],
        rtol=0,
        atol=0,
    )
