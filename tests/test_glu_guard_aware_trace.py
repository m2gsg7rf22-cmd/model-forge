import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest


def load():
    p = Path(__file__).parents[1] / "examples/isolated_glu/guard_aware_trace.py"
    assert p.exists(), "Complete cone trace is absent"
    s = importlib.util.spec_from_file_location("guard_aware_trace", p)
    assert s and s.loader
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def problem():
    return (
        np.array([100.0, 0.0]),
        np.array([-6.0, -2.0]),
        np.array([[2.0, 1.0], [1.0, 2.0]]),
        np.eye(2),
    )


def test_all_subset_solutions_and_rejected_duals_are_saved_without_math_change():
    m = load()
    g, d, k, a = problem()
    old, info = m.frozen.cone_direction(g, d, k, a)
    z, trace = m.traced_direction(g, d, k, a, proposal_id="screen-0")
    np.testing.assert_array_equal(z, old)
    assert trace["result"]["selected_mask"] == info["selected_mask"] == 3
    assert len(trace["subsets"]) == 4 and trace["linear_solves_attempted"] == 4
    for v in trace["subsets"]:
        primal = np.asarray(v["solution"])
        mu = np.asarray(v["normalized_duals"])
        kn = k / trace["normalization"]["objective_scale"]
        dn = d / trace["normalization"]["objective_scale"]
        np.testing.assert_allclose(2 * kn @ primal + dn + a.T @ mu, 0, atol=1e-13)
        np.testing.assert_allclose(a[v["active"]] @ primal, 0, atol=1e-13)
        assert v["solution_finite"] is True
    assert any(v["status"] == "numerical_kkt_rejected" for v in trace["subsets"])
    json.dumps(trace, allow_nan=False)


def test_fresh_synthetic_results_match_frozen_solver_bitwise_and_clip():
    m = load()
    rng = np.random.default_rng(9250)
    for n in [1, 3, 5]:
        for scale in [100.0, 0.001]:
            g = rng.normal(size=n) * scale
            h = rng.normal(size=(n, n))
            k = h.T @ h + np.eye(n)
            d = rng.normal(size=n)
            a = rng.normal(size=(min(n, 3), n))
            before = [v.copy() for v in [g, d, k, a]]
            old, info = m.frozen.cone_direction(g, d, k, a)
            new, trace = m.traced_direction(g, d, k, a)
            np.testing.assert_array_equal(old, new)
            assert trace["result"]["score"] == info["score"]
            assert trace["result"]["selected_mask"] == info["selected_mask"]
            for actual, copy in zip([g, d, k, a], before):
                np.testing.assert_array_equal(actual, copy)


def test_global_geometry_failure_keeps_current_mask_and_previous_solutions():
    m = load()
    g, d, k, _ = problem()
    a = np.array([[1.0, 0.0], [-1.0, 1e-11]])
    with pytest.raises(m.TraceError) as caught:
        m.traced_direction(g, d, k, a)
    trace = caught.value.trace
    assert (
        trace["terminal_status"] == "failed" and trace["failure_kind"] == "computation"
    )
    assert [v["mask"] for v in trace["subsets"]] == [0, 1, 2, 3]
    assert trace["subsets"][-1]["status"] == "aborted"
    assert trace["subsets"][-1]["solution"] is None
    failed = trace["subsets"][-1]
    assert len(failed["singular_values"]) == 2
    assert failed["exact_original_rank"] == 2
    assert failed["singular_ratio"] < m.GEOMETRY_TOL
    assert failed["reason"] == "exact_independent_near_dependency"
    assert trace["linear_solves_attempted"] == 3
    json.dumps(trace, allow_nan=False)


def test_nonfinite_subset_solutions_keep_bits_without_json_nan(monkeypatch):
    m = load()
    g, d, k, a = problem()
    monkeypatch.setattr(
        np.linalg, "solve", lambda matrix, rhs: np.full(rhs.shape, np.nan)
    )
    with pytest.raises(m.TraceError) as caught:
        m.traced_direction(g, d, k, a)
    trace = caught.value.trace
    assert len(trace["subsets"]) == 4
    for record in trace["subsets"]:
        assert record["nonfinite_solution_encoding"] == {
            "dtype": "<f8",
            "layout": "z_then_mu",
            "z_count": 2,
            "mu_count": 2,
        }
        assert record["solution_finite"] is False and record["solution"] is None
        assert np.isnan(
            np.frombuffer(
                bytes.fromhex(record["nonfinite_solution_ieee754_hex"]),
                dtype=record["nonfinite_solution_encoding"]["dtype"],
            )
        ).any()
    json.dumps(trace, allow_nan=False)


def test_sink_failure_stops_without_attempting_remaining_subsets():
    m = load()
    g, d, k, a = problem()
    events = []

    def sink(event):
        if event["kind"] == "subset_completed":
            raise OSError("synthetic write failure")
        events.append(event)

    with pytest.raises(m.TraceError) as caught:
        m.traced_direction(g, d, k, a, event_sink=sink, proposal_id="screen-1")
    t = caught.value.trace
    assert t["failure_kind"] == "trace_write" and t["linear_solves_attempted"] == 1
    assert len(t["subsets"]) == 1 and t["events_delivered"] == len(events)
    assert [v["ordinal"] for v in events] == list(range(len(events)))


def test_invalid_input_trace_does_not_serialize_nonfinite_values():
    m = load()
    g, d, k, a = problem()
    with pytest.raises(m.TraceError) as caught:
        m.traced_direction(g * np.nan, d, k, a)
    assert caught.value.trace["subsets"] == []
    json.dumps(caught.value.trace, allow_nan=False)
