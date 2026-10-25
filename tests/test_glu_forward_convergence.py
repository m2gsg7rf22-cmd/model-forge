import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / ("examples/isolated_glu/" + name + ".py")
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m = load("forward_convergence")
e = load("forward_convergence_evaluate")


def fixture():
    rng = np.random.default_rng(818)
    g = np.array([[1.0, 0.2]])
    c = np.array([[[0.3, 0.6]]])
    tx = rng.normal(size=(16, 2))
    gx = rng.normal(size=(16, 2))
    ones = np.ones(16)
    return (
        g,
        c,
        tx,
        gx,
        m.core.factor.operator(g[0], c[0], tx, ones),
        m.core.factor.operator(g[0], c[0], gx, ones),
    )


def test_initial_guard_bound_prevents_cumulative_tolerance_drift():
    assert m.guard_ok([0.1 + 5e-11] * 4, [0.1] * 4, [0.1] * 4)
    assert not m.guard_ok([0.1 + 1.5e-10] * 4, [0.1 + 1e-10] * 4, [0.1] * 4)


def test_exact_unit_recovery_stops_after_no_acceptance_and_saves_proposals():
    g, c, tx, gx, ty, gy = fixture()
    new, report, arrays = m.converge(g, c * 0.7, tx, ty, gx, gy, 4, 4)
    np.testing.assert_allclose(new, c, atol=1e-12)
    assert report["stop_reason"] == "zero_accepted" and len(report["sweeps"]) == 2
    assert report["local_rankone_solves"] == 3
    assert arrays["screen_0"].shape == (1, 1, 2) and arrays["sequential_1"].shape == (
        0,
        1,
        2,
    )
    np.testing.assert_array_equal(
        arrays["weighted_target"], ty * arrays["weights"][:, None]
    )


def test_guard_conflict_stops_first_sweep_without_change():
    g, c, tx, gx, ty, gy = fixture()
    new, report, _ = m.converge(g, c * 0.7, tx, ty, gx, gy * 0.7, 4, 4)
    np.testing.assert_array_equal(new, c * 0.7)
    assert len(report["sweeps"]) == 1 and report["accepted_update_count"] == 0
    assert report["local_rankone_solves"] == 2


def test_sweep_budget_and_stable_score_ties():
    g, c, tx, gx, ty, gy = fixture()
    g = np.repeat(g, 2, axis=0)
    c = np.repeat(c, 2, axis=0)
    _, report, _ = m.converge(g, c, tx, ty * 3, gx, gy * 3, 4, 1)
    assert len(report["sweeps"]) == 1 and report["sweeps"][0]["selected_units"] == [
        0,
        1,
    ]
    assert report["local_rankone_solves"] == 4
    with pytest.raises(ValueError):
        m.converge(g, c, tx, ty * 3, gx, gy * 3, 4, 5)


def test_exclusive_failure_marker_and_evaluation_before_teacher(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        m.build({}, tmp_path / "fit", {})
    with pytest.raises(FileExistsError):
        m.build({}, tmp_path / "fit", {})
    (tmp_path / "fit-report.json").write_text(json.dumps({"candidate_built": False}))

    def forbidden():
        raise AssertionError("teacher access")

    monkeypatch.setattr(e, "public_reference", forbidden)
    with pytest.raises(ValueError, match="No development candidate"):
        e.evaluate(tmp_path)
