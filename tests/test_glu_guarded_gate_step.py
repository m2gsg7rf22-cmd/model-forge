import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest


def load(name):
    path = Path(__file__).parents[1] / "examples/isolated_glu" / (name + ".py")
    assert path.exists(), "New bounded gate-step implementation is absent"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_newton_half_factor_trust_and_zero_branch():
    m = load("guarded_gate_step")
    g = np.array([3.0, 4.0])
    k = np.diag([2.0, 8.0])
    d = np.array([4.0, 16.0])
    delta, record = m.direction(g, d, k)
    np.testing.assert_allclose(record["unclipped"], [-1.0, -1.0])
    np.testing.assert_allclose(delta, np.array([-1.0, -1.0]) * (0.25 / np.sqrt(2)))
    assert record["score"] == pytest.approx(-(d @ delta + delta @ k @ delta))
    zero, info = m.direction(g, np.zeros(2), k)
    np.testing.assert_array_equal(zero, np.zeros(2))
    assert info["score"] == 0 and info["zero_direction"] is True
    small, _ = m.direction(g, d * 1e-4, k)
    np.testing.assert_allclose(small, [-1e-4, -1e-4])


def test_singular_indefinite_nonfinite_and_zero_gate_rejected():
    m = load("guarded_gate_step")
    for k in [
        np.diag([0.0, 1.0]),
        np.diag([-1.0, 1.0]),
        np.diag([1e-12, 1.0]),
        np.eye(2) * np.nan,
    ]:
        with pytest.raises(ValueError):
            m.direction(np.ones(2), np.ones(2), k)
    with pytest.raises(ValueError):
        m.direction(np.zeros(2), np.ones(2), np.eye(2))


def test_every_group_relative_and_absolute_current_and_initial_bounds():
    m = load("guarded_gate_step")
    rel = [1.0, 2.0, 3.0, 4.0]
    absolute = [10.0, 20.0, 30.0, 40.0]
    assert m.guard_ok(rel, absolute, rel, absolute, rel, absolute)
    for index in range(4):
        worse = rel.copy()
        worse[index] += 1e-6
        assert not m.guard_ok(worse, absolute, rel, absolute, rel, absolute)
        worse_abs = absolute.copy()
        worse_abs[index] += 1e-6
        assert not m.guard_ok(rel, worse_abs, rel, absolute, rel, absolute)
    assert not m.guard_ok(rel, absolute, [2.0] * 4, [100.0] * 4, rel, absolute)
    assert not m.guard_ok(rel, absolute, rel, absolute, [0.0] * 4, absolute)
    assert not m.guard_ok(rel, absolute, rel, [100.0] * 4, rel, [1.0] * 4)
    with pytest.raises(ValueError):
        m.guard_ok([np.nan] * 4, absolute, rel, absolute, rel, absolute)


def fixture(m):
    rng = np.random.default_rng(9217)
    g = np.array([[0.4, -0.2]])
    c = np.array([[[0.3, 0.5], [0.6, 1.0]]])
    x = rng.normal(size=(64, 2))
    y = m.gradient.forward(g * 1.02, c, x)
    return g, c, x, y


def test_real_synthetic_step_keeps_c_and_first_accepted_trial():
    m = load("guarded_gate_step")
    g, c, x, y = fixture(m)
    originals = [a.copy() for a in [g, c, x, y]]
    new, report, arrays = m.step(g, c, x, y, x, y, 16)
    assert report["accepted_update_count"] == 1
    assert report["local_linear_solves"] == 2
    assert report["final_cost"] < report["initial_cost"]
    assert len(report["trials"]) == 1
    assert report["trials"][0]["alpha"] == 1.0
    np.testing.assert_allclose(new, g * 1.02, rtol=2e-4, atol=2e-5)
    for actual, original in zip([g, c, x, y], originals):
        np.testing.assert_array_equal(actual, original)
    assert arrays["trial_gates"].shape == (1, 2)
    np.testing.assert_allclose(
        arrays["final_train_residual"],
        (y - m.gradient.forward(new, c, x)) * arrays["weights"][:, None],
        atol=1e-14,
    )


def test_guard_veto_exhausts_fixed_eight_trials_without_update():
    m = load("guarded_gate_step")
    g, c, x, y = fixture(m)
    guard_y = m.gradient.forward(g, c, x)
    new, report, arrays = m.step(g, c, x, y, x, guard_y, 16)
    assert report["accepted_update_count"] == 0
    assert [v["alpha"] for v in report["trials"]] == [2.0**-i for i in range(8)]
    assert all(
        v["training_decreased"] and not v["guard_passed"] for v in report["trials"]
    )
    np.testing.assert_array_equal(new, g)
    assert len(arrays["trial_gates"]) == 8


def test_attempt_reserved_before_invalid_fit_and_no_retry(tmp_path):
    m = load("guarded_gate_step")
    out = tmp_path / "fit"
    with pytest.raises(ValueError):
        m.build({}, out, {})
    assert (
        json.loads((out / "gate-step-attempt.json").read_text())["automatic_retry"]
        is False
    )
    with pytest.raises(FileExistsError):
        m.build({}, out, {})


def test_evaluator_no_candidate_before_teacher_and_reserves_attempt(tmp_path):
    m = load("guarded_gate_step_evaluate")
    (tmp_path / "fit-report.json").write_text(json.dumps({"candidate_built": False}))

    def forbidden():
        pytest.fail("Teacher must not be accessed without changed committed candidate")

    m.public_reference = forbidden
    with pytest.raises(ValueError, match="No development candidate"):
        m.evaluate(tmp_path)
    assert (tmp_path / "evaluation-attempt.json").exists()
    with pytest.raises(FileExistsError):
        m.evaluate(tmp_path)


def test_stable_ties_top_eight_single_pass_and_zero_score_no_trials():
    m = load("guarded_gate_step")
    g, c, x, _ = fixture(m)
    g = np.repeat(g, 10, axis=0)
    c = np.repeat(c / 10, 10, axis=0)
    y = m.gradient.forward(g * 1.02, c, x)
    _, report, _ = m.step(g, c, x, y, x, y, 16)
    assert report["screen_order"] == list(range(10))
    assert report["selected_units"] == list(range(8))
    assert report["local_linear_solves"] == 18
    assert len(report["sequential"]) == 8
    assert report["trial_count"] <= 64
    exact = m.gradient.forward(g, c, x)
    unchanged, zero, _ = m.step(g, c, x, exact, x, exact, 16)
    assert zero["selected_units"] == [] and zero["trials"] == []
    assert zero["accepted_update_count"] == 0
    np.testing.assert_array_equal(unchanged, g)
