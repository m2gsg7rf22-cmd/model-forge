import importlib.util
import json
from pathlib import Path
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


m = load("hessian_polish")
e = load("hessian_polish_evaluate")


def test_one_fixed_gate_synthetic_recovery():
    rng = np.random.default_rng(881)
    g = rng.normal(size=(1, 3))
    c = rng.normal(size=(1, 2, 1)) * rng.normal(size=(1, 1, 3))
    x = rng.normal(size=(7, 3))
    y = m.factor.operator(g[0], c[0], x)
    new, report, initial, final, proposals = m.polish(g, c * 0.7, y, x, 8)
    np.testing.assert_allclose(new, c, atol=1e-12)
    np.testing.assert_allclose(proposals["screen"][0], c[0], atol=1e-12)
    assert proposals["sequential"].shape == (1, 2, 3)
    assert report["accepted_units"] == [0] and np.linalg.norm(
        final
    ) < 1e-12 * np.linalg.norm(initial)


def test_zero_residual_does_not_change_or_reserve_new_candidate():
    g = np.array([[1.0, 0.2]])
    c = np.array([[[0.3, 0.6]]])
    x = np.array([[1.0, 2.0], [-1.0, 0.1]])
    y = m.factor.operator(g[0], c[0], x)
    new, report, _, _, _ = m.polish(g, c, y, x, 8)
    np.testing.assert_array_equal(new, c)
    assert report["selected_units"] == report["accepted_units"] == []


def test_identical_initial_scores_use_stable_unit_index():
    g = np.array([[1.0, 0.2], [1.0, 0.2]])
    c = np.array([[[0.3, 0.6]], [[0.3, 0.6]]])
    x = np.array([[1.0, 2.0], [-1.0, 0.1], [0.2, 0.3]])
    y = 3 * m.factor.operator(g[0], c[0], x)
    _, report, _, _, _ = m.polish(g, c, y, x, 1)
    assert report["selected_units"] == [0] and report["accepted_units"] == [0]


def test_fit_attempt_is_exclusive_on_bad_input(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        m.build({}, output, {})
    assert (output / "polish-attempt.json").exists()
    with pytest.raises(FileExistsError):
        m.build({}, output, {})


def test_evaluator_rejects_unchanged_candidate_before_teacher(tmp_path, monkeypatch):
    (tmp_path / "fit-report.json").write_text(json.dumps({"candidate_built": False}))

    def forbidden():
        raise AssertionError("teacher access")

    monkeypatch.setattr(e, "public_reference", forbidden)
    with pytest.raises(ValueError, match="No development candidate"):
        e.evaluate(tmp_path)
