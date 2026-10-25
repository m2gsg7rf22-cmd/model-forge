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


m = load("guarded_forward")
o = load("forward_observe")
e = load("guarded_forward_evaluate")


def test_scale_balancing_weights_targets_once():
    y = np.arange(1, 25, dtype=np.float64).reshape(12, 2)
    w, weighted, norms = m.prepare_weights(y, 3)
    np.testing.assert_array_equal(weighted, y * w[:, None])
    for i in range(4):
        assert np.linalg.norm(weighted[i * 3 : i * 3 + 3]) == pytest.approx(1.0)
    np.testing.assert_allclose(w, np.repeat(1 / np.array(norms), 3))
    with pytest.raises(ValueError):
        m.prepare_weights(y * 0, 3)


def fixture():
    rng = np.random.default_rng(718)
    g = np.array([[1.0, 0.2]])
    c = np.array([[[0.3, 0.6]]])
    x = rng.normal(size=(16, 2))
    gx = rng.normal(size=(16, 2))
    ones = np.ones(16)
    return (
        g,
        c,
        x,
        gx,
        m.factor.operator(g[0], c[0], x, ones),
        m.factor.operator(g[0], c[0], gx, ones),
    )


def test_train_improvement_rejected_when_guard_worsens():
    g, c, x, gx, y, gy = fixture()
    updated, report, arrays = m.polish(g, c * 0.7, x, y, gx, gy * 0.7, 4, 8)
    np.testing.assert_array_equal(updated, c * 0.7)
    assert report["selected_units"] == [0] and report["accepted_units"] == []
    assert (
        report["decisions"][0]["training_decreased"]
        and not report["decisions"][0]["guard_passed"]
    )
    assert arrays["sequential"].shape == (1, 1, 2)


def test_shared_truth_synthetic_recovery_and_saved_proposals():
    g, c, x, gx, y, gy = fixture()
    updated, report, arrays = m.polish(g, c * 0.7, x, y, gx, gy, 4, 8)
    np.testing.assert_allclose(updated, c, atol=1e-12)
    assert report["accepted_units"] == [0] and report["local_rankone_solves"] == 2
    np.testing.assert_allclose(arrays["screen"][0], c[0], atol=1e-12)
    np.testing.assert_array_equal(
        arrays["weighted_target"], y * arrays["weights"][:, None]
    )


def test_attempts_exclusive_and_collector_fails_before_teacher(tmp_path):
    with pytest.raises(ValueError):
        o.collect(tmp_path / "obs", {})
    with pytest.raises(FileExistsError):
        o.collect(tmp_path / "obs", {})
    with pytest.raises(ValueError):
        m.build({}, tmp_path / "absent", tmp_path / "fit", {}, "invalid")
    with pytest.raises(FileExistsError):
        m.build({}, tmp_path / "absent", tmp_path / "fit", {}, "invalid")


def test_evaluator_rejects_nochange_before_teacher(tmp_path, monkeypatch):
    (tmp_path / "fit-report.json").write_text(json.dumps({"candidate_built": False}))

    def forbidden():
        raise AssertionError("teacher access")

    monkeypatch.setattr(e, "public_reference", forbidden)
    with pytest.raises(ValueError, match="No development candidate"):
        e.evaluate(tmp_path)
