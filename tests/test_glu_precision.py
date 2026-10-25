"""Numerical controls for a public precision diagnostic, not sealed recovery."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "precision", Path(__file__).parents[1] / "examples/isolated_glu/precision.py"
)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_analytic_hessian_and_layout():
    rng = np.random.default_rng(91)
    g, u, d = (
        rng.normal(size=(4, 3)) * 0.2,
        rng.normal(size=(4, 3)) * 0.2,
        rng.normal(size=(2, 4)) * 0.2,
    )
    x = rng.normal(size=(2, 3))
    exact = m.analytic_hessian(g, u, d, x)

    def q(z):
        return m.forward(g, u, d, z, "fp64")

    approx, calls = m.curvature(q, x, 0.001)
    assert calls == 2 * 6 * 4
    assert exact.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(approx, exact, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(approx, approx.swapaxes(-1, -2), rtol=0, atol=0)


def test_richardson_removes_quartic_truncation():
    x = np.array([[0.2], [0.7]])

    def q(z):
        return z**4

    a, _ = m.curvature(q, x, 0.1)
    b, _ = m.curvature(q, x, 0.2)
    exact = (12 * x * x).reshape(2, 1, 1, 1)
    assert np.linalg.norm(a - exact) > 0.01
    np.testing.assert_allclose(m.richardson(a, b), exact, atol=1e-12, rtol=1e-12)


def test_rounding_channels_and_compute_reference():
    rng = np.random.default_rng(92)
    g, u, d = rng.normal(size=(4, 3)), rng.normal(size=(4, 3)), rng.normal(size=(2, 4))
    x = rng.normal(size=(5, 3))
    baseline = m.forward(g, u, d, x, "fp64")
    np.testing.assert_array_equal(
        m.forward(g, u, d, x, "fp64_return_fp32"),
        baseline.astype(np.float32).astype(np.float64),
    )
    np.testing.assert_array_equal(
        m.forward(g, u, d, x, "fp64_return_fp16"),
        baseline.astype(np.float16).astype(np.float64),
    )
    weights = m.reference_weights(g, u, d, "fp32_compute")
    for a, b in zip(weights, (g, u, d)):
        np.testing.assert_array_equal(a, b.astype(np.float32).astype(np.float64))
    z = x.astype(np.float32) @ g.astype(np.float32).T
    expected = (
        z / (1 + np.exp(-z)) * (x.astype(np.float32) @ u.astype(np.float32).T)
    ) @ d.astype(np.float32).T
    np.testing.assert_array_equal(
        m.forward(g, u, d, x, "fp32_compute"), expected.astype(np.float64)
    )


def test_subspace_metrics_controls():
    exact = np.zeros((1, 2, 2, 2))
    exact[0, 0, 0, 0] = 2
    exact[0, 1, 1, 1] = 1
    same = m.metrics(exact, exact, 2)
    assert same["hessian_relative_error"] == 0
    assert same["subspace_rms_sine"] < 1e-7
    changed = exact.copy()
    changed[0, 1, 1, 1] = 0
    changed[0, 1, 0, 1] = 1
    changed[0, 1, 1, 0] = 1
    assert m.metrics(changed, exact, 2)["subspace_rms_sine"] > 0.5


def test_reference_rank_must_be_separated():
    exact = np.zeros((1, 3, 2, 2))
    exact[0, 0, 0, 0] = 2
    exact[0, 1, 1, 1] = 1
    exact[0, 2, 0, 1] = exact[0, 2, 1, 0] = 0.02
    with pytest.raises(ValueError, match="not spectrally separated"):
        m.metrics(exact, exact, 2)
