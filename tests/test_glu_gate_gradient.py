import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "gate_gradient",
    Path(__file__).parents[1] / "examples/isolated_glu/gate_gradient.py",
)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rng = np.random.default_rng(9211)
    return (
        rng,
        rng.normal(size=(2, 3)) * 0.4,
        rng.normal(size=(2, 2, 3)) * 0.3,
        rng.normal(size=(9, 3)),
        rng.normal(size=(9, 2)),
        np.exp(rng.normal(size=9) * 0.3),
    )


def reference(g, c, x):
    t = x @ g.T
    s = t / (1 + np.exp(-t))
    v = np.einsum("nd,wod->nwo", x, c)
    return np.einsum("nw,nwo->no", s, v)


def loss(g, c, x, y, w):
    return float(np.sum(((reference(g, c, x) - y) * w[:, None]) ** 2))


def test_forward_matches_independent_vectorized_sigmoid():
    _, g, c, x, _, _ = fixture()
    np.testing.assert_allclose(
        m.forward(g, c, x), reference(g, c, x), atol=1e-14, rtol=1e-14
    )


def test_gate_coordinate_gradients_match_independent_finite_differences():
    _, g, c, x, y, w = fixture()
    value, dg, _ = m.loss_gradient(g, c, x, y, w)
    expected = np.empty_like(g)
    h = 1e-6
    assert value == pytest.approx(loss(g, c, x, y, w), rel=1e-14)
    for index in np.ndindex(g.shape):
        step = np.zeros_like(g)
        step[index] = h
        expected[index] = (loss(g + step, c, x, y, w) - loss(g - step, c, x, y, w)) / (
            2 * h
        )
    np.testing.assert_allclose(dg, expected, atol=2e-8, rtol=2e-6)


def test_coupling_coordinate_gradients_match_independent_finite_differences():
    _, g, c, x, y, w = fixture()
    _, _, dc = m.loss_gradient(g, c, x, y, w)
    expected = np.empty_like(c)
    h = 1e-6
    for index in np.ndindex(c.shape):
        step = np.zeros_like(c)
        step[index] = h
        expected[index] = (loss(g, c + step, x, y, w) - loss(g, c - step, x, y, w)) / (
            2 * h
        )
    np.testing.assert_allclose(dc, expected, atol=2e-8, rtol=2e-6)


def test_joint_direction_derivative_and_squared_weight_scaling():
    rng, g, c, x, y, w = fixture()
    value, dg, dc = m.loss_gradient(g, c, x, y, w)
    gg = rng.normal(size=g.shape)
    cc = rng.normal(size=c.shape)
    h = 1e-6
    finite = (
        loss(g + h * gg, c + h * cc, x, y, w) - loss(g - h * gg, c - h * cc, x, y, w)
    ) / (2 * h)
    assert float(np.sum(dg * gg) + np.sum(dc * cc)) == pytest.approx(
        finite, abs=2e-8, rel=2e-6
    )
    scaled, sg, sc = m.loss_gradient(g, c, x, y, 2 * w)
    assert scaled == pytest.approx(4 * value)
    np.testing.assert_allclose(sg, 4 * dg, atol=1e-13)
    np.testing.assert_allclose(sc, 4 * dc, atol=1e-13)


def test_zero_residual_and_no_input_mutation():
    _, g, c, x, _, w = fixture()
    y = m.forward(g, c, x)
    copies = [v.copy() for v in [g, c, x, y, w]]
    value, dg, dc = m.loss_gradient(g, c, x, y, w)
    assert value == 0.0
    np.testing.assert_array_equal(dg, np.zeros_like(g))
    np.testing.assert_array_equal(dc, np.zeros_like(c))
    for actual, expected in zip([g, c, x, y, w], copies):
        np.testing.assert_array_equal(actual, expected)


def test_finite_shape_precision_and_positive_weight_guards():
    _, g, c, x, y, w = fixture()
    for bad in [w * 0, -w, w * np.nan, w.astype(np.float32), w[:-1]]:
        with pytest.raises(ValueError):
            m.loss_gradient(g, c, x, y, bad)
    with pytest.raises(ValueError):
        m.loss_gradient(g, c, x, y.astype(np.float32), w)
    with pytest.raises(ValueError):
        m.forward(g * np.nan, c, x)
    with pytest.raises(ValueError):
        m.forward(g, c, np.ones((4097, 3)))


def test_gate_jvp_matches_independent_directional_difference():
    rng, g, c, x, _, w = fixture()
    direction = rng.normal(size=3)
    step = np.zeros_like(g)
    step[0] = direction
    h = 1e-6
    expected = (
        (reference(g + h * step, c, x) - reference(g - h * step, c, x))
        / (2 * h)
        * w[:, None]
    )
    np.testing.assert_allclose(
        m.gate_jvp(g[0], c[0], x, w, direction), expected, atol=2e-8, rtol=2e-6
    )


def test_gate_gram_matches_independent_finite_difference_design():
    _, g, c, x, _, w = fixture()
    h = 1e-6
    columns = []
    for index in range(3):
        step = np.zeros_like(g)
        step[0, index] = h
        derivative = (
            (reference(g + step, c, x) - reference(g - step, c, x))
            / (2 * h)
            * w[:, None]
        )
        columns.append(derivative.reshape(-1))
    design = np.column_stack(columns)
    actual = m.gate_gram(g[0], c[0], x, w)
    np.testing.assert_allclose(actual, design.T @ design, atol=2e-8, rtol=2e-6)
    np.testing.assert_array_equal(actual, actual.T)
    assert np.linalg.eigvalsh(actual)[0] >= -1e-12
