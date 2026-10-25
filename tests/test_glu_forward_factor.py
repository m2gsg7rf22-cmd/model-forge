import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "forward_factor",
    Path(__file__).parents[1] / "examples/isolated_glu/forward_factor.py",
)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rng = np.random.default_rng(6011)
    return (
        rng,
        rng.normal(size=3),
        rng.normal(size=(13, 3)),
        np.exp(rng.normal(size=13)),
    )


def test_weighted_operator_matches_independent_silu():
    rng, g, x, w = fixture()
    c = rng.normal(size=(2, 3))
    t = x @ g
    expected = (w * t / (1 + np.exp(-t)))[:, None] * (x @ c.T)
    np.testing.assert_allclose(m.operator(g, c, x, w), expected, atol=1e-13)


def test_adjoint_inner_product_and_explicit_gram():
    rng, g, x, w = fixture()
    c = rng.normal(size=(2, 3))
    y = rng.normal(size=(13, 2))
    t = x @ g
    phi = (w * t / (1 + np.exp(-t)))[:, None] * x
    np.testing.assert_allclose(m.adjoint(g, y, x, w), y.T @ phi, atol=1e-12)
    np.testing.assert_allclose(m.gram(g, x, w), phi.T @ phi, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(m.operator(g, c, x, w) * y),
        np.sum(c * m.adjoint(g, y, x, w)),
        atol=1e-12,
    )


def test_exact_rankone_recovery_and_zero_partial():
    rng, g, x, w = fixture()
    c = rng.normal(size=(2, 1)) * rng.normal(size=(1, 3))
    recovered, report = m.rankone_update(g, m.operator(g, c, x, w), x, w)
    np.testing.assert_allclose(recovered, c, atol=1e-12)
    assert report["weighted_objective"] < 1e-24
    zero, report = m.rankone_update(g, np.zeros((13, 2)), x, w)
    np.testing.assert_array_equal(zero, np.zeros((2, 3)))
    assert report["weighted_objective"] == 0.0


def test_general_solution_matches_independent_cholesky():
    rng, g, x, w = fixture()
    t = x @ g
    phi = (w * t / (1 + np.exp(-t)))[:, None] * x
    y = rng.normal(size=(13, 2))
    lower = np.linalg.cholesky(phi.T @ phi)
    target = y.T @ phi
    white = np.linalg.solve(lower, target.T).T
    u, s, v = np.linalg.svd(white, full_matrices=False)
    z = (u[:, 0] * s[0])[:, None] * v[0][None, :]
    expected = np.linalg.solve(lower.T, z.T).T
    actual, _ = m.rankone_update(g, y, x, w)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_residual_multiplier_sqrt_count_equals_repeated_observations():
    rng, g, x, _ = fixture()
    counts = 1 + np.arange(13) % 3
    physical = rng.normal(size=(13, 2))
    w = np.sqrt(counts.astype(np.float64))
    weighted, _ = m.rankone_update(g, physical * w[:, None], x, w)
    repeated, _ = m.rankone_update(
        g,
        np.repeat(physical, counts, axis=0),
        np.repeat(x, counts, axis=0),
        np.ones(sum(counts)),
    )
    np.testing.assert_allclose(weighted, repeated, atol=1e-12, rtol=1e-12)


def test_invalid_weights_dimensions_and_singular_whitening():
    _, g, x, w = fixture()
    for bad in [w * 0, -w, w * np.nan, w.astype(np.float32), w[:-1]]:
        with pytest.raises(ValueError):
            m.features(g, x, bad)
    with pytest.raises(ValueError):
        m.features(g, np.ones((4097, 3)), np.ones(4097))
    with pytest.raises(ValueError, match="whitening"):
        m.rankone_update(np.zeros(3), np.ones((13, 2)), x, w)
