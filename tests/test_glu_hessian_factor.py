import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "factor", Path(__file__).parents[1] / "examples/isolated_glu/hessian_factor.py"
)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rng = np.random.default_rng(5501)
    return rng, rng.normal(size=3), rng.normal(size=(2, 3)), rng.normal(size=(5, 3))


def test_hessian_matches_independent_forward_difference():
    _, g, c, x = fixture()

    def forward(z):
        t = z @ g
        return (t / (1 + np.exp(-t)))[:, None] * (z @ c.T)

    expected = np.empty((len(x), 2, 3, 3))
    h = 1e-4
    eye = np.eye(3) * h
    for i in range(3):
        for j in range(3):
            expected[:, :, i, j] = (
                forward(x + eye[i] + eye[j])
                - forward(x + eye[i] - eye[j])
                - forward(x - eye[i] + eye[j])
                + forward(x - eye[i] - eye[j])
            ) / (4 * h * h)
    np.testing.assert_allclose(m.operator(g, c, x), expected, atol=2e-7, rtol=2e-6)


def test_adjoint_and_gram_match_explicit_operator_basis():
    rng, g, c, x = fixture()
    y = rng.normal(size=(5, 2, 3, 3))
    a = np.column_stack(
        [m.operator(g, np.eye(3)[i : i + 1], x).reshape(-1) for i in range(3)]
    )
    expected = y.transpose(1, 0, 2, 3).reshape(2, -1) @ a
    np.testing.assert_allclose(m.adjoint(g, y, x), expected, atol=1e-12)
    np.testing.assert_allclose(m.gram(g, x), a.T @ a, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(m.operator(g, c, x) * y), np.sum(c * m.adjoint(g, y, x)), atol=1e-12
    )


def test_rankone_update_recovers_exact_unit_and_rejects_singular_whitening():
    rng, g, _, x = fixture()
    c = rng.normal(size=(2, 1)) * rng.normal(size=(1, 3))
    y = m.operator(g, c, x)
    recovered, report = m.rankone_update(g, y, x)
    np.testing.assert_allclose(recovered, c, atol=1e-12)
    assert report["objective"] < 1e-24
    with pytest.raises(ValueError, match="whitening"):
        m.rankone_update(np.zeros(3), y, x)


def test_general_rankone_solution_matches_independent_cholesky_coordinates():
    rng, g, _, x = fixture()
    y = rng.normal(size=(5, 2, 3, 3))
    a = np.column_stack(
        [m.operator(g, np.eye(3)[i : i + 1], x).reshape(-1) for i in range(3)]
    )
    t = y.transpose(1, 0, 2, 3).reshape(2, -1) @ a
    lower = np.linalg.cholesky(a.T @ a)
    w = np.linalg.solve(lower, t.T).T
    u, s, v = np.linalg.svd(w, full_matrices=False)
    z = (u[:, 0] * s[0])[:, None] * v[0][None, :]
    expected = np.linalg.solve(lower.T, z.T).T
    actual, _ = m.rankone_update(g, y, x)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_finite_shape_bounds():
    _, g, c, x = fixture()
    with pytest.raises(ValueError):
        m.operator(g, c, x * np.nan)
    with pytest.raises(ValueError):
        m.operator(g, c, np.ones((65, 3)))
    with pytest.raises(ValueError):
        m.adjoint(g, np.ones((5, 2, 3, 2)), x)
