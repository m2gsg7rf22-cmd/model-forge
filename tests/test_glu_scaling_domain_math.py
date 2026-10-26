import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    p = Path(__file__).parents[1] / "examples/isolated_glu/scaling_domain_math.py"
    assert p.exists(), "Distinct dimension-domain mathematics absent"
    spec = importlib.util.spec_from_file_location("scaling_domain_math", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_svec_order_and_frobenius_inner_product():
    m = load()
    a = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    b = np.array([[2.0, -1.0, 1.0], [-1.0, 3.0, -2.0], [1.0, -2.0, 4.0]])
    np.testing.assert_allclose(
        m.svec(a), [1, 2 * np.sqrt(2), 3 * np.sqrt(2), 4, 5 * np.sqrt(2), 6]
    )
    assert np.dot(m.svec(a), m.svec(b)) == pytest.approx(np.sum(a * b))
    assert m.svec(np.stack([a, b])).shape == (2, 6)


def test_analytic_hessian_product_rule_and_atom_containment():
    m = load()
    g = np.array([[1.0, 0.0], [0.0, 1.0]])
    u = np.array([[0.0, 2.0], [3.0, 1.0]])
    d = np.array([[3.0, 2.0], [-1.0, 4.0]])
    x = np.array([[0.0, 0.0], [0.2, -0.4], [0.8, 0.3]])
    h = m.analytic_hessian(g, u, d, x)
    expected = 0.5 * np.einsum("oi,ij,ik->ojk", d, g, u)
    np.testing.assert_allclose(h[0], expected + expected.swapaxes(-1, -2))
    atoms = m.symmetric_atoms(g, u)
    np.testing.assert_allclose(atoms[0], [1, 0, 0])
    np.testing.assert_allclose(atoms[1], [0, 2 * np.sqrt(2), 0])
    sp = m.spectrum(atoms)
    a = m.svec(h).reshape(-1, 3)
    np.testing.assert_allclose(a @ sp["basis"].T @ sp["basis"], a, atol=1e-13)
    # Different implementation: mixed central differences of explicit SiLU products.
    delta = 1e-4

    def f(v):
        z = v @ g.T
        return (z / (1 + np.exp(-z)) * (v @ u.T)) @ d.T

    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * delta, np.eye(2)[j] * delta
            fd = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                4 * delta**2
            )
            np.testing.assert_allclose(fd, h[:, :, i, j], rtol=2e-6, atol=1e-7)


def test_rank_strict_boundary_full_spectrum_and_gap_semantics():
    m = load()
    a = np.diag([1.0, 1e-10, 0.0])
    sp = m.spectrum(a)
    assert sp["rank"] == 1
    assert sp["gap"] == pytest.approx(1e10)
    np.testing.assert_array_equal(sp["singular_values"], [1.0, 1e-10, 0.0])
    assert m.spectrum(np.diag([1.0, np.nextafter(1e-10, np.inf), 0.0]))["rank"] == 2
    for a, rank, reason in [
        (np.zeros((2, 3)), 0, "zero_rank"),
        (np.eye(3), 3, "full_space"),
        (np.array([[1.0, 0.0, 0.0]]), 1, "zero_next_singular"),
    ]:
        sp = m.spectrum(a)
        assert sp["rank"] == rank and sp["gap"] is None and sp["gap_reason"] == reason
        assert len(sp["singular_values"]) == 3
        assert sp["basis"].shape == (rank, 3)


def test_full_space_cannot_discriminate_rank_one_directions():
    m = load()
    v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [-3.0, 0.4]])
    full = m.spectrum(np.eye(3))
    np.testing.assert_allclose(m.direction_residuals(full["basis"], v), 0, atol=1e-14)
    deficient = m.spectrum(np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(m.direction_residuals(deficient["basis"], v[:2]), [0, 1])
    np.testing.assert_allclose(m.direction_residuals(np.zeros((0, 3)), v), 1)
    np.testing.assert_allclose(
        m.direction_residuals(deficient["basis"], v * 1e-200),
        m.direction_residuals(deficient["basis"], v),
    )


def test_four_corner_order_diagonal_step_and_richardson():
    m = load()
    x = np.array([0.2, -0.4])

    def f(q):
        return np.column_stack([q[:, 0] ** 4 + q[:, 1] ** 4, q[:, 0] * q[:, 1]])

    hs = []
    for step in [0.03, 0.06]:
        q = m.four_corner_inputs(x, step)
        assert q.shape == (4, 3, 2)
        np.testing.assert_allclose(
            q[:, 0, 0], [x[0] + 2 * step, x[0], x[0], x[0] - 2 * step]
        )
        np.testing.assert_allclose(
            q[:, 1],
            [
                x + [step, step],
                x + [step, -step],
                x + [-step, step],
                x + [-step, -step],
            ],
        )
        h = m.curvature_from_responses(np.stack([f(t) for t in q]), 2, step)
        hs.append(h)
        np.testing.assert_allclose(
            h[0].diagonal(), 12 * x * x + 8 * step**2, atol=1e-12
        )
        np.testing.assert_allclose(h[1], [[0, 1], [1, 0]], atol=1e-12)
    r = m.richardson(*hs)
    np.testing.assert_allclose(r[0], np.diag(12 * x * x), atol=1e-12)


@pytest.mark.parametrize(
    "bad", [np.zeros((0, 3)), np.ones(3), np.array([[np.nan]]), np.array([[np.inf]])]
)
def test_invalid_span_matrices_rejected(bad):
    with pytest.raises(ValueError):
        load().spectrum(bad)


def test_invalid_shapes_asymmetry_nonfinite_and_orthogonality_rejected():
    m = load()
    with pytest.raises(ValueError):
        m.svec(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        m.svec(np.ones((2, 3)))
    with pytest.raises(ValueError):
        m.analytic_hessian(
            np.ones((1, 2)), np.ones((2, 2)), np.ones((1, 1)), np.ones((1, 2))
        )
    with pytest.raises(ValueError):
        m.direction_residuals(np.ones((1, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        m.direction_residuals(np.eye(3), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        m.four_corner_inputs(np.array([np.inf]), 0.3)
    with pytest.raises(ValueError):
        m.four_corner_inputs(np.ones(2), 0)
    with pytest.raises(ValueError):
        m.curvature_from_responses(np.ones((3, 3, 2)), 2, 0.3)
    with pytest.raises(ValueError):
        m.richardson(np.ones((2, 2)), np.ones((1, 2)))


def test_large_and_tiny_geometry_is_finite_or_explicit_failure():
    m = load()
    for scale in [1e-200, 1e200]:
        sp = m.spectrum(np.eye(3) * scale)
        assert sp["rank"] == 3 and np.isfinite(sp["singular_values"]).all()
    # Finite inputs whose products overflow may not silently manufacture a zero span.
    with pytest.raises(ValueError):
        m.symmetric_atoms(np.array([[1e200]]), np.array([[1e200]]))
    with pytest.raises(ValueError, match="underflow"):
        m.symmetric_atoms(np.array([[1e-200]]), np.array([[1e-200]]))
    with pytest.raises(ValueError):
        m.four_corner_inputs(np.array([1e308]), 1e308)
    with pytest.raises(ValueError):
        m.curvature_from_responses(np.ones((4, 1, 1)), 1, 1e-200)


def test_direction_products_underflow_and_tiny_residual_not_false_zero():
    m = load()
    # Projection leaves a tiny but representable residual. Squaring it underflows.
    residual = m.direction_residuals(
        np.array([[1.0, 1e-200, 0.0]]), np.array([[1.0, 0.0]])
    )
    assert residual[0] == pytest.approx(1e-200, rel=1e-12, abs=0)
    # Forming vv^T would lose its tiny diagonal; explicitly reject partial underflow.
    with pytest.raises(ValueError, match="underflow"):
        m.direction_residuals(np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 1e-200]]))


def test_analytic_contracted_computed_zero_is_not_an_exact_rank_certificate():
    m = load()
    # Ideal value1e-400 is not representable. This smooth FP64 diagnostic may
    # return computed zero; it is never advertised as exact arithmetic truth.
    h = m.analytic_hessian(
        np.array([[1e-100]]),
        np.array([[1e-100]]),
        np.array([[1e-200]]),
        np.array([[0.0]]),
    )
    np.testing.assert_array_equal(h, 0)
