import importlib.util
from pathlib import Path
import numpy as np
import pytest


def load():
    path = Path(__file__).parents[1] / "examples/isolated_glu/guard_aware_direction.py"
    assert path.exists(), "Bounded guard-aware direction is not implemented"
    spec = importlib.util.spec_from_file_location("guard_aware_direction", path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_unconstrained_half_factor_and_zero_gradient():
    m = load()
    k = np.diag([2.0, 8.0])
    d = np.array([4.0, 16.0])
    g = np.array([100.0, 0.0])
    z, info = m.cone_direction(g, d, k, np.empty((0, 2)))
    np.testing.assert_allclose(z, [-1.0, -1.0])
    assert info["preclip_objective"] == pytest.approx(-10.0)
    assert info["subsets_considered"] == 1 and info["numerical_kkt_checked_preclip"]
    zero, zero_info = m.cone_direction(
        g, np.zeros(2), k, np.array([[1.0, 0.0], [-1.0, 0.0]])
    )
    np.testing.assert_array_equal(zero, np.zeros(2))
    assert zero_info["score"] == 0.0


def test_boundary_solution_has_original_duals_and_half_cost_convention():
    m = load()
    k = np.array([[2.0, 1.0], [1.0, 2.0]])
    d = np.array([-6.0, -2.0])
    g = np.array([100.0, 0.0])
    z, info = m.cone_direction(g, d, k, np.array([[2.0, 0.0]]))
    np.testing.assert_allclose(z, [0.0, 0.5], atol=1e-14)
    np.testing.assert_allclose(info["original_duals"], [2.5])
    assert info["preclip_objective"] == pytest.approx(-0.5)
    np.testing.assert_allclose(
        2 * k @ z + d + np.array([[2.0, 0.0]]).T @ np.asarray(info["original_duals"]),
        0,
        atol=1e-13,
    )


def test_duplicate_opposite_rows_and_blocked_empty_interior_cone():
    m = load()
    g = np.array([100.0])
    k = np.ones((1, 1))
    d = np.array([-2.0])
    z, info = m.cone_direction(g, d, k, np.array([[1.0], [2.0], [-1.0]]))
    np.testing.assert_allclose(z, [0.0], atol=1e-14)
    assert info["subsets_considered"] == 8
    assert any(v["status"] == "dependent_active_rows" for v in info["subsets"])
    assert info["numerical_kkt_checked_preclip"]
    np.testing.assert_allclose(
        2 * k @ np.asarray(info["unclipped"])
        + d
        + np.array([[1.0], [2.0], [-1.0]]).T @ info["original_duals"],
        0,
        atol=1e-13,
    )


def test_radial_cap_preserves_cone_but_certificate_is_preclip_only():
    m = load()
    g = np.array([0.03, 0.04])
    k = np.array([[2.0, 1.0], [1.0, 2.0]])
    d = np.array([-6.0, -2.0])
    a = np.array([[1.0, 0.0]])
    z, info = m.cone_direction(g, d, k, a)
    assert np.linalg.norm(z) == pytest.approx(0.0025)
    assert np.max(a @ z) <= 1e-14 and info["clip_factor"] < 1
    assert info["preclip_objective"] == pytest.approx(-0.5)
    assert info["postclip_objective"] < 0
    assert info["trust_region_minimizer_claimed"] is False
    assert info["postclip_kkt_claimed"] is False


def test_normalization_is_invariant_and_does_not_mutate_inputs():
    m = load()
    g = np.array([100.0, 0.0])
    d = np.array([-6.0, -2.0])
    k = np.array([[2.0, 1.0], [1.0, 2.0]])
    a = np.array([[1.0, 0.0], [0.0, -1.0]])
    originals = [v.copy() for v in [g, d, k, a]]
    z, info = m.cone_direction(g, d, k, a)
    scaled, other = m.cone_direction(
        g, 10 * d, 10 * k, np.array([[1e6, 0.0], [0.0, -1e-6]])
    )
    np.testing.assert_allclose(z, scaled, atol=1e-13)
    np.testing.assert_allclose(
        info["original_duals"] * np.array([1e-5, 1e7]),
        other["original_duals"],
        atol=1e-10,
    )
    for v, original in zip([g, d, k, a], originals):
        np.testing.assert_array_equal(v, original)


def test_bounds_nonfinite_conditioning_and_zero_normal_rejected():
    m = load()
    g = np.ones(2)
    d = np.ones(2)
    k = np.eye(2)
    a = np.empty((0, 2))
    for bad in [
        np.diag([0.0, 1.0]),
        np.diag([-1.0, 1.0]),
        np.diag([1e-12, 1.0]),
        k.astype(np.float32),
        np.array([[1.0, 1.0], [0.0, 1.0]]),
    ]:
        with pytest.raises(ValueError):
            m.cone_direction(g, d, bad, a)
    for bad in [
        np.zeros((1, 2)),
        np.ones((9, 2)),
        np.full((1, 2), np.nan),
        np.ones((1, 3)),
    ]:
        with pytest.raises(ValueError):
            m.cone_direction(g, d, k, bad)
    with pytest.raises(ValueError):
        m.cone_direction(np.zeros(2), d, k, a)
    with pytest.raises(ValueError):
        m.cone_direction(g, d * np.nan, k, a)


def test_all_256_subsets_bounded_and_redundancy_disclosed():
    m = load()
    a = np.ones((8, 1))
    z, info = m.cone_direction(np.array([100.0]), np.array([-2.0]), np.ones((1, 1)), a)
    np.testing.assert_allclose(z, [0.0], atol=1e-14)
    assert info["subsets_considered"] == 256
    assert len(info["subsets"]) == 256
    assert info["dependent_subsets"] == 247


def test_overflowed_certificate_scale_cannot_hide_stationarity_error():
    m = load()
    # force norm overflows although force entries are finite. Dividing a
    # substantial stationarity residual by infinity must never certify it.
    checks = m.certificate(
        np.array([7e153]),
        np.empty(0),
        np.array([-1.3e154]),
        np.ones((1, 1)),
        np.empty((0, 1)),
    )
    assert checks["stationarity"] > m.TOL


def test_large_other_dual_cannot_hide_negative_multiplier():
    m = load()
    checks = m.certificate(
        np.zeros(2),
        np.array([1e12, -1.0]),
        np.array([-1e12, 1.0]),
        np.eye(2),
        np.eye(2),
    )
    assert checks["dual"] > m.TOL


def test_large_unrelated_coordinate_cannot_hide_infeasible_point():
    m = load()
    checks = m.certificate(
        np.array([1.0, 1e12]),
        np.zeros(1),
        np.array([-2.0, -2e12]),
        np.eye(2),
        np.array([[1.0, 0.0]]),
    )
    assert checks["primal"] > m.TOL


@pytest.mark.parametrize("epsilon", [1e-14, 1e-11, 1e-6])
def test_nearly_opposite_independent_normals_fail_closed_not_false_optimum(epsilon):
    m = load()
    # Exact cone forces z[1]<=0, hence the closest point to [0,1] is zero.
    # A tolerance-only check would accept [0,1] with violation merely1e-14.
    with pytest.raises(ValueError, match="near-dependent"):
        m.cone_direction(
            np.array([100.0, 0.0]),
            np.array([0.0, -2.0]),
            np.eye(2),
            np.array([[1.0, 0.0], [-1.0, epsilon]]),
        )


def test_exact_general_dependency_keeps_independent_support_solution():
    m = load()
    a = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    z, info = m.cone_direction(
        np.array([100.0, 0.0, 0.0]), -2 * np.ones(3), np.eye(3), a
    )
    np.testing.assert_allclose(z, [1 / 3, 1 / 3, -1 / 3], atol=1e-13)
    assert info["dependent_subsets"] == 1


def test_separated_opposed_normals_recover_blocked_solution():
    m = load()
    z, info = m.cone_direction(
        np.array([100.0, 0.0]),
        np.array([0.0, -2.0]),
        np.eye(2),
        np.array([[1.0, 0.0], [-1.0, 1e-3]]),
    )
    np.testing.assert_allclose(z, [0.0, 0.0], atol=1e-7)
    assert info["numerical_kkt_checked_preclip"]
