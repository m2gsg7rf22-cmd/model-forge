"""Public synthetic geometric controls, not teacher recovery certification."""
from pathlib import Path
import importlib.util

import numpy as np


def module():
    spec = importlib.util.spec_from_file_location(
        "glu_coverage", Path(__file__).resolve().parents[1] / "examples/isolated_glu/coverage.py"
    )
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_tangent_isometry_and_line_degeneracy():
    m = module()
    rng = np.random.default_rng(87)
    g, u = np.linalg.qr(rng.normal(size=(8, 2)))[0].T
    b = np.linalg.qr(np.stack([m.pack(np.outer(g, g)), m.pack(np.outer(g, u) + np.outer(u, g))]).T)[0].T
    x = g + 0.03 * u
    x /= np.linalg.norm(x)
    t = m.tangent(x)
    np.testing.assert_allclose(t.T @ t, np.eye(8), atol=1e-14)
    sv = np.linalg.svd(t - b.T @ (b @ t), compute_uv=False)
    assert sv[-1] < 1e-13
    assert sv[-2] > 0.01


def test_refinement_uses_original_subspace_and_reduces_loss():
    m = module()
    g = np.eye(8)[0]
    u = np.eye(8)[1]
    b = np.linalg.qr(np.stack([m.pack(np.outer(g, g)), m.pack(np.outer(g, u) + np.outer(u, g))]).T)[0].T
    x = g + 0.1 * u + 0.03 * np.eye(8)[2]
    x /= np.linalg.norm(x)
    y, before, after = m.refine(x, b)
    assert after < before * 1e-6
    assert abs(y @ g) > 0.999999


def test_residual_dimension_is_not_truth_certificate():
    m = module()
    b = np.eye(6)
    wrong = np.array([[0.6, 0.8, 0.0]])
    residual, diagnostics = m.residual_subspace(wrong, b)
    assert residual.shape == (4, 6)
    assert diagnostics['component_rank'] == 2
    assert diagnostics['coverage_certified'] is False
    np.testing.assert_allclose(residual @ residual.T, np.eye(4), atol=1e-13)
