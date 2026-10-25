"""Synthetic controls for multi-direction residual development, not sealed science."""

from pathlib import Path
import importlib.util
import numpy as np


def module():
    p = Path(__file__).resolve().parents[1] / "examples/isolated_glu/coverage_multi.py"
    spec = importlib.util.spec_from_file_location("multi_coverage", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_multi_missing_directions_on_public_synthetic_subspace():
    m = module()
    rng = np.random.default_rng(8153)
    g = rng.normal(size=(3, 8))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    u = rng.normal(size=(3, 8))
    components = np.stack(
        [
            m.pack(a)
            for x, y in zip(g, u)
            for a in [np.outer(x, x), np.outer(x, y) + np.outer(y, x)]
        ]
    )
    basis = np.linalg.qr(components.T)[0].T
    result, rounds = m.search(g[:1].copy(), basis, target=3, mixtures=32)
    assert len(result) == 3
    assert np.min(np.max(np.abs(g @ result.T), axis=1)) > 0.99999
    assert len(rounds) <= 4


def test_filter_never_truncates_extra_nonduplicates():
    m = module()
    known = np.eye(5)[:1]
    candidates = np.eye(5)[1:]
    result = m.merge(known, candidates, np.zeros(4))
    assert result.shape == (5, 5)
