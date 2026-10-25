import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "residual_sensitivity",
    Path(__file__).parents[1] / "examples/isolated_glu/residual_sensitivity.py",
)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_original_sorted_greedy_threshold_rule():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    losses = np.array([1e-9, 1e-10, 2e-8])
    assert m.select(v, losses, 1e-8) == [1]
    assert m.select(v, losses, 2e-8) == [1, 2]


def test_no_64_truncation():
    assert len(m.select(np.eye(70), np.zeros(70), 1e-8)) == 70


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        m.select(np.eye(2), np.array([0.0, np.nan]), 1e-8)
    with pytest.raises(ValueError):
        m.select(np.eye(2), np.array([0.0, -1.0]), 1e-8)
    with pytest.raises(ValueError):
        m.select(np.eye(2) * 2, np.zeros(2), 1e-8)
    with pytest.raises(ValueError):
        m.select(np.eye(2), np.zeros(2), np.inf)


def test_duplicate_cosine_boundary_is_strict():
    cosine = 0.9995
    v = np.array([[1.0, 0.0], [cosine, np.sqrt(1 - cosine * cosine)]])
    assert m.select(v, np.array([0.0, 1e-9]), 1e-8) == [0]
    cosine = np.nextafter(0.9995, 0)
    v[1] = [cosine, np.sqrt(1 - cosine * cosine)]
    assert m.select(v, np.array([0.0, 1e-9]), 1e-8) == [0, 1]
