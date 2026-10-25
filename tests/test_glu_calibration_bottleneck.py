import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "bottleneck",
    Path(__file__).parents[1] / "examples/isolated_glu/calibration_bottleneck.py",
)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_even_odd_formula_matches_silu_sum():
    rng = np.random.default_rng(451)
    x = rng.normal(size=(13, 3))
    g = rng.normal(size=(4, 3))
    c = rng.normal(size=(4, 2, 3))

    def direct(z):
        result = np.zeros((len(z), 2))
        for gate, coupling in zip(g, c):
            t = z @ gate
            result += (t / (1 + np.exp(-t)))[:, None] * (z @ coupling.T)
        return result

    a = m.components(g, c, x)
    np.testing.assert_allclose(a["even"], (direct(x) + direct(-x)) * 0.5, atol=1e-14)
    np.testing.assert_allclose(a["odd"], (direct(x) - direct(-x)) * 0.5, atol=1e-14)


def test_cubic_limit_and_saturation_are_distinct():
    g = np.array([[1.0, -0.3]])
    c = np.array([[[0.2, 0.4]]])
    x = np.array([[0.4, 0.8], [1.0, -0.2]])
    a = m.components(g, c, x)
    small = m.components(g, c, x * 1e-4)
    np.testing.assert_allclose(small["odd"] / (0.25 * 1e-12), a["cubic"], rtol=1e-8)
    assert not np.allclose(a["odd"], a["saturation"])


def test_metrics_reject_zero_reference_and_bad_shapes():
    with pytest.raises(ValueError):
        m.metrics(np.ones((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        m.metrics(np.ones((2, 3)), np.ones((3, 2)))
    a = m.metrics(np.array([[2.0, 4.0]]), np.array([[1.0, 2.0]]))
    assert a["relative_frobenius"] == 1.0 and a["max_absolute"] == 2.0


def test_attempt_exclusive_on_invalid_inputs(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        m.diagnose({}, output, {})
    assert (output / "diagnostic-attempt.json").exists()
    with pytest.raises(FileExistsError):
        m.diagnose({}, output, {})
