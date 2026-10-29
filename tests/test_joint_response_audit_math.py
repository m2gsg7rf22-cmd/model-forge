import importlib
import math
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_audit_math")


def test_independent_scalar_forward_and_physical_coordinates(m):
    x = np.array([[1.0], [-2.0], [0.0]])
    g = np.array([[0.5]])
    u = np.array([[2.0]])
    d = np.array([[3.0]])
    expected = np.array(
        [[3 * (z * 0.5 / (1 + math.exp(-z * 0.5))) * (z * 2)] for z in x[:, 0]]
    )
    np.testing.assert_allclose(m.forward(x, g, u, d), expected, rtol=2e-15, atol=0)
    np.testing.assert_allclose(
        m.forward(x / 2, g, u, d) * 5,
        m.forward(x, g / 2, u / 2, d * 5),
        rtol=2e-15,
        atol=0,
    )


def test_streaming_metrics_match_literal_and_partition(m):
    y = np.array([[3.0, 4.0], [0.0, 0.0]])
    p = np.array([[4.0, 2.0], [2.0, 0.0]])
    a = m.metrics([(p, y)])
    b = m.metrics([(p[:1], y[:1]), (p[1:], y[1:])])
    assert a == b
    assert a["mse"] == pytest.approx(9 / 4)
    assert a["half_mse"] == pytest.approx(9 / 8)
    assert a["relative_error"] == pytest.approx(3 / 5)
    assert m.rms([y[:1], y[1:]]) == pytest.approx(2.5)


def test_scaled_rms_does_not_overflow_square(m):
    assert m.rms([np.array([[1e200, -1e200]])]) == pytest.approx(1e200)


def test_genuine_zero_reference_and_exact_residual(m):
    z = np.zeros((1, 2))
    o = np.ones((1, 2))
    assert m.metrics([(z, z)]) == {
        "mse": 0.0,
        "half_mse": 0.0,
        "relative_error": None,
        "components": 2,
    }
    assert m.metrics([(o, z)])["relative_error"] is None
    assert m.metrics([(o, o)])["relative_error"] == 0


def test_underflowed_nonzero_loss_and_overflow_rejected(m):
    with pytest.raises(FloatingPointError):
        m.metrics([(np.array([[1e-200]]), np.zeros((1, 1)))])
    with pytest.raises(FloatingPointError):
        m.metrics([(np.array([[1e200]]), np.zeros((1, 1)))])
    with pytest.raises(FloatingPointError):
        m.forward(
            np.array([[1e308]]), np.array([[2.0]]), np.array([[1.0]]), np.array([[1.0]])
        )


def test_selection_earliest_and_near_ties_uncertified(m):
    a = m.selection([3.0, 1.0, 2.0], [3.0, 1.0, 2.0])
    assert a == {"index": 1, "ambiguous": False, "ambiguous_with": []}
    a = m.selection([1.0, 1.0, 2.0], [1.0, 1.0, 2.0])
    assert a["index"] == 0 and a["ambiguous_with"] == [1]
    a = m.selection([1.0, 1.0 + 1e-11], [1.0, 1.0 + 1e-11])
    assert a["ambiguous"]


def test_scalar_and_both_array_comparisons(m):
    assert m.scalar_agrees(1.0, 1.0 + 1e-11)
    assert not m.scalar_agrees(1.0, 1.01)
    assert m.array_agrees(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0 + 1e-11]]))
    assert not m.array_agrees(np.zeros((1, 2)), np.array([[0.0, 1e-8]]))
    assert not m.array_agrees(np.zeros((1, 2)), np.zeros((2, 1)))


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[float("nan")]]),
        np.array([[float("inf")]]),
        np.array([[1.0]], dtype=np.float32),
        np.array([1.0]),
    ],
)
def test_invalid_numeric_profile(m, bad):
    with pytest.raises(ValueError):
        m.rms([bad])


def test_empty_and_invalid_metric_shape_and_selection(m):
    with pytest.raises(ValueError):
        m.metrics([])
    with pytest.raises(ValueError):
        m.metrics([(np.ones((1, 2)), np.ones((2, 1)))])
    with pytest.raises(ValueError):
        m.selection([], [])
    with pytest.raises(ValueError):
        m.selection([1.0], [float("nan")])


def test_positive_saturation_does_not_raise_for_negligible_log_correction(m):
    one = np.ones((1, 1))
    assert m.forward(np.array([[1000.0]]), one, one, one)[0, 0] == pytest.approx(
        1e6, rel=2e-15
    )
    with pytest.raises(FloatingPointError):
        m.forward(np.array([[-1000.0]]), one, one, one)
