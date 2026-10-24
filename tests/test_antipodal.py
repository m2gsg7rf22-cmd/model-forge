"""Owned synthetic teachers test only the even component and conditional signs."""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from modelforge.antipodal import (
    canonical_probe_points,
    predict_even,
    recover_even_quadratic,
    recover_orientations,
)

Array = NDArray[np.float64]


def teacher(x: Array, gates: Array, up: Array, down: Array) -> Array:
    z = x @ gates.T
    return ((z / (1.0 + np.exp(-z))) * (x @ up.T)) @ down.T


def analytic_q(gates: Array, up: Array, down: Array) -> Array:
    return np.asarray(
        0.25
        * (
            np.einsum("oh,hi,hj->oij", down, up, gates)
            + np.einsum("oh,hi,hj->oji", down, up, gates)
        ),
        dtype=np.float64,
    )


def test_owned_width64_teacher_even_recovery_on_unseen_inputs() -> None:
    rng = np.random.default_rng(24101)
    gates = rng.normal(scale=0.2, size=(64, 32))
    up = rng.normal(scale=0.2, size=(64, 32))
    down = rng.normal(scale=0.2, size=(32, 64))
    probes = canonical_probe_points(32)
    assert probes.shape == (528, 32)
    assert 2 * len(probes) == 32 * 33
    q = recover_even_quadratic(
        teacher(probes, gates, up, down), teacher(-probes, gates, up, down), 32
    )
    np.testing.assert_allclose(q, analytic_q(gates, up, down), atol=2e-15, rtol=2e-13)
    unseen = rng.normal(scale=0.5, size=(53, 32))
    actual_even = (
        teacher(unseen, gates, up, down) + teacher(-unseen, gates, up, down)
    ) / 2
    np.testing.assert_allclose(
        predict_even(q, unseen), actual_even, atol=2e-13, rtol=2e-12
    )
    # Odd behavior is not reconstructed by Q.
    assert (
        np.max(np.abs(predict_even(q, unseen) - teacher(unseen, gates, up, down)))
        > 0.01
    )


def test_geglu_even_recovery() -> None:
    rng = np.random.default_rng(24102)
    gates, up = rng.normal(size=(2, 5, 3))
    down = rng.normal(size=(2, 5))
    probes = canonical_probe_points(3)

    def geglu(x: Array) -> Array:
        z = x @ gates.T
        phi = np.asarray(
            [[0.5 * (1 + math.erf(v / math.sqrt(2))) for v in row] for row in z]
        )
        return np.asarray(((z * phi) * (x @ up.T)) @ down.T, dtype=np.float64)

    q = recover_even_quadratic(geglu(probes), geglu(-probes), 3)
    np.testing.assert_allclose(q, analytic_q(gates, up, down), atol=2e-14)


def test_probe_order_and_scalar_input_prediction() -> None:
    np.testing.assert_array_equal(canonical_probe_points(2), [[1, 0], [0, 1], [1, 1]])
    q = recover_even_quadratic(np.array([[6.0]]), np.array([[2.0]]), 1)
    np.testing.assert_array_equal(q, [[[4.0]]])
    np.testing.assert_array_equal(predict_even(q, np.array([3.0])), [36.0])


def test_conditional_orientations_recover_known_independently_supplied_candidates() -> (
    None
):
    rng = np.random.default_rng(24103)
    gates = rng.normal(size=(7, 4))
    up = rng.normal(size=(7, 4))
    down = rng.normal(size=(3, 7))
    signs = np.array([1, -1, -1, 1, -1, 1, 1], dtype=np.float64)
    couplings = np.einsum("oh,hi->hoi", down, up)
    q = analytic_q(signs[:, None] * gates, up, down)
    np.testing.assert_array_equal(recover_orientations(gates, couplings, q), signs)


def test_orientation_rejects_rank_deficient_and_zero_candidates() -> None:
    gates = np.ones((2, 2))
    couplings = np.ones((2, 1, 2))
    with pytest.raises(ValueError, match="rank"):
        recover_orientations(gates, couplings, np.ones((1, 2, 2)))
    with pytest.raises(ValueError, match="zero"):
        recover_orientations(np.zeros((1, 2)), couplings[:1], np.ones((1, 2, 2)))
    with pytest.raises(ValueError, match="zero"):
        recover_orientations(gates[:1], np.zeros((1, 1, 2)), np.ones((1, 2, 2)))


def test_orientation_rejects_non_sign_and_residual_mismatches() -> None:
    gates = np.array([[1.0, 0.0]])
    couplings = np.array([[[1.0, 0.0]]])
    with pytest.raises(ValueError, match="sign"):
        recover_orientations(gates, couplings, np.array([[[1.0, 0.0], [0.0, 0.0]]]))
    with pytest.raises(ValueError, match="residual"):
        recover_orientations(gates, couplings, np.array([[[0.5, 0.0], [0.0, 1.0]]]))


def test_orientation_rejects_ill_conditioned_candidates() -> None:
    gates = np.array([[1.0, 0.0], [1.0, 1e-10]])
    couplings = np.ones((2, 1, 2))
    with pytest.raises(ValueError, match="condition"):
        recover_orientations(gates, couplings, np.ones((1, 2, 2)))


def test_bias_contamination_and_wrong_odd_mapping_exposed_by_unseen_points() -> None:
    rng = np.random.default_rng(24104)
    gates, up = rng.normal(scale=0.2, size=(2, 5, 3))
    down = rng.normal(size=(2, 5))
    probes = canonical_probe_points(3)
    pos, neg = teacher(probes, gates, up, down), teacher(-probes, gates, up, down)
    q = recover_even_quadratic(pos + 0.7, neg + 0.7, 3)
    unseen = rng.normal(size=(8, 3))
    biased_even = (
        teacher(unseen, gates, up, down) + teacher(-unseen, gates, up, down)
    ) / 2 + 0.7
    assert np.max(np.abs(predict_even(q, unseen) - biased_even)) > 0.1
    # An odd linear addition is invisible to antipodal recovery but changes the full function.
    wrong_pos, wrong_neg = pos + probes[:, :2], neg - probes[:, :2]
    np.testing.assert_allclose(
        recover_even_quadratic(wrong_pos, wrong_neg, 3),
        analytic_q(gates, up, down),
        atol=1e-14,
    )
    assert (
        np.max(
            np.abs(
                teacher(unseen, gates, up, down)
                + unseen[:, :2]
                - predict_even(q, unseen)
            )
        )
        > 0.1
    )


@pytest.mark.parametrize("dimension", [0, -1, 65, True, 1.5])
def test_rejects_invalid_dimension(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension"):
        canonical_probe_points(dimension)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_outputs(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        recover_even_quadratic(np.array([[bad]]), np.ones((1, 1)), 1)


def test_shape_symmetry_complex_and_resource_validation() -> None:
    with pytest.raises(ValueError):
        recover_even_quadratic(np.zeros((3, 1)), np.zeros((2, 1)), 2)
    with pytest.raises(ValueError):
        recover_even_quadratic(np.zeros((1, 65)), np.zeros((1, 65)), 1)
    with pytest.raises(ValueError):
        recover_even_quadratic(np.ones((1, 1), dtype=complex), np.ones((1, 1)), 1)
    with pytest.raises(ValueError, match="symmetric"):
        predict_even(np.array([[[1.0, 1.0], [0.0, 1.0]]]), np.ones((1, 2)))
    with pytest.raises(ValueError):
        predict_even(np.ones((1, 2, 2)), np.ones((3, 3)))
    with pytest.raises(ValueError, match="resource"):
        recover_orientations(
            np.ones((128, 64)), np.ones((128, 64, 64)), np.ones((64, 64, 64))
        )
    with pytest.raises(ValueError):
        recover_orientations(
            np.ones((129, 1)), np.ones((129, 1, 1)), np.ones((1, 1, 1))
        )


def test_rejects_non_rank_one_glu_coupling_candidate() -> None:
    gates = np.array([[1.0, 1.0]])
    # This full-rank coupling can fit a quadratic but is not one GLU neuron.
    coupling = np.eye(2)[None, :, :]
    q = np.array([[[0.5, 0.25], [0.25, 0.0]], [[0.0, 0.25], [0.25, 0.5]]])
    with pytest.raises(ValueError, match="rank.one"):
        recover_orientations(gates, coupling, q)


def test_rejects_nonfinite_and_overflow_predictions_and_empty_shapes() -> None:
    with pytest.raises(ValueError, match="finite"):
        predict_even(np.ones((1, 1, 1)), np.array([np.nan]))
    with pytest.raises(ValueError, match="finite"):
        predict_even(np.ones((1, 1, 1)), np.array([1e308]))
    with pytest.raises(ValueError):
        predict_even(np.ones((1, 1, 1)), np.ones((0, 1)))
    with pytest.raises(ValueError, match="finite"):
        recover_orientations(
            np.array([[np.inf]]), np.ones((1, 1, 1)), np.ones((1, 1, 1))
        )
    with pytest.raises(ValueError):
        recover_orientations(np.ones((1, 1)), np.ones((1, 2, 1)), np.ones((1, 1, 1)))
