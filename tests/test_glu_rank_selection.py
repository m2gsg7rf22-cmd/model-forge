import importlib.util
from pathlib import Path
import numpy as np
import pytest


def load():
    p = Path(__file__).parents[1] / "examples/isolated_glu/rank_selection.py"
    assert p.exists(), "Observation-only estimator absent"
    spec = importlib.util.spec_from_file_location("rank_selection", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_lowrank_gap_clipped_and_zero_full_null():
    m = load()
    a = np.diag([10.0, 1.0, 0.0])
    r = m.select(a, a, a)
    assert [r["choices"][k]["rank"] for k in ["strict", "gap", "stable"]] == [2, 2, 2]
    assert r["choices"]["gap"]["next_resolution_clipped"]
    assert not any(v["certificate"] for v in r["choices"].values())
    z = m.select(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)))
    assert (
        z["choices"]["gap"]["rank"] is None and z["choices"]["stable"]["rank"] is None
    )
    full = m.select(np.eye(3), np.eye(3), np.eye(3))
    assert full["choices"]["strict"]["rank"] == 3
    assert full["choices"]["strict"]["observed_membership_domain"] == "full_space"
    assert (
        full["choices"]["gap"]["rank"] is None
        and full["choices"]["stable"]["rank"] is None
    )


def test_agreement_does_not_certify_spiky_full_signal_or_common_bias():
    m = load()
    observed = np.diag([1.0, 1e-4, 1e-4])
    # Both fullsignal+zeroerror and rank1signal+commonbias fit these SAME observations.
    low = np.diag([1.0, 0.0, 0.0])
    bias = observed - low
    np.testing.assert_array_equal(low + bias, observed)
    r = m.select(observed, observed, observed)
    assert r["choices"]["strict"]["rank"] == 3
    assert r["choices"]["stable"]["rank"] == 1
    assert r["choices"]["stable"]["certificate"] is False
    assert r["choices"]["stable"]["status"] == "heuristic_only"


def test_prefix_projector_stability_rejects_rotating_signal():
    m = load()
    a = np.diag([1.0, 1e-4, 1e-4])
    b = np.diag([1e-4, 1.0, 1e-4])
    r = m.select(a, b, a)
    assert r["choices"]["gap"]["rank"] == 1
    assert r["choices"]["stable"]["rank"] is None
    assert r["diagnostics"]["stability"][0] > 0.99


def test_scale_invariance_and_exact_tie_smallest_rank():
    m = load()
    a = np.diag([100.0, 10.0, 1.0])
    # Both un-clipped adjacent gaps exactlylog10; deterministic smallest index.
    choice = m.choose(np.array([2.0, 2.0, 1.0]), np.array([True, True, True]))
    assert choice == 0
    base = m.select(a, a, a)
    for scale in [1e-200, 1e200]:
        r = m.select(a * scale, a * scale, a * scale)
        assert r["choices"] == base["choices"]
        np.testing.assert_allclose(
            r["spectra"]["richardson"], base["spectra"]["richardson"] * scale
        )


def test_row_limited_space_and_no_width_or_privileged_metadata():
    m = load()
    a = np.array([[1.0, 0.0, 0.0]])
    r = m.select(a, a, a)
    assert r["choices"]["stable"]["rank"] == 1
    with pytest.raises(TypeError):
        m.select(a, a, a, width=17)
    with pytest.raises(TypeError):
        m.select(a, a, a, reference=np.eye(3))


@pytest.mark.parametrize(
    "a",
    [
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.zeros((0, 3)),
        np.ones(3),
        np.ones((2, 2), dtype=complex),
    ],
)
def test_invalid_matrices_fail(a):
    with pytest.raises(ValueError):
        load().select(a, a, a)


def test_mismatched_views_rejected():
    m = load()
    with pytest.raises(ValueError):
        m.select(np.eye(3), np.eye(2), np.eye(3))


def test_exact_computed_gap_ten_passes_unchanged_boundary():
    m = load()
    a = np.diag([10.0, 1.0])
    r = m.select(a, a, a)
    assert r["choices"]["gap"]["rank"] == 1
    assert r["choices"]["stable"]["rank"] == 1


@pytest.mark.parametrize("mask", [np.array([np.nan, 0.0]), np.array([1, 0])])
def test_candidate_mask_requires_real_booleans(mask):
    with pytest.raises(ValueError):
        load().choose(np.array([3.0, 2.0]), mask)
