import importlib.util
from pathlib import Path
import numpy as np
import pytest


def load():
    path = Path(__file__).parents[1] / "examples/isolated_glu/guard_metric_normals.py"
    assert path.exists(), "Guard metric derivatives are not implemented"
    spec = importlib.util.spec_from_file_location("guard_metric_normals", path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def forward(g, c, x):
    t = x @ g.T
    return np.einsum("nw,nwo->no", t / (1 + np.exp(-t)), np.einsum("nd,wod->nwo", x, c))


def fixture():
    rng = np.random.default_rng(9229)
    g = rng.normal(size=3) * 0.3
    c = rng.normal(size=(2, 3)) * 0.2
    x = rng.normal(size=(9, 3))
    y = rng.normal(size=(9, 2))
    other = rng.normal(size=(9, 2)) * 0.2
    error = forward(g[None], c[None], x) + other - y
    return g, c, x, y, other, error


def test_squared_relative_and_unique_max_normals_match_independent_differences():
    m = load()
    g, c, x, y, other, e = fixture()
    a, report = m.metric_group(g, c, x, y, e, 2, 18)
    assert len(report["records"]) == 2
    h = 1e-6
    expected_relative = []
    expected_absolute = []
    for j in range(3):
        delta = np.zeros(3)
        delta[j] = h
        plus = forward((g + delta)[None], c[None], x) + other - y
        minus = forward((g - delta)[None], c[None], x) + other - y
        expected_relative.append(
            (np.sum(plus**2) - np.sum(minus**2)) / (2 * h * np.sum(y**2))
        )
        expected_absolute.append(
            (np.max(np.abs(plus)) - np.max(np.abs(minus))) / (2 * h)
        )
    np.testing.assert_allclose(a[0], expected_relative, atol=2e-9, rtol=2e-6)
    np.testing.assert_allclose(a[1], expected_absolute, atol=2e-9, rtol=2e-6)
    rec = report["records"][1]
    row, col = np.unravel_index(np.argmax(np.abs(e)), e.shape)
    assert rec["row"] == 18 + row and rec["output"] == col and rec["group"] == 2
    assert rec["exact_maximum"] is True
    assert rec["signed_residual"] == e[row, col]
    assert report["relative_squared"] == pytest.approx(np.sum(e**2) / np.sum(y**2))


def tied_fixture():
    g = np.array([0.2, 0.1])
    c = np.array([[1.0, 0.0]])
    x = np.array([[1.0, 0.0], [-1.0, 0.0]])
    e = np.array([[1.0], [-1.0]])
    y = forward(g[None], c[None], x) - e
    return g, c, x, y, e


def test_all_tied_normals_catch_direction_missed_by_first_argmax():
    m = load()
    g, c, x, y, e = tied_fixture()
    a, r = m.metric_group(g, c, x, y, e, 0, 0)
    assert len(r["records"]) == 3 and all(v["exact_maximum"] for v in r["records"][1:])
    direction = np.array([-1.0, 0.0])
    assert a[1] @ direction < 0 and a[2] @ direction > 0
    h = 1e-7
    new_error = forward((g + h * direction)[None], c[None], x) - y
    derivative = (np.max(np.abs(new_error)) - 1) / h
    assert derivative == pytest.approx(max(a[1:] @ direction), rel=2e-6, abs=1e-8)


def test_near_max_is_conservative_and_provenance_distinguishes_exact():
    m = load()
    g, c, x, _, e = tied_fixture()
    e[1, 0] = -(1 - 5e-13)
    y = forward(g[None], c[None], x) - e
    _, r = m.metric_group(g, c, x, y, e, 0, 0)
    assert [v["exact_maximum"] for v in r["records"][1:]] == [True, False]
    assert r["near_max_relative_tolerance"] == 1e-12
    assert r["maximum_absolute"] == 1.0
    assert r["records"][2]["signed_residual"] < 0


def test_response_rescaling_preserves_relative_and_scales_absolute_normals():
    m = load()
    g, c, x, y, _, e = fixture()
    a, r = m.metric_group(g, c, x, y, e, 0, 0)
    scaled, other = m.metric_group(g, -3.5 * c, x, -3.5 * y, -3.5 * e, 0, 0)
    np.testing.assert_allclose(scaled[0], a[0], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(scaled[1:], 3.5 * a[1:], rtol=1e-13, atol=1e-14)
    assert other["relative_squared"] == pytest.approx(r["relative_squared"])
    assert other["maximum_absolute"] == pytest.approx(3.5 * r["maximum_absolute"])


def test_computed_zero_normals_retained_as_records_not_fabricated_constraints():
    m = load()
    g = np.array([0.2])
    c = np.zeros((1, 1))
    x = np.array([[1.0], [2.0]])
    a, r = m.metric_group(g, c, x, np.ones((2, 1)), -np.ones((2, 1)), 0, 0)
    assert a.shape == (0, 1) and len(r["records"]) == 3
    assert all(
        v["matrix_row"] is None and v["omitted_reason"] == "computed_FP64_zero"
        for v in r["records"]
    )
    with pytest.raises(ValueError, match="zero maximum"):
        m.metric_group(g, c, x, np.ones((2, 1)), np.zeros((2, 1)), 0, 0)


def test_full_profile_derivatives_use_selected_unit_and_group_row_offsets():
    m = load()
    rng = np.random.default_rng(9230)
    g = rng.normal(size=(2, 3)) * 0.2
    c = rng.normal(size=(2, 2, 3)) * 0.3
    x = rng.normal(size=(32, 3))
    y = rng.normal(size=(32, 2))
    copies = [v.copy() for v in [g, c, x, y]]
    a, r = m.guard_normals(g, c, x, y, 8, 1)
    assert a.shape == (8, 3) and len(r["records"]) == 8
    h = 1e-6
    for group in range(4):
        block = slice(8 * group, 8 * (group + 1))
        expected = []
        for j in range(3):
            delta = np.zeros_like(g)
            delta[1, j] = h
            plus = forward(g + delta, c, x)[block] - y[block]
            minus = forward(g - delta, c, x)[block] - y[block]
            expected.append(
                (np.sum(plus**2) - np.sum(minus**2)) / (2 * h * np.sum(y[block] ** 2))
            )
        np.testing.assert_allclose(a[2 * group], expected, rtol=2e-6, atol=2e-9)
        rec = r["records"][2 * group + 1]
        assert rec["group"] == group and group * 8 <= rec["row"] < (group + 1) * 8
    for actual, before in zip([g, c, x, y], copies):
        np.testing.assert_array_equal(actual, before)


def test_full_profile_ties_reject_logical_budget_without_truncation():
    m = load()
    g, c, x, _, e = tied_fixture()
    x = np.tile(x, (4, 1))
    e = np.tile(e, (4, 1))
    y = forward(g[None], c[None], x) - e
    with pytest.raises(ValueError, match="logical plane budget"):
        m.guard_normals(g[None], c[None], x, y, 2, 0)
    # Zero-normal omission cannot permit12 logical records through the cap.
    with pytest.raises(ValueError, match="logical plane budget"):
        m.guard_normals(g[None], np.zeros((1, 1, 2)), x, np.ones((8, 1)), 2, 0)


def test_group_invalid_physical_reference_error_and_profile_rejected():
    m = load()
    g, c, x, y, _, e = fixture()
    for bad in [
        np.zeros_like(y),
        y.astype(np.float32),
        y * np.nan,
        y * 1e200,
        y * 1e-200,
    ]:
        with pytest.raises(ValueError):
            m.metric_group(g, c, x, bad, e, 0, 0)
    with pytest.raises(ValueError):
        m.metric_group(g, c, x, y, e.astype(np.float32), 0, 0)
    with pytest.raises(ValueError):
        m.metric_group(g, c, x, y, e, True, 0)
    with pytest.raises(ValueError):
        m.metric_group(g, c, x, y, e, 0, -1)
    with pytest.raises(ValueError):
        m.guard_normals(g[None], c[None], x, y, True, 0)
    with pytest.raises(ValueError):
        m.guard_normals(g[None], c[None], x, y, 2, True)
    with pytest.raises(ValueError):
        m.guard_normals(g[None], c[None], x, y, 2, 0)


@pytest.mark.parametrize("reference,error", [(1.0, 1e-200), (1e100, 1e-100)])
def test_nonzero_residual_with_underflowed_squared_norm_fails_closed(reference, error):
    m = load()
    with pytest.raises(ValueError, match="underflow"):
        m.metric_group(
            np.array([0.2]),
            np.zeros((1, 1)),
            np.array([[1.0]]),
            np.array([[reference]]),
            np.array([[error]]),
            0,
            0,
        )
