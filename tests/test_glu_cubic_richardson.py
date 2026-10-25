import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / ("examples/isolated_glu/" + name + ".py")
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m = load("cubic_candidate")
o = load("cubic_observe")
e = load("cubic_evaluate")


def responses(even, cubic, quintic, rho, dtype=np.float64):
    pos = even * rho**2 + 0.25 * cubic * rho**3 + quintic * rho**5
    neg = even * rho**2 - 0.25 * cubic * rho**3 - quintic * rho**5
    return pos.astype(dtype).astype(np.float64), neg.astype(dtype).astype(np.float64)


def test_normalized_cubic_richardson_cancels_quintic():
    rng = np.random.default_rng(891)
    even, cubic, quintic = rng.normal(size=(3, 64, 3))
    fine = responses(even, cubic, quintic, 0.03)
    coarse = responses(even, cubic, quintic, 0.06)
    np.testing.assert_allclose(
        m.cubic_rhs(*fine, *coarse, 0.03), cubic, atol=1e-11, rtol=1e-11
    )


def test_synthetic_fp32_cancellation_control():
    rng = np.random.default_rng(892)
    even, cubic, quintic = rng.normal(size=(3, 256, 3))
    baseline = responses(even, cubic, quintic, 0.001, np.float32)
    base = (baseline[0] - baseline[1]) * 0.5 / (0.25 * 0.001**3)
    fine = responses(even, cubic, quintic, 0.03, np.float32)
    coarse = responses(even, cubic, quintic, 0.06, np.float32)
    result = m.cubic_rhs(*fine, *coarse, 0.03)
    assert np.linalg.norm(result - cubic) < np.linalg.norm(base - cubic)


def test_bridge_is_derived_rhs_not_a_target_query():
    rhs = np.random.default_rng(893).normal(size=(64, 3))
    np.testing.assert_allclose(
        m.bridge(rhs) / (0.25 * 0.001**3), rhs, atol=1e-15, rtol=1e-15
    )


def test_observe_uses_four_signed_queries_per_input():
    count = 0

    def query(x):
        nonlocal count
        count += len(x)
        return (x * x + x * x * x).astype(np.float32).astype(np.float64)

    arrays = o.observe(np.ones((7, 3)), query)
    assert count == 28 and set(arrays) == {
        "small_x",
        "fine_pos",
        "fine_neg",
        "coarse_pos",
        "coarse_neg",
    }
    np.testing.assert_array_equal(arrays["fine_pos"], query(np.ones((7, 3)) * 0.03))


def test_collector_reserves_attempt_before_input_failure(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        o.collect(tmp_path / "missing", output, {})
    assert (output / "collection-attempt.json").exists()
    with pytest.raises(FileExistsError):
        o.collect(tmp_path / "missing", output, {})


def test_new_evaluation_refuses_rejected_model_before_teacher(tmp_path, monkeypatch):
    (tmp_path / "fit-report.json").write_text(json.dumps({"candidate_built": False}))

    def forbidden():
        raise AssertionError("teacher access")

    monkeypatch.setattr(e, "public_reference", forbidden)
    with pytest.raises(ValueError, match="No development candidate"):
        e.evaluate(tmp_path)


def test_raw_digest_cannot_disable_commitment():
    for value in [None, "", "x" * 64, "A" * 64]:
        with pytest.raises(ValueError, match="SHA256"):
            m.required_digest(value)
    assert m.required_digest("a" * 64) == "a" * 64
