import importlib.util
from pathlib import Path
import sys
import numpy as np

spec = importlib.util.spec_from_file_location(
    "count_diagnostic",
    Path(__file__).parents[1] / "examples/isolated_glu/count_diagnostic.py",
)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_return_telemetry_preserves_output_and_counts():
    def merge(known, candidates):
        return np.concatenate([known, candidates])

    def operation():
        return merge(np.zeros((2, 3)), np.ones((4, 3)))

    value, error, events, snapshot = m.capture(operation, {merge.__code__: "merge"})
    assert error is None and value.shape == (6, 3)
    np.testing.assert_array_equal(value, operation())
    assert events[0]["input_count"] == 2 and events[0]["retained_count"] == 6


def test_exception_telemetry_and_profiler_restoration():
    def directions():
        known = np.zeros((7, 3))
        assert known.shape == (7, 3)
        raise ValueError("fixed rejection")

    before = sys.getprofile()
    value, error, events, snapshot = m.capture(
        directions, {directions.__code__: "directions"}
    )
    assert value is None and error == "fixed rejection"
    assert snapshot["known"].shape == (7, 3)
    assert events[-1]["final_count"] == 7
    assert sys.getprofile() is before


def test_known_subspace_angle():
    a = np.eye(3)[:2]
    b = a.copy()
    b[1] = [0, np.cos(0.3), np.sin(0.3)]
    assert abs(m.distance(a, b) - np.sin(0.3) / np.sqrt(2)) < 1e-14
