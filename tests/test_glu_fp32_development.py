import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "fp32_fullmap", Path(__file__).parents[1] / "examples/isolated_glu/fp32_fullmap.py"
)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_multiscale_collects_exact_channels_and_counts():
    count = 0

    def query(x):
        nonlocal count
        count += len(x)
        return (2 * x).astype(np.float32).astype(np.float64)

    obs = m.collect_multiscale(query)
    assert count == 10272
    assert obs["small_x"].shape == (4096, 32)
    assert obs["large_x"].shape == (512, 32)
    assert obs["even_pos"].shape == (528, 32)
    np.testing.assert_array_equal(obs["small_odd"], query(obs["small_x"] * 0.001))
    np.testing.assert_array_equal(obs["large_odd"], query(obs["large_x"] * 1000))


def test_rejected_fit_never_evaluates(tmp_path):
    m.write(
        tmp_path / "fit-report.json",
        {"candidate_built": False, "stage": "directions", "error": "test rejection"},
    )
    with pytest.raises(ValueError, match="No accepted development candidate"):
        m.evaluate(tmp_path)


def test_tampered_observations_rejected_before_fit(tmp_path):
    p = tmp_path / "public"
    p.mkdir()
    (p / "hessians.npz").write_bytes(b"tamper")
    m.write(
        p / "manifest.json",
        {
            "channel": "fp32_compute",
            "files": {
                "hessians.npz": "0" * 64,
                "multiscale.npz": "1" * 64,
                "curvature-pair.npz": "2" * 64,
            },
        },
    )
    with pytest.raises(ValueError, match="commitment"):
        m.fit(tmp_path)


def test_fit_attempt_blocks_repeat_after_input_failure(tmp_path):
    with pytest.raises(ValueError):
        m.fit(tmp_path)
    with pytest.raises(FileExistsError):
        m.fit(tmp_path)
