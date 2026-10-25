import importlib.util
from pathlib import Path
import json
import sys
import numpy as np
import pytest
from modelforge.execution import run_json


def source():
    p = Path(__file__).parents[1] / "examples/isolated_glu/guard_normal_geometry.py"
    assert p.exists(), "Read-only guard geometry diagnostic is absent"
    return p


def load():
    s = importlib.util.spec_from_file_location("guard_normal_geometry", source())
    assert s and s.loader
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_all_subsets_distinguish_independent_exact_dependency_and_near_geometry():
    m = load()
    info = m.inspect_geometry(np.eye(3))
    assert len(info["subsets"]) == 8 and info["in_domain"] is True
    assert info["subsets"][0]["status"] == "empty"
    dep = m.inspect_geometry(np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]]))
    assert len(dep["subsets"]) == 8 and dep["in_domain"] is True
    assert any(v["status"] == "exact_dependent" for v in dep["subsets"])
    near = m.inspect_geometry(np.array([[1.0, 0.0], [-1.0, 1e-11]]))
    assert len(near["subsets"]) == 4 and near["in_domain"] is False
    assert near["subsets"][3]["status"] == "near_independent_outside_domain"
    assert near["subsets"][3]["exact_original_rank"] == 2


def test_empty_matrix_preserved_and_invalid_rows_rejected():
    m = load()
    info = m.inspect_geometry(np.empty((0, 3)))
    assert info["empty_unconstrained"] and len(info["subsets"]) == 1
    for a in [
        np.zeros((1, 3)),
        np.ones((9, 3)),
        np.full((1, 3), np.nan),
        np.eye(2, dtype=np.float32),
    ]:
        with pytest.raises(ValueError):
            m.inspect_geometry(a)


def fixture():
    rng = np.random.default_rng(9240)
    return (
        rng.normal(size=(2, 3)) * 0.3,
        rng.normal(size=(2, 2, 3)) * 0.2,
        rng.normal(size=(32, 3)),
        rng.normal(size=(32, 2)),
    )


def test_collection_preserves_every_unit_and_matches_reviewed_mapping():
    m = load()
    g, c, x, y = fixture()
    copies = [v.copy() for v in [g, c, x, y]]
    arrays, details = m.collect(g, c, x, y, 8)
    assert (
        details["counts"]["units"] == 2 and details["counts"]["mapping_succeeded"] == 2
    )
    assert len(details["units"]) == 2 and details["direction_solves"] == 0
    for i in range(2):
        expected, report = m.metrics.guard_normals(g, c, x, y, 8, i)
        np.testing.assert_array_equal(arrays["normals"][i, : len(expected)], expected)
        assert details["units"][i]["metrics"]["records"] == report["records"]
        assert arrays["subset_counts"][i] == 256
    for actual, before in zip([g, c, x, y], copies):
        np.testing.assert_array_equal(actual, before)


def test_mapping_failure_retains_tie_records_but_never_runs_geometry():
    m = load()
    g = np.array([[0.2, 0.1]])
    c = np.array([[[1.0, 0.0]]])
    x = np.tile(np.array([[1.0, 0.0], [-1.0, 0.0]]), (4, 1))
    e = np.tile(np.array([[1.0], [-1.0]]), (4, 1))
    y = m.metrics.gradient.forward(g, c, x) - e
    arrays, d = m.collect(g, c, x, y, 2)
    assert d["counts"]["mapping_failed"] == 1
    assert d["units"][0]["mapping_status"] == "failed"
    assert len(d["units"][0]["metrics"]["records"]) == 12
    assert arrays["subset_counts"][0] == 0
    assert sum(len(v["near_coordinates"]) for v in d["guard_profile"]) == 8


def test_computed_zero_profile_keeps_unit_and_empty_geometry():
    m = load()
    g = np.array([[0.2]])
    c = np.zeros((1, 1, 1))
    x = np.arange(1.0, 9.0)[:, None]
    y = np.tile(np.array([[1.0], [2.0]]), (4, 1))
    arrays, d = m.collect(g, c, x, y, 2)
    assert d["counts"]["mapping_succeeded"] == 1 and d["counts"]["empty_geometry"] == 1
    assert arrays["normal_counts"][0] == 0 and arrays["subset_counts"][0] == 1
    assert len(d["units"][0]["metrics"]["records"]) == 8


def test_invalid_inputs_reserve_attempt_before_validation(tmp_path):
    m = load()
    out = tmp_path / "attempt"
    with pytest.raises(ValueError):
        m.build({}, out, {})
    assert json.loads((out / "attempt.json").read_text())["automatic_retry"] is False
    with pytest.raises(FileExistsError):
        m.build({}, out, {})


def test_worstsize_64_by_256_full_detail_transport_through_2k_stdout(tmp_path):
    p = source()
    code = f"""from pathlib import Path
import importlib.util,json
s=importlib.util.spec_from_file_location('geometry',{str(p)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps(m.synthetic_transport(Path('diagnostic'))))
"""
    result = run_json(
        [sys.executable, "-c", code],
        {},
        cwd=tmp_path,
        memory_limit_mb=512,
        timeout_seconds=60,
        output_limit_bytes=2048,
    )
    assert result["units"] == 64 and result["subsets"] == 16384
    assert result["details_bytes"] > 2_000_000
    full = json.loads((tmp_path / "diagnostic/evidence/details.json").read_text())
    assert len(full["units"]) == 64
    assert all(
        len(v["geometry"]["subsets"]) == 256 and len(v["metrics"]["records"]) == 8
        for v in full["units"]
    )
    with np.load(
        tmp_path / "diagnostic/evidence/numeric-evidence.npz", allow_pickle=False
    ) as data:
        assert data["subset_singular_values"].shape == (64, 256, 8)
    assert len(json.dumps(result)) < 2048


def test_zero_maximum_failure_does_not_invent_logical_plane_count():
    m = load()
    g = np.array([[0.2]])
    c = np.ones((1, 1, 1))
    x = np.arange(1.0, 9.0)[:, None]
    y = m.metrics.gradient.forward(g, c, x)
    arrays, details = m.collect(g, c, x, y, 2)
    assert details["counts"]["mapping_failed"] == 1
    assert details["units"][0]["metrics"]["logical_plane_count"] is None
    assert arrays["subset_counts"][0] == 0
