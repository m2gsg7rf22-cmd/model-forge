import importlib
import math
from pathlib import Path

import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_inventory")


def test_worst_case_has_every_required_numeric_trace_and_bounded_shards(m):
    specs = m.trace_specs(96)
    assert len(specs) == 31
    names = {s.name for s in specs}
    assert {
        "initial",
        "schedule",
        "split-indices",
        "selected-physical",
        "failure-state",
    } <= names
    assert {f"checkpoint-{i:04d}" for i in (0, *range(256, 4000, 256), 4000)} <= names
    assert len([s for s in specs if s.name.startswith("prediction-")]) == 9
    assert len([s for s in specs if s.name.startswith("checkpoint-")]) == 17
    for s in specs:
        payload = sum(math.prod(f.shape) * 8 for f in s.fields)
        assert s.encoded_bound == payload + 4096 * (len(s.fields) + 1)
        assert s.encoded_bound <= 8 * 1024**2
    schedule = next(s for s in specs if s.name == "schedule")
    assert (
        schedule.fields[0].shape == (4000, 256) and schedule.fields[0].dtype == "int64"
    )
    predictions = [s for s in specs if s.name.startswith("prediction-")]
    assert sum(s.fields[0].shape[0] for s in predictions) == 71808
    assert max(s.fields[0].shape[0] for s in predictions) == 8192


def test_scalar_total_includes_parent_progress_failure_metadata_and_transport(m):
    v = m.inventory()
    assert v["prior_retained_upper_bound"] == 671428083
    assert v["trace_shards"] == 6 * 31
    assert v["trace_encoded_upper_bound"] == 6 * sum(
        s.encoded_bound for s in m.trace_specs(96)
    )
    assert v["progress_upper_bound"] == 6 * 4000 * 512
    assert v["transport_upper_bound"] == 2 * v["largest_shard_upper_bound"]
    assert v["interruption_reserve"] == 2 * v["largest_shard_upper_bound"]
    assert v["metadata_reserve"] == 8 * 1024**2
    assert v["combined_artifacts_upper_bound"] == sum(
        v[k]
        for k in (
            "prior_retained_upper_bound",
            "trace_encoded_upper_bound",
            "progress_upper_bound",
            "transport_upper_bound",
            "interruption_reserve",
            "metadata_reserve",
        )
    )
    assert v["combined_artifacts_upper_bound"] < 1024**3
    assert v["minimum_free_disk_bytes"] == 2 * 1024**3
    assert v["memory_planning_bytes"] < 512 * 1024**2
    assert v["memory_is_measured"] is False


@pytest.mark.parametrize("bad", [0, 97, True, 1.0, "96"])
def test_width_cap_strict(m, bad):
    with pytest.raises(ValueError):
        m.trace_specs(bad)
