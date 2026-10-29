import importlib
from pathlib import Path


def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_audit_inventory")


def test_explicit_prior_marker_and_complete_retained_budget(monkeypatch):
    m = module(monkeypatch)
    a = m.inventory()
    assert a["prior_output_and_ancestor_bytes"] == 857667407
    assert a["prior_development_marker_bytes"] == 1209
    assert a["prior_known_bytes"] == 857668616
    assert a["older_marker_reserve"] == 3 * 128 * 1024
    assert a["root_metadata_bytes"] == 6 * 128 * 1024
    assert a["case_metadata_bytes"] == 6 * (128 * 1024 + 17 * 4096)
    assert (
        a["temporary_metadata_bytes"]
        == a["root_metadata_bytes"] + a["case_metadata_bytes"]
    )
    assert a["combined_artifacts_upper_bound"] == sum(
        a[k]
        for k in (
            "prior_known_bytes",
            "older_marker_reserve",
            "root_metadata_bytes",
            "case_metadata_bytes",
            "temporary_metadata_bytes",
            "audit_marker_and_temp_bytes",
            "transport_encoded_upper_bound",
            "transport_partial_reserve",
        )
    )
    assert a["combined_artifacts_upper_bound"] < 1024**3
    assert a["largest_shard_upper_bound"] < 8 * 1024**2


def test_memory_includes_assembly_overlap_and_runtime(monkeypatch):
    m = module(monkeypatch)
    p = m.memory_plan()
    a = m.inventory()
    assert p["original_and_immutable_assembly_overlap"] == 4 * 71808 * 32 * 8
    assert p["forward_workspace"] == 24 * 4096 * 96 * 8
    assert p["rank_loader_allowance"] == 64 * 1024**2
    assert p["runtime_allocator_monitor_allowance"] == 96 * 1024**2
    assert a["memory_planning_bytes"] == sum(p.values()) < 512 * 1024**2
    assert a["preflight_literal_bytes"] == sum(p.values()) - 96 * 1024**2
    assert a["resource_proof"] is False and a["execution_approved"] is False
