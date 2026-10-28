import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest


def load(monkeypatch):
    root = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(root))
    p = root / "fresh_rank_inventory.py"
    assert p.exists(), "Fresh combined inventory contract absent"
    spec = importlib.util.spec_from_file_location("fresh_rank_inventory", p)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "fresh_rank_inventory", m)
    spec.loader.exec_module(m)
    return m


def test_combined_counts_include_all_four_stores_and_preflight(monkeypatch):
    m = load(monkeypatch)
    p = m.inventory_plan()
    assert p["teachers"] == 12 and p["cases"] == 24
    assert p["vector_calls"] == 545280 and p["callback_batches"] == 768
    assert p["scientific_shards"] == 1824 and p["preflight_shards"] == 7
    assert {k: v["shards"] for k, v in p["stores"].items()} == dict(
        truth=1608, public=144, estimator=48, evaluator=24
    )
    assert p["stores"]["public"]["manifest_bytes_upper_bound"] < 128 * 1024
    assert (
        p["bytes_upper_bound"]
        == sum(v["bytes_upper_bound"] for v in p["stores"].values())
        + p["preflight_bytes_upper_bound"]
        + 8 * 1024**2
    )
    assert p["bytes_upper_bound"] < 1024**3
    assert p["minimum_free_disk_bytes"] == 2 * 1024**3


def test_max_width_shapes_and_rowlimited_basis_are_explicit(monkeypatch):
    m = load(monkeypatch)
    s = {x.kind: x for x in m.profile_schemas(32, 96)}
    assert s["context"].fields["atoms"] == (192, 528)
    assert s["context"].fields["preactivations"] == (16, 96)
    assert s["bases"].fields["richardson_basis"] == (512, 528)
    assert s["bases"].fields["richardson_spectrum"] == (528,)
    assert s["choices"].fields["stable_projector"] == (528, 528)
    assert s["choices"].fields["stability"] == (512,)
    assert s["evaluation"].fields["gate_residuals"] == (6, 96)
    assert s["evaluation"].fields["reference_ranks"] == (3,)
    assert s["inputs"].fields == {"inputs": (2112, 32)}
    assert s["inputs"].count == s["responses"].count == 32


def test_shapes_and_scalar_counts_are_independently_consistent(monkeypatch):
    m = load(monkeypatch)
    raw_in = raw_out = hessians = 0
    for n, widths in [
        (4, [2, 4, 8]),
        (8, [4, 12, 20]),
        (16, [8, 24, 48]),
        (32, [16, 48, 96]),
    ]:
        for width in widths:
            schemas = m.profile_schemas(n, width)
            for s in schemas:
                assert 1 <= len(s.fields) <= 16 and 1 <= s.count <= 32
                payload = 0
                for shape in s.fields.values():
                    product = 1
                    for size in shape:
                        assert type(size) is int and size > 0
                        product *= size
                    payload += 8 * product
                assert s.payload_bytes == payload
                assert s.bytes_upper_bound == payload + 4096 * len(s.fields)
                assert s.bytes_upper_bound < 8 * 1024**2
                if s.kind == "inputs":
                    raw_in += 2 * s.count * payload
                if s.kind == "responses":
                    raw_out += 2 * s.count * payload
                if s.kind in ("reference", "observations"):
                    hessians += 2 * s.count * payload
    assert raw_in == raw_out == 119193600
    assert hessians == 115015680


@pytest.mark.parametrize(
    "n,width", [(True, 8), (4, False), (3, 2), (64, 4), (32, 0), (32, 97)]
)
def test_invalid_profiles_rejected_without_allocation(monkeypatch, n, width):
    m = load(monkeypatch)
    with pytest.raises(ValueError):
        m.profile_schemas(n, width)


def test_manifest_bound_matches_maximum_bounded_record(monkeypatch):
    m = load(monkeypatch)
    encoded = (
        json.dumps(
            dict(name="x" * 80 + ".npz", bytes=8 * 1024**2, sha256="0" * 64),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    assert len(encoded) == m.manifest_record_upper_bound() == 188


def test_maximum_literal_shard_families_roundtrip_within_all_declared_caps(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    from scaling_domain_store import ShardStore, verify_manifest

    store = ShardStore(tmp_path / "transport")
    total = 0
    for schema in m.worst_schemas():
        fields = {
            k: np.zeros(shape, dtype=np.float64) for k, shape in schema.fields.items()
        }
        r = store.save(schema.kind + ".npz", **fields)
        assert r["bytes"] <= schema.bytes_upper_bound
        with np.load(store.root / r["name"], allow_pickle=False) as loaded:
            assert set(loaded.files) == set(fields)
            assert all(np.array_equal(loaded[k], value) for k, value in fields.items())
        total += r["bytes"]
    actual = verify_manifest(store.root)
    assert actual["shards"] == 7
    assert (
        actual["bytes_reserved"]
        == total + (store.root / "manifest.jsonl").stat().st_size
    )
    assert actual["bytes_reserved"] <= m.inventory_plan()["preflight_bytes_upper_bound"]


def test_mutable_shape_cannot_change_a_verified_budget(monkeypatch):
    m = load(monkeypatch)
    dims = [1]
    with pytest.raises(ValueError):
        m.ShardSpec("truth", "literal", {"value": dims}, 1)


@pytest.mark.parametrize("field", ["file", "allow_pickle", "x" * 65, "שם"])
def test_schema_field_names_follow_bounded_store_contract(monkeypatch, field):
    m = load(monkeypatch)
    with pytest.raises(ValueError):
        m.ShardSpec("truth", "literal", {field: (1,)}, 1)


def test_original_field_mapping_cannot_mutate_a_frozen_schema(monkeypatch):
    m = load(monkeypatch)
    original = {"value": (1,)}
    schema = m.ShardSpec("truth", "literal", original, 1)
    original["value"] = (10**9,)
    assert schema.payload_bytes == 8
    with pytest.raises(TypeError):
        schema.fields["value"] = (3,)


def test_dimension_count_cannot_bypass_header_overhead_bound(monkeypatch):
    m = load(monkeypatch)
    with pytest.raises(ValueError):
        m.ShardSpec("truth", "literal", {"value": (1,) * 2000}, 1)


def test_planned_limits_construct_every_real_store(tmp_path, monkeypatch):
    m = load(monkeypatch)
    from scaling_domain_store import ShardStore

    plan = m.inventory_plan()
    for name, limits in plan["stores"].items():
        store = ShardStore(
            tmp_path / name,
            shard_limit_bytes=limits["shard_limit_bytes"],
            total_limit_bytes=limits["bytes_upper_bound"],
        )
        assert store.shard_limit_bytes <= store.total_limit_bytes
        assert store.shard_limit_bytes <= 8 * 1024**2
