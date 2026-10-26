import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load():
    p = Path(__file__).parents[1] / "examples/isolated_glu/scaling_domain_store.py"
    assert p.exists(), "Bounded dimension-domain transport absent"
    spec = importlib.util.spec_from_file_location("scaling_domain_store", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_shard_roundtrip_manifest_bytes_and_exclusive_roots(tmp_path):
    m = load()
    root = tmp_path / "store"
    s = m.ShardStore(root)
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    item = s.save("group-0.npz", inputs=a, responses=a + 1)
    with np.load(root / "group-0.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["inputs"], a)
        np.testing.assert_array_equal(z["responses"], a + 1)
    manifest = json.loads((root / "manifest.jsonl").read_text())
    assert item == manifest
    result = m.verify_manifest(root)
    assert result["shards"] == 1 and result["bytes_reserved"] == s.bytes_reserved
    assert (
        s.bytes_reserved
        == (root / "group-0.npz").stat().st_size
        + (root / "manifest.jsonl").stat().st_size
    )
    with pytest.raises(FileExistsError):
        s.save("group-0.npz", inputs=a)
    with pytest.raises(FileExistsError):
        m.ShardStore(root)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(2, dtype=np.float32),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([], dtype=np.float64),
        np.array(["x"], dtype=object),
    ],
)
def test_strict_finite_float64_arrays_before_file_write(tmp_path, bad):
    s = load().ShardStore(tmp_path / "store")
    with pytest.raises(ValueError):
        s.save("bad.npz", data=bad)
    assert not (s.root / "bad.npz").exists()
    assert s.bytes_reserved == 0


@pytest.mark.parametrize(
    "name",
    [
        "../outside.npz",
        "/tmp/outside.npz",
        "nested/out.npz",
        "manifest.jsonl",
        "bad name.npz",
    ],
)
def test_filename_and_symlink_ancestry_rejected(tmp_path, name):
    m = load()
    s = m.ShardStore(tmp_path / "store")
    with pytest.raises(ValueError):
        s.save(name, data=np.ones(1))
    (tmp_path / "link").symlink_to(s.root, target_is_directory=True)
    with pytest.raises(ValueError):
        m.ShardStore(tmp_path / "link/new")


def test_array_and_total_limits_fail_before_any_shard_write(tmp_path):
    m = load()
    s = m.ShardStore(tmp_path / "store", shard_limit_bytes=1024, total_limit_bytes=1200)
    with pytest.raises(ValueError):
        s.save("big.npz", data=np.ones(1024))
    assert s.bytes_reserved == 0
    s.save("first.npz", data=np.ones(40))
    reserved = s.bytes_reserved
    with pytest.raises(ValueError):
        s.save("second.npz", data=np.ones(80))
    assert s.bytes_reserved == reserved and not (s.root / "second.npz").exists()


def test_partial_write_is_terminal_preserved_and_not_retried(tmp_path, monkeypatch):
    m = load()
    s = m.ShardStore(tmp_path / "store")
    real = Path.open

    # Failure before writing the shard still consumes the immutable name/byte reservation.
    def injected(path, mode="r", *args, **kwargs):
        if path.name == "first.npz" and mode == "xb":
            with real(path, "wb") as f:
                f.write(b"partial")
            raise OSError("injected write failure")
        return real(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", injected)
    with pytest.raises(OSError):
        s.save("first.npz", data=np.ones(2))
    assert s.terminal and s.bytes_reserved > 0
    assert (s.root / "first.npz").read_bytes() == b"partial"
    with pytest.raises(ValueError):
        s.save("second.npz", data=np.ones(2))
    with pytest.raises(ValueError):
        m.verify_manifest(s.root)


@pytest.mark.parametrize(
    "damage", ["bytes", "extra", "missing", "truncated", "duplicate", "symlink"]
)
def test_manifest_detects_tampering_and_inventory(tmp_path, damage):
    m = load()
    s = m.ShardStore(tmp_path / "store")
    s.save("first.npz", data=np.ones(2))
    p = s.root / "first.npz"
    if damage == "bytes":
        p.write_bytes(b"corrupt")
    elif damage == "extra":
        (s.root / "extra.npz").write_bytes(b"extra")
    elif damage == "missing":
        p.unlink()
    elif damage == "truncated":
        (s.root / "manifest.jsonl").write_text('{"name":')
    elif damage == "duplicate":
        q = s.root / "manifest.jsonl"
        q.write_text(q.read_text() * 2)
    else:
        p.rename(s.root / "original")
        p.symlink_to(s.root / "original")
    with pytest.raises(ValueError):
        m.verify_manifest(s.root)


def test_synthetic_transport_and_disk_preflight_no_registered_data(tmp_path):
    m = load()
    with pytest.raises(ValueError):
        m.synthetic_preflight(tmp_path / "low", free_bytes=2 * 1024**3 - 1)
    assert not (tmp_path / "low").exists()
    r = m.synthetic_preflight(tmp_path / "ok", free_bytes=2 * 1024**3)
    assert r["status"] == "passed" and r["registered_teachers_created"] == 0
    assert r["reference_calls"] == 0 and r["largest_shard_bytes"] <= 8 * 1024**2
    assert r["shards"] == 3


@pytest.mark.parametrize("field", ["file", "allow_pickle"])
def test_numpy_savez_reserved_keywords_never_drop_a_field(tmp_path, field):
    s = load().ShardStore(tmp_path / "store")
    with pytest.raises(ValueError, match="field"):
        s.save("reserved.npz", **{field: np.ones(1)})
    assert s.bytes_reserved == 0 and not (s.root / "reserved.npz").exists()


def test_payload_bound_precedes_any_elementwise_allocation(tmp_path, monkeypatch):
    m = load()
    s = m.ShardStore(tmp_path / "store")
    huge = np.broadcast_to(np.ones(1), (1_000_000_000,))

    def forbidden(*a, **k):
        pytest.fail("Elementwise validation must not precede byte-size guard")

    monkeypatch.setattr(m.np, "isfinite", forbidden)
    with pytest.raises(ValueError, match="payload"):
        s.save("oversized.npz", data=huge)
    assert s.bytes_reserved == 0


def test_replaced_manifest_symlink_never_appends_external_file(tmp_path):
    m = load()
    s = m.ShardStore(tmp_path / "store")
    target = tmp_path / "outside.jsonl"
    target.write_bytes(b"preserve")
    manifest = s.root / "manifest.jsonl"
    manifest.unlink()
    manifest.symlink_to(target)
    with pytest.raises(ValueError):
        s.save("new.npz", data=np.ones(1))
    assert target.read_bytes() == b"preserve" and not (s.root / "new.npz").exists()


def test_duplicate_json_keys_are_invalid_not_last_value_wins(tmp_path):
    m = load()
    s = m.ShardStore(tmp_path / "store")
    s.save("first.npz", data=np.ones(2))
    p = s.root / "manifest.jsonl"
    original = p.read_text().strip()
    p.write_text('{"name":"ignored.npz",' + original[1:] + "\n")
    with pytest.raises(ValueError, match="uplicate"):
        m.verify_manifest(s.root)
