import importlib.util
import json
from pathlib import Path
import pytest


def load(monkeypatch):
    root = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(root))
    p = root / "rank_analysis_run.py"
    assert p.exists(), "Rank analysis worker absent"
    spec = importlib.util.spec_from_file_location("rank_analysis_run", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_synthetic_worst_archive_roundtrip_without_source_data(tmp_path, monkeypatch):
    m = load(monkeypatch)
    r = m.synthetic_transport(tmp_path / "transport", free_bytes=2 * 1024**3)
    assert r["status"] == "passed" and r["shards"] == 2
    assert 6_600_000 < r["largest_shard_bytes"] < 8 * 1024**2
    assert r["new_reference_queries"] == 0
    assert r["registered_data_opened"] is False


def test_low_disk_blocks_preflight_before_output(tmp_path, monkeypatch):
    m = load(monkeypatch)
    out = tmp_path / "transport"
    with pytest.raises(ValueError, match="GiB"):
        m.synthetic_transport(out, free_bytes=1024**3 - 1)
    assert not out.exists()


@pytest.mark.parametrize("failure", ["approval", "scope", "transport", "marker"])
def test_worker_rejects_before_original_arrays_or_truth(tmp_path, monkeypatch, failure):
    m = load(monkeypatch)
    repo = tmp_path / "repository"
    repo.mkdir()
    (tmp_path / "experiments").mkdir()
    good = dict(
        process_monitor_scope="process_and_children", memory_samples=1, status="passed"
    )
    request = dict(
        output=str(tmp_path / "output"),
        repo=str(repo),
        policy=m.POLICY,
        full_source_protocol_approved=True,
        source_commit="synthetic",
        config_sha256="synthetic",
        scope_preflight=dict(good),
        transport_preflight=dict(good),
    )
    if failure == "approval":
        request["full_source_protocol_approved"] = False
    elif failure == "scope":
        request["scope_preflight"]["process_monitor_scope"] = "process_only"
    elif failure == "transport":
        request["transport_preflight"]["status"] = "failed"
    else:
        (tmp_path / "experiments" / m.MARKER).write_text("partial")
    monkeypatch.setattr(m, "check_legacy_commitments", lambda *a: None)

    def forbidden(*a, **k):
        pytest.fail("Original arrays must not be opened")

    monkeypatch.setattr(m, "Inputs", forbidden)
    with pytest.raises((ValueError, FileExistsError)):
        m.build(request)
    r = json.loads((tmp_path / "output/worker-report.json").read_text())
    assert r["status"] == "failed" and r["new_reference_queries"] == 0
    assert r["registered_data_opened"] is False
