import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_audit_runtime")


def specs():
    from joint_response_inventory import Field, Shard

    return (
        Shard(
            "parameters",
            (
                Field("g", "float64", (1, 1)),
                Field("u", "float64", (1, 1)),
                Field("d", "float64", (1, 1)),
            ),
        ),
        Shard("predictions", (Field("predictions", "float64", (2, 1)),)),
    )


def test_tiny_transport_same_bytes_hash_schema_and_literal_values(m, tmp_path):
    receipt = m._transport(tmp_path / "transport", specs())
    m._validate_transport(
        tmp_path / "transport", receipt, specs(), require_memory=False
    )
    p = tmp_path / "transport/parameters.npz"
    p.write_bytes(p.read_bytes() + b"X")
    with pytest.raises(ValueError):
        m._validate_transport(
            tmp_path / "transport", receipt, specs(), require_memory=False
        )


def test_memory_false_or_bad_TREE_receipt_stops_before_decode(m, tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("transport read before TREE gate")

    monkeypatch.setattr(m, "_read", forbidden)
    with pytest.raises(ValueError):
        m.validate_transport(
            tmp_path, {"status": "passed", "process_monitor_scope": "process_only"}
        )


def test_small_memory_exercise_fixture_not_actual_registered_allocation(m, tmp_path):
    receipt = m._preflight(
        tmp_path / "transport", specs(), {"fixture": 16}, hold_seconds=0
    )
    manifest = json.loads((tmp_path / "transport/manifest.json").read_text())
    assert manifest["memory_exercise"] and manifest["allocated_literal_bytes"] == 16
    assert manifest["allocation_plan"] == {"fixture": 16}
    assert receipt["status"] == "passed"


def test_layout_registered_partial_outputs_and_marker(m, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "controller-report.json").write_text("{}")
    (output / "cases/job00").mkdir(parents=True)
    (output / "cases/job00/checkpoint-metrics.jsonl").write_text("{}\n")
    marker = tmp_path / m.MARKER
    marker.write_text("{}")
    a = m.layout(output)
    assert a["retained_bytes"] == 5 and a["audit_marker_bytes"] == 2
    assert a["combined_retained_bytes"] == 857668616 + 393216 + 7
    assert a["files"] == 3


def test_layout_forbids_unknown_files_nested_case_and_oversize(m, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    p = output / "unknown"
    p.write_text("x")
    with pytest.raises(ValueError):
        m.layout(output)
    p.unlink()
    (output / "cases/job00/nested").mkdir(parents=True)
    with pytest.raises(ValueError):
        m.layout(output)


def test_live_engine_requests_count_against_root_metadata_aggregate(m, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    for name in m.ROOT_METADATA:
        (output / (name + ".json")).write_bytes(b"X" * (128 * 1024))
    engine = output / "preflight-engine/engine-fixture"
    engine.mkdir(parents=True)
    (engine / "request.json").write_text("{}")
    with pytest.raises(ValueError):
        m.layout(output)


def test_short_transport_write_and_output_directory_no_retry(m, tmp_path):
    root = tmp_path / "transport"
    root.mkdir()
    with pytest.raises(FileExistsError):
        m._transport(root, specs())


def test_short_transport_write_is_terminal_and_partial_file_preserved(
    m, tmp_path, monkeypatch
):
    original = Path.open

    class Short:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.f.close()

        def write(self, blob):
            return self.f.write(blob[:-1])

        def flush(self):
            return self.f.flush()

        def fileno(self):
            return self.f.fileno()

    def opened(path, *args, **kwargs):
        f = original(path, *args, **kwargs)
        return (
            Short(f)
            if path.name == "parameters.npz" and args and args[0] == "xb"
            else f
        )

    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(OSError):
        m._transport(tmp_path / "transport", specs())
    assert (tmp_path / "transport/parameters.npz").exists() and not (
        tmp_path / "transport/manifest.json"
    ).exists()
    with pytest.raises(FileExistsError):
        m._transport(tmp_path / "transport", specs())
