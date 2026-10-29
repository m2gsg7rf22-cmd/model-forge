import hashlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_audit_storage")


def build(m, tmp_path):
    from test_joint_response_audit_case import fixture

    case = importlib.import_module("joint_response_audit_case")
    data, meta, records, checkpoints, physical, predictions = fixture(case)
    root = tmp_path / "source"
    root.mkdir(parents=True)

    def js(name, value):
        (root / name).write_text(json.dumps(value, allow_nan=False))

    def numeric(name, values):
        blob = io.BytesIO()
        np.savez(blob, **values)
        (root / (name + ".npz")).write_bytes(blob.getvalue())

    js("case.json", meta)
    js(
        "metrics.json",
        dict(
            checkpoints=records,
            selected_step=0,
            selected_guard_mse=records[0]["guard_mse"],
            initial=records[0],
            final=records[-1],
            selected=records[0],
            checkpoint_metric_records=17,
            updates_completed=4000,
            guard_is_consumed_calibration=True,
            schema="joint-response-metrics-v1",
        ),
    )
    (root / "checkpoint-metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    for step, p in checkpoints.items():
        values = {
            prefix + k: v if not prefix else np.zeros_like(v)
            for prefix in ("", "m_", "v_")
            for k, v in p.items()
        }
        numeric(f"checkpoint-{step:04d}", values)
    numeric("split-indices", {"train": data.train, "guard": data.guard})
    numeric("selected-physical", physical)
    for i, p in enumerate(predictions):
        numeric(f"prediction-{i:02d}", {"predictions": p})
    metrics = json.loads((root / "metrics.json").read_text())
    metrics["metric_journal_bytes"] = (root / "checkpoint-metrics.jsonl").stat().st_size
    js("metrics.json", metrics)
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    reader = m._fixture_reader(
        root, pins, width=1, n=1, outputs=1, probes=4, rows=2, prediction_rows=8
    )
    return reader, data, pins, root


def test_bound_reader_full_case_and_no_schedule_or_moments_decode(m, tmp_path):
    reader, data, _, _ = build(m, tmp_path)
    case = importlib.import_module("joint_response_audit_case")
    metadata = reader.metadata()
    result = case.audit_case(
        data=data,
        **metadata,
        checkpoint=reader.checkpoint,
        predictions=reader.predictions(),
        prediction_rows=8,
    )
    assert result["status"] == "agreed"
    receipt = reader.finish()
    assert receipt["numeric_decodes"] == 21  # 17checkpoints+2predictions+split+physical
    assert receipt["reads"] == 24 and receipt["completion_hashes"] == 24
    assert receipt["status"] == "completed"


def test_hash_changed_before_decode_and_terminal_no_retry(m, tmp_path, monkeypatch):
    reader, _, _, root = build(m, tmp_path)
    reader.metadata()
    p = root / "checkpoint-0000.npz"
    p.write_bytes(p.read_bytes() + b"X")

    def forbidden(*a, **k):
        raise AssertionError("decode before hash")

    monkeypatch.setattr(m, "decode_checkpoint", forbidden)
    with pytest.raises(ValueError):
        reader.checkpoint(0)
    with pytest.raises(ValueError):
        reader.metadata()
    assert reader.report()["failed"]


def test_exact_order_no_extra_or_missing_file_requests(m, tmp_path):
    reader, *_ = build(m, tmp_path)
    with pytest.raises(ValueError):
        reader.checkpoint(256)
    reader, *_ = build(m, tmp_path / "new")
    with pytest.raises(ValueError):
        reader.finish()


def test_metadata_binding_and_completion_rehash(m, tmp_path):
    reader, data, _, root = build(m, tmp_path)
    metadata = reader.metadata()
    case = importlib.import_module("joint_response_audit_case")
    case.audit_case(
        data=data,
        **metadata,
        checkpoint=reader.checkpoint,
        predictions=reader.predictions(),
        prediction_rows=8,
    )
    p = root / "case.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError):
        reader.finish()


def test_journal_prefix_survives_failure_no_resume_or_success_prefix(m, tmp_path):
    store = m.AuditStore(tmp_path / "case")
    store.append({"step": 0, "status": "checked", "metrics_agreed": False})
    store.fail("ValueError")
    report = json.loads((tmp_path / "case/result.json").read_text())
    assert report["status"] == "failed" and report["committed_checkpoint_records"] == 1
    assert not report["certificate"]
    with pytest.raises(ValueError):
        store.append({"step": 256})
    with pytest.raises(FileExistsError):
        m.AuditStore(tmp_path / "case")


def test_journal_exact_order_size_and_complete_binding(m, tmp_path):
    store = m.AuditStore(tmp_path / "case")
    with pytest.raises(ValueError):
        store.append({"step": 256})
    store = m.AuditStore(tmp_path / "other")
    records = []
    for step in m.CHECKPOINTS:
        row = {"step": step, "metrics_agreed": True}
        store.append(row)
        records.append(row)
    with pytest.raises(ValueError):
        store.complete({"checkpoints": records[:-1], "status": "agreed"})


def test_final_result_binds_actual_disk_journal_not_only_memory(m, tmp_path):
    store = m.AuditStore(tmp_path / "case")
    records = []
    for step in m.CHECKPOINTS:
        row = {"step": step, "metrics_agreed": True}
        store.append(row)
        records.append(row)
    (tmp_path / "case/checkpoint-metrics.jsonl").write_bytes(b"")
    with pytest.raises(ValueError):
        store.complete(
            {
                "checkpoints": records,
                "status": "agreed",
                "certificate": False,
                "private_gate": "closed",
            }
        )
    assert store.failed and not store.done
    assert not (tmp_path / "case/result.json").exists()
