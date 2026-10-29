import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_audit_protocol")


def audit(m, actual):
    roots = (
        "controller-report.json",
        "outer-monitor.json",
        "preflight-receipt.json",
        "input-binding.json",
        "terminal-binding.json",
        "progress.json",
    )
    numeric = (
        {"initial", "schedule", "split-indices", "selected-physical"}
        | {f"checkpoint-{s:04d}" for s in m.POLICY["checkpoints"]}
        | {f"prediction-{i:02d}" for i in range(9)}
    )
    metadata = {
        "case.json",
        "metrics.json",
        "terminal.json",
        "rng.json",
        "manifest.jsonl",
        "checkpoint-intents.jsonl",
        "checkpoint-metrics.jsonl",
        "progress.jsonl",
    }
    pins = {str(actual / n): "a" * 64 for n in roots}
    for i in range(6):
        pins.update(
            {
                str(actual / "cases" / f"job{i:02d}" / n): "b" * 64
                for n in ({n + ".npz" for n in numeric} | metadata)
            }
        )
    return dict(
        saved_engineering_receipt_audit_approved=True,
        numeric_metric_audit_approved=False,
        actual_source_commit=m.ACTUAL_COMMIT,
        private_gate="closed",
        certificate=False,
        artifact_sha256=pins,
    )


def test_bindings_exact_source_case_inventory_and_missing_rejected(m, tmp_path):
    a = audit(m, tmp_path)
    b = m.input_bindings(a, tmp_path)
    assert len(b["case_files"]) == 6 and all(
        len(p) == 38 for p in b["case_files"].values()
    )
    a["artifact_sha256"].pop(str(tmp_path / "cases/job02/initial.npz"))
    with pytest.raises(ValueError):
        m.input_bindings(a, tmp_path)


def test_unapproved_source_stops_before_config_runtime_or_input_reads(
    m, tmp_path, monkeypatch
):
    approval = tmp_path / "review.json"
    approval.write_text("{}")

    def forbidden(*a, **k):
        raise AssertionError("config touched before approval")

    monkeypatch.setattr(m, "config_template", forbidden)
    with pytest.raises(ValueError, match="independent"):
        m.verify_source(tmp_path, tmp_path / "missingconfig", approval)


def test_partial_source_approval_or_gate_open_is_not_execution_approval(m, tmp_path):
    approval = tmp_path / "review.json"
    approval.write_text(
        json.dumps(
            dict(
                full_joint_response_numeric_audit_protocol_approved=True,
                execution_after_gates_approved=False,
                private_gate="closed",
                certificate=False,
            )
        )
    )
    with pytest.raises(ValueError):
        m.verify_source(tmp_path, tmp_path / "none", approval)


def test_prior_approval_or_hash_format_invalid_rejected(m, tmp_path):
    a = audit(m, tmp_path)
    a["private_gate"] = "open"
    with pytest.raises(ValueError):
        m.input_bindings(a, tmp_path)
    a = audit(m, tmp_path)
    a["artifact_sha256"][str(tmp_path / "cases/job00/case.json")] = "bad"
    with pytest.raises(ValueError):
        m.input_bindings(a, tmp_path)
