import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / ("examples/isolated_glu/" + name + ".py")
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m = load("pool_candidate")
e = load("pool_evaluate")


def test_uncut_hypotheses_include_high_loss_and_minimum_representative():
    v = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    assert m.pool_indices(v, np.array([10.0, 9.0, 8.0]), 2) == [2, 1]


def test_known_width_is_checked_after_selection_without_truncation():
    with pytest.raises(ValueError, match="width"):
        m.pool_indices(np.eye(70), np.zeros(70), 64)


def test_attempt_is_reserved_before_input_failure(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        m.build(tmp_path / "missingcal", tmp_path / "missingnpz", output, {})
    assert (output / "hypothesis-attempt.json").exists()
    with pytest.raises(FileExistsError):
        m.build(tmp_path / "missingcal", tmp_path / "missingnpz", output, {})


def test_rejected_candidate_never_accesses_reference(tmp_path, monkeypatch):
    (tmp_path / "fit-report.json").write_text(json.dumps({"candidate_built": False}))

    def forbidden():
        raise AssertionError("reference access")

    monkeypatch.setattr(e, "public_reference", forbidden)
    with pytest.raises(ValueError, match="No development candidate"):
        e.evaluate(tmp_path)
    with pytest.raises(FileExistsError):
        e.evaluate(tmp_path)


def test_tampered_candidate_never_accesses_reference(tmp_path, monkeypatch):
    (tmp_path / "candidate.npz").write_bytes(b"tamper")
    (tmp_path / "fit-report.json").write_text(
        json.dumps({"candidate_built": True, "candidate_sha256": "0" * 64})
    )

    def forbidden():
        raise AssertionError("reference access")

    monkeypatch.setattr(e, "public_reference", forbidden)
    with pytest.raises(ValueError, match="Candidate commitment"):
        e.evaluate(tmp_path)
