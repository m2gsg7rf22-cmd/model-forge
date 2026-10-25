import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest


def load():
    path = Path(__file__).parents[1] / "examples/isolated_glu/guard_aware_worker.py"
    spec = importlib.util.spec_from_file_location("step_worker", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def inputs(tmp_path):
    paths = {
        name: tmp_path / name
        for name in [
            "candidate",
            "compact_metadata",
            "full_fit_report",
            "observations",
            "manifest",
            "original_manifest",
        ]
    }

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def npz(path, **arrays):
        with path.open("wb") as output:
            np.savez(output, **arrays)

    arrays = {}
    for group, seed in [("train", 882030), ("guard", 882031)]:
        rng = np.random.default_rng(seed)
        arrays[group + "_x"] = np.vstack(
            [rng.normal(size=(256, 32)) * s for s in [0.01, 1.0, 10.0, 100.0]]
        )
        arrays[group + "_y"] = np.ones((1024, 32))
    npz(paths["observations"], **arrays)
    npz(paths["candidate"], gates=np.ones((64, 32)), couplings=np.ones((64, 32, 32)))
    paths["full_fit_report"].write_bytes(b"hash-only, deliberately not JSON" * 100_000)
    paths["original_manifest"].write_text(json.dumps({"channel": "fp32_compute"}))
    paths["manifest"].write_text(
        json.dumps(
            {
                "method": "isolated-glu-fp32-guarded-forward-polish-v1",
                "channel": "fp32_compute",
                "train_seed": 882030,
                "guard_seed": 882031,
                "scales": [0.01, 1.0, 10.0, 100.0],
                "group_rows": 256,
                "vector_calls": 2048,
                "baseline_sha256": "b2d5fdd4cf8be40954562c2ce9393cd5b9a3c3cbd3cd8cbfa9bc18728d1de41e",
                "raw_sha256": sha(paths["observations"]),
            }
        )
    )
    paths["compact_metadata"].write_text(
        json.dumps(
            {
                "method": "isolated-glu-fp32-guarded-gate-step-v1",
                "candidate_built": True,
                "candidate_sha256": sha(paths["candidate"]),
                "forward_raw_sha256": sha(paths["observations"]),
                "forward_manifest_sha256": sha(paths["manifest"]),
                "calibration_manifest_sha256": sha(paths["original_manifest"]),
                "full_fit_report_sha256": sha(paths["full_fit_report"]),
                "full_fit_details_preserved_externally": True,
            }
        )
    )
    return paths, {"inputs": {k: sha(p) for k, p in paths.items()}}


def test_bounded_saved_loader_verifies_all_provenance_and_hashes_large_report_only(
    tmp_path, monkeypatch
):
    m = load()
    paths, config = inputs(tmp_path)
    original = m.read_json
    parsed = []

    def parse(path):
        parsed.append(path)
        assert path != paths["full_fit_report"]
        return original(path)

    monkeypatch.setattr(m, "read_json", parse)
    before, candidate, observed = m.read_data(paths, config)
    assert before == config["inputs"] and len(parsed) == 3
    assert candidate["gates"].shape == (64, 32) and observed["train_x"].shape == (
        1024,
        32,
    )


def test_changed_input_hash_stops_before_any_metadata_or_array_parse(
    tmp_path, monkeypatch
):
    m = load()
    paths, config = inputs(tmp_path)
    paths["observations"].write_bytes(b"changed")

    def forbidden(*args):
        pytest.fail("Input digest mismatch must precede scientific parsing")

    monkeypatch.setattr(m, "read_json", forbidden)
    monkeypatch.setattr(m, "read_arrays", forbidden)
    with pytest.raises(ValueError, match="commitments changed"):
        m.read_data(paths, config)


def test_worker_refuses_missing_preflight_before_reading_any_input(
    tmp_path, monkeypatch
):
    m = load()

    def forbidden(*args):
        pytest.fail("Data loader cannot run before preflight")

    monkeypatch.setattr(m, "read_data", forbidden)
    with pytest.raises(ValueError, match="pre-data"):
        m.build({}, tmp_path / "worker", {}, {})
    assert not (tmp_path / "worker").exists()


@pytest.mark.parametrize("mutate", [False, True])
def test_worker_keeps_before_and_after_on_fitting_failure(
    tmp_path, monkeypatch, mutate
):
    m = load()
    paths, config = inputs(tmp_path)

    def fail(*args, **kwargs):
        if mutate:
            paths["observations"].write_bytes(b"injected concurrent mutation")
        raise ValueError("original injected fit failure")

    monkeypatch.setattr(m.step, "run_step", fail)
    with pytest.raises(ValueError, match="original injected fit failure"):
        m.build(
            paths,
            tmp_path / "failed-worker",
            config,
            {"process_monitor_scope": "process_and_children", "memory_samples": 2},
        )
    report = json.loads((tmp_path / "failed-worker/worker-report.json").read_text())
    assert report["status"] == "failed" and report["inputs_before"] == config["inputs"]
    assert report["inputs_unchanged"] is (not mutate)
    assert report["error"] == "original injected fit failure"
    assert (report["inputs_after"] == report["inputs_before"]) is (not mutate)
