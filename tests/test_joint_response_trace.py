import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_trace")


def arrays(spec):
    return {f.name: np.zeros(f.shape, dtype=f.dtype) for f in spec.fields}


def test_trace_schema_hash_immutable_files_and_no_retry(m, tmp_path):
    store = m.TraceStore(tmp_path / "trace", 1)
    spec = store.specs["initial"]
    receipt = store.save("initial", arrays(spec))
    assert receipt["bytes"] <= spec.encoded_bound
    assert receipt["sha256"] == m.digest(store.root / "initial.npz")
    assert store.reserved_bytes == spec.encoded_bound
    with pytest.raises(FileExistsError):
        store.save("initial", arrays(spec))
    with pytest.raises(ValueError):
        store.save("schedule", arrays(store.specs["schedule"]))
    assert store.failed


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "field"])
def test_trace_rejects_before_output_and_is_terminal(m, tmp_path, bad):
    s = m.TraceStore(tmp_path / bad, 1)
    values = arrays(s.specs["initial"])
    if bad == "shape":
        values["g"] = np.zeros((1, 31))
    if bad == "dtype":
        values["g"] = values["g"].astype(np.float32)
    if bad == "nan":
        values["g"][0, 0] = np.nan
    if bad == "field":
        values["context"] = np.ones((1, 1))
    with pytest.raises(ValueError):
        s.save("initial", values)
    assert not (s.root / "initial.npz").exists() and s.failed


def test_checkpoint_intent_and_progress_order(m, tmp_path):
    s = m.TraceStore(tmp_path / "trace", 1)
    for name in ("initial", "schedule", "split-indices"):
        v = arrays(s.specs[name])
        if name == "split-indices":
            indices = np.arange(71808, dtype=np.int64)
            guard = np.isin((indices // 2112) % 17, (3, 7, 11, 15))
            v = {"train": indices[~guard], "guard": indices[guard]}
        s.save(name, v)
    s.metadata("rng.json", {"fixture": "no RNG"})
    s.metadata("case.json", {"fixture": "no observations"})
    s.intent(0)
    s.save("checkpoint-0000", arrays(s.specs["checkpoint-0000"]))
    s.progress(1, 0.25)
    assert s.completed_updates == 1 and s.progress_bytes <= 512
    assert (s.root / "progress.jsonl").read_text().count("\n") == 1
    with pytest.raises(ValueError):
        s.progress(3, 0.2)
    assert s.failed


def test_no_successful_selected_prefix_or_checkpoint_without_intent(m, tmp_path):
    for name in ("selected-physical", "checkpoint-0000", "prediction-00"):
        s = m.TraceStore(tmp_path / name, 1)
        with pytest.raises(ValueError):
            s.save(name, arrays(s.specs[name]))
        assert s.failed


def test_metadata_caps_paths_and_disk_write_failure_are_terminal(
    m, tmp_path, monkeypatch
):
    s = m.TraceStore(tmp_path / "trace", 1)
    with pytest.raises(ValueError):
        s.metadata("case.json", {"huge": "a" * (128 * 1024)})
    assert not (s.root / "case.json").exists() and s.failed
    s = m.TraceStore(tmp_path / "other", 1)

    def fail(path, blob):
        raise OSError("literal disk failure")

    monkeypatch.setattr(s, "_exclusive", fail)
    with pytest.raises(OSError):
        s.save("initial", arrays(s.specs["initial"]))
    assert s.failed and s.reserved_bytes == s.specs["initial"].encoded_bound


def test_trace_root_symlink_is_rejected(m, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        m.TraceStore(link / "trace", 1)


def test_complete_literal_trace_enforces_caps_through_4000_progress_and_17_checkpoints(
    m, tmp_path
):
    import json

    s = m.TraceStore(tmp_path / "complete", 1)
    for name in ("initial", "schedule", "split-indices"):
        v = arrays(s.specs[name])
        if name == "split-indices":
            indices = np.arange(71808, dtype=np.int64)
            guard = np.isin((indices // 2112) % 17, (3, 7, 11, 15))
            v = {"train": indices[~guard], "guard": indices[guard]}
        s.save(name, v)
    s.metadata("rng.json", {"fixture": "zero arrays; no random generator"})
    s.metadata("case.json", {"fixture": "no original observations or fitting"})
    for step in range(4001):
        if step:
            s.progress(step, 0.0)
        if step in m.CHECKPOINTS:
            s.intent(step)
            name = f"checkpoint-{step:04d}"
            s.save(name, arrays(s.specs[name]))
    s.save("selected-physical", arrays(s.specs["selected-physical"]))
    for name in s.specs:
        if name.startswith("prediction-"):
            s.save(name, arrays(s.specs[name]))
    s.metadata("metrics.json", {"fixture_steps": list(m.CHECKPOINTS)})
    s.metadata("terminal.json", {"status": "completed", "fixture": True})
    assert s.finished and not s.failed
    records = [
        json.loads(line)
        for line in (s.root / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(records) == 30 and s.completed_updates == 4000
    assert len((s.root / "checkpoint-intents.jsonl").read_text().splitlines()) == 17
    assert s.metadata_bytes <= 1024**2 and s.progress_bytes <= 4000 * 512
    for record in records:
        path = s.root / (record["name"] + ".npz")
        assert (
            path.stat().st_size
            == record["bytes"]
            <= s.specs[record["name"]].encoded_bound
        )
        assert m.digest(path) == record["sha256"]
    with np.load(s.root / "schedule.npz", allow_pickle=False) as saved:
        assert saved["indices"].dtype == np.int64
    with pytest.raises(ValueError):
        s.progress(4001, 0.0)


def test_short_write_is_terminal_and_not_committed(m, tmp_path, monkeypatch):
    s = m.TraceStore(tmp_path / "short", 1)
    original = Path.open

    class ShortWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, blob):
            return self.stream.write(blob[:2])

        def flush(self):
            return self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

    def patched(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return (
            ShortWriter(stream) if path.name == "rng.json" and mode == "xb" else stream
        )

    monkeypatch.setattr(Path, "open", patched)
    with pytest.raises(OSError):
        s.metadata("rng.json", {"literal": "abcdefgh"})
    assert s.failed and not s.finished
    assert (s.root / "rng.json").stat().st_size == 2


def test_short_append_does_not_advance_counters(m, tmp_path, monkeypatch):
    s = m.TraceStore(tmp_path / "shortappend", 1)
    original = Path.open

    class ShortWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, blob):
            return self.stream.write(blob[:2])

        def flush(self):
            return self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

    def patched(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return (
            ShortWriter(stream)
            if path.name == "manifest.jsonl" and mode == "ab"
            else stream
        )

    monkeypatch.setattr(Path, "open", patched)
    with pytest.raises(OSError):
        s.save("initial", arrays(s.specs["initial"]))
    assert s.failed and s.sizes["manifest.jsonl"] == 0
    assert (s.root / "manifest.jsonl").stat().st_size == 2
