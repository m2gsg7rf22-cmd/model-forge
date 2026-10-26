import importlib.util
from pathlib import Path
import numpy as np
import pytest


def load(monkeypatch):
    root = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(root))
    p = root / "rank_analysis.py"
    assert p.exists(), "Rank analysis pipeline absent"
    s = importlib.util.spec_from_file_location("rank_analysis", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def fixture(m, tmp_path):
    s = m.ShardStore(tmp_path / "source")
    case = "p0-t0-fp64"
    n = 2
    o = 2
    probes = 2
    width = 4
    h = np.tile(np.eye(n), (probes, o, 1, 1))
    for kind in ["fine", "coarse", "richardson", "reference"]:
        s.save(f"{case}-{kind}-h00.npz", hessian=h)
    g = np.tile(np.array([[1.0, 0.2]]), (width, 1))
    u = g * 0.3
    d = np.ones((o, width)) * 0.2
    x = np.ones((probes, n))
    s.save(
        case + "-context.npz",
        g=g,
        u=u,
        d=d,
        reference_g=g,
        reference_u=u,
        reference_d=d,
        probes=x,
        random_directions=np.tile([[1.0, -1.0]], (64, 1)),
        atoms=np.zeros((2 * width, 3)),
        preactivations=x @ g.T,
    )
    return s, [(case, dict(n=n, outputs=o, probes=probes))]


def test_privileged_loader_is_blocked_before_all_sealed_choices(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    with pytest.raises(ValueError, match="seal"):
        reader.context(*specs[0])
    with pytest.raises(ValueError, match="seal"):
        reader.hessians(specs[0][0], "reference", specs[0][1])
    obs = reader.observations(*specs[0])
    assert set(obs) == {"fine", "coarse", "richardson"}
    assert all(a.shape == (4, 3) for a in obs.values())
    assert not any("context" in p or "reference" in p for p in reader.opened)


def test_views_sealed_before_evaluator_and_immutable_outputs(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    out = tmp_path / "output"
    out.mkdir()
    store = m.ShardStore(out / "arrays")
    original = reader.context
    sealed = []

    def context(case, meta):
        assert (out / "choices-sealed.json").exists()
        sealed.append((out / "choices-sealed.json").read_bytes())
        assert len(list(store.root.glob("*-bases.npz"))) == len(specs)
        return original(case, meta)

    monkeypatch.setattr(reader, "context", context)
    r = m.run_views(reader, store, out, specs)
    assert r["status"] == "completed" and r["new_reference_queries"] == 0
    assert r["estimator_has_width_metadata"] is False
    assert len(r["evaluations"]) == 1 and len(r["evaluations"][0]["methods"]) == 6
    assert r["inputs_unchanged"] is True
    assert (out / "choices-sealed.json").read_bytes() == sealed[0]


def test_extra_width_metadata_rejected_before_any_input(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    specs[0][1]["width"] = 4
    with pytest.raises(ValueError, match="metadata"):
        m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)


def test_source_corruption_rejected_without_privileged_fallback(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    (s.root / (specs[0][0] + "-fine-h00.npz")).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        reader.observations(*specs[0])
    assert not any("context" in p for p in reader.opened)


def test_failed_estimator_never_reads_privileged_evaluation(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    out = tmp_path / "output"
    out.mkdir()
    store = m.ShardStore(out / "arrays")

    def fail(*a):
        raise RuntimeError("injected estimator")

    monkeypatch.setattr(m, "select", fail)
    with pytest.raises(RuntimeError):
        m.run_views(reader, store, out, specs)
    assert not (out / "choices-sealed.json").exists()
    assert not any("context" in p or "reference" in p for p in reader.opened)


def test_failed_seal_never_reads_truth(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    out = tmp_path / "output"
    out.mkdir()
    store = m.ShardStore(out / "arrays")
    original = m.save_json

    def fail(path, value):
        if path.name == "choices-sealed.json":
            raise OSError("injected seal")
        return original(path, value)

    monkeypatch.setattr(m, "save_json", fail)
    with pytest.raises(OSError):
        m.run_views(reader, store, out, specs)
    assert not any("context" in p or "reference" in p for p in reader.opened)


def test_manifest_change_detected_before_any_privileged_read(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    out = tmp_path / "output"
    out.mkdir()
    store = m.ShardStore(out / "arrays")
    original = reader.seal

    def seal(path, report):
        original(path, report)
        with (s.root / "manifest.jsonl").open("ab") as f:
            f.write(b"\n")

    monkeypatch.setattr(reader, "seal", seal)
    with pytest.raises(ValueError, match="manifest"):
        m.run_views(reader, store, out, specs)
    assert not any("context" in p or "reference" in p for p in reader.opened)


def test_changed_choice_archive_stops_before_truth(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    out = tmp_path / "output"
    out.mkdir()
    store = m.ShardStore(out / "arrays")
    original = reader.seal

    def seal(path, report):
        original(path, report)
        (store.root / report["selections"][0]["selected"]["name"]).write_bytes(
            b"changed"
        )

    monkeypatch.setattr(reader, "seal", seal)
    with pytest.raises(ValueError, match="commitment|Artifact"):
        m.run_views(reader, store, out, specs)
    assert not any("context" in p or "reference" in p for p in reader.opened)


def test_clipped_metadata_controls_do_not_claim_distinct_ranks(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    out = tmp_path / "output"
    out.mkdir()
    r = m.run_views(reader, m.ShardStore(out / "arrays"), out, specs)
    controls = r["evaluations"][0]["methods"][3:]
    assert [x["metadata"]["requested_rank"] for x in controls] == [8, 4, 16]
    assert all(x["metadata"]["effective_rank"] == 3 for x in controls)
    assert all(x["metadata"]["clipped"] for x in controls)
    assert all(x["metadata"]["same_effective_rank_as_known"] for x in controls)
    assert all(x["certificate"] is False for x in controls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", None),
        ("sha256", "F" * 64),
        ("bytes", True),
        ("bytes", 0),
        ("name", "../escape.npz"),
    ],
)
def test_invalid_manifest_schema_fails_before_loader(
    tmp_path, monkeypatch, field, value
):
    import json

    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    path = s.root / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0][field] = value
    path.write_text("".join(json.dumps(x) + "\n" for x in records))
    with pytest.raises(ValueError, match="manifest"):
        m.Inputs(s.root, m.sha(path), specs)


def test_changed_phase1_input_stops_before_truth(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    out = tmp_path / "output"
    out.mkdir()
    store = m.ShardStore(out / "arrays")
    original = reader.seal

    def seal(path, report):
        original(path, report)
        (s.root / (specs[0][0] + "-fine-h00.npz")).write_bytes(b"changed")

    monkeypatch.setattr(reader, "seal", seal)
    with pytest.raises(ValueError, match="input"):
        m.run_views(reader, store, out, specs)
    assert not any("context" in p or "reference" in p for p in reader.opened)


def test_short_seal_write_is_terminal_without_truth(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s, specs = fixture(m, tmp_path)
    reader = m.Inputs(s.root, m.sha(s.root / "manifest.jsonl"), specs)
    out = tmp_path / "output"
    out.mkdir()
    store = m.ShardStore(out / "arrays")
    original = Path.write_bytes

    def short(path, data):
        if path.name == "choices-sealed.tmp":
            original(path, data[:-1])
            return len(data) - 1
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", short)
    with pytest.raises(OSError, match="Short"):
        m.run_views(reader, store, out, specs)
    assert reader.seal_path is None
    assert not any("context" in p or "reference" in p for p in reader.opened)
