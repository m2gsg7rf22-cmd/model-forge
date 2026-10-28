import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def load(monkeypatch):
    root = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(root))
    path = root / "fresh_rank_view.py"
    assert path.exists(), "Dedicated observation-only view absent"
    spec = importlib.util.spec_from_file_location("fresh_rank_view", path)
    m = importlib.util.module_from_spec(spec)
    import sys

    monkeypatch.setitem(sys.modules, "fresh_rank_view", m)
    spec.loader.exec_module(m)
    return m


def fixture(m, tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    from scaling_domain_store import ShardStore

    store = ShardStore(root / "arrays")
    name = "c00-fp64"
    h = np.tile(np.array([[2.0, 1.0], [1.0, 3.0]]), (2, 1, 1, 1))
    for kind in ("fine", "coarse", "richardson"):
        store.save(name + "-" + kind + "-h00.npz", hessian=h)
    meta = dict(
        schema="modelforge-fresh-rank-observations-v1",
        steps=[0.3, 0.6],
        cases=[dict(name=name, n=2, outputs=1, probes=2, compute="fp64")],
    )
    (root / "cases.json").write_text(json.dumps(meta))
    truth = tmp_path / "truth"
    truth.mkdir()
    (truth / "context.npz").write_bytes(b"TRUTH MUST NOT BE OPENED")
    return root, meta


def view(m, root):
    return m.FDView(
        root, m.sha(root / "cases.json"), m.sha(root / "arrays/manifest.jsonl")
    )


def test_literal_public_view_packs_without_truth_and_returns_readonly(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    root, _ = fixture(m, tmp_path)
    v = view(m, root)
    assert v.cases[0].n == 2 and not hasattr(v.cases[0], "width")
    arrays = v.packed("c00-fp64")
    assert set(arrays) == {"fine", "coarse", "richardson"}
    for a in arrays.values():
        assert a.shape == (2, 3)
        np.testing.assert_allclose(a, np.tile([2.0, np.sqrt(2.0), 3.0], (2, 1)))
        assert not a.flags.writeable
    assert len(v.verify_unchanged()["used_input_sha256"]) == 3
    assert (tmp_path / "truth/context.npz").read_bytes() == b"TRUTH MUST NOT BE OPENED"


@pytest.mark.parametrize("extra", ["width", "seed", "teacher_type", "reference_rank"])
def test_privileged_metadata_rejected_before_array_read(tmp_path, monkeypatch, extra):
    m = load(monkeypatch)
    root, meta = fixture(m, tmp_path)
    meta["cases"][0][extra] = 3
    (root / "cases.json").write_text(json.dumps(meta))
    monkeypatch.setattr(
        m,
        "read_arrays",
        lambda *a, **k: pytest.fail("NPZ read before metadata rejection"),
    )
    with pytest.raises(ValueError):
        view(m, root)


def test_full_manifest_truth_size_record_rejected_before_array_read(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    root, _ = fixture(m, tmp_path)
    record = dict(name="c00-fp64-context.npz", bytes=12345, sha256="0" * 64)
    with (root / "arrays/manifest.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    monkeypatch.setattr(
        m, "read_arrays", lambda *a, **k: pytest.fail("NPZ read before view rejection")
    )
    with pytest.raises(ValueError):
        view(m, root)


@pytest.mark.parametrize(
    "field,value",
    [("sha256", None), ("sha256", "F" * 64), ("bytes", True), ("bytes", 0)],
)
def test_invalid_public_record_schema_rejected(tmp_path, monkeypatch, field, value):
    m = load(monkeypatch)
    root, _ = fixture(m, tmp_path)
    path = root / "arrays/manifest.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines()]
    records[0][field] = value
    path.write_text("".join(json.dumps(x) + "\n" for x in records))
    with pytest.raises(ValueError):
        view(m, root)


@pytest.mark.parametrize("where", ["root", "arrays", "symlink"])
def test_extra_inventory_or_symlink_is_rejected(tmp_path, monkeypatch, where):
    m = load(monkeypatch)
    root, _ = fixture(m, tmp_path)
    if where == "root":
        (root / "truth.json").write_text("{}")
    elif where == "arrays":
        (root / "arrays/context.npz").write_bytes(b"private")
    else:
        p = root / "arrays/c00-fp64-fine-h00.npz"
        p.unlink()
        p.symlink_to(tmp_path / "truth/context.npz")
    with pytest.raises(ValueError):
        view(m, root)


def test_changed_used_input_and_metadata_fail_final_check(tmp_path, monkeypatch):
    m = load(monkeypatch)
    root, _ = fixture(m, tmp_path)
    v = view(m, root)
    v.packed("c00-fp64")
    (root / "arrays/c00-fp64-fine-h00.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        v.verify_unchanged()


@pytest.mark.parametrize("which", ["cases", "manifest"])
def test_external_commitment_cannot_be_disabled(tmp_path, monkeypatch, which):
    m = load(monkeypatch)
    root, _ = fixture(m, tmp_path)
    pins = [m.sha(root / "cases.json"), m.sha(root / "arrays/manifest.jsonl")]
    pins[0 if which == "cases" else 1] = None
    with pytest.raises(ValueError):
        m.FDView(root, *pins)


def test_adjacent_truth_is_never_opened(tmp_path, monkeypatch):
    m = load(monkeypatch)
    root, _ = fixture(m, tmp_path)
    original = Path.open

    def guarded(path, *a, **kw):
        if tmp_path / "truth" in [path, *path.parents]:
            pytest.fail("Truth filesystem access from public view")
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "open", guarded)
    v = view(m, root)
    assert len(v.packed("c00-fp64")) == 3


@pytest.mark.parametrize("kind", ["shape", "field", "dtype"])
def test_invalid_npz_header_fails_without_usable_output(tmp_path, monkeypatch, kind):
    m = load(monkeypatch)
    root, _ = fixture(m, tmp_path)
    target = root / "arrays/c00-fp64-fine-h00.npz"
    fields = {"hessian": np.ones((2, 1, 2, 2))}
    if kind == "shape":
        fields = {"hessian": np.ones((2, 1, 3, 3))}
    elif kind == "field":
        fields = {"reference": np.ones((2, 1, 2, 2))}
    else:
        fields = {"hessian": np.ones((2, 1, 2, 2), dtype=np.float32)}
    np.savez(target, **fields)
    manifest = root / "arrays/manifest.jsonl"
    records = [json.loads(x) for x in manifest.read_text().splitlines()]
    records[0].update(bytes=target.stat().st_size, sha256=m.sha(target))
    manifest.write_text("".join(json.dumps(x) + "\n" for x in records))
    v = view(m, root)
    with pytest.raises(ValueError):
        v.packed("c00-fp64")


@pytest.mark.parametrize(
    "change", ["duplicate_key", "truncated_manifest", "cases_mutation"]
)
def test_metadata_failures_are_not_array_fallbacks(tmp_path, monkeypatch, change):
    m = load(monkeypatch)
    root, meta = fixture(m, tmp_path)
    if change == "cases_mutation":
        v = view(m, root)
        (root / "cases.json").write_text(json.dumps(meta) + " ")
        with pytest.raises(ValueError):
            v.packed("c00-fp64")
        return
    if change == "duplicate_key":
        (root / "cases.json").write_text(
            json.dumps(meta).replace('"schema":', '"schema":"duplicate","schema":', 1)
        )
    else:
        p = root / "arrays/manifest.jsonl"
        p.write_bytes(p.read_bytes()[:-1])
    monkeypatch.setattr(
        m,
        "read_arrays",
        lambda *a, **k: pytest.fail("Array read after metadata failure"),
    )
    with pytest.raises(ValueError):
        view(m, root)
