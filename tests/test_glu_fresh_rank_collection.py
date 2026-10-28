import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest


def load(monkeypatch):
    base = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(base))
    p = base / "fresh_rank_collection.py"
    assert p.exists(), "Fresh collection stage absent"
    spec = importlib.util.spec_from_file_location("fresh_rank_collection", p)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "fresh_rank_collection", m)
    spec.loader.exec_module(m)
    return m


def test_literal_counter_persists_input_and_intent_before_callback(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    store = m.ShardStore(tmp_path / "arrays")
    ledger = m.RawCounter(tmp_path / "progress.json", 4)

    def callback(x):
        assert not x.flags.writeable
        assert (store.root / "literal-inputs.npz").exists()
        assert json.loads(ledger.path.read_text())["calls_attempted"] == 4
        return np.ones((4, 2))

    y = ledger.query(store, "literal", np.zeros((4, 2)), callback, 2)
    assert y.shape == (4, 2)
    assert ledger.counts == dict(
        calls_reserved=4,
        calls_attempted=4,
        calls_returned=4,
        calls_persisted=4,
        callback_batches=1,
    )


@pytest.mark.parametrize("failure", ["callback", "shape", "nan", "mutate"])
def test_counter_partial_failure_is_terminal_without_retry(
    tmp_path, monkeypatch, failure
):
    m = load(monkeypatch)
    store = m.ShardStore(tmp_path / "arrays")
    ledger = m.RawCounter(tmp_path / "progress.json", 4)

    def callback(x):
        if failure == "callback":
            raise RuntimeError("injected callback")
        if failure == "shape":
            return np.zeros((1, 2))
        if failure == "nan":
            return np.full((4, 2), np.nan)
        x.setflags(write=True)
        x[0, 0] = 99
        return np.ones((4, 2))

    with pytest.raises((ValueError, RuntimeError)):
        ledger.query(store, "literal", np.zeros((4, 2)), callback, 2)
    assert ledger.terminal and ledger.counts["calls_attempted"] == 4
    with pytest.raises(ValueError):
        ledger.query(store, "retry", np.zeros((4, 2)), callback, 2)
    with np.load(store.root / "literal-inputs.npz") as f:
        assert np.all(f["inputs"] == 0)


def test_callback_buffer_reuse_cannot_change_persisted_response(tmp_path, monkeypatch):
    m = load(monkeypatch)
    store = m.ShardStore(tmp_path / "arrays")
    ledger = m.RawCounter(tmp_path / "progress.json", 4)
    shared = np.ones((4, 2))
    out = ledger.query(store, "literal", np.zeros((4, 2)), lambda x: shared, 2)
    shared[:] = 99
    assert np.all(out == 1)
    with np.load(store.root / "literal-responses.npz") as f:
        assert np.all(f["responses"] == 1)


def test_supplied_literal_case_has_separate_truth_and_public_manifests(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    truth = m.ShardStore(tmp_path / "truth")
    public = m.ShardStore(tmp_path / "public")
    ledger = m.RawCounter(tmp_path / "progress.json", 24)
    ctx = dict(
        g=np.zeros((3, 2)),
        u=np.zeros((3, 2)),
        d=np.ones((2, 3)),
        probes=np.zeros((1, 2)),
        random_directions=np.tile([[1.0, 1.0]], (64, 1)),
    )
    m.collect_case(
        truth, public, ledger, "c00-fp64", "fp64", ctx, lambda x: np.ones((len(x), 2))
    )
    assert (
        ledger.counts["calls_persisted"] == 24
        and ledger.counts["callback_batches"] == 2
    )
    assert len(public.names) == 3 and all(
        any("-" + k + "-" in name for k in ("fine", "coarse", "richardson"))
        for name in public.names
    )
    assert any("context" in name for name in truth.names)


@pytest.mark.parametrize(
    "failure", ["approval", "policy", "scope", "transport", "marker"]
)
def test_factory_is_blocked_before_entropy_or_teacher(tmp_path, monkeypatch, failure):
    m = load(monkeypatch)
    repo = tmp_path / "repository"
    repo.mkdir()
    (tmp_path / "experiments").mkdir()
    root = tmp_path / "run"
    root.mkdir()
    good = dict(
        process_monitor_scope="process_and_children", memory_samples=1, status="passed"
    )
    request = dict(
        root=str(root),
        repo=str(repo),
        policy=m.POLICY,
        full_source_protocol_approved=True,
        source_commit="0" * 40,
        config_sha256="1" * 64,
        scope_preflight=dict(good),
        transport_preflight=dict(good),
    )
    marker = tmp_path / "experiments" / m.MARKER
    marker.write_text(
        json.dumps(
            dict(
                policy=m.POLICY,
                source_commit=request["source_commit"],
                config_sha256=request["config_sha256"],
                output=str(root),
            )
        )
    )
    if failure == "approval":
        request["full_source_protocol_approved"] = False
    elif failure == "policy":
        request["policy"] = {}
    elif failure == "scope":
        request["scope_preflight"]["process_monitor_scope"] = "process_only"
    elif failure == "transport":
        request["transport_preflight"]["status"] = "failed"
    else:
        marker.write_text("{}")
    monkeypatch.setattr(m.os, "urandom", lambda *a: pytest.fail("Entropy before gates"))
    monkeypatch.setattr(
        m.np.random,
        "default_rng",
        lambda *a, **k: pytest.fail("Teacher RNG before gates"),
    )
    with pytest.raises(ValueError):
        m.build(request)
    assert not (root / "collection").exists()


def test_hash_assignment_is_deterministic_and_does_not_expose_width_in_case_names(
    monkeypatch,
):
    m = load(monkeypatch)
    a = m.assignment(bytes(range(32)))
    assert a == m.assignment(bytes(range(32))) and len(a) == 12
    assert [x["id"] for x in a] == [f"c{i:02d}" for i in range(12)]
    for i, n in enumerate((4, 8, 16, 32)):
        assert {x["n"] for x in a[3 * i : 3 * i + 3]} == {n}
        assert len({x["width"] for x in a[3 * i : 3 * i + 3]}) == 3


def test_callback_input_shape_mutation_cannot_change_accounting(tmp_path, monkeypatch):
    m = load(monkeypatch)
    store = m.ShardStore(tmp_path / "arrays")
    ledger = m.RawCounter(tmp_path / "progress.json", 4)

    def callback(x):
        with pytest.warns(DeprecationWarning):
            x.shape = (2, 4)
        return np.ones((4, 2))

    with pytest.raises(ValueError):
        ledger.query(store, "literal", np.zeros((4, 2)), callback, 2)
    assert ledger.terminal and ledger.counts["calls_returned"] == 4


def test_context_mutation_marks_collection_terminal(tmp_path, monkeypatch):
    m = load(monkeypatch)
    truth = m.ShardStore(tmp_path / "truth")
    public = m.ShardStore(tmp_path / "public")
    ledger = m.RawCounter(tmp_path / "progress.json", 24)
    ctx = dict(
        g=np.zeros((3, 2)),
        u=np.zeros((3, 2)),
        d=np.ones((2, 3)),
        probes=np.zeros((1, 2)),
        random_directions=np.tile([[1.0, 1.0]], (64, 1)),
    )

    def callback(x):
        ctx["g"].setflags(write=True)
        ctx["g"][0, 0] = 7
        return np.ones((len(x), 2))

    with pytest.raises(ValueError):
        m.collect_case(truth, public, ledger, "c00-fp64", "fp64", ctx, callback)
    assert ledger.terminal


def test_context_key_deletion_is_terminal(tmp_path, monkeypatch):
    m = load(monkeypatch)
    truth, public = m.ShardStore(tmp_path / "truth"), m.ShardStore(tmp_path / "public")
    ledger = m.RawCounter(tmp_path / "progress.json", 24)
    ctx = dict(
        g=np.zeros((3, 2)),
        u=np.zeros((3, 2)),
        d=np.ones((2, 3)),
        probes=np.zeros((1, 2)),
        random_directions=np.ones((64, 2)),
    )

    def callback(x):
        del ctx["random_directions"]
        return np.ones((len(x), 2))

    with pytest.raises(ValueError, match="context mutation"):
        m.collect_case(truth, public, ledger, "c00-fp64", "fp64", ctx, callback)
    assert ledger.terminal and ledger.counts["callback_batches"] == 1
