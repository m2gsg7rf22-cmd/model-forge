import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load(monkeypatch):
    root = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(root))
    p = root / "scaling_domain_run.py"
    assert p.exists(), "Full sequential dimension runner absent"
    spec = importlib.util.spec_from_file_location("scaling_domain_run", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fixed_shape_inventory_count_and_budget(monkeypatch):
    m = load(monkeypatch)
    p = m.inventory_plan()
    assert p["vector_calls"] == 1454080 and p["shards"] == 4704 and p["cases"] == 16
    assert p["bytes_upper_bound"] < 1024**3
    assert p["manifest_bytes_upper_bound"] < 1024**2
    assert p["largest_shard_upper_bound"] <= 8 * 1024**2


def context():
    return dict(
        g=np.array([[0.2, 0.3], [0.1, -0.2]]),
        u=np.array([[0.3, -0.1], [0.4, 0.2]]),
        d=np.array([[0.1, 0.2], [-0.2, 0.3]]),
        probes=np.array([[0.1, 0.2], [-0.3, 0.4]]),
        random_directions=np.array([[1.0, 0.0], [1.0, 1.0]]),
    )


def test_literal_case_raw_inventory_metrics_and_counts(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s = m.ShardStore(tmp_path / "arrays")
    q = m.QueryLedger(tmp_path / "progress.json", 48)
    c = context()
    seen = []

    def query(x):
        # All signed inputs persisted BEFORE query attempt.
        assert len(list(s.root.glob("*inputs.npz"))) == len(seen) + 1
        assert json.loads(q.path.read_text())["calls_attempted"] == 12 * (len(seen) + 1)
        seen.append(x.copy())
        return m.forward(c["g"], c["u"], c["d"], x, "fp64")

    r = m.run_case(s, q, "literal", query=query, channel="fp64", **c)
    assert q.counts == dict(
        calls_reserved=48, calls_attempted=48, calls_returned=48, calls_persisted=48
    )
    assert r["n"] == 2 and r["p"] == 3 and len(r["spaces"]) == 5
    assert len(list(s.root.glob("*.npz"))) == 18
    assert m.verify_manifest(s.root)["shards"] == 18
    for a in seen:
        assert a.shape == (12, 2)
    for k in ["fine", "coarse", "richardson"]:
        assert np.isfinite(r["spaces"][k]["hessian_relative_error"])


def test_query_failure_preserves_inputs_attempted_and_prior_responses(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    s = m.ShardStore(tmp_path / "arrays")
    q = m.QueryLedger(tmp_path / "progress.json", 48)
    c = context()
    calls = []

    def query(x):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected query")
        return m.forward(c["g"], c["u"], c["d"], x, "fp64")

    with pytest.raises(RuntimeError, match="injected query"):
        m.run_case(s, q, "literal", query=query, channel="fp64", **c)
    assert q.counts["calls_attempted"] == 24
    assert q.counts["calls_returned"] == q.counts["calls_persisted"] == 12
    assert len(list(s.root.glob("*responses.npz"))) == 1
    assert len(list(s.root.glob("*inputs.npz"))) == 2
    assert not list(s.root.glob("*spectrum*.npz"))


def test_mutated_inputs_are_detected_after_returned_raw_is_preserved(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    s = m.ShardStore(tmp_path / "arrays")
    q = m.QueryLedger(tmp_path / "progress.json", 48)

    def mutate(x):
        x.setflags(write=True)
        x[:] = 7
        return np.ones((len(x), 2))

    with pytest.raises(ValueError, match="mutation"):
        m.run_case(s, q, "literal", query=mutate, channel="fp64", **context())
    assert q.counts["calls_persisted"] == 12
    p = next(s.root.glob("*inputs.npz"))
    with np.load(p, allow_pickle=False) as z:
        assert not np.all(z["inputs"] == 7)
    assert len(list(s.root.glob("*responses.npz"))) == 1


def test_query_reservation_is_strict_and_failed_persistence_not_counted(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    s = m.ShardStore(tmp_path / "arrays")
    q = m.QueryLedger(tmp_path / "progress.json", 12)
    x = np.ones((12, 2))

    def fail(name, **arrays):
        if name.endswith("responses.npz"):
            raise OSError("injected persistence")
        return original(name, **arrays)

    original = s.save
    monkeypatch.setattr(s, "save", fail)
    with pytest.raises(OSError):
        q.query(s, "literal", x, lambda a: a.copy(), 2)
    assert q.counts["calls_attempted"] == q.counts["calls_returned"] == 12
    assert q.counts["calls_persisted"] == 0
    with pytest.raises(ValueError, match="reservation"):
        q.query(s, "other", x, lambda a: pytest.fail("No retry"), 2)


def test_bad_channel_rejected_before_query_or_context_shard(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s = m.ShardStore(tmp_path / "arrays")
    q = m.QueryLedger(tmp_path / "progress.json", 48)
    with pytest.raises(ValueError):
        m.run_case(
            s,
            q,
            "literal",
            query=lambda a: pytest.fail("No query"),
            channel="bad",
            **context(),
        )
    assert not list(s.root.glob("*.npz"))


def gated_request(m, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "experiments").mkdir()
    receipt = dict(
        process_monitor_scope="process_and_children", memory_samples=1, status="passed"
    )
    return dict(
        output=str(tmp_path / "worker"),
        repo=str(repo),
        policy=m.POLICY,
        full_source_protocol_approved=True,
        scope_preflight=receipt,
        transport_preflight=receipt,
        source_commit="synthetic-not-actual",
        config_sha256="0" * 64,
    )


@pytest.mark.parametrize(
    "gate", ["policy", "source", "tree", "transport", "existing_marker"]
)
def test_registered_factory_never_called_after_invalid_gate(
    tmp_path, monkeypatch, gate
):
    m = load(monkeypatch)
    request = gated_request(m, tmp_path)
    monkeypatch.setattr(
        m, "registered_contexts", lambda: pytest.fail("No registered generation")
    )
    if gate == "policy":
        request["policy"] = {}
    elif gate == "source":
        request["full_source_protocol_approved"] = False
    elif gate == "tree":
        request["scope_preflight"] = {
            "process_monitor_scope": "process_only_children_unavailable",
            "memory_samples": 1,
        }
    elif gate == "transport":
        request["transport_preflight"] = {
            **request["transport_preflight"],
            "status": "failed",
        }
    else:
        (
            tmp_path / "experiments/isolated-glu-dimension-domain-v1-attempt.json"
        ).write_text("partial previous marker")
    with pytest.raises((ValueError, FileExistsError)):
        m.build(request)
    r = json.loads((tmp_path / "worker/worker-report.json").read_text())
    assert r["status"] == "failed" and r["calls_reserved"] is None


def test_partial_marker_serialization_blocks_new_output_and_generation(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    request = gated_request(m, tmp_path)
    monkeypatch.setattr(
        m, "registered_contexts", lambda: pytest.fail("No registered generation")
    )

    def fail(*a, **k):
        raise TypeError("injected marker serialization")

    monkeypatch.setattr(m.json, "dump", fail)
    with pytest.raises(TypeError):
        m.build(request)
    marker = tmp_path / "experiments/isolated-glu-dimension-domain-v1-attempt.json"
    assert marker.exists()
    request["output"] = str(tmp_path / "other")
    with pytest.raises(FileExistsError):
        m.build(request)


def test_oversized_callback_input_fails_before_copy(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s = m.ShardStore(tmp_path / "arrays")
    q = m.QueryLedger(tmp_path / "progress.json", 1454080)
    x = np.broadcast_to(np.ones(1), (1000000, 32))
    original = m.np.array

    def guard(a, *args, **kwargs):
        if a is x:
            pytest.fail("Oversized query copied before guard")
        return original(a, *args, **kwargs)

    monkeypatch.setattr(m.np, "array", guard)
    with pytest.raises(ValueError):
        q.query(s, "oversized", x, lambda a: pytest.fail("No query"), 32)
    assert q.counts["calls_attempted"] == 0


def test_query_cannot_mutate_privileged_context_unnoticed(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s = m.ShardStore(tmp_path / "arrays")
    q = m.QueryLedger(tmp_path / "progress.json", 48)
    c = context()

    def mutate(x):
        c["g"][:] = 0.7
        return np.ones((len(x), 2))

    with pytest.raises(ValueError, match="context mutation"):
        m.run_case(s, q, "literal", query=mutate, channel="fp64", **c)
    assert q.counts["calls_persisted"] == 12


def test_oversized_or_complex_callback_output_rejected_before_copy(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    s = m.ShardStore(tmp_path / "arrays")
    q = m.QueryLedger(tmp_path / "progress.json", 12)
    huge = np.broadcast_to(np.ones(1), (1_000_000_000, 2))
    original = m.np.array

    def guard(a, *args, **kwargs):
        if a is huge:
            pytest.fail("Huge callback output copied before guard")
        return original(a, *args, **kwargs)

    monkeypatch.setattr(m.np, "array", guard)
    with pytest.raises(ValueError):
        q.query(s, "huge", np.ones((12, 2)), lambda a: huge, 2)
    assert q.counts["calls_returned"] == 12 and q.counts["calls_persisted"] == 0
    assert q.terminal
    r = json.loads(q.path.read_text())
    assert r["status"] == "failed"


def test_complex_output_never_silently_discards_imaginary_part(tmp_path, monkeypatch):
    m = load(monkeypatch)
    s = m.ShardStore(tmp_path / "arrays")
    q = m.QueryLedger(tmp_path / "progress.json", 12)
    with pytest.raises(ValueError):
        q.query(
            s,
            "complex",
            np.ones((12, 2)),
            lambda a: np.ones((12, 2), dtype=complex) * (1 + 2j),
            2,
        )
    assert q.counts["calls_persisted"] == 0


def test_relative_ratio_underflow_is_not_false_zero(monkeypatch):
    m = load(monkeypatch)
    truth = np.ones(10)
    truth[-1] = 0
    estimate = truth.copy()
    estimate[-1] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="underflow"):
        m._relative(estimate, truth)
