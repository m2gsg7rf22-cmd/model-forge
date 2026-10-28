import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest


def load(monkeypatch, name):
    base = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(base))
    p = base / (name + ".py")
    assert p.exists(), "Fresh stage absent: " + name
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, m)
    spec.loader.exec_module(m)
    return m


def public_fixture(monkeypatch, tmp_path):
    v = load(monkeypatch, "fresh_rank_view")
    from scaling_domain_store import ShardStore, sha

    root = tmp_path / "public"
    root.mkdir()
    store = ShardStore(root / "arrays")
    cases = []
    for name in ("c00-fp64", "c00-fp32"):
        for kind in ("fine", "coarse", "richardson"):
            store.save(
                name + "-" + kind + "-h00.npz", hessian=np.tile(np.eye(2), (2, 1, 1, 1))
            )
        cases.append(
            dict(
                name=name,
                n=2,
                outputs=1,
                probes=2,
                compute="fp64" if name.endswith("fp64") else "fp32_compute",
            )
        )
    (root / "cases.json").write_text(
        json.dumps(
            dict(
                schema="modelforge-fresh-rank-observations-v1",
                steps=[0.3, 0.6],
                cases=cases,
            )
        )
    )
    return v.FDView(root, sha(root / "cases.json"), sha(root / "arrays/manifest.jsonl"))


def test_all_public_choices_sealed_without_adjacent_truth_read(tmp_path, monkeypatch):
    m = load(monkeypatch, "fresh_rank_estimation")
    view = public_fixture(monkeypatch, tmp_path)
    from scaling_domain_store import ShardStore, sha

    truth = tmp_path / "truth.json"
    truth.write_text("MUST NOT OPEN")
    original = Path.open

    def guarded(path, *a, **kw):
        if path == truth:
            pytest.fail("Estimator read truth")
        return original(path, *a, **kw)

    out = tmp_path / "estimator"
    out.mkdir()
    store = ShardStore(out / "arrays")
    monkeypatch.setattr(Path, "open", guarded)
    r = m.estimate(view, store, out, registered=False)
    seal = json.loads((out / "choices-sealed.json").read_text())
    assert len(seal["selections"]) == 2 and r["seal_sha256"] == sha(
        out / "choices-sealed.json"
    )
    assert len(seal["input_commitments"]["used_input_sha256"]) == 6
    assert len(store.names) == 4 and r["certificate"] is False


def test_registered_public_schema_rejects_small_fixture_before_svd(
    tmp_path, monkeypatch
):
    m = load(monkeypatch, "fresh_rank_estimation")
    view = public_fixture(monkeypatch, tmp_path)
    from scaling_domain_store import ShardStore

    out = tmp_path / "estimator"
    out.mkdir()
    store = ShardStore(out / "arrays")
    monkeypatch.setattr(
        m, "select", lambda *a: pytest.fail("SVD before registered metadata check")
    )
    with pytest.raises(ValueError):
        m.estimate(view, store, out)


def test_estimator_failure_never_creates_complete_seal(tmp_path, monkeypatch):
    m = load(monkeypatch, "fresh_rank_estimation")
    view = public_fixture(monkeypatch, tmp_path)
    from scaling_domain_store import ShardStore

    out = tmp_path / "estimator"
    out.mkdir()
    store = ShardStore(out / "arrays")
    monkeypatch.setattr(
        m, "select", lambda *a: (_ for _ in ()).throw(ValueError("injected"))
    )
    with pytest.raises(ValueError):
        m.estimate(view, store, out, registered=False)
    assert not (out / "choices-sealed.json").exists()


def test_sensitive_full_reference_cannot_pass_on_null_candidate(monkeypatch):
    m = load(monkeypatch, "fresh_rank_evaluation")
    ref = np.diag([1.0, 1.0, 5e-10])
    r = m.reference_summary(ref)
    assert r["ranks"] == [3, 3, 2] and r["sensitive"] is True
    assert m.qualify(r, 3, None, None, None, None, None) is False


def test_full_reference_accepts_only_null_and_zero_reference_fails(monkeypatch):
    m = load(monkeypatch, "fresh_rank_evaluation")
    full = m.reference_summary(np.eye(3))
    assert m.qualify(full, 3, None, None, None, None, None) is True
    assert m.qualify(full, 3, 3, 0.0, 0.0, 0.0, 1.0) is False
    zero = m.reference_summary(np.zeros((3, 3)))
    assert m.qualify(zero, 3, None, None, None, None, None) is False


def test_energy_loss_cannot_hide_weak_geometric_direction(monkeypatch):
    m = load(monkeypatch, "fresh_rank_evaluation")
    a = np.diag([1.0, 0.0, 1e-5])
    basis = np.array([[1.0, 0.0, 0.0]])
    r = m.reference_summary(a)
    assert r["rank"] == 2 and r["sensitive"] is False
    assert m.energy_loss(a, basis) == pytest.approx(1e-5, rel=1e-9)
    assert m.qualify(r, 3, 1, np.sqrt(0.5), 0.0, 0.0, 1.0) is False


def test_literal_metrics_qualify_without_any_rank_certificate(monkeypatch):
    m = load(monkeypatch, "fresh_rank_evaluation")
    a = np.array([[1.0, 0.0, 0.0]])
    basis = np.array([[1.0, 0.0, 0.0]])
    choices = {k: dict(rank=1) for k in ("strict", "gap", "stable")}
    report, arrays = m.evaluate_case(
        a, a, basis, choices, np.array([[1.0, 0.0]]), np.tile([[0.0, 1.0]], (64, 1))
    )
    assert report["certificate"] is False
    assert all(x["qualifies"] for x in report["methods"][:3])
    assert set(arrays) == {
        "reference_spectrum",
        "reference_ranks",
        "ranks",
        "gate_residuals",
        "random_residuals",
        "lost_subspace",
        "extra_subspace",
        "reference_relative_residual",
        "observed_relative_residual",
        "reference_component_residuals",
        "qualifies",
        "reference_sensitive",
    }
    assert all(a.dtype == np.float64 and np.isfinite(a).all() for a in arrays.values())


def test_energy_is_globally_scale_invariant_and_zero_is_null(monkeypatch):
    m = load(monkeypatch, "fresh_rank_evaluation")
    a = np.diag([1.0, 0.2, 0.0])
    b = np.array([[1.0, 0.0, 0.0]])
    for scale in (1e-200, 1.0, 1e200):
        assert m.energy_loss(a * scale, b) == pytest.approx(0.2 / np.sqrt(1.04))
    assert m.energy_loss(np.zeros((3, 3)), b) is None


def test_component_amplitudes_square_to_energy_fractions(monkeypatch):
    m = load(monkeypatch, "fresh_rank_evaluation")
    b = np.array([[1.0, 0.0, 0.0]])
    for scale in (1e-200, 1.0, 1e200):
        a = np.diag([3.0, 4.0, 0.0]) * scale
        ref = m.reference_summary(a)
        c = m.component_residuals(ref, b)
        assert c.shape == (3,)
        assert sorted(c) == pytest.approx([0.0, 0.0, 0.8])
        assert float(c @ c) == pytest.approx(0.64)
        assert np.linalg.norm(c) == pytest.approx(m.energy_loss(a, b))
    assert m.component_residuals(m.reference_summary(np.zeros((1, 3))), b) is None


def test_protocol_inventory_adds_components_preserving_frozen_base(monkeypatch):
    m = load(monkeypatch, "fresh_rank_protocol_inventory")
    base = m.base
    plan = m.inventory_plan()
    assert plan["bytes_upper_bound"] == 480302932
    assert plan["preflight_bytes_upper_bound"] == 17621380
    assert plan["stores"]["evaluator"]["bytes_upper_bound"] == 1531968
    assert plan["stores"]["evaluator"]["shard_limit_bytes"] == 86720
    assert plan["scientific_shards"] == 1824
    for n, widths in base.PROFILES:
        for width in widths:
            old = base.profile_schemas(n, width)
            new = m.profile_schemas(n, width)
            assert old[:-1] == new[:-1]
            fields = new[-1].fields
            assert len(fields) == 12
            assert fields["reference_component_residuals"] == (6, n * (n + 1) // 2)
    assert base.inventory_plan()["bytes_upper_bound"] == 479970708


@pytest.mark.parametrize(
    "mutation", ["case", "index", "meta", "rank", "descriptor", "projector", "none"]
)
def test_public_seal_validation_precedes_truth(tmp_path, monkeypatch, mutation):
    est = load(monkeypatch, "fresh_rank_estimation")
    ev = load(monkeypatch, "fresh_rank_evaluation")
    view = public_fixture(monkeypatch, tmp_path)
    from scaling_domain_store import ShardStore

    out = tmp_path / "estimator"
    out.mkdir()
    report = est.estimate(view, ShardStore(out / "arrays"), out, registered=False)
    seal = json.loads((out / "choices-sealed.json").read_text())
    sel = seal["selections"][1]
    if mutation == "case":
        sel["case"] = "wrong"
    if mutation == "index":
        sel["index"] = 0
    if mutation == "meta":
        sel["meta"]["outputs"] = 2
    if mutation == "rank":
        sel["choices"]["strict"]["rank"] = True
    if mutation == "descriptor":
        sel["selected"]["sha256"] = "0" * 64
    if mutation == "projector":
        sel["choices"]["strict"]["rank"] = 0
    report["selections"] = seal["selections"]
    if mutation == "none":
        ev.validate_public_seal(out, view, seal, report)
    else:
        with pytest.raises(ValueError):
            ev.validate_public_seal(out, view, seal, report)


def test_component_norm_keeps_representable_tiny_residual(monkeypatch):
    m = load(monkeypatch, "fresh_rank_evaluation")
    ref = m.reference_summary(np.diag([1.0, 0.0]))
    b = np.array([[1.0, 1e-200]])
    c = m.component_residuals(ref, b)
    assert c[0] == pytest.approx(1e-200, rel=1e-12, abs=0)
    assert c[0] == m.energy_loss(np.diag([1.0, 0.0]), b)


def test_component_weighted_underflow_fails_instead_of_zero(monkeypatch):
    m = load(monkeypatch, "fresh_rank_evaluation")
    ref = m.reference_summary(np.diag([1.0, 1e-200]))
    with pytest.raises(ValueError, match="underflow"):
        m.component_residuals(ref, np.array([[1e-200, 1.0]]))
