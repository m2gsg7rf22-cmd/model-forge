import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest


def load(monkeypatch, name):
    b = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(b))
    p = b / (name + ".py")
    assert p.exists(), "Missing new row-acquisition source: " + name
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, m)
    spec.loader.exec_module(m)
    return m


def test_exact_new_inventory_keeps_v1_frozen(monkeypatch):
    m = load(monkeypatch, "row_acquisition_inventory")
    p = m.inventory_plan()
    assert p["scientific_shards"] == 522
    assert p["vector_calls"] == 430848 and p["callback_batches"] == 204
    assert p["bytes_upper_bound"] == 534980568
    assert p["preflight_bytes_upper_bound"] == 31403888
    assert p["preflight_shards"] == 10
    assert [
        p["stores"][n]["shards"] for n in ("truth", "public", "estimator", "evaluator")
    ] == [432, 54, 24, 12]
    from fresh_rank_protocol_inventory import inventory_plan

    assert inventory_plan()["bytes_upper_bound"] == 480302932


def fixture(monkeypatch, tmp_path, extra=None):
    m = load(monkeypatch, "row_acquisition_view")
    from fresh_rank_common import save_json, sha
    from scaling_domain_store import ShardStore

    root = tmp_path / "public"
    root.mkdir()
    store = ShardStore(root / "arrays")
    name = "c00-fp64"
    h = np.tile(np.eye(2), (17, 1, 1, 1))
    h[-1] *= 3
    for kind in ("fine", "coarse", "richardson"):
        for start in (0, 8, 16):
            store.save(
                f"{name}-{kind}-h{start // 8:02d}.npz", hessian=h[start : start + 8]
            )
    case = dict(name=name, n=2, outputs=1, probes=17, compute="fp64")
    if extra:
        case.update(extra)
    save_json(
        root / "cases.json",
        dict(
            schema="modelforge-row-acquisition-observations-v1",
            steps=[0.3, 0.6],
            cases=[case],
        ),
    )
    return m, root, sha(root / "cases.json"), sha(root / "arrays/manifest.jsonl")


def test_public_prefixes_share_exact_source_archives(tmp_path, monkeypatch):
    m, root, c, h = fixture(monkeypatch, tmp_path)
    v = m.RowView(root, c, h)
    assert [x.probes for x in v.cases] == [16, 17]
    a = v.packed(v.cases[0].name)["richardson"]
    assert len(v.used) == 6
    b = v.packed(v.cases[1].name)["richardson"]
    np.testing.assert_array_equal(a, b[:16])
    assert not np.array_equal(a[-1], b[-1]) and len(v.used) == 9
    assert v.verify_unchanged()["inputs_unchanged"] is True


@pytest.mark.parametrize(
    "extra", [dict(width=48), dict(seed=1), dict(probes=16), dict(probes=True)]
)
def test_public_metadata_rejects_privilege_or_wrong_probe_count(
    tmp_path, monkeypatch, extra
):
    m, root, c, h = fixture(monkeypatch, tmp_path, extra)
    with pytest.raises(ValueError):
        m.RowView(root, c, h)


def test_unregistered_small_view_rejected_before_selection(tmp_path, monkeypatch):
    m, root, c, h = fixture(monkeypatch, tmp_path)
    v = m.RowView(root, c, h)
    with pytest.raises(ValueError):
        m.registered_view(v)


def test_17_probe_supplied_context_stores_tail_and_all_calls(tmp_path, monkeypatch):
    m = load(monkeypatch, "row_acquisition_collection")
    from scaling_domain_store import ShardStore

    truth, public = ShardStore(tmp_path / "truth"), ShardStore(tmp_path / "public")
    ledger = m.RawCounter(tmp_path / "progress.json", 408)
    ctx = dict(
        g=np.zeros((3, 2)),
        u=np.zeros((3, 2)),
        d=np.ones((2, 3)),
        probes=np.zeros((17, 2)),
        random_directions=np.ones((64, 2)),
    )
    m.collect_case(
        truth, public, ledger, "c00-fp64", "fp64", ctx, lambda x: np.ones((len(x), 2))
    )
    assert (
        ledger.counts["calls_persisted"] == 408
        and ledger.counts["callback_batches"] == 34
    )
    assert len(public.names) == 9
    with np.load(public.root / "c00-fp64-fine-h02.npz") as f:
        assert f["hessian"].shape == (1, 2, 2, 2)


def test_new_estimator_seals_both_conditions_from_shared_bytes(tmp_path, monkeypatch):
    m = load(monkeypatch, "row_acquisition_estimation")
    vm, root, c, h = fixture(monkeypatch, tmp_path)
    view = vm.RowView(root, c, h)
    from scaling_domain_store import ShardStore

    out = tmp_path / "estimator"
    out.mkdir()
    r = m.estimate(view, ShardStore(out / "arrays"), out, registered=False)
    assert (
        len(r["selections"]) == 2
        and len(r["input_commitments"]["used_input_sha256"]) == 9
    )
    assert [x["meta"]["probes"] for x in r["selections"]] == [16, 17]


def test_544_row_metric_accepts_real_capacity_but_full_is_failure(monkeypatch):
    m = load(monkeypatch, "row_acquisition_evaluation")
    a = np.zeros((544, 3))
    a[0, 0] = 1.0
    ref = m.reference_summary(a)
    assert ref["rank"] == 1 and ref["all_right"].shape == (3, 3)
    assert m.qualify(ref, 3, 1, 0.0, 0.0, 0.0, 1.0) is True
    full = m.reference_summary(np.tile(np.eye(3), (181, 1)))
    assert m.qualify(full, 3, None, None, None, None, None) is False
    with pytest.raises(ValueError):
        m.reference_summary(np.zeros((545, 3)))


@pytest.mark.parametrize("failure", ["approval", "marker", "scope", "inventory"])
def test_new_factory_cannot_pass_gates_with_forged_v1_request(
    tmp_path, monkeypatch, failure
):
    m = load(monkeypatch, "row_acquisition_collection")
    from row_acquisition_inventory import inventory_plan
    from row_acquisition_common import sha

    repo = tmp_path / "repo"
    repo.mkdir()
    ex = tmp_path / "experiments"
    ex.mkdir()
    root = ex / "run"
    root.mkdir()
    marker = ex / m.MARKER
    value = dict(
        policy=m.POLICY,
        source_commit="a" * 40,
        config_sha256="b" * 64,
        output=str(root),
    )
    marker.write_text(json.dumps(value))
    good = dict(
        status="passed", process_monitor_scope="process_and_children", memory_samples=1
    )
    req = dict(
        root=str(root),
        repo=str(repo),
        policy=m.POLICY,
        full_source_protocol_approved=True,
        source_commit="a" * 40,
        config_sha256="b" * 64,
        marker_sha256=sha(marker),
        scope_preflight=dict(good),
        transport_preflight=dict(good),
        inventory_plan=inventory_plan(),
    )
    if failure == "approval":
        req["full_source_protocol_approved"] = False
    elif failure == "marker":
        marker.write_text("{}")
    elif failure == "scope":
        req["scope_preflight"]["process_monitor_scope"] = "process_only"
    else:
        req["inventory_plan"] = {}
    monkeypatch.setattr(
        m.os, "urandom", lambda *a: pytest.fail("Entropy before full gates")
    )
    monkeypatch.setattr(
        m.np.random,
        "default_rng",
        lambda *a, **kw: pytest.fail("RNG before full gates"),
    )
    with pytest.raises(ValueError):
        m.build(req)


def test_reference_uses_own_16_or_17_prefix(monkeypatch):
    m = load(monkeypatch, "row_acquisition_evaluation")
    from scaling_domain_math import svec

    h = np.zeros((17, 1, 2, 2))
    h[0, 0, 0, 0] = 1.0
    h[16, 0, 1, 1] = 1.0
    ctx = dict(reference_g=np.eye(2), random_directions=np.tile([[1.0, 1.0]], (64, 1)))
    ranks = []
    for budget in (16, 17):
        obs = svec(h[:budget]).reshape(budget, 3)
        _, _, v = np.linalg.svd(obs, full_matrices=False)
        choices = {k: dict(rank=budget - 15) for k in ("strict", "gap", "stable")}
        result, arrays = m.condition_evaluation(h, obs, v, choices, ctx, budget)
        ranks.append(result["reference_rank"])
    assert ranks == [1, 2]
    with pytest.raises(ValueError):
        m.condition_evaluation(h, obs, v, choices, ctx, 18)


def test_all_public_condition_seals_checked_without_truth_open(tmp_path, monkeypatch):
    em = load(monkeypatch, "row_acquisition_estimation")
    ev = load(monkeypatch, "row_acquisition_evaluation")
    vm, root, c, h = fixture(monkeypatch, tmp_path)
    view = vm.RowView(root, c, h)
    from scaling_domain_store import ShardStore

    out = tmp_path / "estimator"
    out.mkdir()
    r = em.estimate(view, ShardStore(out / "arrays"), out, registered=False)
    seal = json.loads((out / "choices-sealed.json").read_text())
    ev.validate_public_seal(out, view, seal, r)
    seal["selections"][1]["meta"]["probes"] = 16
    r["selections"] = seal["selections"]
    with pytest.raises(ValueError):
        ev.validate_public_seal(out, view, seal, r)


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "summary",
        "duplicate",
        "probe",
        "paired",
        "reference",
        "boolean",
        "summary_boolean",
    ],
)
def test_controller_recomputes_paired_summary_from_exact_cases(monkeypatch, mutation):
    m = load(monkeypatch, "row_acquisition_controller")
    mapping = [
        dict(id=f"c{i:02d}", n=32, width=w, kind="generic")
        for i, w in enumerate((16, 48, 96))
    ]
    contexts = {
        f"c{i:02d}-{c}": str(i) * 64 for i in range(3) for c in ("fp64", "fp32")
    }
    cases = []
    for teacher in mapping:
        for channel in ("fp64", "fp32"):
            base = teacher["id"] + "-" + channel
            for budget in (16, 17):
                cases.append(
                    dict(
                        case=base + f"-p{budget}",
                        base_case=base,
                        probes=budget,
                        width_for_evaluation_only=teacher["width"],
                        teacher_kind="generic",
                        shared_context_sha256=contexts[base],
                        reference_rank=32,
                        reference_ranks=[32, 32, 32],
                        reference_sensitive=False,
                        certificate=False,
                        methods=[
                            dict(
                                method=k, qualifies=False, rank=None, certificate=False
                            )
                            for k in (
                                "strict",
                                "gap",
                                "stable",
                                "known_width",
                                "wrong_width_half",
                                "wrong_width_double",
                            )
                        ],
                    )
                )
    pairs = []
    for i, base in enumerate(contexts):
        pairs.append(
            dict(
                base_case=base,
                reference_ranks16=[32, 32, 32],
                reference_ranks17=[32, 32, 32],
                reference_sensitive16=False,
                reference_sensitive17=False,
                rules={k: dict(pass16=False, pass17=False) for k in ("gap", "stable")},
            )
        )
    r = dict(
        cases=cases,
        paired=pairs,
        by_budget={str(b): dict(gap=False, stable=False) for b in (16, 17)},
        rule_passes=dict(gap=False, stable=False),
    )
    if mutation == "summary":
        r["rule_passes"]["gap"] = True
    if mutation == "duplicate":
        r["cases"][1] = r["cases"][0]
    if mutation == "probe":
        r["cases"][1]["probes"] = 16
    if mutation == "paired":
        r["paired"][0]["rules"]["gap"]["pass17"] = True
    if mutation == "reference":
        r["cases"][0]["reference_rank"] = 96
    if mutation == "summary_boolean":
        r["rule_passes"]["gap"] = 0
    if mutation == "boolean":
        r["cases"][0]["methods"][0]["qualifies"] = 0
    if mutation == "none":
        m.validate_evaluation_report(r, mapping, contexts)
    else:
        with pytest.raises(ValueError):
            m.validate_evaluation_report(r, mapping, contexts)
