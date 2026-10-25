"""V4 boundaries; uses only the already public development fixture."""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "examples/isolated_glu/sealed_v4.py"
    spec = importlib.util.spec_from_file_location("sealed_glu_v4", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def direction_fixture(m, public, count):
    rng = np.random.default_rng(572041)
    d = rng.normal(size=(count, 32))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    all_d = np.tile(d, (256 // count + 1, 1))[:256] if count else np.zeros((256, 32))
    np.savez(
        public / "refined-solution.npz",
        directions=d,
        all_directions=all_d,
        residual=np.zeros(256),
    )
    m.write(public / "refine.json", {"directions_retained": count})


def test_direction64_skips_without_launching_rescue(tmp_path, monkeypatch):
    m = module()
    direction_fixture(m, tmp_path, 64)
    before = (tmp_path / "refined-solution.npz").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Skip path launched rescue")

    monkeypatch.setattr(m, "run_json", forbidden)
    result = m.select_directions(tmp_path)
    assert result["branch"] == "skip64"
    assert (tmp_path / "refined-solution.npz").read_bytes() == before
    assert "rescue/report.json" not in m.audit_names(tmp_path)
    with pytest.raises(FileExistsError):
        m.select_directions(tmp_path)


@pytest.mark.parametrize("count", [0, 65])
def test_other_counts_fail_before_rescue(tmp_path, monkeypatch, count):
    m = module()
    direction_fixture(m, tmp_path, count)
    monkeypatch.setattr(m, "run_json", lambda *a, **k: pytest.fail("Unexpected rescue"))
    with pytest.raises(ValueError, match="1..64"):
        m.select_directions(tmp_path)


def test_v4_readiness_precedes_entropy(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m.secrets, "randbits", lambda *a: pytest.fail("Early entropy"))
    with pytest.raises(ValueError, match="readiness"):
        m.prepare(
            tmp_path / "public",
            tmp_path / "private",
            0,
            readiness=tmp_path / "missing.json",
        )
    assert not (tmp_path / "private").exists()


def test_v4_repeat_consumption_precedes_any_read(tmp_path, monkeypatch):
    m = module()
    (tmp_path / "evaluation-consumed.json").write_text("{}")
    monkeypatch.setattr(m, "read", lambda *a: pytest.fail("Read before repeat refusal"))
    with pytest.raises(FileExistsError):
        m.verify(tmp_path / "public", tmp_path)


@pytest.fixture(scope="module")
def known_case_v4(tmp_path_factory):
    m = module()
    root = tmp_path_factory.mktemp("v4-owned-public-fixture")
    public, private = root / "public", root / "private"
    rng = np.random.default_rng(572041)
    t = {
        "gate": rng.normal(size=(64, 32)) * 0.2,
        "up": rng.normal(size=(64, 32)) * 0.2,
        "down": rng.normal(size=(32, 64)) * 0.2,
    }
    rng = np.random.default_rng(73191)
    held = np.stack([rng.normal(size=(512, 32)) * s for s in [0.04, 1, 4, 30]])
    m.materialize(public, private, 1, t, held)
    g = t["gate"]
    c = np.einsum("ow,wd->wod", t["down"], t["up"])
    directions = g / np.linalg.norm(g, axis=1, keepdims=True)
    np.savez(public / "solution.npz", all_directions=np.tile(directions, (4, 1)))
    np.savez(
        public / "refined-solution.npz",
        directions=directions,
        all_directions=np.tile(directions, (4, 1)),
        residual=np.zeros(256),
    )
    h = np.load(public / "hessians.npz", allow_pickle=False)["hessian"]
    i, j = np.triu_indices(32)
    sv = np.linalg.svd(
        (h[..., i, j] * np.where(i == j, 1, np.sqrt(2))).reshape(-1, 528),
        compute_uv=False,
    )
    m.write(
        public / "solve.json",
        {
            "subspace_rank": 128,
            "singular_128": float(sv[127]),
            "singular_129": float(sv[128]),
        },
    )
    m.write(public / "refine.json", {"directions_retained": 64})
    m.select_directions(public)
    m.write(
        public / "multiscale-solve.json",
        {
            "small_design_rank": 2048,
            "small_design_columns": 2048,
            "magnitude_design_rank": 64,
            "orientation_rank": 64,
            "magnitude_min": float(np.linalg.norm(g, axis=1).min()),
        },
    )
    np.savez(public / "multiscale-solution.npz", gates=g, couplings=c)
    fixture_report(m, public)
    m.seal(public, private)
    return root


def fixture_report(m, public):
    m.write(
        public / "learner-run.json",
        {
            "unit_fixture_not_research": True,
            "ok": True,
            "source_sha256": m.sources(),
            "solution_sha256": m.digest(public / "multiscale-solution.npz"),
            "manifest_sha256": m.digest(public / "manifest.json"),
            "audit_sha256": {n: m.digest(public / n) for n in m.audit_names(public)},
            "seconds": 1.0,
            "stages": [
                {
                    "script": name,
                    "process_monitor_scope": "process_and_children",
                    "memory_samples": 1,
                    "peak_sampled_rss_bytes": 1000,
                }
                for name in [
                    "hessian_pilot.py",
                    "refine.py",
                    "direction-selection",
                    "multiscale.py",
                ]
            ],
        },
    )


def copy_case(root, target):
    import shutil

    shutil.copytree(root, target / "case")
    return target / "case/public", target / "case/private"


def clear_selection(public):
    for name in [
        "selection-attempt.json",
        "direction-selection.json",
        "base-refine.json",
        "base-refined-solution.npz",
    ]:
        (public / name).unlink()


def test_v4_exact_public_fixture_and_seal_tampering(known_case_v4, tmp_path):
    m = module()
    public, private = copy_case(known_case_v4, tmp_path)
    result = m.verify(public, private)
    assert result["criteria_met"] is True
    # Tampering in even the skipped-rescue selection must be sealed.
    public2, private2 = copy_case(known_case_v4, tmp_path / "tamper")
    (public2 / "direction-selection.json").write_text("{}")
    with pytest.raises(ValueError, match="Audit commitment"):
        m.verify(public2, private2)
    assert (private2 / "evaluation-consumed.json").exists()


def test_actual_public_only_v4_learner(known_case_v4, tmp_path):
    m = module()
    public, private = copy_case(known_case_v4, tmp_path)
    for n in [*m.audit_names(public), "multiscale-solution.npz", "learner-run.json"]:
        (public / n).unlink()
    result = m.solve(public)
    assert result["ok"] is True
    assert [s["script"] for s in result["stages"]] == [
        "hessian_pilot.py",
        "refine.py",
        "direction-selection",
        "multiscale.py",
    ]
    assert m.read(public / "direction-selection.json")["branch"] == "skip64"
    assert not (private / "evaluation-consumed.json").exists()


def test_actual60_rescue_original_subspace_and_prefix_tamper(known_case_v4, tmp_path):
    m = module()
    public, private = copy_case(known_case_v4, tmp_path)
    clear_selection(public)
    with np.load(public / "refined-solution.npz", allow_pickle=False) as z:
        d = z["directions"][4:]
    np.savez(
        public / "refined-solution.npz",
        directions=d,
        all_directions=np.tile(d, (5, 1))[:256],
        residual=np.zeros(256),
    )
    (public / "refine.json").write_text('{"directions_retained":60}')
    report = m.select_directions(public)
    assert report["branch"] == "multi"
    assert m.selection_audit(public) == 768
    m.fit_audit(public)
    assert "rescue/round-0.npz" in m.audit_names(public)
    assert not (private / "evaluation-consumed.json").exists()
    r = m.read(public / "rescue/report.json")
    saved_geometry = r["rounds"][0]["geometry"].copy()
    r["rounds"][0]["geometry"]["component_rank"] = 0
    (public / "rescue/report.json").write_text(json.dumps(r))
    with pytest.raises(ValueError, match="geometry"):
        m.selection_audit(public)
    r["rounds"][0]["geometry"] = saved_geometry
    (public / "rescue/report.json").write_text(json.dumps(r))
    with np.load(public / "rescue/round-0.npz", allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    a["input_directions"][0] *= -1
    np.savez(public / "rescue/round-0.npz", **a)
    r = m.read(public / "rescue/report.json")
    r["rounds"][0]["artifact_sha256"] = m.digest(public / "rescue/round-0.npz")
    (public / "rescue/report.json").write_text(json.dumps(r))
    with pytest.raises(ValueError, match="prefix"):
        m.selection_audit(public)


def test_rescue_selection_rejects_unlisted_candidate_and_negative_loss():
    m = module()
    rng = np.random.default_rng(7438)
    d = rng.normal(size=(65, 32))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    base = {"directions": d[:63]}
    all_d = np.concatenate([np.tile(d[:63], (2, 1)), d[63:64], d[63:64]])
    rescue = {
        "directions": d[:64].copy(),
        "all_directions": all_d,
        "residual": np.zeros(128),
    }
    # V=Sym(32) makes every unit direction's exact rank1 residual zero.
    basis = np.eye(528)
    m.audit_rescue_selection(base, rescue, basis)
    rescue["directions"][-1] = d[64]
    with pytest.raises(ValueError, match="selection"):
        m.audit_rescue_selection(base, rescue, basis)
    rescue["directions"][-1] = d[63]
    rescue["residual"][0] = -1e-20
    with pytest.raises(ValueError, match="nonnegative"):
        m.audit_rescue_selection(base, rescue, basis)


def test_readiness_rejects_old_review_before_git_or_entropy(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "REG", tmp_path / "reg.json")
    monkeypatch.setattr(m, "sources", lambda: {"new-source.py": "b" * 64})
    review = tmp_path / "old-review.json"
    m.write(
        review,
        {
            "status": "approved_for_fresh_teacher_generation",
            "sources": {"old-source.py": "c" * 64},
            "registration_sha256": "d" * 64,
        },
    )
    ready = tmp_path / "ready.json"
    m.write(
        ready,
        {
            "ready": True,
            "sources": m.sources(),
            "registration_sha256": m.REG_SHA,
            "review_path": "old-review.json",
            "review_sha256": m.digest(review),
        },
    )
    monkeypatch.setattr(
        m.subprocess,
        "check_output",
        lambda *a, **k: pytest.fail("Old review reached Git gate"),
    )
    with pytest.raises(ValueError, match="review source"):
        m.require_readiness(ready)


def test_round_inventory_bounded_and_no_traversal(tmp_path):
    m = module()
    for rounds in [[], [{}] * 5, [{"round": "../private"}], [{"round": 1}]]:
        with pytest.raises(ValueError, match="round"):
            m.round_names({"rounds": rounds})
    assert m.round_names({"rounds": [{"round": i} for i in range(4)]}) == [
        f"rescue/round-{i}.npz" for i in range(4)
    ]


@pytest.mark.parametrize("count", [59, 65])
def test_selector_rejects_stalled_or_excess_result_without_truncating(
    tmp_path, monkeypatch, count
):
    m = module()
    direction_fixture(m, tmp_path, 60)
    monkeypatch.setattr(
        m,
        "run_json",
        lambda *a, **k: {
            "process_monitor_scope": "process_and_children",
            "retained_count": count,
        },
    )
    with pytest.raises(ValueError, match="exactly64"):
        m.select_directions(tmp_path)
    assert (tmp_path / "base-refined-solution.npz").exists()
    assert not (tmp_path / "direction-selection.json").exists()


def test_selector_budget_stop_preserves_failure_and_attempt(tmp_path, monkeypatch):
    m = module()
    direction_fixture(m, tmp_path, 60)

    def stopped(*args, **kwargs):
        raise TimeoutError("controlled process-tree budget stop")

    monkeypatch.setattr(m, "run_json", stopped)
    with pytest.raises(TimeoutError):
        m.select_directions(tmp_path)
    with pytest.raises(FileExistsError):
        m.select_directions(tmp_path)


def test_actual_four_round_audit_chain_and_rehashed_deep_tamper(tmp_path):
    m = module()
    spec = importlib.util.spec_from_file_location(
        "four_round_fixture", ROOT / "tests/glu_v4_fixture.py"
    )
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    public = tmp_path / "synthetic-public"
    report, refresh = fixture.make_case(m, public)
    assert m.selection_audit(public) == 2304
    assert len(m.audit_names(public)) == len(m.AUDIT_FILES) + 6
    geometry = report["rounds"][2]["geometry"].copy()
    report["rounds"][2]["geometry"]["residual_dimension"] = 999
    refresh()
    with pytest.raises(ValueError, match="geometry"):
        m.selection_audit(public)
    report["rounds"][2]["geometry"] = geometry
    refresh()
    round_path = public / "rescue/round-2.npz"
    with np.load(round_path, allow_pickle=False) as z:
        original = {k: z[k].copy() for k in z.files}
    for field, expected in [
        ("input_directions", "prefix"),
        ("starts", "initialization"),
        ("residual", "nonnegative"),
    ]:
        arrays = {k: v.copy() for k, v in original.items()}
        if field == "residual":
            arrays[field][0] = -1
        else:
            arrays[field][0] = arrays[field][1]
        np.savez(round_path, **arrays)
        refresh()
        with pytest.raises(ValueError, match=expected):
            m.selection_audit(public)
    np.savez(round_path, **original)
    refresh()
    (public / "rescue/round-4.npz").write_bytes(b"not an allowed fifth round")
    with pytest.raises(ValueError, match="inventory"):
        m.selection_audit(public)
