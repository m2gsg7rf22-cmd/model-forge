"""Test verifier integrity using the already public development teacher only."""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "examples/isolated_glu/sealed.py"
    assert path.exists(), "Missing separated sealed-experiment wrapper"
    spec = importlib.util.spec_from_file_location("sealed_glu", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_unreviewed_runner_cannot_generate_teacher(tmp_path: Path) -> None:
    m = module()
    with pytest.raises(ValueError, match="readiness"):
        m.prepare(
            tmp_path / "public",
            tmp_path / "private",
            0,
            readiness=tmp_path / "missing.json",
        )
    assert not (tmp_path / "private").exists()


def test_repeat_verifier_refuses_before_read(tmp_path: Path, monkeypatch) -> None:
    m = module()
    private = tmp_path / "private"
    private.mkdir()
    (private / "evaluation-consumed.json").write_text("{}")

    def forbidden(*args, **kwargs):
        raise AssertionError("Repeated verification read a file")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(FileExistsError):
        m.verify(tmp_path / "public", private)


def test_nonfinite_json_and_zero_denominators_rejected(tmp_path: Path) -> None:
    m = module()
    p = tmp_path / "bad.json"
    p.write_text('{"result":NaN}')
    with pytest.raises(ValueError):
        m.read(p)
    with pytest.raises(ValueError):
        m.relative(np.zeros((3, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        m.relative(np.ones((3, 2)), np.full((3, 2), float("nan")))


def test_summary_needs_three_distinct_successes() -> None:
    m = module()
    with pytest.raises(ValueError):
        m.aggregate([{"index": 0, "criteria_met": True}] * 3)
    assert (
        m.aggregate([{"index": i, "criteria_met": False} for i in range(3)])[
            "criteria_met"
        ]
        is False
    )


def test_summary_does_not_trust_true_flag_without_metrics() -> None:
    m = module()
    rows = [
        {
            "index": i,
            "criteria_met": True,
            "process_monitor_scope": "process_and_children",
        }
        for i in range(3)
    ]
    assert m.aggregate(rows)["criteria_met"] is False


@pytest.fixture(scope="module")
def known_case(tmp_path_factory):
    m = module()
    root = tmp_path_factory.mktemp("public-development-verifier-fixture")
    public, private = root / "public", root / "private"
    rng = np.random.default_rng(572041)
    teacher = {
        "gate": rng.normal(size=(64, 32)) * 0.2,
        "up": rng.normal(size=(64, 32)) * 0.2,
        "down": rng.normal(size=(32, 64)) * 0.2,
    }
    # Public test inputs and the previously public teacher, never fresh secret teachers.
    rng = np.random.default_rng(73191)
    held = np.stack([rng.normal(size=(512, 32)) * scale for scale in [0.04, 1, 4, 30]])
    m.materialize(public, private, 1, teacher, held)
    g = teacher["gate"]
    c = np.einsum("ow,wd->wod", teacher["down"], teacher["up"])
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
    weights = np.where(i == j, 1.0, np.sqrt(2.0))
    sv = np.linalg.svd(h[:, :, i, j].reshape(-1, len(i)) * weights, compute_uv=False)
    m.write(
        public / "solve.json",
        {
            "subspace_rank": 128,
            "singular_128": float(sv[127]),
            "singular_129": float(sv[128]),
        },
    )
    m.write(public / "refine.json", {"directions_retained": 64})
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
    # Controlled verifier fixtures do not represent measured research execution.
    m.write(
        public / "learner-run.json",
        {
            "unit_fixture_not_research": True,
            "ok": True,
            "source_sha256": m.sources(),
            "solution_sha256": m.digest(public / "multiscale-solution.npz"),
            "manifest_sha256": m.digest(public / "manifest.json"),
            "audit_sha256": {n: m.digest(public / n) for n in m.AUDIT_FILES},
            "seconds": 1.0,
            "stages": [
                {
                    "script": name,
                    "process_monitor_scope": "process_and_children",
                    "memory_samples": 1,
                    "peak_sampled_rss_bytes": 1000,
                }
                for name in ["hessian_pilot.py", "refine.py", "multiscale.py"]
            ],
        },
    )
    m.seal(public, private)
    return root


def copy_case(root, tmp_path):
    import shutil

    target = tmp_path / "case"
    shutil.copytree(root, target)
    return target / "public", target / "private"


def test_verifier_accepts_exact_known_teacher_without_fitting(known_case, tmp_path):
    m = module()
    public, private = copy_case(known_case, tmp_path)
    report = m.verify(public, private)
    assert report["criteria_met"] is True
    assert max(x["relative_frobenius_error"] for x in report["validation"]) < 1e-12
    assert report["orientations_correct"] == 64
    assert (private / "evaluation-consumed.json").exists()


def test_verifier_rejects_changed_audit_after_seal(known_case, tmp_path):
    m = module()
    public, private = copy_case(known_case, tmp_path)
    p = public / "refine.json"
    p.write_text('{"directions_retained":64,"tampered":true}')
    with pytest.raises(ValueError, match="Audit commitment"):
        m.verify(public, private)
    assert (private / "evaluation-consumed.json").exists()


def test_verifier_rejects_nonfinite_student_even_with_updated_hashes(
    known_case, tmp_path
):
    m = module()
    public, private = copy_case(known_case, tmp_path)
    sol = np.load(public / "multiscale-solution.npz", allow_pickle=False)
    g, c = sol["gates"], sol["couplings"]
    g[0, 0] = float("nan")
    np.savez(public / "multiscale-solution.npz", gates=g, couplings=c)
    report = m.read(public / "learner-run.json")
    report["solution_sha256"] = m.digest(public / "multiscale-solution.npz")
    (public / "learner-run.json").write_text(json.dumps(report))
    fixed = m.read(private / "seal.json")
    fixed["solution_sha256"] = report["solution_sha256"]
    fixed["learner_sha256"] = m.digest(public / "learner-run.json")
    (private / "seal.json").write_text(json.dumps(fixed))
    with pytest.raises(ValueError, match="finite"):
        m.verify(public, private)
    assert (private / "evaluation-consumed.json").exists()


def test_all_worker_failures_still_persist_negative_summary(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "require_readiness", lambda _: None)

    def fail(*args, **kwargs):
        raise RuntimeError("controlled worker startup failure")

    monkeypatch.setattr(m, "run_json", fail)
    output = tmp_path / "negative-run"
    result = m.run(output)
    assert result["criteria_met"] is False
    assert len(result["results"]) == 3
    assert (
        json.loads((output / "public/report.json").read_text())["criteria_met"] is False
    )


def test_actual_public_only_learner_contract(known_case, tmp_path):
    m = module()
    public, private = copy_case(known_case, tmp_path)
    for name in [*m.AUDIT_FILES, "multiscale-solution.npz", "learner-run.json"]:
        (public / name).unlink()
    result = m.solve(public)
    assert result["ok"] is True
    assert result["heldout_read"] is False
    assert [x["script"] for x in result["stages"]] == [
        "hessian_pilot.py",
        "refine.py",
        "multiscale.py",
    ]
    assert all(x["memory_samples"] > 0 for x in result["stages"])
    # Solver receives public only and cannot create a private consumption marker.
    assert not (private / "evaluation-consumed.json").exists()
