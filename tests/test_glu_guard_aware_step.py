import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def load():
    path = Path(__file__).parents[1] / "examples/isolated_glu/guard_aware_step.py"
    assert path.exists(), "Actual bounded guard-aware step runner absent"
    spec = importlib.util.spec_from_file_location("tested_guard_step", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(m, units=1):
    g = np.ones((units, 1))
    c = np.ones((units, 1, 1))
    x = np.linspace(0.2, 1.6, 8)[:, None]
    y = m.gradient.forward(g * 1.02, c, x)
    return g, c, x, y, x * 0.9, m.gradient.forward(g * 1.02, c, x * 0.9)


def test_accepted_step_full_trials_frozen_c_and_auditable_provenance(tmp_path):
    m = load()
    args = fixture(m)
    before = [v.copy() for v in args]
    candidate, report = m.run_step(tmp_path / "step", *args, rows=2)
    assert report["status"] == "completed" and report["changed"]
    assert report["accepted"] == [0] and len(report["trials"]) <= 8
    assert report["final_cost"] < report["initial_cost"]
    assert all(v["full_forward_checked"] for v in report["trials"])
    np.testing.assert_array_equal(candidate["C"], args[1])
    for original, copy in zip(args, before):
        np.testing.assert_array_equal(original, copy)
    with np.load(
        tmp_path / "step/evidence/numeric-evidence.npz", allow_pickle=False
    ) as arrays:
        np.testing.assert_allclose(
            arrays["weighted_target"], args[3] * arrays["weights"][:, None]
        )
        np.testing.assert_allclose(
            arrays["final_train_residual"],
            arrays["weighted_target"]
            - m.gradient.forward(candidate["G"], candidate["C"], args[2])
            * arrays["weights"][:, None],
        )
    assert (
        json.loads((tmp_path / "step/proposals/screen-0.json").read_text())["normals"][
            "logical_plane_count"
        ]
        == 8
    )


def test_equal_scores_keep_unit_order_and_nochange_has_no_candidate(tmp_path):
    m = load()
    args = fixture(m, units=2)
    _, report = m.run_step(tmp_path / "ties", *args, rows=2)
    assert report["ordered"] == [0, 1] and report["selected"] == [0, 1]
    args = list(fixture(m))
    args[3] = m.gradient.forward(args[0], args[1], args[2])
    candidate, report = m.run_step(tmp_path / "nochange", *args, rows=2)
    assert not report["changed"] and not report["selected"] and candidate is None


def test_geometry_failure_preserves_initial_problem_and_blocks_output(tmp_path):
    m = load()
    args = list(fixture(m))
    args[5] = m.gradient.forward(args[0], args[1], args[4])
    with pytest.raises(m.StepError):
        m.run_step(tmp_path / "failed", *args, rows=2)
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "failed" and report["stage"] == "guard_normals"
    assert (tmp_path / "failed/initial-problem.npz").exists()
    assert not (tmp_path / "failed/candidate.npz").exists()
    with pytest.raises(FileExistsError):
        m.run_step(tmp_path / "failed", *args, rows=2)


def test_invalid_profile_and_singular_gram_fail_without_candidate(
    tmp_path, monkeypatch
):
    m = load()
    args = fixture(m)
    monkeypatch.setattr(m.gradient, "gate_gram", lambda *args: np.zeros((1, 1)))
    with pytest.raises(m.StepError):
        m.run_step(tmp_path / "singular", *args, rows=2)
    report = json.loads((tmp_path / "singular/report.json").read_text())
    assert report["status"] == "failed" and report["stage"] == "cone"
    assert (
        json.loads(
            (tmp_path / "singular/cones/proposals/screen-0/trace.json").read_text()
        )["terminal_status"]
        == "failed"
    )


def test_corrupt_cached_trial_stops_before_retention(tmp_path, monkeypatch):
    m = load()
    original = m.cached_trial
    monkeypatch.setattr(m, "cached_trial", lambda *a: original(*a) + 1)
    with pytest.raises(m.StepError):
        m.run_step(tmp_path / "corrupt", *fixture(m), rows=2)
    report = json.loads((tmp_path / "corrupt/report.json").read_text())
    assert report["status"] == "failed" and not report["accepted"]
    assert report["trials"][-1]["status"] == "started"
    assert not (tmp_path / "corrupt/candidate.npz").exists()
    assert (tmp_path / "corrupt/derived-initial.npz").exists()
    with np.load(
        tmp_path / "corrupt/trials/0-predictions.npz", allow_pickle=False
    ) as arrays:
        assert (
            np.max(
                np.abs(
                    arrays["cached_train_prediction"] - arrays["full_train_prediction"]
                )
            )
            > 0.9
        )
    assert (
        json.loads((tmp_path / "corrupt/artifact-manifest.json").read_text())["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "guard_offset, accepted, trials", [(0.0002, True, 8), (0.00005, False, 8)]
)
def test_real_nonlinear_last_alpha_and_all_rejected(
    tmp_path, guard_offset, accepted, trials
):
    m = load()
    args = list(fixture(m))
    args[3] = m.gradient.forward(args[0] * 1.04, args[1], args[2])
    args[5] = m.gradient.forward(args[0] * (1 + guard_offset), args[1], args[4])
    candidate, report = m.run_step(tmp_path / "step", *args, rows=2)
    assert report["status"] == "completed" and report["changed"] is accepted
    assert len(report["trials"]) == trials
    assert all(v["training_decreased"] for v in report["trials"])
    assert not any(v["accepted"] for v in report["trials"][:-1])
    assert report["trials"][-1]["accepted"] is accepted
    assert (candidate is not None) is accepted
    if accepted:
        assert report["trials"][-1]["alpha"] == 1 / 128


def test_second_proposal_uses_changed_full_state_and_fixed_initial_bounds(
    tmp_path, monkeypatch
):
    m = load()
    calls = []
    original = m.previous.guard_ok

    def capture(nr, na, cr, ca, ir, ia):
        calls.append((list(cr), list(ca), list(ir), list(ia)))
        return original(nr, na, cr, ca, ir, ia)

    monkeypatch.setattr(m.previous, "guard_ok", capture)
    args = fixture(m, 2)
    candidate, report = m.run_step(tmp_path / "step", *args, rows=2)
    assert report["accepted"][0] == 0
    with np.load(
        tmp_path / "step/proposals/sequential-1.npz", allow_pickle=False
    ) as arrays:
        assert arrays["current_G"][0, 0] != args[0][0, 0]
        np.testing.assert_array_equal(
            arrays["train_prediction"],
            m.gradient.forward(arrays["current_G"], args[1], args[2]),
        )
    assert all(
        ir == report["initial_guard_relative"]
        and ia == report["initial_guard_max_absolute"]
        for _, _, ir, ia in calls
    )
    assert any(cr != report["initial_guard_relative"] for cr, _, _, _ in calls)
    assert candidate is not None


def test_late_guard_group_failure_keeps_earlier_normals(tmp_path, monkeypatch):
    m = load()
    original = m.metric.metric_group

    def fail(*args):
        if args[-2] == 2:
            raise ValueError("injected late group failure")
        return original(*args)

    monkeypatch.setattr(m.metric, "metric_group", fail)
    with pytest.raises(m.StepError):
        m.run_step(tmp_path / "step", *fixture(m), rows=2)
    p = json.loads((tmp_path / "step/proposals/screen-0.json").read_text())
    assert p["active_guard_group"] == 2 and len(p["normals"]["groups"]) == 2
    assert p["normals"]["logical_plane_count"] == 4
    assert not (tmp_path / "step/candidate.npz").exists()


def test_manifest_failure_dominates_already_written_candidate(tmp_path, monkeypatch):
    m = load()

    def fail(self):
        raise OSError("injected final inventory write failure")

    monkeypatch.setattr(m.Study, "inventory", fail)
    with pytest.raises(m.StepError):
        m.run_step(tmp_path / "step", *fixture(m), rows=2)
    assert (tmp_path / "step/candidate.npz").exists()
    report = json.loads((tmp_path / "step/report.json").read_text())
    assert report["status"] == "failed" and report["changed"] is False


def test_journal_failure_blocks_retention_and_preserves_report(tmp_path, monkeypatch):
    m = load()
    original = m.Study.event

    def fail(self, kind, **payload):
        if kind == "trial_completed":
            raise OSError("injected journal failure")
        return original(self, kind, **payload)

    monkeypatch.setattr(m.Study, "event", fail)
    with pytest.raises(m.StepError):
        m.run_step(tmp_path / "step", *fixture(m), rows=2)
    report = json.loads((tmp_path / "step/report.json").read_text())
    assert report["status"] == "failed" and not report["accepted"]
    assert (tmp_path / "step/trials/0.npz").exists()
