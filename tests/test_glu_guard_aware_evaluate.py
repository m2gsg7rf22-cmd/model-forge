import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def load():
    p = Path(__file__).parents[1] / "examples/isolated_glu/guard_aware_evaluate.py"
    assert p.exists(), "Distinct paired evaluation source absent"
    spec = importlib.util.spec_from_file_location("paired_evaluation", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_paired_points_counts_references_saved_before_prediction(tmp_path):
    m = load()
    seen = []

    def query(x):
        seen.append(x.copy())
        return x.copy()

    def student(x):
        group = len(seen) - 1
        assert (tmp_path / "paired" / f"group-{group}-reference.npz").exists()
        return x * 1.1

    report = m.evaluate_batches(
        tmp_path / "paired",
        query,
        student,
        lambda x: x * 1.2,
        seed=19,
        rows=3,
        dimension=2,
    )
    assert report["status"] == "completed"
    assert [
        report[k]
        for k in [
            "calls_reserved",
            "calls_attempted",
            "calls_returned",
            "calls_persisted",
        ]
    ] == [12] * 4
    assert all(
        v["relative_difference"] < 0 and v["max_absolute_difference"] < 0
        for v in report["groups"]
    )
    with np.load(
        tmp_path / "paired/paired-validation.npz", allow_pickle=False
    ) as arrays:
        np.testing.assert_array_equal(arrays["inputs"], np.vstack(seen))
        np.testing.assert_allclose(arrays["prediction"], arrays["inputs"] * 1.1)
        np.testing.assert_allclose(
            arrays["baseline_prediction"], arrays["inputs"] * 1.2
        )


def test_relative_improvement_does_not_hide_absolute_regression(tmp_path):
    m = load()
    truth = np.ones((2, 2))
    student = truth + np.array([[0.15, 0.0], [0.0, 0.0]])
    report = m.evaluate_batches(
        tmp_path / "paired",
        lambda x: truth.copy(),
        lambda x: student.copy(),
        lambda x: truth + 0.1,
        seed=20,
        rows=2,
        dimension=2,
    )
    assert all(
        v["relative_difference"] < 0 and v["max_absolute_regressed"]
        for v in report["groups"]
    )
    assert report["all_groups_relative_at_most_1e_minus3"] is False


@pytest.mark.parametrize(
    "prediction, passes", [(1001.0, True), (np.nextafter(1001.0, np.inf), False)]
)
def test_fixed_quality_threshold_boundary(tmp_path, prediction, passes):
    m = load()
    report = m.evaluate_batches(
        tmp_path / "paired",
        lambda x: np.full_like(x, 1000.0),
        lambda x: np.full_like(x, prediction),
        lambda x: np.full_like(x, 1002.0),
        seed=21,
        rows=2,
        dimension=2,
    )
    assert report["all_groups_relative_at_most_1e_minus3"] is passes


def test_failed_second_query_preserves_first_group_and_attempt_intent(tmp_path):
    m = load()
    calls = []

    def query(x):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected failed query")
        return x.copy()

    with pytest.raises(m.EvaluationError):
        m.evaluate_batches(
            tmp_path / "paired",
            query,
            lambda x: x,
            lambda x: x,
            seed=22,
            rows=3,
            dimension=2,
        )
    r = json.loads((tmp_path / "paired/progress.json").read_text())
    assert r["status"] == "failed" and r["calls_attempted"] == 6
    assert r["calls_returned"] == r["calls_persisted"] == 3
    assert len(r["groups"]) == 1
    assert (tmp_path / "paired/group-1-inputs.npz").exists()
    assert not (tmp_path / "paired/paired-validation.npz").exists()
    with pytest.raises(FileExistsError):
        m.evaluate_batches(
            tmp_path / "paired",
            query,
            lambda x: x,
            lambda x: x,
            seed=22,
            rows=3,
            dimension=2,
        )


def test_predictor_failure_never_discards_consumed_reference(tmp_path):
    m = load()

    def fail(x):
        raise ValueError("injected predictor failure")

    with pytest.raises(m.EvaluationError):
        m.evaluate_batches(
            tmp_path / "paired",
            lambda x: x + 1,
            fail,
            lambda x: x,
            seed=23,
            rows=3,
            dimension=2,
        )
    r = json.loads((tmp_path / "paired/progress.json").read_text())
    assert r["status"] == "failed" and r["calls_returned"] == r["calls_persisted"] == 3
    assert (tmp_path / "paired/group-0-reference.npz").exists()
    assert not (tmp_path / "paired/paired-validation.npz").exists()


def test_nonzero_error_norm_underflow_is_failure():
    m = load()
    with pytest.raises(ValueError, match="underflow"):
        m.metrics(np.array([[1.0, 1e-200]]), np.array([[1.0, 0.0]]))


@pytest.mark.parametrize("mutator", ["query", "baseline", "student"])
def test_callback_input_mutation_fails_with_original_input_evidence(tmp_path, mutator):
    m = load()

    def mutate(x):
        x.setflags(write=True)
        x[:] = 7.0
        return x.copy()

    callbacks = {
        name: (mutate if name == mutator else lambda x: x.copy())
        for name in ["query", "baseline", "student"]
    }
    with pytest.raises(m.EvaluationError, match="input mutation"):
        m.evaluate_batches(
            tmp_path / "paired",
            callbacks["query"],
            callbacks["student"],
            callbacks["baseline"],
            seed=26,
            rows=3,
            dimension=2,
        )
    r = json.loads((tmp_path / "paired/progress.json").read_text())
    assert r["status"] == "failed"
    assert (tmp_path / "paired/group-0-reference.npz").exists()
    assert not (tmp_path / "paired/paired-validation.npz").exists()


def test_unreadable_progress_preserves_original_error_and_input_after(
    tmp_path, monkeypatch
):
    m = load()
    fit = tmp_path / "fakefit"
    fit.mkdir()
    paths = {name: fit / name for name in m.INPUT_NAMES}
    for p in paths.values():
        p.write_text("{}")
    before = {name: m.sha(p) for name, p in paths.items()}
    monkeypatch.setattr(m, "validate_fit", lambda *a: (before, {}, {}))
    token = tmp_path / "synthetic-marker-26.json"
    token.write_text("{}")
    monkeypatch.setattr(m, "reserve_once", lambda *a, **k: token)

    def fail_factory():
        batches = tmp_path / "worker/batches"
        batches.mkdir()
        (batches / "progress.json").write_text("corrupted")
        raise RuntimeError("original synthetic factory failure")

    monkeypatch.setattr(m, "public_query", fail_factory)
    with pytest.raises(RuntimeError, match="original synthetic factory failure"):
        m.build(
            paths,
            tmp_path / "worker",
            {"inputs": before},
            {"process_monitor_scope": "process_and_children", "memory_samples": 2},
        )
    report = json.loads((tmp_path / "worker/worker-report.json").read_text())
    assert report["status"] == "failed" and report["inputs_unchanged"] is True
    assert report["calls_attempted"] is None and report["calls_returned"] is None
    assert report["progress_check_error"]


def test_predictors_reusing_buffers_cannot_change_earlier_saved_groups(tmp_path):
    m = load()
    shared_student = np.empty((3, 2))
    shared_baseline = np.empty((3, 2))

    def student(x):
        shared_student[:] = x * 1.1
        return shared_student

    def baseline(x):
        shared_baseline[:] = x * 1.2
        return shared_baseline

    m.evaluate_batches(
        tmp_path / "paired",
        lambda x: x.copy(),
        student,
        baseline,
        seed=29,
        rows=3,
        dimension=2,
    )
    with np.load(
        tmp_path / "paired/paired-validation.npz", allow_pickle=False
    ) as all_arrays:
        for group in range(4):
            for name, field in [
                ("student", "prediction"),
                ("baseline", "baseline_prediction"),
            ]:
                with np.load(
                    tmp_path / "paired" / f"group-{group}-{name}.npz",
                    allow_pickle=False,
                ) as saved:
                    np.testing.assert_array_equal(
                        saved["prediction"],
                        all_arrays[field][group * 3 : (group + 1) * 3],
                    )


def test_persistent_marker_blocks_new_output_and_partial_serialization(tmp_path):
    m = load()
    root = tmp_path / "synthetic-fit"
    root.mkdir()
    marker = m.reserve_once(root, {"calls_reserved": 12}, seed=31)
    with pytest.raises(FileExistsError):
        m.reserve_once(
            root, {"calls_reserved": 12, "different_output": "other"}, seed=31
        )
    assert marker.exists()
    with pytest.raises(TypeError):
        m.reserve_once(root, {"bad": object()}, seed=32)
    assert (root / "public32-evaluation-attempt.json").exists()
    with pytest.raises(FileExistsError):
        m.reserve_once(root, {}, seed=32)


def test_failed_reservation_blocks_factory_and_does_not_claim_zero_reserved(
    tmp_path, monkeypatch
):
    m = load()
    paths = {name: tmp_path / name for name in m.INPUT_NAMES}
    for p in paths.values():
        p.write_text("{}")
    before = {name: m.sha(p) for name, p in paths.items()}
    monkeypatch.setattr(m, "validate_fit", lambda *a: (before, {}, {}))

    def exists(*args, **kwargs):
        raise FileExistsError("Synthetic earlier marker")

    monkeypatch.setattr(m, "reserve_once", exists)

    def forbidden():
        pytest.fail("No target factory after failed reservation")

    monkeypatch.setattr(m, "public_query", forbidden)
    with pytest.raises(FileExistsError):
        m.build(
            paths,
            tmp_path / "worker",
            {"inputs": before},
            {"process_monitor_scope": "process_and_children", "memory_samples": 2},
        )
    r = json.loads((tmp_path / "worker/worker-report.json").read_text())
    assert r["status"] == "failed" and r["calls_reserved"] is None
    assert r["calls_attempted"] == r["calls_returned"] == r["calls_persisted"] == 0
